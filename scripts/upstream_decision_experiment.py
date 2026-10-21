"""Three bounded A1 arms, invoked through run_upstream_t11.py.

Preparation/description never releases the training pause. Each arm requires
the launcher's explicit release flag, reserves whole policy decisions, and
keeps old signal100k admission and checkpoints separate.
"""
from copy import copy
import gc
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import torch

from _common import ROOT
from diagnose_upstream_environment import event_data
from planner_evidence import file_sha256, json_write, source_snapshot
from upstream_checkpoint import read_checkpoint, save_checkpoint
from upstream_run_state import save_state
from upstream_process_rollout import ProcessUpstreamRollout
from upstream_decision_evaluation import evaluate_panels, compare_results
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_decision import LearningTimeConfig, remaining_decisions
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_training import TrainingOptions
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_skrl import SkrlUpstreamTrainer

ARMS = {'A': (LearningTimeConfig(1., .2), 125),
        'B': (LearningTimeConfig(5., .2), 25),
        'C': (LearningTimeConfig(5., 1.), 25)}
BASE_BUDGET = 200000
EVALUATION_BUDGET_PER_ARM = 6000


def specification():
    return {'version':'t11_decision_credit_experiment_v1','representation':'A1','seed':23,
        'owners':4,'envs_per_owner':4,'base_steps_per_arm':BASE_BUDGET,
        'evaluation_base_step_reservation_per_arm':EVALUATION_BUDGET_PER_ARM,
        'evaluation_base_step_reservation_suite':3*EVALUATION_BUDGET_PER_ARM,
        'arms':{name:{'learning_time':timing.metadata(.2),'rollout_decisions':length}
                for name,(timing,length) in ARMS.items()},
        'training_paused_until_explicit_release':True}


def parameter_digest(policy):
    digest=hashlib.sha256()
    for name,value in policy.state_dict().items():
        if name.startswith('value_normalizer.'):
            continue  # Initialization comparison covers identical network weights.
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def initialize_exploration(policy,scale):
    """Set the fresh policy distribution itself, never rescale sampled actions."""
    if scale not in (1.,.25):
        raise ValueError('Exploration candidate must be the original or quarter initial standard deviation')
    if scale!=1.:
        with torch.no_grad():
            policy.log_std.add_(math.log(scale))


def validate_request(args):
    if args.decision_credit_arm not in (*ARMS, 'all'):
        raise ValueError('Select decision-credit arm A/B/C/all')
    if args.initial_continuous_std_scale not in (1.,.25):
        raise ValueError('Unknown initial exploration candidate')
    if (args.representation!='A1' or args.seed!=23 or args.budget!=BASE_BUDGET
            or args.samples!=512 or args.env_owners!=4 or args.num_envs!=16
            or args.owner_cpu_workers!=2 or args.trainer_backend!='skrl'
            or args.policy_device!='cpu' or args.gpu_service):
        raise ValueError('Decision-credit scope is A1/seed23/K512/4x4/CPU learner/200000 base steps')
    if args.import_checkpoint or args.import_learner or args.authorize_signal100k:
        raise ValueError('Decision-credit arms start fresh; old learner/signal imports are forbidden')
    if args.resume_state and args.decision_credit_arm=='all':
        raise ValueError('Resume one explicit arm at its own saved boundary')
    if not args.run_id or Path(args.run_id).name!=args.run_id or args.run_id in ('.','..'):
        raise ValueError('Expected a new run directory name')


def _evaluation(path, phase, collector, args, *, final=False, repair_validation=False):
    result_path=path/'metrics'/f'evaluation_{phase}.json'
    if result_path.exists():
        return json.loads(result_path.read_text())
    reserved_path=path/'metrics'/f'evaluation_{phase}_reservation.json'
    if reserved_path.exists():
        raise ValueError('Interrupted evaluation reservation retained; no automatic budget-consuming retry')
    reserve=2800 if final and not repair_validation else 1600
    previous=sum(json.loads(p.read_text())['reserved_base_steps']
                 for p in (path/'metrics').glob('evaluation_*_reservation.json'))
    if previous+reserve>(4000 if repair_validation else EVALUATION_BUDGET_PER_ARM):
        raise ValueError('Evaluation budget exhausted')
    json_write(reserved_path,{'phase':phase,'reserved_base_steps':reserve,
        'training_base_steps':collector.base_steps,'optimizer_updates':0})
    evaluator=evaluate_panels
    if repair_validation:
        from upstream_learning_cohort import evaluate_panels as evaluator
    results=evaluator(args.config,args.far_checkpoint,collector.policy,collector.learning_time,final=final)
    actual=sum(row['actual_base_steps'] for row in results.values())
    if actual>reserve:
        raise AssertionError('Evaluation exceeded its reservation')
    result={'phase':phase,'training_base_steps':collector.base_steps,'actual_base_steps':actual,
        'reserved_base_steps':reserve,'results':results}
    json_write(result_path,event_data(result))
    return event_data(result)


def run_arm(args, *, repair_validation=False):
    timing, length=ARMS[args.decision_credit_arm]
    budget=20000 if repair_validation else BASE_BUDGET
    if repair_validation:
        length=4
    scope='learning_repair_validation' if repair_validation else 'decision_credit'
    path=ROOT/'outputs/runs/upstream_scale_t11'/args.run_id
    g,l=timing.discounts(.2)
    options=TrainingOptions(rollout_macro_steps=length,gamma=g,lam=l,gradient_clipping=args.gradient_clipping,
                            terminal_actor_credit=args.terminal_actor_credit,value_normalization=args.value_normalization)
    spec=specification()
    if repair_validation:
        from upstream_learning_validation import specification as repair_specification
        spec=repair_specification(value_normalization=args.value_normalization)
    identity={'version':spec['version'],'arm':args.decision_credit_arm,
        'specification':spec,'options':options.metadata,'source_snapshot':source_snapshot(),
        'base_config':str(Path(args.config).resolve()),'base_config_sha256':file_sha256(args.config),
        'far_checkpoint':str(Path(args.far_checkpoint).resolve()),'far_checkpoint_sha256':file_sha256(args.far_checkpoint)}
    if args.initial_continuous_std_scale!=1.:
        identity['initial_continuous_std_scale']=args.initial_continuous_std_scale
    if repair_validation and args.comparison_run:
        reference=Path(args.comparison_run).resolve()
        reference_identity=json.loads((reference/'run_manifest.json').read_text())['identity']
        if {k:v for k,v in options.metadata.items() if k!='value_normalization'}!=reference_identity['options']:
            raise ValueError('Comparison changes optimizer options beyond value normalization')
    if path.exists():
        if not args.resume_state or json.loads((path/'run_manifest.json').read_text())['identity']!=identity:
            raise ValueError('Run exists or resume identity changed')
    else:
        if args.resume_state:
            raise ValueError('Cannot resume into a new run')
        for name in ('config','metrics','logs','state','checkpoints'):
            (path/name).mkdir(parents=True,exist_ok=True)
        json_write(path/'run_manifest.json',{'identity':identity,'command':sys.argv,
            'formal_training':False,'bounded_research_learning':True,'system_acceptance_complete':False})
    run=SimpleNamespace(path=path)
    collector=None;trainer=None;updates=0;start=time.perf_counter();last_checkpoint=None;phase='initializing'
    try:
        collector=ProcessUpstreamRollout(args.config,'A1',23,owners=4,envs_per_owner=4,
            samples=512,mode='signal',cpu_workers=2,device='cpu',learning_time=timing,
            cache_dir=path/'state/solver_cache',initial_cohort='near_far_balanced_v1' if repair_validation else None)
        trainer=SkrlUpstreamTrainer(collector,options)
        config=path/'config/effective_configuration.json'
        if not config.exists():
            json_write(config,{'environment':json.loads(collector.env.cfg.configuration_json),
                'learning_time':timing.metadata(collector.env.spec.grid.dt),'options':options.metadata,
                'initial_continuous_std_scale':args.initial_continuous_std_scale})
        if args.resume_state:
            checkpoint=Path(args.resume_state).resolve()
            if not checkpoint.is_relative_to((path/'checkpoints').resolve()):
                raise ValueError('Checkpoint must belong to this arm')
            saved=read_checkpoint(checkpoint)
            if saved['identity']!=identity:
                raise ValueError('Cross-arm checkpoint restore is forbidden')
            trainer.load_state_dict(saved['trainer']);updates=saved['session_updates']
            last_checkpoint=str(checkpoint)
            del saved
        else:
            unscaled_digest=parameter_digest(collector.policy)
            if repair_validation and args.comparison_run:
                reference_digest=json.loads((reference/'metrics/initialization.json').read_text())['before_exploration_scale_sha256']
                if unscaled_digest!=reference_digest:
                    raise ValueError('Initial network weights differ from the reference combo')
            initialize_exploration(collector.policy,args.initial_continuous_std_scale)
            json_write(path/'metrics/initialization.json',{'seed':23,
                'actor_critic_sha256':parameter_digest(collector.policy),'optimizer':'fresh_adam',
                'before_exploration_scale_sha256':unscaled_digest,
                'inherited_checkpoint':None,'initial_continuous_std_scale':args.initial_continuous_std_scale})
            checkpoint=path/'checkpoints/update_000000.pt'
            save_checkpoint(trainer,identity,0,checkpoint,scope=scope)
            last_checkpoint=str(checkpoint)
        # Never silently replay already-spent but uncommitted sampling, or an
        # older checkpoint over newer results. Recovery needs reconciliation.
        later=[p.name for p in (path/'metrics').glob('collection_*.json')
               if int(p.name.split('_')[1])>updates]
        if later:
            raise ValueError('Uncommitted sampling evidence exists; refuse automatic budget reuse: '+later[0])
        if any(json.loads(p.read_text()).get('phase')=='sampling'
               for p in (path/'metrics').glob('failure_*.json')):
            raise ValueError('Interrupted sampling has an uncertain physical-step count; reconcile before resuming')
        existing_summary=path/'metrics'/f'summary_{updates:06d}.json'
        if existing_summary.exists() and json.loads(existing_summary.read_text())['budget_complete']:
            return json.loads(existing_summary.read_text()),json.loads((path/'metrics/evaluation_final.json').read_text())
        phase='evaluation'
        initial_evaluation=_evaluation(path,'initial',collector,args,repair_validation=repair_validation)
        if repair_validation and args.comparison_run:
            from upstream_learning_validation import check_initial_reference
            check_initial_reference(initial_evaluation['results'],reference)
        audit={'schema_version':'t11_learning_repair_validation_v1' if repair_validation else 't11_decision_credit_readiness_v1','scope':scope,
            'maximum_base_steps':budget,'profile_fingerprint':collector.env.spec.fingerprint,
            'learning_time':timing.metadata(collector.env.spec.grid.dt),'formal_training_ready':False}
        initial_updates=updates
        decision_steps=timing.steps(collector.env.spec.grid.dt)
        while True:
            count=remaining_decisions(budget,collector.base_steps,16,decision_steps,length)
            if not count:
                break
            started=time.perf_counter()
            def on_chunk(batch, episodes, records, remaining):
                # Preserve sampling progress even if a later chunk or PPO fails.
                json_write(path/'metrics'/f'collection_{updates+1:06d}_{count-remaining:03d}.json',
                    event_data({'actual_base_steps':collector.base_steps,'policy_decisions':collector.decisions,
                        'valid_transitions':collector.transitions,'episodes':episodes,
                        'finite_observations':bool(torch.isfinite(batch['obs']).all()),
                        'finite_log_prob':all(bool(torch.isfinite(s.log_prob).all()) for s in batch['samples'])}))
            phase='sampling'
            batch,episodes,records=trainer.collect(count,on_chunk=on_chunk)
            phase='updating'
            if (not torch.isfinite(batch['obs']).all() or not torch.isfinite(batch['reward']).all()
                    or not batch['learning_valid'][batch['duration']>0].all()
                    or not all(torch.isfinite(s.log_prob).all() for s in batch['samples'])):
                save_state(run,f'invalid_batch_{updates+1:06d}',{'batch':batch,'execution':records})
                raise ValueError('Nonfinite or invalid training data; arm stopped')
            collect_seconds=time.perf_counter()-started
            # Archive the original on-policy evidence before changing the weights.
            save_state(run,f'interactions_{updates+1:06d}',{'samples':batch['samples'],'execution':records,
                'learning':{k:batch[k].cpu() for k in ('obs','state','hidden','reward','value','next_value','duration','action_effective',
                    'terminated','truncated','bootstrap_allowed','learning_valid','terminal_reason')},
                'update_rng':torch.get_rng_state().clone()})
            holds=torch.stack([s.action.hold for s in batch['samples']]).float()
            duration=batch['duration']
            hold_fraction=float((holds.mean(-1)*duration).sum()/duration.sum())
            started=time.perf_counter()
            result=trainer.update(batch,readiness=audit,training_authorized=True)
            updates+=1
            json_write(path/'metrics'/f'update_{updates:06d}.json',event_data({
                'actual_base_steps':collector.base_steps,'policy_decisions':collector.decisions,
                'valid_transitions':collector.transitions,'sampling_updates':updates,'ppo_epochs':trainer.updates,
                'hold_fraction':hold_fraction,'episodes':episodes,'learning':result,
                'ineffective_terminal_decisions':int(((~batch['action_effective']) & (duration>0)).sum()),
                'collection_seconds':collect_seconds,'update_seconds':time.perf_counter()-started,
                'wall_seconds':time.perf_counter()-start}))
            del batch,episodes,records
            trainer.release_rollout();gc.collect()
            phase='checkpoint'
            checkpoint=path/'checkpoints'/f'update_{updates:06d}.pt'
            save_checkpoint(trainer,identity,updates,checkpoint,scope=scope)
            last_checkpoint=str(checkpoint)
            print(json.dumps({'arm':args.decision_credit_arm,'updates':updates,'base_steps':collector.base_steps}),flush=True)
            if not repair_validation and collector.base_steps>=100000:
                phase='evaluation'
                _evaluation(path,'midpoint',collector,args)
            if args.updates_this_invocation is not None and updates-initial_updates>=args.updates_this_invocation:
                break
        complete=remaining_decisions(budget,collector.base_steps,16,decision_steps,1)==0
        phase='evaluation'
        final=_evaluation(path,'final',collector,args,final=True,repair_validation=repair_validation) if complete else None
        summary={'arm':args.decision_credit_arm,'budget_complete':complete,'actual_base_steps':collector.base_steps,
            'unused_base_steps':budget-collector.base_steps,'policy_decisions':collector.decisions,
            'valid_transitions':collector.transitions,'sampling_updates':updates,'ppo_epochs':trainer.updates,
            'last_checkpoint':last_checkpoint,'wall_seconds_this_invocation':time.perf_counter()-start,
            'evaluation_actual_base_steps':sum(json.loads(p.read_text())['actual_base_steps']
                for p in (path/'metrics').glob('evaluation_*.json') if not p.name.endswith('_reservation.json')),
            'evaluation_reserved_base_steps':sum(json.loads(p.read_text())['reserved_base_steps']
                for p in (path/'metrics').glob('evaluation_*_reservation.json')),
            'full_gathering_acceptance':False,'physx_acceptance':False}
        json_write(path/'metrics'/f'summary_{updates:06d}.json',summary)
        return summary,final
    except BaseException as error:
        json_write(path/'metrics'/f'failure_{time.time_ns()}.json',{
            'error':repr(error),'last_complete_checkpoint':last_checkpoint,'completed_sampling_updates':updates,
            'phase':phase,
            'observed_base_steps':None if collector is None else collector.base_steps,
            'automatic_retry':False})
        raise
    finally:
        if collector is not None:
            collector.close()


def run_experiment(args):
    validate_request(args)
    if args.describe_decision_credit:
        print(json.dumps({**specification(),'optimizer_overrides':{
            'gradient_clipping':args.gradient_clipping,'terminal_actor_credit':args.terminal_actor_credit},
            'initial_continuous_std_scale':args.initial_continuous_std_scale},indent=2));return
    if not args.release_training_pause:
        raise ValueError('Training remains paused: explicit --release-training-pause is required')
    if not args.far_checkpoint:
        raise ValueError('Supply the historical A1 checkpoint for the fixed far panel')
    if args.decision_credit_arm!='all':
        run_arm(args);return
    suite=ROOT/'outputs/runs/upstream_scale_t11'/args.run_id
    if suite.exists():
        raise FileExistsError(suite)
    json_write(suite/'run_manifest.json',{'specification':specification(),'command':sys.argv,
        'system_acceptance_complete':False})
    summaries={};results={};initializations=[]
    for arm in ARMS:
        child=copy(args);child.decision_credit_arm=arm;child.run_id=f'{args.run_id}_{arm.lower()}'
        summary,final=run_arm(child);summaries[arm]=summary
        initializations.append(json.loads((suite.parent/child.run_id/'metrics/initialization.json').read_text())['actor_critic_sha256'])
        if len(set(initializations))!=1:
            raise AssertionError('Arms did not start from identical actor/critic weights')
        if not summary['budget_complete']:
            json_write(suite/'metrics/stopped.json',{'completed_arms':summaries,'reason':'invocation_limit'})
            return
        results[arm]=final['results']
    report={'arms':summaries,**compare_results(results)}
    if sum(s['evaluation_reserved_base_steps'] for s in summaries.values())>20000:
        raise AssertionError('Suite evaluation budget exceeded')
    json_write(suite/'metrics/summary.json',report)
