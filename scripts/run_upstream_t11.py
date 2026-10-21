#!/usr/bin/env python3
"""Recurrent T11 session: bounded diagnostics or explicitly authorized training.

The same collector, PPO epochs, independent evaluation and public recovery are
used in both modes. Diagnostic states are never eligible training checkpoints.
"""
import argparse
import gc
from copy import deepcopy
from dataclasses import replace
import json
import importlib.metadata
from pathlib import Path
import sys
from types import SimpleNamespace
import time

import numpy as np
import torch

from _common import ROOT, cfg_from_experiment
from diagnose_upstream_environment import event_data
from diagnose_upstream_planners import archive_inputs
from evaluate_upstream_gathering import terrain_for
from planner_evidence import file_sha256, json_write, source_snapshot
from upstream_local_closure import runtime_config
from upstream_run_state import save_state, restore_state
from upstream_checkpoint import save_checkpoint, read_checkpoint
from upstream_s05_cases import gathering_scene
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.action_ablation import build_ablation_policies
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_ablation_learning import UpstreamAblationRollout
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_environment import UpstreamEnvironment
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_training import PreparedUpstreamTrainer, TrainingOptions
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_spec import parse_upstream_experiment

SESSION_VERSION = 'upstream_t11_session_v2'


def validate_signal_sampling(protocol, *, owners, num_envs, rollout_steps, cpu_workers):
    protocols={'legacy2':(0,2,128),'owners48_v1':(6,48,128),'owners24_v1':(6,24,128),'owners16_v1':(4,16,128)}
    if protocol not in protocols:raise ValueError('Unknown signal sampling protocol')
    if (owners,num_envs,rollout_steps)!=protocols[protocol] or (protocol!='legacy2' and cpu_workers!=2):
        raise ValueError('Signal sampling protocol/configuration mismatch')


def remaining_rollout_steps(budget, transitions, num_envs, rollout_steps):
    """Never exceed the budget to finish a vectorized environment step."""
    return min(rollout_steps, max(0, (budget-transitions)//num_envs))


def session_configuration(cfg, mode):
    """Label the execution role; authorization is checked separately by main."""
    if mode == 'diagnostic':
        if not cfg.training_paused:raise ValueError('Diagnostic source configuration must remain paused')
        return cfg
    if mode not in ('train','signal'):raise ValueError('Unknown T11 session role')
    data=json.loads(cfg.configuration_json)
    data['experiment']['training_paused']=False
    data['local_goal']['training_paused']=False
    data['upstream_scale']['training_session_version']='upstream_t11_learning_v1'
    return parse_upstream_experiment(data,ROOT)


def readiness(path, cfg, *, scope='system', budget=None, representation=None, seed=None, base_config=None):
    """Require immutable complete evidence, not a caller-supplied ready boolean."""
    if path is None:raise ValueError('Missing independent training readiness audit')
    p=Path(path).resolve()
    if not p.is_relative_to((ROOT/'outputs/runs').resolve()):raise ValueError('Readiness must be an owned run artifact')
    audit=json.loads(p.read_text())
    if scope=='signal100k':
        from audit_upstream_t11_readiness import audit_signal_readiness
        if (audit.get('schema_version')!='upstream_t11_signal_readiness_v1'
                or audit.get('signal_training_ready') is not True
                or audit.get('formal_training_ready') is not False
                or audit.get('source_snapshot')!=source_snapshot()
                or audit.get('profile_fingerprint')!=cfg.spec.fingerprint
                or budget!=100000 or seed!=23 or representation not in ('A0','A1','A2','A3')
                or base_config is None):
            raise ValueError('Signal admission is stale or outside authorized100k/seed23 scope')
        base=cfg_from_experiment(base_config)
        validation=audit.get('validation',{})
        if file_sha256(validation['path'])!=validation.get('sha256'):
            raise ValueError('Signal validation evidence changed')
        expected=audit_signal_readiness(base,validation['path'],authorized=True)
        if expected!=audit:raise ValueError('Signal admission/configuration mismatch')
        return audit
    if scope!='system':raise ValueError('Unknown admission scope')
    if (audit.get('schema_version')!='upstream_t11_readiness_v1'
            or audit.get('formal_training_ready') is not True
            or audit.get('checks',{}).get('source_reconciled') is not True
            or audit.get('profile_fingerprint')!=cfg.spec.fingerprint
            or audit.get('source_snapshot')!=source_snapshot()):
        raise ValueError('Training readiness is missing, stale or belongs to another source/profile')
    reconciliation=audit.get('dependency_audit') or {}
    reconciliation_path=Path(reconciliation.get('path','')).resolve()
    if (not reconciliation_path.is_relative_to((ROOT/'outputs/runs').resolve())
            or file_sha256(reconciliation_path)!=reconciliation.get('sha256')):
        raise ValueError('Missing or changed dependency reconciliation')
    records=audit.get('evidence',{})
    required={'gate_b','gathering','contracts','cuda','training_entry'}
    if set(records)!=required:raise ValueError('Incomplete prerequisite evidence set')
    results={}
    for name,record in records.items():
        target=Path(record['path']).resolve()
        if (not target.is_relative_to((ROOT/'outputs/runs').resolve())
                or file_sha256(target)!=record.get('sha256')):
            raise ValueError('Prerequisite evidence checksum/path mismatch')
        results[name]=json.loads(target.read_text())
    if (results['gate_b'].get('counts')!={'plan':7000,'repeat':630,'local':252,'batch':14}
            or results['gate_b'].get('gate_b_functional') is not True
            or results['gathering'].get('count')!=144
            or results['gathering'].get('s05_gathering_passed') is not True
            or any(results[n].get('passed') is not True for n in ('contracts','cuda','training_entry'))):
        raise ValueError('Required matrix or executed contract evidence did not pass')
    return audit


def make_collector(base, representation, seed, *, samples=512, num_envs=2,
                   episode_seconds=360., family='compact', policy=None, cache_dir=None,
                   mode='diagnostic', learning_time=None, initial_cohort=None, curriculum_config=None):
    native=representation=='A3'
    cfg,_=runtime_config(base,name='upstream_t11_'+representation.lower(),seed=seed,
        adapter='primitive47_upstream_v1' if native else None,
        device='cpu' if native else None,samples=samples,num_envs=num_envs,
        episode_seconds=episode_seconds,safety=True)
    cfg=session_configuration(cfg,mode)
    torch.manual_seed(seed)
    if policy is None:policy=build_ablation_policies(991,cfg.spec)[representation]
    poses=np.stack([gathering_scene(replace(cfg,simulation=replace(cfg.simulation,num_envs=1)),family,seed+i)[0]
                    for i in range(num_envs)])
    terrain,runtime,maps=terrain_for(family,seed,num_envs)
    initial_control=None
    if initial_cohort is not None:
        from upstream_learning_cohort import mixed_initial_conditions
        poses,initial_control=mixed_initial_conditions(poses,seed,initial_cohort)
    from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_reward import UpstreamRewardSettings
    rewards=None
    if learning_time is not None:
        g,l=learning_time.discounts(cfg.spec.grid.dt)
        rewards=UpstreamRewardSettings(gamma=g,lam=l)
    curriculum=None
    if curriculum_config is not None:
        if initial_cohort is not None or family!='compact' or learning_time is None:
            raise ValueError('Curriculum requires its own flat spawn sampler and explicit learning time')
        from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_curriculum import SpawnCurriculum
        curriculum=SpawnCurriculum(cfg.spec,curriculum_config,seed,num_envs)
        poses=curriculum.sample(range(num_envs))
        initial_control=np.zeros((num_envs,4,2));initial_control[...,0]=.1
        rewards=replace(rewards,gather_scaling=curriculum_config['reward_scaling'])
    env=UpstreamEnvironment(cfg,poses,terrain_cfg=terrain,terrain_runtime=runtime,
                           map_identities=maps,cache_dir=cache_dir,reward_settings=rewards,initial_control=initial_control,curriculum=curriculum)
    return UpstreamAblationRollout(env,policy,learning_time)


def evaluate(collector, base, representation, *, max_macro_steps=None, samples=512):
    """Independent environments and RNG scope; never reset the training collector."""
    rows=[]
    devices=list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        cases=[(family,seed) for family in ('compact','irregular','reverse','crossing','slope','npb')
               for seed in (151,157,163)]
        if max_macro_steps is not None:cases=cases[:1]
        for family,seed in cases:
            evaluation=make_collector(base,representation,seed,samples=samples,num_envs=1,
                                      family=family,policy=deepcopy(collector.policy).eval())
            episodes=[];steps=0;violations=[]
            while not episodes and (max_macro_steps is None or steps<max_macro_steps):
                _,ends,records=evaluation.collect(1,deterministic=True);episodes.extend(ends);steps+=1
                for record in records:
                    execution=record.get('execution',{})
                    for key in ('geometry_overlap','clearance_violation'):
                        if np.any(execution.get(key,False)):violations.append(key)
            rows.append({'family':family,'seed':seed,'macro_steps':steps,'episodes':episodes,
                         'violations':violations,'complete':bool(episodes),
                         'successful':bool(episodes and episodes[0]['terminal_reason']=='success')})
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--run-id',required=True)
    parser.add_argument('--representation',choices=('A0','A1','A2','A3'),required=True)
    parser.add_argument('--mode',choices=('diagnostic','signal','train','decision-credit','repair-validation'),default='diagnostic')
    parser.add_argument('--seed',type=int,default=23);parser.add_argument('--rollout-steps',type=int,default=128)
    parser.add_argument('--samples',type=int,default=512);parser.add_argument('--num-envs',type=int,default=2)
    parser.add_argument('--diagnostic-updates',type=int,choices=(1,2),default=1)
    parser.add_argument('--budget',type=int,choices=(20000,100000,200000,500000,1000000,2000000),default=100000)
    parser.add_argument('--evaluate-every',type=int,default=10)
    parser.add_argument('--updates-this-invocation',type=int,help='Stop at a saved update boundary for controlled resume')
    parser.add_argument('--readiness');parser.add_argument('--release-training-pause',action='store_true')
    parser.add_argument('--authorize-signal100k',action='store_true')
    parser.add_argument('--gpu-service',action='store_true',help='Experimental centralized planner evaluation; benchmark before use')
    parser.add_argument('--env-owners',type=int,default=0,help='Persistent process owners; explicit new sampling protocol')
    parser.add_argument('--owner-cpu-workers',type=int,default=2)
    parser.add_argument('--sampling-protocol',choices=('legacy2','owners48_v1','owners24_v1','owners16_v1'),default='legacy2')
    parser.add_argument('--import-learner',help='Explicit weights/Adam/counter import; start a fresh owner environment cohort')
    parser.add_argument('--trainer-backend',choices=('skrl','legacy'),default='skrl')
    parser.add_argument('--policy-device',choices=('cpu','cuda'),default='cpu')
    parser.add_argument('--import-checkpoint',help='Explicit checkpoint-boundary import into a new session; legacy to skrl supported')
    parser.add_argument('--resume-state',help='Owned session snapshot; same mode/source/config required')
    parser.add_argument('--decision-credit-arm',choices=('A','B','C','all'))
    parser.add_argument('--gradient-clipping',choices=('global','actor_critic'),default='global',
                        help='Explicit optimizer semantics for decision-credit experiments; legacy default is global')
    parser.add_argument('--initial-continuous-std-scale',type=float,choices=(1.,.25),default=1.,
                        help='Fresh learnable latent std multiplier; sampling and log-prob use the same distribution')
    parser.add_argument('--terminal-actor-credit',choices=('all','effective'),default='all',
                        help='Exclude actor updates for terminal actions that cannot activate before termination')
    parser.add_argument('--value-normalization',choices=('none','running'),default='none',
                        help='Single-variable 20k repair-validation candidate; original reward units in GAE')
    parser.add_argument('--comparison-run',help='Completed original combo run reused by the value-normalization candidate')
    parser.add_argument('--far-checkpoint',help='Historical A1 owner-0 state for the frozen evaluation panel')
    parser.add_argument('--describe-decision-credit',action='store_true',help='Print bounded experiment specification without allocating environments or training')
    parser.add_argument('--curriculum-config',help='Explicit isolated 1km long-learning curriculum YAML')
    args=parser.parse_args();torch.set_num_threads(1)
    if args.curriculum_config:
        from upstream_curriculum_experiment import run_experiment
        run_experiment(args)
        return
    if args.mode=='repair-validation':
        from upstream_learning_validation import run_experiment
        run_experiment(args)
        return
    if args.value_normalization!='none' or args.comparison_run:
        raise ValueError('Value normalization comparison requires repair-validation mode')
    if args.mode=='decision-credit':
        from upstream_decision_experiment import run_experiment
        run_experiment(args)
        return
    if (args.budget in (20000,200000) or args.decision_credit_arm or args.describe_decision_credit
            or args.far_checkpoint or args.gradient_clipping!='global' or args.initial_continuous_std_scale!=1.
            or args.terminal_actor_credit!='all'):
        raise ValueError('Decision-credit options cannot change legacy training semantics')
    if sum(bool(v) for v in (args.resume_state,args.import_checkpoint,args.import_learner))>1:
        raise ValueError('Choose resume or one explicit new-session import')
    if args.import_learner and (args.sampling_protocol=='legacy2' or args.trainer_backend!='skrl'):
        raise ValueError('Learner-only import requires the explicit skrl owner protocol')
    if args.import_checkpoint and args.env_owners:raise ValueError('Legacy checkpoints cannot change owner sampling protocol')
    if (not 1<=args.rollout_steps<=128 or args.evaluate_every<1
            or args.updates_this_invocation is not None and args.updates_this_invocation<1):
        raise ValueError('Invalid session batch/evaluation interval')
    if args.gpu_service and not args.env_owners:raise ValueError('GPU service requires environment owners')
    if args.num_envs<1 or args.env_owners<0 or args.owner_cpu_workers<0 or (args.env_owners and args.num_envs%args.env_owners):
        raise ValueError('Owners must evenly partition the requested environments')
    if args.mode!='diagnostic' and args.sampling_protocol=='legacy2' and args.budget%args.num_envs:
        raise ValueError('The transition budget must be divisible by the environment batch; do not exceed an authorized budget')
    if args.mode=='signal':
        validate_signal_sampling(args.sampling_protocol,owners=args.env_owners,num_envs=args.num_envs,
                                 rollout_steps=args.rollout_steps,cpu_workers=args.owner_cpu_workers)
    if args.mode=='train' and not args.release_training_pause:
        raise ValueError('Formal training remains paused; this entry requires explicit release')
    if args.mode=='signal' and not args.authorize_signal100k:
        raise ValueError('Signal learning requires explicit100k authorization')
    # Check authorization/evidence before allocating the full execution stack.
    cfg,_=runtime_config(args.config,name='upstream_t11_'+args.representation.lower(),seed=args.seed,
                        samples=args.samples,num_envs=args.num_envs,safety=True,episode_seconds=360.,
                        adapter='primitive47_upstream_v1' if args.representation=='A3' else None,
                        device='cpu' if args.representation=='A3' else None)
    audit=(readiness(args.readiness,cfg,scope='signal100k',budget=args.budget,
                     representation=args.representation,seed=args.seed,base_config=args.config)
           if args.mode=='signal' else readiness(args.readiness,cfg) if args.mode=='train' else None)
    if args.mode!='diagnostic' and args.samples!=512:raise ValueError('The prepared training definition fixes K512')
    cfg=session_configuration(cfg,args.mode)
    path=ROOT/'outputs/runs/upstream_scale_t11'/args.run_id
    dependencies={'python':sys.version}
    for package in ('torch','numpy','scipy','osqp','casadi','acados-template','skrl'):
        try:dependencies[package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:dependencies[package]=None
    identity={'session_version':SESSION_VERSION,'trainer_backend':args.trainer_backend,'policy_device':args.policy_device,'env_owners':args.env_owners,'gpu_service':args.gpu_service,'owner_cpu_workers':args.owner_cpu_workers,'mode':args.mode,'representation':args.representation,
              'source_snapshot':source_snapshot(),'dependencies':dependencies,'contracts':cfg.metadata,'options':TrainingOptions(rollout_macro_steps=args.rollout_steps).metadata,
              'budget':args.budget if args.mode!='diagnostic' else None,'diagnostic_updates':args.diagnostic_updates,
              'import_checkpoint':None if not args.import_checkpoint else {'path':str(Path(args.import_checkpoint).resolve()),'sha256':file_sha256(args.import_checkpoint)},
              'admission_scope':'signal100k' if args.mode=='signal' else 'system' if args.mode=='train' else 'diagnostic'}
    if args.sampling_protocol!='legacy2':identity['sampling_protocol']=args.sampling_protocol
    manifest=path/'run_manifest.json'
    if manifest.exists():
        if not args.resume_state or json.loads(manifest.read_text())['identity']!=identity:
            raise ValueError('Session exists or resume identity changed')
    else:
        for name in ('config','metrics','logs','state','checkpoints'):(path/name).mkdir(parents=True,exist_ok=True)
        json_write(manifest,{'identity':identity,'formal_training':args.mode=='train',
                            'signal_learning':args.mode=='signal','system_acceptance_complete':args.mode=='train',
                            'readiness':None if audit is None else {'path':args.readiness,'sha256':file_sha256(args.readiness)}})
    run=SimpleNamespace(path=path,identity=identity,planner_device=cfg.planner_device)
    archive_inputs(run,cfg)
    config_target=path/'config/effective_configuration.json'
    if not config_target.exists():json_write(config_target,json.loads(cfg.configuration_json))
    if args.env_owners:
        from upstream_process_rollout import ProcessUpstreamRollout
        collector=ProcessUpstreamRollout(args.config,args.representation,args.seed,owners=args.env_owners,
            envs_per_owner=args.num_envs//args.env_owners,samples=args.samples,mode=args.mode,
            cpu_workers=args.owner_cpu_workers,device=args.policy_device,cache_dir=path/'state/solver_cache',gpu_service=args.gpu_service)
    else:
        collector=make_collector(args.config,args.representation,args.seed,samples=args.samples,num_envs=args.num_envs,
                                 cache_dir=path/'state/solver_cache',mode=args.mode)
        if args.policy_device!='cpu':
            collector.policy.to(args.policy_device);collector.device=torch.device(args.policy_device)
            collector.hidden=collector.hidden.to(args.policy_device)
    if args.trainer_backend=='skrl':
        from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_skrl import SkrlUpstreamTrainer
        trainer_type=SkrlUpstreamTrainer
    else:trainer_type=PreparedUpstreamTrainer
    trainer=trainer_type(collector,TrainingOptions(rollout_macro_steps=args.rollout_steps))
    updates=0;start=time.perf_counter()
    if args.import_learner:
        parent=Path(args.import_learner).resolve()
        imported=read_checkpoint(parent,learner_only=True)
        trainer.import_learner_state(imported['trainer'])
        updates=imported['session_updates']
        del imported
        json_write(path/'metrics/learner_import.json',{'path':str(parent),'sha256':file_sha256(parent),
            'inherited_transitions':collector.transitions,'inherited_updates':updates,
            'environment_and_hidden_reset':True,'sampling_protocol':args.sampling_protocol})
        target=path/'checkpoints'/f'update_{updates:06d}.pt'
        save_checkpoint(trainer,identity,updates,target,scope='signal100k')
    if args.import_checkpoint:
        imported=read_checkpoint(args.import_checkpoint)
        if imported['trainer']['metadata']['trainer_version']==trainer.metadata['trainer_version']:
            trainer.load_state_dict(imported['trainer'])
        elif args.trainer_backend=='skrl':trainer.migrate_legacy_state(imported['trainer'])
        else:raise ValueError('Unsupported checkpoint backend conversion')
        updates=imported['session_updates']
        del imported
    if args.resume_state:
        snapshot=Path(args.resume_state).resolve()
        if not snapshot.is_relative_to(path.resolve()):raise ValueError('Resume state must belong to this session')
        if args.mode=='diagnostic':saved=restore_state(snapshot)
        else:
            saved=read_checkpoint(snapshot)
        if saved['identity']!=identity:raise ValueError('Session snapshot identity mismatch')
        trainer.load_state_dict(saved['trainer']);updates=saved['session_updates']
        del saved
    initial_updates=updates
    while ((updates<args.diagnostic_updates) if args.mode=='diagnostic' else (collector.transitions<args.budget)):
        length=args.rollout_steps if args.mode=='diagnostic' else remaining_rollout_steps(
            args.budget,collector.transitions,args.num_envs,args.rollout_steps)
        if length==0:break
        phase_start=time.perf_counter()
        def on_chunk(part,ends,trace,remaining):
            if args.mode!='signal':return
            applied=[r['execution']['applied_control'] for r in trace if 'execution' in r]
            commands=np.asarray(applied)
            valid=part['learning_valid']
            holds=torch.stack([s.action.hold for s in part['samples']])
            sample={'update':updates+1,'remaining_macro_steps':remaining,
                'team_transitions':collector.transitions,'valid_transitions':int(valid.sum()),
                'total_transitions':valid.numel(),'reward_mean':float(part['reward'][valid].mean()) if valid.any() else None,
                'hold_fraction':float(holds.float().mean()),'episodes':ends,
                'maximum_applied_speed':float(np.abs(commands[...,0]).max()) if commands.size else None,
                'maximum_applied_yaw_rate':float(np.abs(commands[...,1]).max()) if commands.size else None,
                'finite_observations':bool(torch.isfinite(part['obs']).all()),
                'finite_log_prob':all(bool(torch.isfinite(s.log_prob).all()) for s in part['samples']),
                'poses':collector.env.core.poses,'task':collector.env.last_task,
                'wall_seconds':time.perf_counter()-start}
            json_write(path/'metrics'/f'collection_{updates+1:06d}_{length-remaining:03d}.json',event_data(sample))
            print(json.dumps({'event':'collection','update':updates+1,'team_transitions':collector.transitions,
                'valid_transitions':int(valid.sum()),'remaining_macro_steps':remaining}),flush=True)
        batch,episodes,records=trainer.collect(length,on_chunk=on_chunk if args.mode=='signal' else None)
        collect_seconds=time.perf_counter()-phase_start;phase_start=time.perf_counter()
        result=(trainer.diagnostic_update(batch) if args.mode=='diagnostic' else
                trainer.update(batch,readiness=audit,training_authorized=True))
        update_seconds=time.perf_counter()-phase_start;updates+=1
        payload={'update':updates,'team_transitions':collector.transitions,'episodes':episodes,'learning':result,
                 'wall_seconds':time.perf_counter()-start,'formal_training':args.mode=='train','signal_learning':args.mode=='signal',
                 'phase_seconds':{'collect':collect_seconds,'ppo_update':update_seconds}}
        json_write(path/'metrics'/f'update_{updates:06d}.json',event_data(payload))
        # Preserve actual original actions/probabilities and execution feedback.
        phase_start=time.perf_counter()
        save_state(run,f'interactions_{updates:06d}',{'samples':batch['samples'],'execution':records,
                   'duration':batch['duration'].cpu(),'learning_valid':batch['learning_valid'].cpu(),
                   'learning':{k:batch[k].cpu() for k in ('reward','value','next_value','terminated','truncated',
                       'bootstrap_allowed','terminal_reason')}})
        interaction_save_seconds=time.perf_counter()-phase_start
        del batch,episodes,records,payload
        trainer.release_rollout()
        gc.collect()
        evaluation_seconds=0.;state_save_seconds=0.
        stop_requested=args.updates_this_invocation is not None and updates-initial_updates>=args.updates_this_invocation
        if args.mode!='signal' and (args.mode=='diagnostic' or updates%args.evaluate_every==0 or collector.transitions>=args.budget or stop_requested):
            phase_start=time.perf_counter()
            rows=evaluate(collector,args.config,args.representation,
                          max_macro_steps=1 if args.mode=='diagnostic' else None,samples=args.samples)
            json_write(path/'metrics'/f'evaluation_{updates:06d}.json',event_data({'scope':'bounded_entry_check' if args.mode=='diagnostic' else 'independent18', 'cases':rows}))
            evaluation_seconds=time.perf_counter()-phase_start;phase_start=time.perf_counter()
            if args.mode=='diagnostic':
                saved={'identity':identity,'trainer':trainer.state_dict(),'session_updates':updates}
                save_state(run,f'diagnostic_session_{updates:06d}',saved)
                del saved
            else:
                target=path/'checkpoints'/f'update_{updates:06d}.pt'
                save_checkpoint(trainer,identity,updates,target,scope='train')
            state_save_seconds=time.perf_counter()-phase_start
        if args.mode=='signal':
            # Save every actual update before continuing. Full18-case evaluation
            # remains separate from this bounded learning-signal experiment.
            phase_start=time.perf_counter()
            target=path/'checkpoints'/f'update_{updates:06d}.pt'
            save_checkpoint(trainer,identity,updates,target,scope='signal100k')
            state_save_seconds=time.perf_counter()-phase_start
        json_write(path/'metrics'/f'timing_{updates:06d}.json',{'version':'t11_session_timing_v1',
            'collect_s':collect_seconds,'ppo_update_s':update_seconds,'interaction_save_s':interaction_save_seconds,
            'evaluation_s':evaluation_seconds,'state_save_s':state_save_seconds,
            'scope':'host elapsed phases; not GPU kernel time; setup and report writes excluded'})
        print(json.dumps({'updates':updates,'transitions':collector.transitions,'mode':args.mode}),flush=True)
        if stop_requested:break
    if args.env_owners:collector.close()
    json_write(path/'metrics'/f'session_summary_{updates:06d}.json',{'mode':args.mode,'updates':updates,
        'actual_team_transitions':collector.transitions,'formal_training_interactions':collector.transitions if args.mode=='train' else 0,
        'signal_learning_interactions':collector.transitions if args.mode=='signal' else 0,
        'wall_seconds':time.perf_counter()-start,'formal_training_ready':args.mode=='train',
        'signal_budget_complete':args.mode=='signal' and 0<=args.budget-collector.transitions<args.num_envs,
        'unused_budget':None if args.mode=='diagnostic' else args.budget-collector.transitions,
        'gate_c_passed':False,'training_checkpoint_created':args.mode!='diagnostic'})


if __name__=='__main__':main()
