"""Explicitly authorized, bounded 1 km curriculum learning through the T11 entry."""
import gc
import json
import math
import multiprocessing as mp
import signal
import sys
import tarfile
import time
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml
from _common import ROOT
from diagnose_upstream_environment import event_data
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_curriculum import (
    update_progress,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_decision import (
    LearningTimeConfig,
    remaining_decisions,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_skrl import (
    SkrlUpstreamTrainer,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_training import (
    TrainingOptions,
    long_training_budget_valid,
)
from planner_evidence import file_sha256, json_write, source_snapshot
from training_tensorboard import TrainingTensorBoard, TENSORBOARD_METADATA
from upstream_checkpoint import read_checkpoint, save_checkpoint, save_tensor_file
from upstream_decision_experiment import initialize_exploration, parameter_digest
from upstream_process_rollout import ProcessUpstreamRollout
from upstream_run_state import save_state


FIXED_VERSION = 'rendezvous_fixed_lunar_1km_v1'


def learning_batch_problem(batch):
    """Keep finite-data checks while honoring the learner's invalid-terminal mask."""
    if (not all(torch.isfinite(batch[k]).all() for k in ('obs', 'state', 'hidden', 'reward', 'value', 'next_value'))
            or not all(torch.isfinite(sample.log_prob).all() for sample in batch['samples'])):
        return 'Nonfinite learning data'
    masked = (batch['duration'] > 0) & ~batch['learning_valid']
    # Code 5 is the existing environment_invalid contract. Such a terminal
    # remains excluded from Actor/Critic targets and cannot bootstrap.
    if (masked & ((batch['terminal_reason'] != 5) | ~batch['terminated']
                  | batch['bootstrap_allowed'])).any():
        return 'Invalid learning mask/terminal contract'
    return None


def obstacle_summary(records,band):
    """Aggregate actual executed/observed feedback; no extra map queries."""
    distances=[];known=[];costs={};rewards={};reads=physical=0
    for record in records:
        if 'obstacle_distance' in record:
            active=np.asarray(record['task']['learning_valid'],bool)
            distances.extend(np.asarray(record['obstacle_distance'])[active].ravel().tolist())
            known.extend(np.asarray(record['obstacle_footprint_known'])[active].ravel().tolist())
            reads+=record['perception']['atlas_cells_read'];physical+=record['perception']['physical_query_points']
            for name,value in record['reward_components'].items():
                rewards.setdefault(name,[]).extend(np.asarray(value)[active].ravel().tolist())
        for name,value in record.get('planner_cost_parts',{}).items():costs.setdefault(name,[]).append(value)
    if not distances:return {}
    d=np.asarray(distances)
    return {'distance_min':float(d.min()),'near_fraction':float((d<band).mean()),'overlap_fraction':float((d<0).mean()),
            'footprint_known_fraction':float(np.mean(known)),'atlas_cells_read':reads,'physical_query_points':physical,
            'planner_cost_mean':{k:float(np.mean(v)) for k,v in costs.items()},
            'executed_reward_mean':{k:float(np.mean(v)) for k,v in rewards.items()}}


def is_fixed(config):
    return config['version'] == FIXED_VERSION


def bounded_evaluation(config):
    return config.get('evaluation_mode') == 'bounded_fixed_window_v1'


def periodic_window_evaluation(config):
    return config.get('evaluation_mode') == 'periodic_fixed_window_v1'


def windowed_evaluation(config):
    return bounded_evaluation(config) or periodic_window_evaluation(config)


def load_curriculum(path):
    config = yaml.safe_load(Path(path).read_text())
    if is_fixed(config):
        short = windowed_evaluation(config)
        periodic = periodic_window_evaluation(config)
        if config.get('evaluation_mode') not in (None, 'bounded_fixed_window_v1', 'periodic_fixed_window_v1'):
            raise ValueError('Unknown fixed evaluation mode')
        if not short and 'evaluation_window_seconds' in config:
            raise ValueError('An evaluation window requires explicit bounded mode')
        if short:
            seconds = config.get('evaluation_window_seconds')
            if (isinstance(seconds, bool) or not isinstance(seconds, (int, float))
                    or not math.isfinite(seconds) or not 0 < seconds <= 1500.
                    or not math.isclose(seconds / config['policy_decision_period_s'],
                                        round(seconds / config['policy_decision_period_s']))):
                raise ValueError('Invalid bounded evaluation window')
            final_reserve = round(8 * seconds / .2)
            minimum_evaluation_budget = 2 * final_reserve
            if periodic:
                final_reserve += 60000  # Eight full 1500-second fixed-start episodes.
                minimum_evaluation_budget += 60000
        else:
            final_reserve, minimum_evaluation_budget = 120000, 360000
        stage = config['stages']
        if (config['terrain'] != 'lunar_scene_pack' or config['terrain_failure_semantics'] != 'known_terrain_terminal_v1' or config['initial_level'] != 0 or len(stage) != 1
                or stage[0]['radius_m'] != [200., 300.] or stage[0]['deadline_s'] != 1500.
                or stage[0]['heading_noise_deg'] != 180. or config['easy_replay_fraction'] != 0
                or config['curriculum_enabled'] or config['promotion_enabled']
                or config['maximum_stage_base_steps'] is not None or config['maximum_spawn_attempts'] < 1
                or not 1 <= config['rollout_decisions'] <= 128
                or not 0 < config['initial_hold_probability'] < 1
                or config['initial_continuous_std_scale'] not in (1., .25)
                or config['terrain_storage'] != 'mmap'
                or {k:len(v) for k,v in config['scene_manifests'].items()} != {'training':2,'fixed_eval':1,'unseen':1}
                or (not periodic and config['evaluation_every_base_steps'] != 500000)
                or (not periodic and config['full_evaluation_at_base_steps'] != [500000,1000000,1500000,2000000])
                or (periodic and (type(config['evaluation_every_base_steps']) is not int
                                  or config['evaluation_every_base_steps'] < 12800
                                  or config['full_evaluation_at_base_steps'] != []))
                or config['final_evaluation_reservation'] < final_reserve
                or config['evaluation_max_base_steps'] < minimum_evaluation_budget):
            raise ValueError('Invalid authorized fixed lunar training configuration')
        timing_for(config)
        return config
    if config['version'] not in ('rendezvous_curriculum_1km_v1', 'rendezvous_curriculum_1km_v2') or config['terrain'] != 'flat_proxy':
        raise ValueError('Expected the explicit flat Proxy 1km curriculum')
    if (not 0 <= config['easy_replay_fraction'] < 1 or not 0 < config['promotion_success_fraction'] <= 1
            or config['promotion_window_episodes'] < 1 or config['minimum_stage_base_steps'] < 0
            or not 1 <= config['rollout_decisions'] <= 128
            or config['evaluation_max_base_steps'] < config['final_evaluation_reservation']):
        raise ValueError('Invalid curriculum scheduling/budget')
    for stage in config['stages']:
        if not (0 < stage['radius_m'][0] <= stage['radius_m'][1] <= 300
                and 0 < stage['deadline_s'] <= 1200 and 0 <= stage['heading_noise_deg'] <= 180):
            raise ValueError('Invalid 1km curriculum stage')
    if config['version'] == 'rendezvous_curriculum_1km_v2':
        levels = len(config['stages'])
        if (type(config['initial_level']) is not int or not 1 <= config['initial_level'] < levels
                or config['maximum_stage_base_steps'] < max(400, config['minimum_stage_base_steps'])
                or not 0 < config['initial_hold_probability'] < 1
                or config['initial_continuous_std_scale'] not in (1., .25)
                or config['evaluation_every_base_steps'] < 400
                or config['checkpoint_every_updates'] < 1):
            raise ValueError('Invalid reviewed curriculum scheduling/initialization')
        checkpoints = config['full_evaluation_at_base_steps']
        if (checkpoints != sorted(set(checkpoints)) or not checkpoints
                or any(type(x) is not int or x < config['evaluation_every_base_steps']
                       or x % config['evaluation_every_base_steps'] for x in checkpoints)):
            raise ValueError('Invalid full evaluation checkpoints')
        for key in ('trend_evaluation_levels', 'full_evaluation_levels', 'initial_full_evaluation_levels', 'final_evaluation_levels'):
            if (not config[key] or len(set(config[key])) != len(config[key])
                    or any(type(i) is not int or not 0 <= i < levels for i in config[key])):
                raise ValueError('Invalid fixed evaluation levels')
        seconds = config['trend_evaluation_seconds']
        if (not math.isfinite(seconds) or seconds <= 0
                or seconds > min(config['stages'][i]['deadline_s'] for i in config['trend_evaluation_levels'])
                or not math.isclose(seconds / config['policy_decision_period_s'], round(seconds / config['policy_decision_period_s']))
                or any(not {0, levels-1}.issubset(config[k]) for k in ('full_evaluation_levels', 'final_evaluation_levels'))):
            raise ValueError('Invalid trend duration or convergence panel coverage')
        reserve = evaluation_reservation(evaluation_panels(config, config['initial_level'], 'final', final=True))
        if config['final_evaluation_reservation'] < reserve:
            raise ValueError('Final evaluation reservation cannot cover the declared panels')
    return config


def timing_for(config):
    return LearningTimeConfig(config['policy_decision_period_s'], config['discount_reference_period_s'],
                              gamma=config['gamma'], lam=config['lambda'])


def collection_decisions(config, progress, budget, used, num_envs, decision_steps):
    """Reserve the full vector interval against BOTH physical-step limits."""
    limit = budget
    if config['version'] == 'rendezvous_curriculum_1km_v2':
        limit = min(limit, progress['stage_start_base_steps'] + config['maximum_stage_base_steps'])
    return remaining_decisions(limit, used, num_envs, decision_steps, config['rollout_decisions'])


def initialize_fresh_policy(policy, config):
    scale = config['initial_continuous_std_scale'] if config['version'] == 'rendezvous_curriculum_1km_v2' or is_fixed(config) else .25
    initialize_exploration(policy, scale)
    if config['version'] == 'rendezvous_curriculum_1km_v2' or is_fixed(config):
        p = config['initial_hold_probability']
        with torch.no_grad():
            policy.hold_logit.bias.fill_(math.log(p / (1-p)))


def evaluation_panels(config, level, phase, *, final=False, full=False):
    """Separate fixed-duration trends from full episodes used for convergence."""
    if is_fixed(config):
        if windowed_evaluation(config):
            panels = [('far', 0, config['evaluation_window_seconds'], False)]
            if periodic_window_evaluation(config) and final:
                panels.append(('full_far', 0, config['stages'][0]['deadline_s'], True))
            return panels
        return [('far', 0, config['stages'][0]['deadline_s'], True)] + ([('unseen', 0, config['stages'][0]['deadline_s'], True)] if final else [])
    if config['version'] == 'rendezvous_curriculum_1km_v1':
        levels = sorted({0, level, len(config['stages'])-1} if final else {0, level})
        if phase == 'initial':
            levels = [0, 1]
        return [(f'level_{i}', i, config['stages'][i]['deadline_s'], True) for i in levels]
    panels = [(f'trend_level_{i}', i, config['trend_evaluation_seconds'], False)
              for i in config['trend_evaluation_levels']]
    levels = (config['final_evaluation_levels'] if final else
              config['initial_full_evaluation_levels'] if phase == 'initial' else
              config['full_evaluation_levels'] if full else [])
    return panels + [(f'level_{i}', i, config['stages'][i]['deadline_s'], True) for i in levels]


def evaluation_reservation(panels):
    return sum(round(8 * seconds / .2) for _, _, seconds, _ in panels)


def evaluation_budget_upper_bound(config, budget):
    """Full-duration reservation, even when every rollout times out."""
    level = config['initial_level']
    if bounded_evaluation(config):
        return 2 * evaluation_reservation(evaluation_panels(config, level, 'initial'))
    trend = evaluation_reservation(evaluation_panels(config, level, 'scheduled'))
    extra = evaluation_reservation(evaluation_panels(config, level, 'scheduled', full=True)) - trend
    return (evaluation_reservation(evaluation_panels(config, level, 'initial'))
            + ((budget-1)//config['evaluation_every_base_steps']) * trend
            + sum(x < budget for x in config['full_evaluation_at_base_steps']) * extra
            + evaluation_reservation(evaluation_panels(config, level, 'final', final=True)))


def admission(collector, budget):
    return {'schema_version': 't11_curriculum_readiness_v1', 'scope': 'curriculum_1km',
            'maximum_base_steps': budget, 'profile_fingerprint': collector.env.spec.fingerprint,
            'learning_time': collector.learning_time.metadata(.2), 'curriculum_config': collector.curriculum_config,
            'formal_training_ready': False, 'system_acceptance_complete': False}


def evaluate(path, phase, collector, args, config, *, final=False, full=False, tensorboard=None):
    """Independent fixed panels; only complete episodes certify success."""
    from run_upstream_t11 import make_collector
    from upstream_decision_evaluation import panel
    destination = path/'metrics'/f'evaluation_{phase}.json'
    if destination.exists():
        return json.loads(destination.read_text())
    level = collector.curriculum_progress['level']
    panels = evaluation_panels(config, level, phase, final=final, full=full)
    reserve = evaluation_reservation(panels)
    reserved = sum(json.loads(p.read_text())['reserved_base_steps']
                   for p in (path/'metrics').glob('evaluation_*_reservation.json'))
    limit = config['evaluation_max_base_steps'] - (0 if final else config['final_evaluation_reservation'])
    if reserved + reserve > limit:
        json_write(path/'metrics'/f'evaluation_{phase}_skipped.json',
                   {'reason': 'evaluation_budget_reserved_for_final', 'required_base_steps': reserve})
        return None
    reservation_path=path/'metrics'/f'evaluation_{phase}_reservation.json'
    if reservation_path.exists():
        # An interrupted attempt keeps its entire reservation. A restart uses
        # the same frozen policy but must reserve its additional physical work.
        reservation_path=path/'metrics'/f'evaluation_{phase}_{time.time_ns()}_reservation.json'
    json_write(reservation_path,
               {'reserved_base_steps': reserve, 'training_base_steps': collector.base_steps,
                'levels': sorted({i for _, i, _, _ in panels}), 'panels': panels})
    results = {}
    with torch.random.fork_rng(devices=[]):
        for prefix, i, seconds, complete_episode in panels:
            for deterministic in (False, True):
                eval_config = deepcopy(config)
                seed, policy_seed = (130363, 130367) if prefix == 'unseen' else (104729, 104759)
                if is_fixed(config):
                    eval_config['scene_role'] = 'unseen' if prefix == 'unseen' else 'fixed_eval'
                c = make_collector(args.config, 'A1', seed, samples=512, num_envs=4,
                    episode_seconds=config['stages'][i]['deadline_s'] if is_fixed(config) else 1200.,
                    policy=deepcopy(collector.policy).cpu().eval(), learning_time=collector.learning_time,
                    curriculum_config=eval_config)
                c.env.curriculum.set_level(i)
                poses = c.env.curriculum.sample(np.arange(4), replay=False)
                c.env.task.deadlines[:] = c.env.curriculum.deadlines
                c.env.reset(np.arange(4), poses)
                torch.manual_seed(policy_seed)
                def progress(elapsed):
                    value = {'phase':phase,'panel':prefix,'mode':'mean' if deterministic else 'sample',
                             'training_base_steps':collector.base_steps,'panel_actual_base_steps':int(elapsed.sum()),
                             'panel_elapsed_seconds':(elapsed*.2).tolist()}
                    target = path/'metrics/evaluation_progress.json'
                    temporary = target.with_suffix('.tmp')
                    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
                    temporary.replace(target)
                    if tensorboard is not None:
                        tensorboard.scalars('evaluation_progress/panel_actual_base_steps', int(elapsed.sum()), collector.base_steps)
                        tensorboard.writer.flush()
                row = panel(c, deterministic=deterministic, seconds=seconds,
                            **({'progress': progress} if is_fixed(config) else {}),
                            **({'bounded_metrics': True} if windowed_evaluation(config) else {}))
                if complete_episode and len(row['episodes']) != 4:
                    raise AssertionError('Curriculum evaluation must finish every held-out episode')
                # Preserve original traces separately; headline metrics stay small.
                row['full_episode_evaluation'] = complete_episode
                row['observation_seconds'] = seconds
                name = f'{prefix}_{"mean" if deterministic else "sample"}'
                json_write(path/'metrics'/f'trace_{phase}_{name}.json', event_data(row.pop('trace')))
                results[name] = row
                del c
                gc.collect()
    actual = sum(row['actual_base_steps'] for row in results.values())
    if actual > reserve:
        raise AssertionError('Evaluation physical-step budget exceeded')
    result = event_data({'training_base_steps': collector.base_steps, 'curriculum_level': level,
                         'reserved_base_steps': reserve, 'actual_base_steps': actual, 'results': results,
                         'optimizer_updates': 0, 'full_episode_evaluation': all(p[3] for p in panels)})
    json_write(destination, result)
    return result


def frozen_evaluation_collector(collector):
    """No environment/Adam/RNG is shared with the evaluator process."""
    return SimpleNamespace(policy=deepcopy(collector.policy).cpu().eval(),
        base_steps=collector.base_steps, learning_time=collector.learning_time,
        curriculum_progress=deepcopy(collector.curriculum_progress))


def _evaluate_frozen(path, snapshot, args, config, final, full):
    from torch.utils.tensorboard import SummaryWriter
    torch.set_num_threads(1)
    saved=torch.load(snapshot,map_location='cpu',weights_only=False)
    writer=SummaryWriter(str(path/'tensorboard'),filename_suffix='.evaluation',flush_secs=10)
    try:
        evaluate(path,saved['phase'],saved['collector'],args,config,final=final,full=full,
                 tensorboard=SimpleNamespace(writer=writer,scalars=writer.add_scalar))
    except BaseException as error:
        json_write(path/'metrics'/f'failure_evaluation_{time.time_ns()}.json',
                   {'phase':saved['phase'],'training_base_steps':saved['collector'].base_steps,
                    'error':repr(error),'automatic_retry':False})
        raise
    finally:writer.close()


class ConcurrentFixedEvaluation:
    """One frozen evaluator overlaps sampling; scheduled evaluations stay ordered."""
    def __init__(self,path,args,config):
        self.path,self.args,self.config=path,args,config
        self.process=None;self.phase=None

    def poll(self):
        if self.process is None:return True
        if self.process.is_alive():return False
        self.process.join()
        code=self.process.exitcode
        self.process=None
        if code!=0:raise RuntimeError(f'Frozen evaluation {self.phase} exited with code {code}')
        if not any((self.path/'metrics'/f'evaluation_{self.phase}{suffix}.json').exists()
                   for suffix in ('','_skipped')):
            raise RuntimeError(f'Frozen evaluation {self.phase} did not publish a result')
        return True

    def wait(self):
        while not self.poll():self.process.join(timeout=1.)

    def start(self,phase,collector,*,final=False,full=False):
        if any((self.path/'metrics'/f'evaluation_{phase}{suffix}.json').exists() for suffix in ('','_skipped')):return
        if self.process is not None and self.phase==phase:return
        self.wait()
        snapshot=self.path/'state'/f'evaluation_{phase}.pt'
        if not snapshot.exists():
            metadata=save_tensor_file(snapshot,{'phase':phase,'collector':frozen_evaluation_collector(collector),
                                                'final':final,'full':full})
            json_write(snapshot.with_suffix('.json'),metadata)
        elif file_sha256(snapshot)!=json.loads(snapshot.with_suffix('.json').read_text())['sha256']:
            raise ValueError('Frozen evaluation snapshot checksum mismatch')
        self.phase=phase
        self.process=mp.get_context('spawn').Process(target=_evaluate_frozen,
            args=(self.path,snapshot,self.args,self.config,final,full),name='rover-fixed-evaluation')
        self.process.start()

    def resume_pending(self):
        pending=[p for p in (self.path/'state').glob('evaluation_*.pt')
                 if not any((self.path/'metrics'/(p.stem+suffix+'.json')).exists() for suffix in ('','_skipped'))]
        if len(pending)>1:raise ValueError('More than one unfinished frozen evaluation')
        if pending:
            saved=torch.load(pending[0],map_location='cpu',weights_only=False)
            self.start(saved['phase'],saved['collector'],final=saved['final'],full=saved['full'])

    def close(self):
        if self.process is not None:
            if self.process.is_alive():self.process.terminate()
            self.process.join(timeout=10.)
            if self.process.is_alive():self.process.kill();self.process.join()
            if not (self.path/'metrics'/f'evaluation_{self.phase}.json').exists():
                progress=self.path/'metrics/evaluation_progress.json'
                row=json.loads(progress.read_text()) if progress.exists() else {}
                json_write(self.path/'metrics'/f'evaluation_interruption_{time.time_ns()}.json',
                    {'phase':self.phase,'last_committed_progress':row,
                     'budget_accounting':'entire reservation retained; retry needs another full reservation',
                     'snapshot_retained':True})
            self.process=None


def convergence_report(path, config):
    if periodic_window_evaluation(config):
        return {'stable_convergence_screen_passed': False,
                'criterion': 'periodic short-window trends and one final full fixed panel; no repeated full or unseen convergence screen',
                'bounded_evaluation': False, 'periodic_window_evaluation': True,
                'unseen_screen_passed': False, 'statistical_significance_claim': False,
                'physx_acceptance': False}
    if bounded_evaluation(config):
        return {'stable_convergence_screen_passed': False,
                'criterion': 'bounded initial/final windows compare survival and approach; no full-episode convergence screen',
                'bounded_evaluation': True, 'final_level_panels': [], 'unseen_screen_passed': False,
                'statistical_significance_claim': False, 'physx_acceptance': False}
    if is_fixed(config):
        panels=[]; unseen=False
        for p in (path/'metrics').glob('evaluation_*.json'):
            row=json.loads(p.read_text()); results=row.get('results',{})
            if all(f'far_{m}' in results for m in ('sample','mean')):
                panels.append({'base_steps':row['training_base_steps'],
                    'passed':all(results[f'far_{m}']['successes'] >= 3 for m in ('sample','mean')), 'evidence':str(p)})
            if all(f'unseen_{m}' in results for m in ('sample','mean')):
                unseen=all(results[f'unseen_{m}']['successes'] >= 3 for m in ('sample','mean'))
        unique={r['base_steps']:r for r in sorted(panels,key=lambda r:r['base_steps'])}
        tail=list(unique.values())[-3:]
        return {'stable_convergence_screen_passed':len(tail)==3 and all(r['passed'] for r in tail) and unseen,
                'criterion':'3 distinct checkpoints >=3/4 full far successes in both modes; final unseen >=3/4 both modes',
                'final_level_panels':panels,'unseen_screen_passed':unseen,'statistical_significance_claim':False,'physx_acceptance':False}
    level = len(config['stages'])-1
    panels = []
    for p in (path/'metrics').glob('evaluation_*.json'):
        row = json.loads(p.read_text())
        results = row.get('results', {})
        if all(f'level_{level}_{mode}' in results for mode in ('sample', 'mean')):
            passed = all(results[f'level_{i}_{mode}']['successes'] >= 3
                         for i in (0, level) for mode in ('sample', 'mean'))
            panels.append({'base_steps': row['training_base_steps'], 'passed': passed, 'evidence': str(p)})
    panels.sort(key=lambda x: x['base_steps'])
    # A final-only panel cannot establish stable convergence. Repeated fixed
    # cases are an engineering screen, never a multi-seed significance claim.
    unique = {p['base_steps']: p for p in panels}
    tail = list(unique.values())[-3:]
    return {'stable_convergence_screen_passed': len(tail) == 3 and all(p['passed'] for p in tail),
            'criterion': '3 distinct checkpoints: >=3/4 near and final-range successes in both sample and mean modes',
            'final_level_panels': panels, 'statistical_significance_claim': False,
            'physx_acceptance': False}


def validate_long_training_budget(config, budget):
    # A replacement run may consume only the unspent part of an existing cap.
    if not long_training_budget_valid(config, budget):
        raise ValueError('Expected at most 2M fixed-lunar base steps, or a 1M/2M curriculum budget')


def run_experiment(args):
    config = load_curriculum(args.curriculum_config)
    validate_long_training_budget(config, args.budget)
    if (args.mode != 'train' or not args.release_training_pause or args.representation != 'A1' or args.seed != 23
            or args.env_owners != 4 or args.num_envs != 16
            or args.owner_cpu_workers != 2 or args.samples != 512 or args.trainer_backend != 'skrl'
            or args.policy_device != 'cpu' or args.gpu_service or args.import_checkpoint or args.import_learner
            or args.authorize_signal100k or args.decision_credit_arm or args.comparison_run):
        raise ValueError('Long learning requires explicitly released A1/seed23/K512/4x4/skrl/CPU')
    if not args.run_id or Path(args.run_id).name != args.run_id or args.run_id in ('.', '..'):
        raise ValueError('Expected an owned run name')
    if ((config['version'] == 'rendezvous_curriculum_1km_v2' or is_fixed(config))
            and evaluation_budget_upper_bound(config, args.budget) > config['evaluation_max_base_steps']):
        raise ValueError('Fixed evaluation schedule exceeds its physical-step budget')
    if not args.readiness:
        raise ValueError('Supply the completed prelaunch audit summary with --readiness')
    evidence_path = Path(args.readiness).resolve()
    if not evidence_path.is_relative_to((ROOT/'outputs/runs').resolve()):
        raise ValueError('Prelaunch audit must be an owned run artifact')
    evidence = json.loads(evidence_path.read_text())
    static_audit = json.loads((evidence_path.parent/'static_audit.json').read_text())
    if evidence.get('passed') is not True or evidence.get('optimizer_updates') != 0:
        raise ValueError('Prelaunch reward/geometry/parking audit did not pass')
    if (config['version'] == 'rendezvous_curriculum_1km_v2' or is_fixed(config)) and static_audit.get('curriculum_config') != config:
        raise ValueError('Prelaunch audit belongs to a different curriculum')
    timing = timing_for(config)
    g, l = timing.discounts(.2)
    options = TrainingOptions(rollout_macro_steps=config['rollout_decisions'], epochs=config['ppo_epochs'],
        gamma=g, lam=l, gradient_clipping='actor_critic', terminal_actor_credit='effective',
        value_normalization=config['value_normalization'])
    path = ROOT/'outputs/runs'/('upstream_1km_fixed_far' if is_fixed(config) else 'upstream_1km_curriculum')/args.run_id
    identity = {'version': config['version'], 'config': config, 'budget_base_steps': args.budget,
                'base_config': str(Path(args.config).resolve()), 'base_config_sha256': file_sha256(args.config),
                'source_snapshot': source_snapshot(), 'options': options.metadata, 'seed': 23,
                'prelaunch_audit': str(evidence_path), 'prelaunch_audit_sha256': file_sha256(evidence_path)}
    if is_fixed(config):
        identity['terrain_source_snapshot'] = {str(p.relative_to(ROOT)):file_sha256(p)
            for p in sorted((ROOT/'source/lunar_rover_tasks/lunar_rover_tasks/terrain_data').glob('*.py'))}
        identity['scene_manifest_sha256'] = {p:file_sha256(p) for paths in config['scene_manifests'].values() for p in paths}
    if path.exists():
        if not args.resume_state or json.loads((path/'run_manifest.json').read_text())['identity'] != identity:
            raise ValueError('Run exists or source/configuration identity changed')
    else:
        if args.resume_state:
            raise ValueError('Cannot resume into a new run')
        for directory in ('config', 'metrics', 'state', 'logs', 'checkpoints'):
            (path/directory).mkdir(parents=True, exist_ok=True)
        json_write(path/'run_manifest.json', {'identity': identity, 'command': sys.argv, 'tensorboard': TENSORBOARD_METADATA,
                   'formal_training': True, 'scope': 'user_authorized_proxy_fixed_far_lunar_learning' if is_fixed(config) else 'user_authorized_proxy_curriculum_long_learning',
                   'system_acceptance_complete': False, 'physx_acceptance': False})
        with tarfile.open(path/'config/source_and_configs.tar.gz', 'w:gz') as archive:
            for directory in ('source/lunar_rover_tasks/lunar_rover_tasks/tasks/multi_rover_gathering', 'source/lunar_rover_tasks/lunar_rover_tasks/terrain_data', 'scripts', 'configs', 'tests'):
                for p in sorted((ROOT/directory).rglob('*')):
                    if p.is_file() and p.suffix in ('.py', '.cpp', '.cu', '.h', '.yaml', '.json'):
                        archive.add(p, arcname=str(p.relative_to(ROOT)))
    stop = []
    signal.signal(signal.SIGTERM, lambda *_: stop.append('SIGTERM'))
    signal.signal(signal.SIGINT, lambda *_: stop.append('SIGINT'))
    collector = None
    last_checkpoint = None
    updates = 0
    phase = 'initialization'
    started = time.perf_counter()
    run = SimpleNamespace(path=path)
    tensorboard = TrainingTensorBoard(path, pinned='bounded_far' if windowed_evaluation(config)
                                     else 'full_far' if is_fixed(config) else 'diagnostic')
    evaluation_worker=ConcurrentFixedEvaluation(path,args,config) if is_fixed(config) else None
    try:
        collector = ProcessUpstreamRollout(args.config, 'A1', 23, owners=4, envs_per_owner=4, samples=512,
            episode_seconds=config['stages'][0]['deadline_s'] if is_fixed(config) else 1200., mode='train', cpu_workers=2, learning_time=timing,
            curriculum_config=config, cache_dir=path/'state/solver_cache')
        trainer = SkrlUpstreamTrainer(collector, options)
        if collector.env.spec.world_width != 1000. or collector.env.spec.world_height != 1000.:
            raise ValueError('Long curriculum needs the isolated 1km physical profile')
        from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_reward import (
            UpstreamReward,
            UpstreamRewardSettings, configured_reward_settings,
        )
        expected_reward = UpstreamReward(collector.env.spec, 1,
            configured_reward_settings(g,l,config), obstacle_input=collector.env.cfg.obstacle_input).metadata
        if static_audit['reward']['settings'] != expected_reward:
            raise ValueError('Audited reward/profile differs from the training configuration')
        if args.resume_state:
            checkpoint = Path(args.resume_state).resolve()
            if not checkpoint.is_relative_to((path/'checkpoints').resolve()):
                raise ValueError('Resume requires this run\'s own full checkpoint')
            saved = read_checkpoint(checkpoint)
            if saved['identity'] != identity:
                raise ValueError('Cross-semantics resume is forbidden')
            trainer.load_state_dict(saved['trainer'])
            updates = saved['session_updates']
            last_checkpoint = str(checkpoint)
            del saved
            if any(int(p.name.split('_')[1]) > updates for p in (path/'metrics').glob('collection_*.json')):
                raise ValueError('Uncommitted physical steps exist; reconcile before resume')
            if list((path/'metrics').glob('failure_*.json')):
                raise ValueError('Failure evidence requires reconciliation before resume')
        else:
            initialize_fresh_policy(collector.policy, config)
            json_write(path/'config/effective_configuration.json', {
                'environment': json.loads(collector.env.cfg.configuration_json), 'collector': collector.metadata,
                'learning_time': timing.metadata(.2), 'options': options.metadata, 'curriculum': config})
            json_write(path/'metrics/initialization.json', {'seed': 23, 'optimizer': 'fresh_Adam',
                'network_sha256': parameter_digest(collector.policy), 'inherited_checkpoint': None,
                'initial_continuous_std_scale': config.get('initial_continuous_std_scale', .25),
                'initial_hold_probability': config.get('initial_hold_probability', .5)})

        def checkpoint_now():
            nonlocal last_checkpoint
            target = path/'checkpoints'/f'update_{updates:06d}.pt'
            if str(target) != last_checkpoint:
                save_checkpoint(trainer, identity, updates, target, scope='curriculum_1km')
                last_checkpoint = str(target)

        checkpoint_now()
        phase = 'evaluation'
        tensorboard.sync()
        if evaluation_worker is None:
            evaluate(path, 'initial', collector, args, config, tensorboard=tensorboard)
        else:
            if args.resume_state:evaluation_worker.resume_pending()
            evaluation_worker.start('initial',collector)
        tensorboard.sync()
        audit = admission(collector, args.budget)
        initial_updates = updates
        while not stop:
            count = collection_decisions(config, collector.curriculum_progress, args.budget,
                                         collector.base_steps, 16, timing.steps(.2))
            if not count:
                if remaining_decisions(args.budget, collector.base_steps, 16, timing.steps(.2), 1):
                    stop.append('stage_budget_exhausted')
                break
            collection_start = time.perf_counter()
            before_steps = collector.base_steps
            phase = 'sampling'
            def on_chunk(batch, episodes, records, remaining, count=count, update=updates+1):
                elapsed=time.perf_counter()-collection_start
                json_write(path/'metrics'/f'collection_{update:06d}_{count-remaining:03d}.json',
                           event_data({'actual_base_steps': collector.base_steps, 'episodes': episodes,
                                       'base_steps_per_second':(collector.base_steps-before_steps)/elapsed}))
                tensorboard.sync()
                if evaluation_worker is not None:evaluation_worker.poll()
            batch, episodes, records = trainer.collect(count, on_chunk=on_chunk,
                **({'chunk_decisions': 1} if is_fixed(config) else {}))
            problem = learning_batch_problem(batch)
            if problem is not None:
                save_state(run, f'invalid_batch_{updates+1:06d}', {'batch': batch, 'execution': records})
                raise ValueError(problem)
            save_state(run, f'interactions_{updates+1:06d}', {'samples': batch['samples'],
                'learning': {k: v.cpu() for k, v in batch.items() if torch.is_tensor(v)},
                'execution': records, 'update_rng': torch.get_rng_state().clone()})
            collect_seconds = time.perf_counter()-collection_start
            phase = 'updating'
            result = trainer.update(batch, readiness=audit, training_authorized=True)
            updates += 1
            collector.curriculum_progress, promoted = update_progress(config, collector.curriculum_progress, episodes, collector.base_steps)
            if promoted:
                collector.set_curriculum_level(collector.curriculum_progress['level'])
            holds = torch.stack([s.action.hold for s in batch['samples']]).float().mean(-1)
            report = event_data({'actual_base_steps': collector.base_steps, 'policy_decisions': collector.decisions,
                'valid_transitions': collector.transitions, 'sampling_updates': updates, 'ppo_epochs': trainer.updates,
                'masked_environment_invalid_transitions': int(((batch['duration'] > 0) & ~batch['learning_valid']).sum()),
                'curriculum': collector.curriculum_progress, 'episodes': episodes, 'learning': result,
                'hold_fraction': float((holds*batch['duration']).sum()/batch['duration'].sum()),
                'collection_seconds': collect_seconds, 'wall_seconds_this_invocation': time.perf_counter()-started})
            if collector.env.cfg.obstacle_input is not None:
                report['obstacle']=obstacle_summary(records,collector.env.cfg.obstacle_input.data['cost']['soft_band_m'])
            json_write(path/'metrics'/f'update_{updates:06d}.json', report)
            tensorboard.sync()
            del batch, episodes, records
            trainer.release_rollout()
            gc.collect()
            due_eval = (not bounded_evaluation(config) and
                        collector.base_steps//config['evaluation_every_base_steps'] > before_steps//config['evaluation_every_base_steps'])
            phase = 'checkpoint'
            if updates <= 2 or promoted or due_eval or updates % config['checkpoint_every_updates'] == 0 or stop:
                checkpoint_now()
            print(json.dumps({k: report[k] for k in ('actual_base_steps', 'sampling_updates', 'hold_fraction')} |
                             {'level': collector.curriculum_progress['level']}), flush=True)
            if due_eval and not stop:
                phase = 'evaluation'
                full = (config['version'] == 'rendezvous_curriculum_1km_v2' and
                        any(before_steps < x <= collector.base_steps for x in config['full_evaluation_at_base_steps']))
                # The final panels already contain the scheduled full panels.
                ending_now = collection_decisions(config, collector.curriculum_progress, args.budget,
                                                  collector.base_steps, 16, timing.steps(.2)) == 0
                if not (ending_now and (config['version'] == 'rendezvous_curriculum_1km_v2' or is_fixed(config))):
                    if evaluation_worker is None:
                        evaluate(path, f'step_{collector.base_steps:07d}', collector, args, config, full=full, tensorboard=tensorboard)
                    else:
                        evaluation_worker.start(f'step_{collector.base_steps:07d}',collector,full=full)
                    tensorboard.sync()
            if args.updates_this_invocation is not None and updates-initial_updates >= args.updates_this_invocation:
                stop.append('requested_update_boundary')
        phase = 'checkpoint'
        checkpoint_now()
        complete = remaining_decisions(args.budget, collector.base_steps, 16, timing.steps(.2), 1) == 0
        if complete or stop == ['stage_budget_exhausted']:
            phase = 'evaluation'
            if evaluation_worker is not None:evaluation_worker.wait()
            evaluate(path, 'final', collector, args, config, final=True, tensorboard=tensorboard)
        summary = {'budget_complete': complete, 'actual_base_steps': collector.base_steps,
            'unused_base_steps': args.budget-collector.base_steps, 'sampling_updates': updates,
            'ppo_epochs': trainer.updates, 'policy_decisions': collector.decisions,
            'curriculum': collector.curriculum_progress, 'last_checkpoint': last_checkpoint,
            'stop_reason': 'budget_complete' if complete else stop,
            'evaluation_actual_base_steps': sum(json.loads(p.read_text()).get('actual_base_steps', 0)
                for p in (path/'metrics').glob('evaluation_*.json')), **convergence_report(path, config)}
        json_write(path/'metrics'/f'summary_{updates:06d}.json', summary)
        tensorboard.sync()
    except BaseException as error:
        json_write(path/'metrics'/f'failure_{time.time_ns()}.json', {'phase': phase, 'error': repr(error),
            'observed_base_steps': None if collector is None else collector.base_steps,
            'last_complete_checkpoint': last_checkpoint, 'automatic_retry': False})
        raise
    finally:
        if evaluation_worker is not None:evaluation_worker.close()
        tensorboard.close()
        if collector is not None:
            collector.close()
