"""Explicitly authorized, bounded 1 km curriculum learning through the T11 entry."""
import gc
import json
import math
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
)
from planner_evidence import file_sha256, json_write, source_snapshot
from upstream_checkpoint import read_checkpoint, save_checkpoint
from upstream_decision_experiment import initialize_exploration, parameter_digest
from upstream_process_rollout import ProcessUpstreamRollout
from upstream_run_state import save_state


def load_curriculum(path):
    config = yaml.safe_load(Path(path).read_text())
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
    scale = config['initial_continuous_std_scale'] if config['version'] == 'rendezvous_curriculum_1km_v2' else .25
    initialize_exploration(policy, scale)
    if config['version'] == 'rendezvous_curriculum_1km_v2':
        p = config['initial_hold_probability']
        with torch.no_grad():
            policy.hold_logit.bias.fill_(math.log(p / (1-p)))


def evaluation_panels(config, level, phase, *, final=False, full=False):
    """Separate fixed-duration trends from full episodes used for convergence."""
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


def evaluate(path, phase, collector, args, config, *, final=False, full=False):
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
    json_write(path/'metrics'/f'evaluation_{phase}_reservation.json',
               {'reserved_base_steps': reserve, 'training_base_steps': collector.base_steps,
                'levels': sorted({i for _, i, _, _ in panels}), 'panels': panels})
    results = {}
    with torch.random.fork_rng(devices=[]):
        for prefix, i, seconds, complete_episode in panels:
            for deterministic in (False, True):
                c = make_collector(args.config, 'A1', 104729, samples=512, num_envs=4, episode_seconds=1200.,
                    policy=deepcopy(collector.policy).cpu().eval(), learning_time=collector.learning_time,
                    curriculum_config=config)
                c.env.curriculum.set_level(i)
                poses = c.env.curriculum.sample(np.arange(4), replay=False)
                c.env.task.deadlines[:] = c.env.curriculum.deadlines
                c.env.reset(np.arange(4), poses)
                torch.manual_seed(104759)
                row = panel(c, deterministic=deterministic, seconds=seconds)
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


def convergence_report(path, config):
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


def run_experiment(args):
    if (args.mode != 'train' or not args.release_training_pause or args.representation != 'A1' or args.seed != 23
            or args.budget not in (1000000, 2000000) or args.env_owners != 4 or args.num_envs != 16
            or args.owner_cpu_workers != 2 or args.samples != 512 or args.trainer_backend != 'skrl'
            or args.policy_device != 'cpu' or args.gpu_service or args.import_checkpoint or args.import_learner
            or args.authorize_signal100k or args.decision_credit_arm or args.comparison_run):
        raise ValueError('Long curriculum requires explicitly released A1/seed23/K512/4x4/skrl/CPU/1M or 2M base steps')
    if not args.run_id or Path(args.run_id).name != args.run_id or args.run_id in ('.', '..'):
        raise ValueError('Expected an owned run name')
    config = load_curriculum(args.curriculum_config)
    if (config['version'] == 'rendezvous_curriculum_1km_v2'
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
    if config['version'] == 'rendezvous_curriculum_1km_v2' and static_audit.get('curriculum_config') != config:
        raise ValueError('Prelaunch audit belongs to a different curriculum')
    timing = timing_for(config)
    g, l = timing.discounts(.2)
    options = TrainingOptions(rollout_macro_steps=config['rollout_decisions'], epochs=config['ppo_epochs'],
        gamma=g, lam=l, gradient_clipping='actor_critic', terminal_actor_credit='effective',
        value_normalization=config['value_normalization'])
    path = ROOT/'outputs/runs/upstream_1km_curriculum'/args.run_id
    identity = {'version': config['version'], 'config': config, 'budget_base_steps': args.budget,
                'base_config': str(Path(args.config).resolve()), 'base_config_sha256': file_sha256(args.config),
                'source_snapshot': source_snapshot(), 'options': options.metadata, 'seed': 23,
                'prelaunch_audit': str(evidence_path), 'prelaunch_audit_sha256': file_sha256(evidence_path)}
    if path.exists():
        if not args.resume_state or json.loads((path/'run_manifest.json').read_text())['identity'] != identity:
            raise ValueError('Run exists or source/configuration identity changed')
    else:
        if args.resume_state:
            raise ValueError('Cannot resume into a new run')
        for directory in ('config', 'metrics', 'state', 'logs', 'checkpoints'):
            (path/directory).mkdir(parents=True, exist_ok=True)
        json_write(path/'run_manifest.json', {'identity': identity, 'command': sys.argv,
                   'formal_training': True, 'scope': 'user_authorized_proxy_curriculum_long_learning',
                   'system_acceptance_complete': False, 'physx_acceptance': False})
        with tarfile.open(path/'config/source_and_configs.tar.gz', 'w:gz') as archive:
            for directory in ('source/lunar_rover_tasks/lunar_rover_tasks/tasks/multi_rover_gathering', 'scripts', 'configs', 'tests'):
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
    try:
        collector = ProcessUpstreamRollout(args.config, 'A1', 23, owners=4, envs_per_owner=4, samples=512,
            episode_seconds=1200., mode='train', cpu_workers=2, learning_time=timing,
            curriculum_config=config, cache_dir=path/'state/solver_cache')
        trainer = SkrlUpstreamTrainer(collector, options)
        if collector.env.spec.world_width != 1000. or collector.env.spec.world_height != 1000.:
            raise ValueError('Long curriculum needs the isolated 1km physical profile')
        from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_reward import (
            UpstreamReward,
            UpstreamRewardSettings,
        )
        expected_reward = UpstreamReward(collector.env.spec, 1,
            UpstreamRewardSettings(gamma=g, lam=l, gather_scaling=config['reward_scaling'])).metadata
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
        evaluate(path, 'initial', collector, args, config)
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
                json_write(path/'metrics'/f'collection_{update:06d}_{count-remaining:03d}.json',
                           event_data({'actual_base_steps': collector.base_steps, 'episodes': episodes}))
            batch, episodes, records = trainer.collect(count, on_chunk=on_chunk)
            if (not all(torch.isfinite(batch[k]).all() for k in ('obs', 'state', 'reward', 'value', 'next_value'))
                    or not batch['learning_valid'][batch['duration'] > 0].all()
                    or not all(torch.isfinite(s.log_prob).all() for s in batch['samples'])):
                save_state(run, f'invalid_batch_{updates+1:06d}', {'batch': batch, 'execution': records})
                raise ValueError('Invalid or nonfinite learning data')
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
                'curriculum': collector.curriculum_progress, 'episodes': episodes, 'learning': result,
                'hold_fraction': float((holds*batch['duration']).sum()/batch['duration'].sum()),
                'collection_seconds': collect_seconds, 'wall_seconds_this_invocation': time.perf_counter()-started})
            json_write(path/'metrics'/f'update_{updates:06d}.json', report)
            del batch, episodes, records
            trainer.release_rollout()
            gc.collect()
            due_eval = collector.base_steps//config['evaluation_every_base_steps'] > before_steps//config['evaluation_every_base_steps']
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
                if not (ending_now and config['version'] == 'rendezvous_curriculum_1km_v2'):
                    evaluate(path, f'step_{collector.base_steps:07d}', collector, args, config, full=full)
            if args.updates_this_invocation is not None and updates-initial_updates >= args.updates_this_invocation:
                stop.append('requested_update_boundary')
        phase = 'checkpoint'
        checkpoint_now()
        complete = remaining_decisions(args.budget, collector.base_steps, 16, timing.steps(.2), 1) == 0
        if complete or stop == ['stage_budget_exhausted']:
            phase = 'evaluation'
            evaluate(path, 'final', collector, args, config, final=True)
        summary = {'budget_complete': complete, 'actual_base_steps': collector.base_steps,
            'unused_base_steps': args.budget-collector.base_steps, 'sampling_updates': updates,
            'ppo_epochs': trainer.updates, 'policy_decisions': collector.decisions,
            'curriculum': collector.curriculum_progress, 'last_checkpoint': last_checkpoint,
            'stop_reason': 'budget_complete' if complete else stop,
            'evaluation_actual_base_steps': sum(json.loads(p.read_text()).get('actual_base_steps', 0)
                for p in (path/'metrics').glob('evaluation_*.json')), **convergence_report(path, config)}
        json_write(path/'metrics'/f'summary_{updates:06d}.json', summary)
    except BaseException as error:
        json_write(path/'metrics'/f'failure_{time.time_ns()}.json', {'phase': phase, 'error': repr(error),
            'observed_base_steps': None if collector is None else collector.base_steps,
            'last_complete_checkpoint': last_checkpoint, 'automatic_retry': False})
        raise
    finally:
        if collector is not None:
            collector.close()
