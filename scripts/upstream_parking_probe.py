#!/usr/bin/env python3
"""Bounded parking/approach acquisition probe; no old training-state import.

Four fixed environment slots give exactly half the sampled team decisions to
parking, irrespective of episode length. This explicit research launcher uses
the production recurrent PPO objective without claiming old formal readiness.
"""
import argparse
from copy import deepcopy
import gc
import json
from pathlib import Path
import signal
import sys
import tarfile
import time

import numpy as np
import torch

from _common import ROOT
from diagnose_upstream_environment import event_data
from planner_evidence import file_sha256, json_write, source_snapshot
from training_tensorboard import TrainingTensorBoard, TENSORBOARD_METADATA
from run_upstream_t11 import session_configuration
from upstream_checkpoint import save_tensor_file
from upstream_curriculum_experiment import initialize_fresh_policy, load_curriculum, timing_for
from upstream_decision_evaluation import panel
from upstream_local_closure import runtime_config
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.action_ablation import (
    build_ablation_policies, replay_action_sequence,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.learning import duration_gae
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_ablation_learning import UpstreamAblationRollout
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_environment import UpstreamEnvironment
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_observation import UpstreamObservationEncoder
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_reward import UpstreamRewardSettings
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_training import TrainingOptions, recurrent_mappo_update


VERSION = 't11_balanced_parking_probe_v1'
BASE_LIMIT = 10000
EVALUATION_LIMIT = 3200
SLOTS = ('parking_moving', 'parking_stopped', 'approach_12m', 'approach_18m')


def initial_conditions(kind):
    """Fixed legal states; no teacher actions or preloaded dwell counter."""
    poses = np.zeros((4, 4, 3))
    controls = np.zeros((4, 4, 2))
    square = np.array([[-3.6, -3.6], [-3.6, 3.6], [3.6, -3.6], [3.6, 3.6]])
    for e in range(4):
        yaw = e * np.pi / 2
        rotation = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        poses[e, :, :2] = square @ rotation.T
        poses[e, :, 2] = (yaw + np.pi) % (2 * np.pi) - np.pi
        controls[e, :, 0] = .1 if e % 2 == 0 else 0.
    deadlines = np.full(4, 30.)
    if kind == 'parking':
        return poses, controls, deadlines
    if kind not in ('training', 'approach'):
        raise ValueError('Unknown probe initial condition panel')
    indices = (2, 3) if kind == 'training' else range(4)
    for e in indices:
        radius = (12., 18., 12., 18.)[e]
        yaw = e * np.pi / 2
        angles = np.arange(4) * np.pi / 2 + np.pi / 4 + yaw
        poses[e, :, :2] = radius * np.stack([np.cos(angles), np.sin(angles)], axis=-1)
        noise = np.deg2rad(60. if e % 2 == 0 else -60.)
        poses[e, :, 2] = (angles + np.pi + noise + np.pi) % (2 * np.pi) - np.pi
        controls[e, :, 0] = .1
        deadlines[e] = 300.
    return poses, controls, deadlines


class ProbeRollout(UpstreamAblationRollout):
    @property
    def metadata(self):
        return {**super().metadata, 'research_probe': VERSION,
                'fixed_slots': SLOTS, 'deadline_seconds': self.env.task.deadlines.tolist(),
                'initial_poses': self.env._initial.tolist(),
                'formal_training': False, 'automatic_resume_supported': False}


def make_probe_collector(base, config, *, policy=None, kind='training'):
    cfg, _ = runtime_config(base, name=VERSION, seed=23, samples=512,
                            num_envs=4, episode_seconds=300., safety=True)
    cfg = session_configuration(cfg, 'signal' if kind == 'training' else 'diagnostic')
    if cfg.spec.world_width != 1000. or cfg.spec.world_height != 1000. or cfg.planner_device != 'cuda':
        raise ValueError('Probe requires the existing 1 km CUDA MPPI runtime')
    timing = timing_for(config)
    if timing.steps(.2) != 25:
        raise ValueError('Probe requires unchanged 5-second decisions')
    if policy is None:
        torch.manual_seed(23)
        policy = build_ablation_policies(991, cfg.spec)['A1']
        initialize_fresh_policy(policy, config)
        policy.enable_value_normalization()
    poses, controls, deadlines = initial_conditions(kind)
    g, l = timing.discounts(.2)
    UpstreamEnvironment.cpu_workers = 2
    env = UpstreamEnvironment(cfg, poses, initial_control=controls,
        reward_settings=UpstreamRewardSettings(gamma=g, lam=l, gather_scaling=config['reward_scaling']))
    env.encoder = UpstreamObservationEncoder(cfg.spec, spatial_scaling=config['observation_scaling'])
    env.task.deadlines[:] = deadlines
    return ProbeRollout(env, policy, timing)


def concatenate(parts, metadata):
    batch = {k: torch.cat([p[k] for p in parts]) for k in parts[0] if torch.is_tensor(parts[0][k])}
    batch.update(samples=[s for p in parts for s in p['samples']], metadata=deepcopy(metadata))
    return batch


def exposure(batch):
    result = {}
    for name, slots in (('parking', slice(0, 2)), ('approach', slice(2, 4))):
        duration = batch['duration'][:, slots]
        result[name] = {'actual_base_steps': int(duration.sum()),
                        'sampled_team_decisions': int((duration > 0).sum()),
                        'valid_team_decisions': int(batch['learning_valid'][:, slots].sum()),
                        'actor_effective_team_decisions': int((batch['learning_valid'][:, slots]
                                                              & batch['action_effective'][:, slots]).sum())}
    if result['parking']['sampled_team_decisions'] != result['approach']['sampled_team_decisions']:
        raise AssertionError('Fixed slots failed to produce balanced actual decisions')
    return result


def behavior(batch, records):
    holds = torch.stack([s.action.hold for s in batch['samples']]).float()
    max_dwell = np.zeros(4, dtype=np.int64)
    resets = np.zeros(4, dtype=np.int64)
    previous = np.zeros(4, dtype=np.int64)
    previous_ended = np.zeros(4, dtype=bool)
    for record in records:
        if 'task' not in record:
            continue
        count = np.asarray(record['task']['hold_count'])
        max_dwell = np.maximum(max_dwell, count)
        ended = np.asarray(record['task']['terminated'])
        resets += (previous > 0) & (count == 0) & ~ended & ~previous_ended
        previous = count
        previous_ended = ended
    return {'parking_hold_fraction': float(holds[:, :2].mean()),
            'approach_hold_fraction': float(holds[:, 2:].mean()),
            'maximum_dwell_steps_by_slot': max_dwell.tolist(),
            'dwell_interrupted_within_batch_by_slot': resets.tolist()}


@torch.no_grad()
def validate_batch(collector, batch, budget):
    if batch['metadata'] != collector.metadata or not 0 < collector.base_steps <= budget:
        raise ValueError('Probe learning identity/budget mismatch')
    if (not batch['learning_valid'].all() or not torch.isfinite(batch['obs']).all()
            or not torch.isfinite(batch['state']).all() or not torch.isfinite(batch['reward']).all()):
        raise ValueError('Invalid learning samples consume physical budget and stop this probe')
    replay = replay_action_sequence(collector.policy, batch['obs'], batch['hidden'][0],
        batch['samples'], batch['learning_valid'], batch['terminated'])
    error = float((replay['ratio'][batch['learning_valid']] - 1).abs().max())
    if error > 2e-5:
        raise ValueError(f'Probability replay mismatch before PPO: {error}')
    return error


def frozen_targets(batch, options):
    _, targets = duration_gae(batch['reward'], batch['value'], batch['next_value'], batch['duration'],
        batch['terminated'], batch['truncated'], gamma=options.gamma, lam=options.lam,
        bootstrap_allowed=batch['bootstrap_allowed'], learning_valid=batch['learning_valid'])
    return targets[batch['learning_valid']].detach()


def extra_critic_fit(policy, optimizer, batch, targets, options, count):
    """Optional isolated test: same fixed returns/moments, no extra actor steps."""
    losses = []
    target = policy.value_normalizer.normalize(targets)
    for _ in range(count):
        optimizer.zero_grad(set_to_none=True)
        loss = options.value_weight * torch.nn.functional.mse_loss(
            policy.critic(batch['state'][batch['learning_valid']]).squeeze(-1), target)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite extra critic loss')
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(policy.critic.parameters(), options.gradient_limit)
        if not torch.isfinite(norm):
            raise FloatingPointError('Nonfinite extra critic gradient')
        optimizer.step()
        losses.append(float(loss.detach()))
    return losses


def load_prefit(policy, path):
    saved = torch.load(path, map_location='cpu', weights_only=False)
    if saved.get('schema_version') != 'parking_probe_critic_prefit_v1':
        raise ValueError('Explicit critic-only prefit format required; old learner checkpoints are not imports')
    policy.critic.load_state_dict(saved['critic_state_dict'], strict=True)
    policy.value_normalizer.load_state_dict(saved['value_normalizer_state_dict'], strict=True)
    if not all(torch.isfinite(x).all() for x in policy.state_dict().values()):
        raise ValueError('Nonfinite critic-only prefit state')


def checkpoint(path, collector, optimizer, updates, critic_updates, identity, name):
    target = path/'checkpoints'/f'{name}.pt'
    descriptor = save_tensor_file(target, {'schema_version': VERSION, 'identity': identity,
        'collector': collector.state_dict(), 'optimizer': optimizer.state_dict(),
        'sampling_updates': updates, 'ppo_epochs': updates * 5, 'extra_critic_updates': critic_updates,
        'formal_training': False, 'automatic_resume_supported': False})
    json_write(target.with_suffix('.json'), {**descriptor, 'actual_base_steps': collector.base_steps,
                                           'policy_decisions': collector.decisions, 'sampling_updates': updates})
    return str(target)


def evaluate(path, phase, collector, base, config):
    # Eight runs × four cases × 100 steps = 3200 for initial AND final combined.
    json_write(path/'metrics'/f'evaluation_{phase}_reservation.json', {'reserved_base_steps': 1600,
        'training_base_steps': collector.base_steps, 'optimizer_updates': 0})
    results = {}
    with torch.random.fork_rng(devices=[]):
        for kind in ('parking', 'approach'):
            for deterministic in (False, True):
                c = make_probe_collector(base, config, policy=deepcopy(collector.policy).eval(), kind=kind)
                torch.manual_seed(20261021)
                row = panel(c, deterministic=deterministic, seconds=20)
                key = f'{kind}_{"mean" if deterministic else "sample"}'
                row['short_screen_only'] = True
                row['full_episode_evaluation'] = False
                row['initial_speed_mps'] = initial_conditions(kind)[1][..., 0]
                json_write(path/'metrics'/f'trace_{phase}_{key}.json', event_data(row.pop('trace')))
                results[key] = row
                del c
                gc.collect()
    actual = sum(row['actual_base_steps'] for row in results.values())
    if actual > 1600:
        raise AssertionError('Independent evaluation exceeded its reservation')
    result = event_data({'actual_base_steps': actual, 'reserved_base_steps': 1600,
                        'training_base_steps': collector.base_steps, 'results': results, 'optimizer_updates': 0})
    json_write(path/'metrics'/f'evaluation_{phase}.json', result)
    return result


def compare(initial, final):
    before, after = initial['results'], final['results']
    checks = {
        'parking_success_improves_one_mode': any(after[f'parking_{m}']['successes'] > before[f'parking_{m}']['successes']
                                               for m in ('sample', 'mean')),
        'parking_no_success_regression': all(after[f'parking_{m}']['successes'] >= before[f'parking_{m}']['successes']
                                            for m in ('sample', 'mean')),
        'parking_hold_above_approach': all(np.mean(after[f'parking_{m}']['hold_probability'])
                                         > np.mean(after[f'approach_{m}']['hold_probability']) for m in ('sample', 'mean')),
        'approach_not_all_hold': all(float(np.mean(after[f'approach_{m}']['hold_fraction'])) < .9
                                     for m in ('sample', 'mean')),
        'approach_no_mean_net_divergence': all(np.mean(after[f'approach_{m}']['dmax_reduction']) >= 0
                                              for m in ('sample', 'mean')),
    }
    return {'screening_checks': checks, 'parking_acquisition_screen_passed': all(checks.values()),
            'full_gathering_acceptance': False, 'physx_acceptance': False, 'statistical_significance_claim': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/experiment/upstream_1km_curriculum_v1.yaml')
    parser.add_argument('--curriculum-config', default='configs/training/upstream_1km_curriculum_v2.yaml')
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--budget', type=int, default=BASE_LIMIT)
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                        help='Adam learning rate for the bounded probe (0 < value <= 1e-3)')
    parser.add_argument('--authorize-bounded-learning', action='store_true')
    parser.add_argument('--critic-prefit', help='Explicit critic-only file; fresh actor and Adam remain mandatory')
    parser.add_argument('--critic-fit-steps', type=int, default=0, help='Extra critic steps on each fixed PPO return target')
    args = parser.parse_args()
    if not args.authorize_bounded_learning or not 100 <= args.budget <= BASE_LIMIT or not 0 <= args.critic_fit_steps <= 100:
        raise ValueError('Explicit authorization and at most 10000 physical learning steps are required')
    if not np.isfinite(args.learning_rate) or not 0 < args.learning_rate <= 1e-3:
        raise ValueError('Probe learning rate must be finite and satisfy 0 < value <= 1e-3')
    if Path(args.run_id).name != args.run_id or args.run_id in ('.', '..'):
        raise ValueError('Expected a new owned run name')
    path = ROOT/'outputs/runs/upstream_1km_curriculum'/args.run_id
    path.mkdir(parents=True, exist_ok=False)
    for folder in ('metrics', 'checkpoints', 'state', 'config', 'logs'):
        (path/folder).mkdir()
    config = load_curriculum(args.curriculum_config)
    if config['version'] != 'rendezvous_curriculum_1km_v2' or config['ppo_epochs'] != 5:
        raise ValueError('Probe expects the reviewed v2 source protocol')
    identity = {'schema_version': VERSION, 'base_config': str(Path(args.config).resolve()),
        'base_config_sha256': file_sha256(args.config), 'source_snapshot': source_snapshot(),
        'curriculum_config': config, 'seed': 23, 'training_base_step_limit': args.budget,
        'evaluation_base_step_limit': EVALUATION_LIMIT, 'fixed_slots': SLOTS,
        'initial_conditions': {k: initial_conditions(k) for k in ('training', 'parking', 'approach')},
        'learning_rate': args.learning_rate, 'critic_fit_steps': args.critic_fit_steps,
        'critic_prefit': None if args.critic_prefit is None else {'file': str(Path(args.critic_prefit).resolve()),
                                                              'sha256': file_sha256(args.critic_prefit)},
        'fresh_adam': True, 'formal_training': False, 'automatic_resume_supported': False,
        'learning_objective': 'production_recurrent_mappo_update', 'budget_counts_invalid_physical_steps': True}
    identity = event_data(identity)
    json_write(path/'run_manifest.json', {'identity': identity, 'command': sys.argv, 'tensorboard': TENSORBOARD_METADATA})
    with tarfile.open(path/'config/source_snapshot.tar.gz', 'w:gz') as archive:
        for name in identity['source_snapshot']:
            archive.add(ROOT/name, arcname=name)
    torch.set_num_threads(1)
    stop = []
    def request_stop(signum, _frame):
        stop.append(signum)
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    collector = None; optimizer = None; last_checkpoint = None
    updates = 0; critic_updates = 0; phase = 'initializing'; started = time.monotonic()
    totals = {name: {key: 0 for key in ('actual_base_steps', 'sampled_team_decisions',
                                     'valid_team_decisions', 'actor_effective_team_decisions')}
              for name in ('parking', 'approach')}
    tensorboard = TrainingTensorBoard(path)
    try:
        collector = make_probe_collector(args.config, config)
        if args.critic_prefit:
            load_prefit(collector.policy, args.critic_prefit)
        g, l = collector.learning_time.discounts(.2)
        options = TrainingOptions(learning_rate=args.learning_rate, rollout_macro_steps=4, epochs=5, gamma=g, lam=l,
            gradient_clipping='actor_critic', terminal_actor_credit='effective', value_normalization='running')
        optimizer = torch.optim.Adam(collector.policy.parameters(), lr=options.learning_rate)
        json_write(path/'config/effective_configuration.json', {'runtime': json.loads(collector.env.cfg.configuration_json),
                   'collector': collector.metadata, 'optimizer': options.metadata})
        last_checkpoint = checkpoint(path, collector, optimizer, updates, critic_updates, identity, 'initial')
        phase = 'initial_evaluation'
        initial = evaluate(path, 'initial', collector, args.config, config)
        tensorboard.sync()
        while not stop and args.budget - collector.base_steps >= 100:
            phase = 'sampling'; parts = []; episodes = []; records = []
            for decision in range(4):
                if stop or args.budget - collector.base_steps < 100:
                    break
                json_write(path/'metrics'/f'reservation_{updates+1:06d}_{decision:02d}.json',
                           {'already_spent_base_steps': collector.base_steps, 'reserved_base_steps': 100})
                part, ends, trace = collector.collect(1)
                parts.append(part); episodes.extend(ends); records.extend(trace)
                json_write(path/'metrics'/f'collection_{updates+1:06d}_{decision:02d}.json',
                           {'actual_base_steps': collector.base_steps, 'policy_decisions': collector.decisions})
                tensorboard.sync()
            if not parts:
                break
            batch = concatenate(parts, collector.metadata)
            save_tensor_file(path/'state'/f'update_{updates+1:06d}_input.pt',
                             {'batch': batch, 'records': records, 'episodes': episodes,
                              'policy': collector.policy.export_diagnostic_state(),
                              'optimizer': optimizer.state_dict(), 'torch_rng': torch.get_rng_state()})
            batch_exposure = exposure(batch)
            for name in totals:
                for key in totals[name]:
                    totals[name][key] += batch_exposure[name][key]
            ratio_error = validate_batch(collector, batch, args.budget)
            phase = 'updating'
            targets = frozen_targets(batch, options)
            result = recurrent_mappo_update(collector.policy, optimizer, batch, options)
            extra = extra_critic_fit(collector.policy, optimizer, batch, targets, options, args.critic_fit_steps)
            updates += 1; critic_updates += len(extra)
            last_checkpoint = checkpoint(path, collector, optimizer, updates, critic_updates, identity, f'update_{updates:06d}')
            json_write(path/'metrics'/f'update_{updates:06d}.json', event_data({
                'actual_base_steps': collector.base_steps, 'sampling_updates': updates, 'ppo_epochs': updates * 5,
                'extra_critic_updates': critic_updates, 'extra_critic_losses': extra, 'exposure': batch_exposure,
                'cumulative_exposure': totals, 'episodes': episodes, 'behavior': behavior(batch, records),
                'pre_update_ratio_max_error': ratio_error, 'learning': result, 'wall_seconds': time.monotonic()-started}))
            tensorboard.sync()
            print(json.dumps({'update': updates, 'actual_base_steps': collector.base_steps,
                              'parking_steps': totals['parking']['actual_base_steps'],
                              'approach_steps': totals['approach']['actual_base_steps']}), flush=True)
            del parts, batch, records, result, targets
            gc.collect()
        final = None
        if not stop:
            phase = 'final_evaluation'
            final = evaluate(path, 'final', collector, args.config, config)
        summary = {'actual_base_steps': collector.base_steps, 'remaining_base_steps': args.budget-collector.base_steps,
            'sampling_updates': updates, 'ppo_epochs': updates * 5, 'extra_critic_updates': critic_updates,
            'exposure': totals, 'last_complete_checkpoint': last_checkpoint,
            'stop_reason': 'signal_requested' if stop else 'physical_budget_exhausted',
            'evaluation_actual_base_steps': initial['actual_base_steps'] + (0 if final is None else final['actual_base_steps']),
            'wall_seconds': time.monotonic()-started, 'formal_training': False,
            **({} if final is None else compare(initial, final))}
        json_write(path/'metrics/summary.json', event_data(summary))
        tensorboard.sync()
    except BaseException as error:
        json_write(path/'metrics/failure.json', {'error': repr(error), 'phase': phase,
            'last_complete_checkpoint': last_checkpoint, 'completed_updates': updates,
            'known_base_steps': None if collector is None else collector.base_steps,
            'uncertain_current_reservation_base_steps': 100 if phase == 'sampling' else 0,
            'automatic_retry': False})
        if collector is not None and optimizer is not None:
            save_tensor_file(path/'state/failure_state.pt', {'collector': collector.state_dict(),
                'optimizer': optimizer.state_dict(), 'error': repr(error), 'phase': phase})
        raise

    finally:
        tensorboard.close()


if __name__ == '__main__':
    main()
