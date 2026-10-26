#!/usr/bin/env python3
"""Bounded on-policy local parking exploration with a prespecified withdrawal.

Reuses the balanced fixed-slot probe, fresh seed-23 weights and unchanged PPO.
No old optimizer, critic prefit, action override or extra direction loss.
"""
import argparse
from copy import deepcopy
from dataclasses import replace
import gc
import json
from pathlib import Path
import signal
import sys
import tarfile
import time

import torch

from _common import ROOT
from diagnose_upstream_environment import event_data
from planner_evidence import file_sha256, json_write, source_snapshot
from upstream_checkpoint import save_tensor_file
from upstream_curriculum_experiment import load_curriculum
from upstream_parking_probe import (
    SLOTS, behavior, compare, concatenate, evaluate, exposure,
    initial_conditions, make_probe_collector, validate_batch,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_parking_exploration import (
    ParkingExploration, exploration_strength,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_training import (
    TrainingOptions, recurrent_mappo_update,
)


VERSION = 't11_local_parking_exploration_probe_v1'
BASE_LIMIT = 10000
EVALUATION_LIMIT = 4800
BASELINE = 'outputs/runs/upstream_1km_curriculum/parking_exposure_seed23_10k_20261021_01'


def checkpoint(path, collector, optimizer, updates, identity, name):
    target = path / 'checkpoints' / f'{name}.pt'
    descriptor = save_tensor_file(target, {'schema_version': VERSION, 'identity': identity,
        'collector': collector.state_dict(), 'optimizer': optimizer.state_dict(),
        'sampling_updates': updates, 'ppo_epochs': updates * 5,
        'formal_training': False, 'automatic_resume_supported': False})
    json_write(target.with_suffix('.json'), {**descriptor, 'actual_base_steps': collector.base_steps,
        'policy_decisions': collector.decisions, 'sampling_updates': updates,
        'parking_exploration': collector.policy.parking_exploration.metadata})
    return str(target)


def verify_baseline(policy, baseline):
    """Reuse initial no-prior evaluation only after exact weight/identity checks."""
    initial = baseline / 'checkpoints/initial.pt'
    if file_sha256(initial) != json.loads(initial.with_suffix('.json').read_text())['sha256']:
        raise ValueError('Reference initial checkpoint checksum mismatch')
    saved = torch.load(initial, map_location='cpu', weights_only=False)
    expected = saved['collector']['policy']
    actual = policy.export_diagnostic_state()
    if actual['metadata'] != expected['metadata'] or set(actual['weights']) != set(expected['weights']):
        raise ValueError('Reference initial policy semantics differ')
    if not all(torch.equal(value, expected['weights'][key]) for key, value in actual['weights'].items()):
        raise ValueError('Reference initial policy tensors differ')
    if saved['optimizer']['state'] or saved['collector']['base_steps'] != 0:
        raise ValueError('Expected a fresh reference, not trained weights/Adam')
    evaluation = baseline / 'metrics/evaluation_initial.json'
    return json.loads(evaluation.read_text()), {
        'checkpoint': str(initial), 'checkpoint_sha256': file_sha256(initial),
        'evaluation': str(evaluation), 'evaluation_sha256': file_sha256(evaluation),
        'all_initial_tensors_exact': True, 'fresh_reference_adam': True,
        'new_environment_steps_for_reused_evaluation': 0,
    }


@torch.no_grad()
def prior_diagnostics(policy, batch):
    obs = batch['obs']
    prior = policy.parking_exploration
    compact = prior.compact_geometry(obs)
    hold = torch.stack([s.action.hold for s in batch['samples']])
    return {'settings': prior.metadata,
            'compact_actor_decisions_by_slot': compact.sum((0, 2)).tolist(),
            'continuation_eligible_by_slot': (compact & (obs[..., 14] > .5)).sum((0, 2)).tolist(),
            'all_four_hold_decisions_by_slot': hold.all(-1).sum(0).tolist()}


def main():
    from training_tensorboard import TrainingTensorBoard, TENSORBOARD_METADATA
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/experiment/upstream_1km_curriculum_v1.yaml')
    parser.add_argument('--curriculum-config', default='configs/training/upstream_1km_curriculum_v2.yaml')
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--baseline-run', default=BASELINE)
    parser.add_argument('--authorize-bounded-learning', action='store_true')
    args = parser.parse_args()
    if not args.authorize_bounded_learning:
        raise ValueError('Explicit bounded-learning authorization required')
    if Path(args.run_id).name != args.run_id or args.run_id in ('.', '..'):
        raise ValueError('Expected a new owned run name')
    config = load_curriculum(args.curriculum_config)
    if (config['version'] != 'rendezvous_curriculum_1km_v2' or config['ppo_epochs'] != 5
            or config['observation_scaling'] != 'log1p'):
        raise ValueError('Probe requires the reviewed v2 protocol')
    path = ROOT / 'outputs/runs/upstream_1km_curriculum' / args.run_id
    path.mkdir(parents=True, exist_ok=False)
    for folder in ('metrics', 'checkpoints', 'state', 'config', 'logs'):
        (path / folder).mkdir()
    baseline = (ROOT / args.baseline_run).resolve()
    identity = event_data({'schema_version': VERSION,
        'source_snapshot': source_snapshot(), 'base_config': str(Path(args.config).resolve()),
        'base_config_sha256': file_sha256(args.config), 'curriculum_config': config,
        'seed': 23, 'training_base_step_limit': BASE_LIMIT,
        'evaluation_base_step_limit': EVALUATION_LIMIT, 'fixed_slots': SLOTS,
        'initial_conditions': {k: initial_conditions(k) for k in ('training', 'parking', 'approach')},
        'baseline_run': str(baseline), 'fresh_actor_critic_and_adam': True,
        'strength_schedule': {'full_until_base_steps': 3000, 'zero_from_base_steps': 8000,
                              'change_only_before_batch': True},
        'extra_critic_updates': 0, 'extra_direction_updates': 0,
        'formal_training': False, 'automatic_resume_supported': False,
        'learning_objective': 'production_recurrent_mappo_update',
        'budget_counts_invalid_physical_steps': True})
    json_write(path / 'run_manifest.json', {'identity': identity, 'command': sys.argv, 'tensorboard': TENSORBOARD_METADATA})
    with tarfile.open(path / 'config/source_snapshot.tar.gz', 'w:gz') as archive:
        for name in identity['source_snapshot']:
            archive.add(ROOT / name, arcname=name)
    torch.set_num_threads(1)
    stop = []
    def request_stop(signum, _frame):
        stop.append(signum)
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    collector = None; optimizer = None; last_checkpoint = None
    updates = 0; phase = 'initializing'; evaluation_steps = 0; started = time.monotonic()
    totals = {name: {key: 0 for key in ('actual_base_steps', 'sampled_team_decisions',
                                     'valid_team_decisions', 'actor_effective_team_decisions')}
              for name in ('parking', 'approach')}
    tensorboard = TrainingTensorBoard(path)
    try:
        collector = make_probe_collector(args.config, config)
        initial_no_prior, reference = verify_baseline(collector.policy, baseline)
        json_write(path / 'metrics/reused_initial_no_prior.json', {'reference': reference, 'evaluation': initial_no_prior})
        settings = ParkingExploration.from_spec(collector.env.spec, spatial_scaling=config['observation_scaling'])
        collector.policy.set_parking_exploration(settings)
        g, l = collector.learning_time.discounts(.2)
        options = TrainingOptions(rollout_macro_steps=4, epochs=5, gamma=g, lam=l,
            gradient_clipping='actor_critic', terminal_actor_credit='effective', value_normalization='running')
        optimizer = torch.optim.Adam(collector.policy.parameters(), lr=options.learning_rate)
        json_write(path / 'config/effective_configuration.json', {
            'runtime': json.loads(collector.env.cfg.configuration_json),
            'collector': collector.metadata, 'optimizer': options.metadata})
        last_checkpoint = checkpoint(path, collector, optimizer, updates, identity, 'initial')
        phase = 'initial_evaluation'
        initial_prior = evaluate(path, 'initial_prior', collector, args.config, config)
        evaluation_steps += initial_prior['actual_base_steps']
        tensorboard.sync()
        while not stop and BASE_LIMIT - collector.base_steps >= 100:
            phase = 'sampling'; parts = []; episodes = []; records = []
            collector.policy.set_parking_exploration(replace(settings, strength=exploration_strength(collector.base_steps)))
            for decision in range(4):
                if stop or BASE_LIMIT - collector.base_steps < 100:
                    break
                json_write(path / 'metrics' / f'reservation_{updates+1:06d}_{decision:02d}.json', {
                    'already_spent_base_steps': collector.base_steps, 'reserved_base_steps': 100,
                    'strength': collector.policy.parking_exploration.strength})
                part, ends, trace = collector.collect(1)
                parts.append(part); episodes.extend(ends); records.extend(trace)
                json_write(path / 'metrics' / f'collection_{updates+1:06d}_{decision:02d}.json', {
                    'actual_base_steps': collector.base_steps, 'policy_decisions': collector.decisions})
                tensorboard.sync()
            if not parts:
                break
            batch = concatenate(parts, collector.metadata)
            save_tensor_file(path / 'state' / f'update_{updates+1:06d}_input.pt', {
                'batch': batch, 'records': records, 'episodes': episodes,
                'policy': collector.policy.export_diagnostic_state(),
                'optimizer': optimizer.state_dict(), 'torch_rng': torch.get_rng_state()})
            batch_exposure = exposure(batch)
            for name in totals:
                for key in totals[name]:
                    totals[name][key] += batch_exposure[name][key]
            ratio_error = validate_batch(collector, batch, BASE_LIMIT)
            prior_stats = prior_diagnostics(collector.policy, batch)
            phase = 'updating'
            result = recurrent_mappo_update(collector.policy, optimizer, batch, options)
            updates += 1
            last_checkpoint = checkpoint(path, collector, optimizer, updates, identity, f'update_{updates:06d}')
            metrics = event_data({'actual_base_steps': collector.base_steps,
                'policy_decisions': collector.decisions, 'sampling_updates': updates, 'ppo_epochs': updates * 5,
                'exposure': batch_exposure, 'cumulative_exposure': totals, 'episodes': episodes,
                'behavior': behavior(batch, records), 'prior': prior_stats,
                'pre_update_ratio_max_error': ratio_error, 'learning': result,
                'wall_seconds': time.monotonic() - started})
            json_write(path / 'metrics' / f'update_{updates:06d}.json', metrics)
            tensorboard.sync()
            print(json.dumps({'update': updates, 'actual_base_steps': collector.base_steps,
                'prior_strength': collector.policy.parking_exploration.strength,
                'successes': sum(e['terminal_reason'] == 'success' for e in episodes),
                'all_hold_by_slot': prior_stats['all_four_hold_decisions_by_slot']}), flush=True)
            del parts, batch, records, result
            gc.collect()
        final_no_prior = None; final_prior = None
        if not stop:
            phase = 'final_evaluation'
            # Evaluate copies. The checkpoint remains the actual final training policy.
            training_policy = collector.policy
            collector.policy = deepcopy(training_policy)
            collector.policy.set_parking_exploration(replace(settings, strength=0.))
            final_no_prior = evaluate(path, 'final_no_prior', collector, args.config, config)
            evaluation_steps += final_no_prior['actual_base_steps']
            tensorboard.sync()
            collector.policy.set_parking_exploration(settings)
            final_prior = evaluate(path, 'final_prior', collector, args.config, config)
            evaluation_steps += final_prior['actual_base_steps']
            collector.policy = training_policy
        if evaluation_steps > EVALUATION_LIMIT:
            raise AssertionError('Independent evaluation exceeded the budget')
        summary = {'actual_base_steps': collector.base_steps, 'remaining_base_steps': BASE_LIMIT - collector.base_steps,
            'policy_decisions': collector.decisions, 'sampling_updates': updates, 'ppo_epochs': updates * 5,
            'extra_critic_updates': 0, 'exposure': totals, 'last_complete_checkpoint': last_checkpoint,
            'stop_reason': 'signal_requested' if stop else 'physical_budget_exhausted',
            'evaluation_actual_base_steps': evaluation_steps, 'wall_seconds': time.monotonic() - started,
            'formal_training': False,
            **({} if final_no_prior is None else {'learned_without_prior': compare(initial_no_prior, final_no_prior),
                'supported_with_prior': compare(initial_no_prior, final_prior),
                'within_prior_learning_change': compare(initial_prior, final_prior)})}
        json_write(path / 'metrics/summary.json', event_data(summary))
        tensorboard.sync()
    except BaseException as error:
        json_write(path / 'metrics/failure.json', {'error': repr(error), 'phase': phase,
            'last_complete_checkpoint': last_checkpoint, 'completed_updates': updates,
            'known_base_steps': None if collector is None else collector.base_steps,
            'completed_evaluation_base_steps': evaluation_steps,
            'uncertain_current_reservation_base_steps': 100 if phase == 'sampling' else (
                1600 if phase in ('initial_evaluation', 'final_evaluation') else 0),
            'automatic_retry': False})
        if collector is not None and optimizer is not None:
            save_tensor_file(path / 'state/failure_state.pt', {'collector': collector.state_dict(),
                'optimizer': optimizer.state_dict(), 'error': repr(error), 'phase': phase})
        raise

    finally:
        tensorboard.close()


if __name__ == '__main__':
    main()
