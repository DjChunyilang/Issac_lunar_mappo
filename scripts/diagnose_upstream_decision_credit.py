#!/usr/bin/env python3
"""Bounded same-state decision/credit counterfactuals; never updates a policy.

Replay one existing owner through the public environment lifecycle. Renewing a
fixed odometry target is an explicit intervention, not an on-policy PPO sample.
The historical runtime identity is retained only to restore the environment.
"""
import argparse
from collections import Counter
from dataclasses import replace
import gc
import json
from pathlib import Path
import time

import numpy as np
import torch

from _common import ROOT
from planner_evidence import EvidenceRun, file_sha256, json_write
from run_upstream_t11 import make_collector
from upstream_checkpoint import read_checkpoint
from upstream_run_state import save_state
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.learning import duration_gae
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_ablation_learning import renew_legal_bounds
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_environment import UpstreamEnvironment
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_planning import UpstreamIntent


def renew_targets(targets, times, validity):
    """New delivery tickets for identical fixed-frame targets, without extending old tickets."""
    return [[replace(intent, timestamp=float(times[e]), expires_at=float(times[e])+validity)
             for intent in row] for e, row in enumerate(targets)]


def local_centroid_requests(env, *, away=False, hold=False):
    """Only own pose and valid received states; no global/other-private state."""
    result = []
    for e in range(env.e):
        row = []
        for a in range(4):
            now = float(env.core.times[e]); pose = env.local_pose(e, a)
            snapshot = env.radio.snapshot(e, a, now, pose)
            peers = [message.pose[:2] for slot, message in enumerate(snapshot.messages)
                     if snapshot.state_valid[slot]]
            delta = np.zeros(2) if not peers else np.mean([pose[:2], *peers], axis=0)-pose[:2]
            if away: delta = -delta
            length = np.linalg.norm(delta)
            is_hold = hold or not peers or length < .2
            goal = pose[:2].copy() if is_hold else pose[:2]+delta*min(1., 4./length)
            row.append(UpstreamIntent(goal, bool(is_hold), now, now+env.spec.grid.request_validity))
        result.append(row)
    return result


def geometry(poses):
    points = poses[..., :2]; center = points.mean(1)
    return {'dmax': np.linalg.norm(points[:, :, None]-points[:, None, :], axis=-1).max((1, 2)),
            'dispersion': np.square(points-center[:, None]).sum(-1).mean(-1)}


def evaluate_branch(collector, env_state, hidden, rng, mode, seconds):
    env, policy = collector.env, collector.policy
    env.load_state_dict(env_state); torch.set_rng_state(rng)
    hidden = hidden.clone(); targets = None
    initial = geometry(env.core.poses)
    arrays = {key: [] for key in ('reward', 'value', 'next_value', 'duration', 'terminated',
                                 'truncated', 'bootstrap_allowed', 'learning_valid')}
    rows = []; records = []; statuses = Counter(); traveled = np.zeros((env.e, 4))
    original_weights = {key: value.clone() for key, value in policy.state_dict().items()}
    start = time.perf_counter()
    with torch.no_grad():
        for step in range(seconds):
            if env._ended.any(): raise ValueError('Chosen short panel crossed an episode boundary')
            renew_legal_bounds(env)
            obs, state = env.observe()
            # Recurrent state advances on every observed macro, even on renewals.
            sample = policy.sample(obs, hidden, deterministic=mode == 'policy_mean')
            hidden = sample.hidden
            sampled, _ = collector.bridge.decode(sample.action)
            if mode in ('policy_sample', 'policy_mean'):
                requests = sampled
            elif mode == 'sample_commit5':
                if step % 5 == 0: targets = sampled
                requests = renew_targets(targets, env.core.times, env.spec.grid.request_validity)
            elif mode in ('toward_commit5', 'away_commit5', 'hold'):
                if step % 5 == 0:
                    targets = local_centroid_requests(env, away=mode == 'away_commit5', hold=mode == 'hold')
                requests = renew_targets(targets, env.core.times, env.spec.grid.request_validity)
            else: raise ValueError(mode)
            value = policy.value(state)
            transition = env.step_intents(requests)
            following = policy.value(transition.terminal_critic)
            for key in arrays:
                item = value if key == 'value' else following if key == 'next_value' else getattr(transition, key)
                arrays[key].append(item.clone())
            for record in transition.records:
                if 'execution' not in record: continue
                ex = record['execution']; traveled += np.linalg.norm(ex['after'][..., :2]-ex['before'][..., :2], axis=-1)
                statuses.update(decision['execution_status'] for decision in record['decisions'])
            records.extend(transition.records)
            rows.append({'step': step, 'requests': requests, 'geometry': geometry(env.core.poses),
                         'reward_components': transition.reward_components, 'task': transition.records[-1]['task']})
            env.discard_delivered_history()
            if (step+1) % 5 == 0:
                print(json.dumps({'event': 'progress', 'mode': mode, 'seconds': step+1,
                                  'dmax_mean': float(geometry(env.core.poses)['dmax'].mean())}), flush=True)
    batch = {key: torch.stack(value) for key, value in arrays.items()}
    advantage, _ = duration_gae(**{
        'rewards': batch['reward'], 'values': batch['value'], 'next_values': batch['next_value'],
        'durations': batch['duration'], 'terminated': batch['terminated'], 'truncated': batch['truncated'],
        'bootstrap_allowed': batch['bootstrap_allowed'], 'learning_valid': batch['learning_valid']})
    duration = batch['duration'].numpy(); reward = batch['reward'].numpy()
    offsets = np.cumsum(duration, axis=0)-duration
    total_duration = duration.sum(0)
    discounted = (np.power(.99, offsets)*reward).sum(0)
    initial_value = batch['value'][0].numpy(); final_value = batch['next_value'][-1].numpy()
    components = {name: sum(.99**offsets[t]*row['reward_components'][name].numpy()
                           for t, row in enumerate(rows)) for name in rows[0]['reward_components']}
    final = geometry(env.core.poses)
    assert all(torch.equal(original_weights[key], value) for key, value in policy.state_dict().items())
    summary = {'mode': mode, 'seconds': seconds, 'wall_seconds': time.perf_counter()-start,
               'initial': initial, 'final': final, 'dmax_reduction': initial['dmax']-final['dmax'],
               'dispersion_reduction': initial['dispersion']-final['dispersion'],
               'distance_traveled_per_rover': traveled, 'discounted_reward': discounted,
               'discounted_components': components, 'initial_value': initial_value, 'final_value': final_value,
               'bootstrapped_advantage': discounted+.99**total_duration*final_value-initial_value,
               'gae_first': advantage[0].numpy(), 'execution_status': dict(statuses),
               'all_learning_valid': bool(batch['learning_valid'].all()),
               'optimizer_updates': 0, 'policy_weights_unchanged': True,
               'interpretation': 'Off-policy counterfactual critic residuals, not PPO training advantages.'}
    return summary, {'rows': rows, 'batch': batch, 'execution': records}


def jsonable(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, (np.bool_, np.integer, np.floating)): return value.item()
    if isinstance(value, dict): return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)): return [jsonable(item) for item in value]
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--config', default='configs/experiment/upstream_scale_s04_braking_lease_cuda_v5.yaml')
    parser.add_argument('--seconds', type=int, default=20)
    parser.add_argument('--modes', nargs='+', default=['hold', 'toward_commit5', 'away_commit5', 'policy_sample', 'policy_mean', 'sample_commit5'])
    args = parser.parse_args()
    if not 1 <= args.seconds <= 30: raise ValueError('This diagnostic is bounded to 30 seconds per branch')
    torch.set_num_threads(1); UpstreamEnvironment.cpu_workers = 2
    saved = read_checkpoint(args.checkpoint)
    central = saved['trainer']['collector']; representation = central['metadata']['policy']['representation']
    shard = central['owner_shards'][0]
    env_state = torch.load(shard['file'], map_location='cpu', weights_only=False)
    # Signal identity is needed to load the existing environment; there is no trainer or optimizer here.
    collector = make_collector(args.config, representation, 23, samples=512, num_envs=4, mode='signal')
    collector.policy.load_diagnostic_state(central['policy'])
    run = EvidenceRun(ROOT/'outputs/runs/upstream_scale_t11'/args.run_id, args.config,
                      settings={'stage': 'decision_credit_counterfactual', 'checkpoint': str(args.checkpoint),
                                'checkpoint_sha256': file_sha256(args.checkpoint), 'owner': shard,
                                'modes': args.modes, 'seconds': args.seconds, 'optimizer_updates': 0,
                                'script_sha256': file_sha256(__file__)})
    run.resolved(collector.env.cfg)
    summaries = []
    for mode in args.modes:
        summary, trace = evaluate_branch(collector, env_state, central['hidden'][:4], central['torch_rng'], mode, args.seconds)
        json_write(run.path/'metrics'/f'{mode}.json', jsonable(summary))
        save_state(run, mode, trace)
        summaries.append(summary)
        print(json.dumps(jsonable(summary)), flush=True)
        del trace; gc.collect()
    json_write(run.path/'metrics/summary.json', jsonable({'representation': representation,
               'branches': summaries, 'training_interactions': 0, 'optimizer_updates': 0,
               'simulated_environment_steps': args.seconds*5*4*len(args.modes)}))


if __name__ == '__main__':
    main()
