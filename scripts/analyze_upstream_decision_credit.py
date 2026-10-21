#!/usr/bin/env python3
"""Prefix credit and target continuity from bounded counterfactual traces."""
import argparse
import json
from pathlib import Path

import numpy as np

from planner_evidence import file_sha256, json_write
from upstream_run_state import restore_state
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_reward import UpstreamRewardSettings


def frozen_hold_reference(dmax, dispersion, remaining_steps, *, dmax_limit=12., dispersion_limit=30.):
    """Flat, safe, zero-velocity, zero-dwell state frozen through task deadline.

    This is a specified HOLD continuation value, not the true learned-policy V.
    Include the terminal potential cancellation and the deadline penalty once.
    """
    s = UpstreamRewardSettings()
    distance = np.maximum(np.asarray(dmax)-dmax_limit, 0.)/dmax_limit
    spread = np.maximum(np.asarray(dispersion)-dispersion_limit, 0.)/dispersion_limit
    phi = -s.distance_progress*distance-s.dispersion_progress*spread-s.hold_potential
    level = s.distance_level*distance+s.dispersion_level*spread
    steps = np.asarray(remaining_steps)
    if (steps < 1).any(): raise ValueError('Frozen HOLD reference requires future deadline')
    return -phi-level*(1-s.gamma**steps)/(1-s.gamma)-s.timeout_penalty*s.gamma**(steps-1)


def analyze(path):
    data = restore_state(path)
    batch, rows = data['batch'], data['rows']
    durations = batch['duration'].numpy(); rewards = batch['reward'].numpy()
    offsets = np.cumsum(durations, axis=0)-durations
    prefixes = {}
    for length in (1, 3, 5, 10, 20):
        if length > len(rows): continue
        ret = (.99**offsets[:length]*rewards[:length]).sum(0)
        bootstrap = .99**durations[:length].sum(0)*batch['next_value'][length-1].numpy()
        prefixes[str(length)] = {'discounted_reward': ret.tolist(),
            'critic_residual': (ret+bootstrap-batch['value'][0].numpy()).tolist()}
    goals = np.asarray([[ [intent.goal_xy for intent in row] for row in item['requests']] for item in rows])
    holds = np.asarray([[ [intent.hold for intent in row] for row in item['requests']] for item in rows])
    jumps = np.linalg.norm(np.diff(goals, axis=0), axis=-1)
    active_move_pairs = ~holds[1:] & ~holds[:-1]
    task_rows = [r for r in data['execution'] if 'execution' in r]
    no_collision = all(not r['execution']['geometry_overlap'].any() for r in task_rows)
    max_control_error = max(float(np.max(abs(r['execution']['desired_control']-r['execution']['applied_control']))) for r in task_rows)
    result = {'trace': str(path), 'sha256': file_sha256(path), 'prefixes': prefixes,
        'requested_hold_fraction': float(holds.mean()),
        'goal_jump_mean_m': float(jumps.mean()),
        'move_pair_goal_jump_mean_m': float(jumps[active_move_pairs].mean()) if active_move_pairs.any() else None,
        'goal_unchanged_fraction': float((jumps < 1e-8).mean()),
        'no_geometry_overlap': no_collision, 'maximum_issued_applied_control_error': max_control_error,
        'final_actual_stopped': rows[-1]['task']['actual_stopped'].tolist(),
        'final_hold_count': rows[-1]['task']['hold_count'].tolist(),
        'counterfactual_not_on_policy': True}
    if path.name == 'hold.pkl.gz':
        cfg = json.loads((path.parent.parent/'config/effective_env_config.json').read_text())
        final = task_rows[-1]; task = final['task']; execution = final['execution']
        assert not np.any(execution['actual_velocity']) and not np.any(task['hold_count'])
        assert task['parking_terrain_satisfied'].all() and task['clearance_satisfied'].all()
        for row in task_rows:
            assert not np.any(row['reward_components']['flatness'])
        now = execution['activation_time']+cfg['spec']['grid']['dt']
        remaining = np.rint((cfg['simulation']['episode_length_s']-now)/cfg['spec']['grid']['dt']).astype(int)
        reference = frozen_hold_reference(task['dmax'], task['dispersion'], remaining,
            dmax_limit=cfg['spec']['task']['dmax'], dispersion_limit=cfg['spec']['task']['dispersion'])
        estimate = batch['next_value'][-1].numpy()
        result['stationary_hold_tail'] = {'remaining_steps': remaining.tolist(),
            'analytic_continuation_value': reference.tolist(), 'critic_estimate': estimate.tolist(),
            'continuation_minus_critic': (reference-estimate).tolist(),
            'assumption': 'Freeze flat safe geometry with zero velocity and no dwell until deadline; this is not V of the stochastic policy.'}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    result = {p.stem.removesuffix('.pkl'): analyze(p) for p in sorted((args.run/'state').glob('*.pkl.gz'))}
    json_write(args.run/'metrics/credit_continuity.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
