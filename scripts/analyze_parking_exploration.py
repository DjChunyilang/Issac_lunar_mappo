#!/usr/bin/env python3
"""Read-only credit and joint-HOLD audit of a completed bounded parking run.

Consumes archived inputs/checkpoints; no environment, optimizer or new samples.
"""
import argparse
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import torch

from _common import ROOT, cfg_from_experiment
from planner_evidence import file_sha256, json_write
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.action_ablation import AblationPolicy, ActionContract
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.learning import duration_gae
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_environment import REASON_CODE
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_parking_exploration import ParkingExploration


def restore(state, spec):
    p = AblationPolicy(991, ActionContract.from_spec('A1', spec))
    if 'value_normalization' in state['metadata']:
        p.enable_value_normalization()
    if 'parking_exploration' in state['metadata']:
        settings = dict(state['metadata']['parking_exploration'])
        settings.pop('version')
        p.set_parking_exploration(ParkingExploration(**settings))
    p.load_diagnostic_state(state)
    return p.eval()


@torch.no_grad()
def head_sequence(policy, batch):
    hidden = batch['hidden'][0].clone()
    result = []
    for obs, ended in zip(batch['obs'], batch['terminated']):
        dist, following = policy.distribution(obs, hidden)
        result.append({'raw_probability': policy.hold_logit(following).squeeze(-1).sigmoid(),
                       'actual_probability': dist.branch.probs})
        hidden = torch.where(ended[:, None, None], 0., following)
    return {key: torch.stack([row[key] for row in result]) for key in result[0]}


def nullable_mean(values):
    return float(np.mean(values)) if len(values) else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    args = parser.parse_args()
    run = (ROOT / args.run).resolve()
    summary = json.loads((run / 'metrics/summary.json').read_text())
    manifest = json.loads((run / 'run_manifest.json').read_text())['identity']
    spec = cfg_from_experiment(manifest['base_config']).spec
    options = json.loads((run / 'config/effective_configuration.json').read_text())['optimizer']
    torch.set_num_threads(1)
    rows = []; all_hold = np.zeros(4, int); pairs = np.zeros(4, int)
    effective_all_hold = np.zeros(4, int); effective_pairs = np.zeros(4, int)
    previous = np.zeros(4, bool); previous_ended = np.ones(4, bool)
    previous_effective = np.zeros(4, bool)
    max_dwell = np.zeros(4, int); episodes = Counter()
    overlap_steps = 0; clearance_steps = 0
    credits = {name: {'raw': [], 'normalized': [], 'count': 0} for name in (
        'parking_all_hold_effective', 'parking_other_effective', 'approach_all_hold_effective',
        'success_terminal_effective', 'success_terminal_excluded')}
    phase_counts = {name: {'steps': 0, 'team_decisions': 0, 'parking_all_hold': 0,
                           'parking_consecutive_pairs': 0, 'successes': 0} for name in ('full', 'fading', 'zero')}
    initial_state = torch.load(run / 'checkpoints/initial.pt', map_location='cpu', weights_only=False)['collector']['policy']
    initial_policy = restore(initial_state, spec)
    for metric_file in sorted((run / 'metrics').glob('update_*.json')):
        update = json.loads(metric_file.read_text())
        name = metric_file.stem
        saved = torch.load(run / 'state' / f'{name}_input.pt', map_location='cpu', weights_only=False)
        batch = saved['batch']; valid = batch['learning_valid']; effective = valid & batch['action_effective']
        adv, _ = duration_gae(batch['reward'], batch['value'], batch['next_value'], batch['duration'],
            batch['terminated'], batch['truncated'], gamma=options['gamma'], lam=options['lam'],
            bootstrap_allowed=batch['bootstrap_allowed'], learning_valid=valid)
        selected = adv[effective]
        norm = torch.where(effective, (adv - selected.mean()) / selected.std(unbiased=False).clamp_min(1e-6), 0.)
        held = torch.stack([s.action.hold for s in batch['samples']]); joint = held.all(-1)
        strength = update['prior']['settings']['strength']
        phase = 'full' if strength == 1. else ('zero' if strength == 0. else 'fading')
        counts = phase_counts[phase]
        counts['steps'] += int(batch['duration'].sum())
        counts['team_decisions'] += int((batch['duration'] > 0).sum())
        for step in range(len(joint)):
            current = joint[step].numpy()
            consecutive = current & previous & ~previous_ended
            current_effective = effective[step].numpy()
            all_hold += current; pairs += consecutive
            effective_all_hold += current & current_effective
            effective_pairs += consecutive & current_effective & previous_effective
            counts['parking_all_hold'] += int(current[:2].sum())
            counts['parking_consecutive_pairs'] += int(consecutive[:2].sum())
            previous = current; previous_ended = batch['terminated'][step].numpy()
            previous_effective = current_effective
        for record in saved['records']:
            if 'task' in record:
                max_dwell = np.maximum(max_dwell, record['task']['hold_count'])
            if 'execution' in record:
                overlap_steps += int(np.count_nonzero(record['execution']['geometry_overlap']))
                clearance_steps += int(np.count_nonzero(record['execution']['clearance_violation']))
        for episode in saved['episodes']:
            episodes[episode['terminal_reason']] += 1
            counts['successes'] += episode['terminal_reason'] == 'success'
        success = batch['terminal_reason'] == REASON_CODE['success']
        masks = {
            'parking_all_hold_effective': effective & joint,
            'parking_other_effective': effective & ~joint,
            'approach_all_hold_effective': effective & joint,
            'success_terminal_effective': success & effective,
            'success_terminal_excluded': success & valid & ~effective,
        }
        masks['parking_all_hold_effective'][:, 2:] = False
        masks['parking_other_effective'][:, 2:] = False
        masks['approach_all_hold_effective'][:, :2] = False
        for key, mask in masks.items():
            credits[key]['raw'].extend(adv[mask].tolist())
            credits[key]['normalized'].extend(norm[mask].tolist())
            credits[key]['count'] += int(mask.sum())
        before = restore(saved['policy'], spec)
        checkpoint = run / 'checkpoints' / f'{name}.pt'
        after_saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        after = restore(after_saved['collector']['policy'], spec)
        pre = head_sequence(before, batch); post = head_sequence(after, batch)
        p = pre['actual_probability']
        # Exact initial-surrogate derivative wrt a common additive HOLD logit.
        # Positive means this group pushes HOLD upward; includes the team credit mask.
        push = norm[..., None] * (held.float() - p) * effective[..., None]
        rows.append({'update': name, 'base_steps': update['actual_base_steps'], 'strength': strength,
            'pre_update_ratio_max_error': update['pre_update_ratio_max_error'],
            'parking_hold_logit_policy_gradient_push': float(push[:, :2].sum() / (effective.sum() * 4)),
            'approach_hold_logit_policy_gradient_push': float(push[:, 2:].sum() / (effective.sum() * 4)),
            'raw_head_probability_change_same_obs_parking': float((post['raw_probability'][:, :2] - pre['raw_probability'][:, :2]).mean()),
            'raw_head_probability_change_same_obs_approach': float((post['raw_probability'][:, 2:] - pre['raw_probability'][:, 2:]).mean()),
            'hold_head_bias_change': float((after.hold_logit.bias - before.hold_logit.bias).detach()),
            'adam_steps': sorted({int(v['step']) for v in after_saved['optimizer']['state'].values() if 'step' in v})})
    for value in credits.values():
        value['raw_mean'] = nullable_mean(value['raw'])
        value['normalized_mean'] = nullable_mean(value['normalized'])
        value['normalized_positive_count'] = sum(x > 0 for x in value['normalized'])
        value.pop('raw'); value.pop('normalized')
    final_file = Path(summary['last_complete_checkpoint'])
    sidecar = json.loads(final_file.with_suffix('.json').read_text())
    final_hash = file_sha256(final_file)
    if final_hash != sidecar['sha256']:
        raise ValueError('Final checkpoint SHA mismatch')
    final_saved = torch.load(final_file, map_location='cpu', weights_only=False)
    final_policy = restore(final_saved['collector']['policy'], spec)
    initial_policy.set_parking_exploration(replace(initial_policy.parking_exploration, strength=0.))
    final_policy.set_parking_exploration(replace(final_policy.parking_exploration, strength=0.))
    reference = torch.load(Path(manifest['baseline_run']) / 'state/update_000001_input.pt', map_location='cpu', weights_only=False)['batch']
    with torch.no_grad():
        p0 = initial_policy.distribution(reference['obs'][0], reference['hidden'][0])[0].branch.probs
        p1 = final_policy.distribution(reference['obs'][0], reference['hidden'][0])[0].branch.probs
    mismatches = [name for name, digest in manifest['source_snapshot'].items()
                  if file_sha256(ROOT / name) != digest]
    result = {'schema_version': 'parking_exploration_credit_audit_v1',
        'new_environment_steps': 0, 'new_optimizer_updates': 0,
        'training_base_steps': summary['actual_base_steps'], 'all_hold_by_slot': all_hold.tolist(),
        'consecutive_all_hold_pairs_same_episode_by_slot': pairs.tolist(),
        'effective_all_hold_by_slot': effective_all_hold.tolist(),
        'both_effective_consecutive_all_hold_pairs_by_slot': effective_pairs.tolist(),
        'maximum_dwell_steps_by_slot': max_dwell.tolist(), 'episodes': dict(episodes),
        'geometry_overlap_steps': overlap_steps, 'clearance_violation_steps': clearance_steps,
        'phase_counts': phase_counts, 'credit': credits, 'updates': rows,
        'fixed_initial_obs_raw_hold_probability_before_by_slot': p0.mean(-1).tolist(),
        'fixed_initial_obs_raw_hold_probability_after_by_slot': p1.mean(-1).tolist(),
        'final_checkpoint_sha256': final_hash, 'final_checkpoint_sha_verified': True,
        'archived_runtime_source_mismatches': mismatches,
        'analysis_script_sha256': file_sha256(Path(__file__))}
    json_write(run / 'metrics/joint_parking_credit_audit.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'updates'}, indent=2))


if __name__ == '__main__':
    main()
