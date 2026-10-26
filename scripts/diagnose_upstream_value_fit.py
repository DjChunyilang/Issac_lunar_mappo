"""Frozen-data value calibration, with complete-environment holdout and no rollout."""
import argparse
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
import pickle
import signal
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'source/lunar_rover_tasks'))
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.learning import duration_gae
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.action_ablation import AblationPolicy, ActionContract


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def weights_sha(weights):
    h = hashlib.sha256()
    for key, value in sorted(weights.items()):
        h.update(key.encode()); h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def extract(source, out):
    rows = {k: [] for k in ('state', 'reward', 'duration', 'terminated', 'value', 'gae', 'stage', 'deadline', 'batch')}
    provenance = []
    gamma = .997 ** .2
    for number in range(1, 21):
        path = source / 'state' / f'interactions_{number:06d}.pkl.gz'
        data = pickle.load(gzip.open(path, 'rb')); b = data['learning']
        assert b['learning_valid'].all() and not b['truncated'].any()
        assert torch.all(b['duration'] == 25)
        count, envs = b['reward'].shape
        deadline = torch.empty((count, envs))
        for owner in range(4):
            records = [x for x in data['execution'] if 'execution' in x and x['owner_id'] == owner]
            assert len(records) == count * 25
            for t in range(count):
                # Both records precede any possible interval terminal (all archived durations=25).
                a, c = records[t * 25], records[t * 25 + 1]
                seconds = .2 / (a['task']['remaining_time'] - c['task']['remaining_time'])
                choices = np.asarray([30., 180., 300.])
                nearest = choices[np.argmin(abs(seconds[:, None] - choices), axis=1)]
                assert np.allclose(seconds, nearest, atol=.0001)
                deadline[t, owner * 4:owner * 4 + 4] = torch.tensor(nearest, dtype=torch.float32)
        _, targets = duration_gae(b['reward'], b['value'], b['next_value'], b['duration'],
                                 b['terminated'], b['truncated'], gamma=gamma, lam=.99 ** .2,
                                 bootstrap_allowed=b['bootstrap_allowed'], learning_valid=b['learning_valid'])
        stage = (deadline == 180).long() + 2 * (deadline == 300).long()
        for key in ('state', 'reward', 'duration', 'terminated', 'value'):
            rows[key].append(b[key].clone())
        for key, value in (('deadline', deadline), ('stage', stage), ('gae', targets),
                           ('batch', torch.full_like(stage, number))):
            rows[key].append(value)
        provenance.append({'path': str(path), 'sha256': sha(path)})
        print(json.dumps({'extract_batch': number}), flush=True)
    data = {k: torch.cat(v) for k, v in rows.items()}
    shape = data['reward'].shape
    mc = torch.full(shape, float('nan')); episode = torch.full(shape, -1, dtype=torch.long)
    episode_count = 0; incomplete = 0
    for env in range(shape[1]):
        start = 0
        for t in range(shape[0]):
            if not data['terminated'][t, env]:
                continue
            assert abs(float(data['state'][start, env, -1]) - 1.) < 1e-5
            assert (data['deadline'][start:t + 1, env] == data['deadline'][start, env]).all()
            assert torch.allclose(data['state'][start:t + 1, env, -1],
                1 - torch.arange(t - start + 1) * 5 / data['deadline'][start, env], atol=2e-6)
            carry = 0.
            for j in range(t, start - 1, -1):
                carry = float(data['reward'][j, env]) + gamma ** float(data['duration'][j, env]) * carry
                mc[j, env] = carry
            episode[start:t + 1, env] = episode_count
            episode_count += 1; start = t + 1
        incomplete += shape[0] - start
    keep = episode >= 0
    data['mc'] = mc; data['episode'] = episode
    data['env'] = torch.arange(shape[1]).expand(shape)
    data['remaining_seconds'] = data['state'][..., -1] * data['deadline']
    selected = {key: value[keep] for key, value in data.items()}
    selected['fit'] = selected['env'] % 4 != 3
    assert torch.isfinite(selected['mc']).all()
    assert torch.allclose(selected['mc'][selected['terminated']], selected['reward'][selected['terminated']])
    torch.save(selected, out / 'state' / 'critic_dataset.pt')
    metadata = {'source_interactions': provenance, 'total_decisions': int(keep.numel()),
                'complete_episode_samples': int(keep.sum()), 'incomplete_tail_excluded': incomplete,
                'complete_episodes': episode_count, 'holdout_environments': [3, 7, 11, 15],
                'fit_samples': int(selected['fit'].sum()), 'holdout_samples': int((~selected['fit']).sum()),
                'target': 'discounted actual reward through terminal; no bootstrap; mixed historical behavior policies',
                'gamma_base': gamma, 'new_environment_steps': 0}
    write(out / 'metrics' / 'critic_dataset.json', metadata)
    return selected, metadata


def stats(predicted, target, mask):
    residual = predicted[mask] - target[mask]
    if not residual.numel():
        return {'n': 0}
    return {'n': int(residual.numel()), 'rmse': float(residual.square().mean().sqrt()),
            'bias': float(residual.mean()), 'prediction_mean': float(predicted[mask].mean()),
            'target_mean': float(target[mask].mean()),
            'explained_variance': float(1 - residual.var(unbiased=False) / target[mask].var(unbiased=False).clamp_min(1e-6))}


def evaluate(critic, state, data, mean, scale):
    with torch.no_grad():
        predicted = critic(state).squeeze(-1) * scale + mean
    report = {}
    for split, select in [('fit', data['fit']), ('holdout', ~data['fit'])]:
        report[split] = {name: stats(predicted, data['mc'], select & group) for name, group in {
            'all': torch.ones_like(select), 'parking': data['stage'] == 0,
            'near': data['stage'] == 1, 'main': data['stage'] == 2,
            'last_10_seconds': data['remaining_seconds'] <= 10.001,
            'terminal_decision': data['terminated'], 'early': data['remaining_seconds'] > 60,
        }.items()}
    return report


def run(args):
    torch.set_num_threads(1); torch.manual_seed(23); np.random.seed(23)
    source, out = Path(args.source), Path(args.run)
    for folder in ('metrics', 'state'):
        (out / folder).mkdir(parents=True, exist_ok=True)
    if (out / 'metrics' / f'critic_fit_{args.arm}.json').exists():
        raise FileExistsError(f'Preserve completed critic diagnostic: {args.arm}')
    dataset_path = out / 'state' / 'critic_dataset.pt'
    if dataset_path.exists():
        data = torch.load(dataset_path, weights_only=False)
        metadata = json.loads((out / 'metrics' / 'critic_dataset.json').read_text())
    else:
        data, metadata = extract(source, out)
    checkpoint = source / 'checkpoints' / 'update_000020.learner.pt'
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    policy_state = saved['trainer']['collector']['policy']; md = policy_state['metadata']
    contract = ActionContract(**{key: md[key] for key in ('representation', 'profile_fingerprint', 'rho_max', 'speed_cap_max', 'beta_max', 'request_seconds')})
    policy = AblationPolicy(991, contract); policy.enable_value_normalization()
    policy.load_state_dict(policy_state['weights'])
    actor_initial = weights_sha({k: v for k, v in policy.state_dict().items() if not k.startswith('critic.')})
    critic = deepcopy(policy.critic)
    state = data['state']
    if args.arm == 'absolute_clock':
        original = critic[0]
        layer = torch.nn.Linear(993, 256)
        with torch.no_grad():
            layer.weight[:, :991].copy_(original.weight); layer.weight[:, 991:].zero_(); layer.bias.copy_(original.bias)
        critic[0] = layer
        state = torch.cat([state, data['remaining_seconds'][:, None] / 300, data['deadline'][:, None] / 300], -1)
    mean = float(policy.value_normalizer.mean); scale = float(policy.value_normalizer.std)
    target = (data['mc'] - mean) / scale
    fit = data['fit']; weights = torch.ones_like(target[fit])
    if args.arm != 'natural':
        stage = data['stage'][fit]
        for level in torch.unique(stage):
            group = stage == level
            weights[group] = len(stage) / (len(torch.unique(stage)) * int(group.sum()))
    optimizer = torch.optim.Adam(critic.parameters(), lr=args.learning_rate)
    initial_sha = weights_sha(critic.state_dict()); history = []
    stop_requested = False
    def request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    started = time.monotonic()
    for step in range(args.steps + 1):
        if step % 50 == 0 or step == args.steps or stop_requested:
            metrics = evaluate(critic, state, data, mean, scale)
            history.append({'step': step, **metrics})
            print(json.dumps({'arm': args.arm, 'step': step, 'fit': metrics['fit']['all'],
                              'holdout': metrics['holdout']}), flush=True)
        if step == args.steps or stop_requested:
            break
        optimizer.zero_grad()
        error = critic(state[fit]).squeeze(-1) - target[fit]
        loss = .5 * (weights * error.square()).mean()
        assert torch.isfinite(loss)
        loss.backward(); torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.)
        optimizer.step()
    actor_final = weights_sha({k: v for k, v in policy.state_dict().items() if not k.startswith('critic.')})
    assert actor_initial == actor_final
    before = history[0]['holdout']; after = history[-1]['holdout']
    gate = {'holdout_rmse_reduction_at_least_half': after['all']['rmse'] <= before['all']['rmse'] * .5,
            'parking_abs_bias_reduction_at_least_half': abs(after['parking']['bias']) <= abs(before['parking']['bias']) * .5,
            'last_10_seconds_abs_bias_reduction_at_least_half': abs(after['last_10_seconds']['bias']) <= abs(before['last_10_seconds']['bias']) * .5}
    report = {'arm': args.arm, 'source_checkpoint': str(checkpoint), 'source_checkpoint_sha256': sha(checkpoint),
              'dataset': metadata, 'command': sys.argv, 'new_environment_steps': 0,
              'optimizer_steps': step, 'requested_optimizer_steps': args.steps,
              'stopped_early': stop_requested, 'learning_rate': args.learning_rate, 'new_adam': True,
              'fixed_normalizer': {'mean': mean, 'std': scale},
              'fit_weighting': 'natural' if args.arm == 'natural' else 'equal total weight per present course stage',
              'actor_weights_sha256_before': actor_initial, 'actor_weights_sha256_after': actor_final,
              'critic_weights_sha256_before': initial_sha, 'critic_weights_sha256_after': weights_sha(critic.state_dict()),
              'initial': history[0], 'final': history[-1], 'history': history, 'acceptance': gate,
              'seconds': time.monotonic() - started,
              'interpretation_limits': 'Offline fitting capacity only. Complete MC uses actual historical policy behavior, not a fresh on-policy target or a causal action advantage.'}
    write(out / 'metrics' / f'critic_fit_{args.arm}.json', report)
    torch.save({'critic': critic.state_dict(), 'normalization_mean': mean, 'normalization_std': scale,
                'diagnostic_only': True, 'actor_included': False, 'source_checkpoint': str(checkpoint),
                'arm': args.arm}, out / 'state' / f'critic_fit_{args.arm}.pt')
    print(json.dumps({'completed': args.arm, 'acceptance': gate, 'seconds': report['seconds']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True); parser.add_argument('--run', required=True)
    parser.add_argument('--arm', choices=('natural', 'balanced', 'absolute_clock'), default='natural')
    parser.add_argument('--steps', type=int, default=500); parser.add_argument('--learning-rate', type=float, default=1e-4)
    args = parser.parse_args()
    if not 1 <= args.steps <= 500:
        parser.error('Each isolated diagnostic is limited to 500 critic steps')
    run(args)
