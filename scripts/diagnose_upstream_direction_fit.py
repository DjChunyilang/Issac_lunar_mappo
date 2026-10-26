"""Bounded direction-capacity diagnosis on existing complete episodes.

This is supervised auxiliary fitting, not PPO or proof of learned rendezvous.
It never resumes training or edits a source run. Fitting is offline; a separate
--execute-eval mode permits at most 1600 independent, no-learning base steps.
Labels use only the Actor's legal relative-neighbor observations. Recurrent
features are replayed from zero at true episode starts, including across saved
batch boundaries; archived hidden states are not reused after fitting.

Run only after the preceding Critic and parking diagnoses have been assessed.
The default prints the protocol; --execute-fit performs at most 500 updates.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import gc
import gzip
import json
import math
from pathlib import Path
import pickle
import sys

import numpy as np
import torch
from torch.nn import functional as F

from _common import ROOT, cfg_from_experiment
from planner_evidence import file_sha256, json_write
from upstream_checkpoint import read_checkpoint
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.action_ablation import (
    AblationPolicy, ActionContract,
)


SOURCE = ROOT / 'outputs/runs/upstream_1km_curriculum/a1_seed23_v2_stage250k_20261020_01'
VALIDATION_ENVS = (3, 7, 11, 15)
SECTORS = ('front', 'side', 'rear')


def load_policy(state):
    metadata = state['metadata']
    contract = ActionContract(**{k: metadata[k] for k in (
        'representation', 'profile_fingerprint', 'rho_max', 'speed_cap_max',
        'beta_max', 'request_seconds')})
    if contract.representation != 'A1':
        raise ValueError('Direction diagnosis requires the A1 conditional MOVE head')
    policy = AblationPolicy(metadata['critic_dim'], contract,
                            terrain_channels=metadata['terrain_channels'])
    if 'value_normalization' in metadata:
        policy.enable_value_normalization()
    policy.load_diagnostic_state(state)
    return policy.cpu().eval()


def legal_labels(obs, spec):
    """Invert the explicit 1km log1p Actor scaling, ignoring invalid senders."""
    neighbors = obs[..., 15:66].reshape(*obs.shape[:-1], 3, 17)
    valid = obs[..., [381, 391, 401]] > .5
    normalized = torch.where(valid[..., None], neighbors[..., :2], 0.)
    relative = normalized.sign() * spec.task.dmax * torch.expm1(
        normalized.abs() * math.log1p(spec.communication.full_radius / spec.task.dmax))
    count = valid.sum(-1)
    center = (relative * valid[..., None]).sum(-2) / count.clamp_min(1)[..., None]
    angle = torch.atan2(center[..., 1], center[..., 0])
    eligible = (count > 0) & (center.norm(dim=-1) > 1e-6)
    # Front = +/-45 degrees; rear = beyond +/-135; both sides share a bin.
    sector = torch.where(angle.abs() <= math.pi/4, 0,
                         torch.where(angle.abs() >= 3*math.pi/4, 2, 1))
    return {'angle': angle, 'eligible': eligible, 'sector': sector,
            'neighbor_count': count}


def complete_episodes(source, batches, spec):
    """Keep whole valid episodes and split by global environment, never rows."""
    buffers = [[] for _ in range(16)]
    starts = [True] * 16
    good = [True] * 16
    episode_ids = [0] * 16
    episodes, provenance = [], []
    dropped = 0
    for batch_index in range(1, batches + 1):
        path = source / 'state' / f'interactions_{batch_index:06d}.pkl.gz'
        provenance.append({'file': str(path), 'sha256': file_sha256(path)})
        with gzip.open(path, 'rb') as stream:
            archive = pickle.load(stream)
        data = archive['learning']
        if data['obs'].shape[1:] != (16, 4, 411):
            raise ValueError('Expected the reviewed four-owner/four-environment Actor layout')
        for t in range(len(data['obs'])):
            for env in range(16):
                if starts[env]:
                    if data['hidden'][t, env].abs().max().item() != 0:
                        raise ValueError('Archived episode start has nonzero recurrent memory')
                    starts[env] = False
                good[env] = good[env] and bool(data['learning_valid'][t, env])
                buffers[env].append((data['obs'][t, env].clone(), batch_index, t))
                if data['terminated'][t, env]:
                    records = buffers[env]
                    if good[env]:
                        obs = torch.stack([r[0] for r in records])
                        episodes.append({
                            'id': f'env_{env:02d}_episode_{episode_ids[env]:04d}',
                            'env': env, 'obs': obs, **legal_labels(obs, spec),
                            'partition': 'validation' if env in VALIDATION_ENVS else 'fit',
                            'first_batch': records[0][1], 'first_t': records[0][2],
                            'last_batch': records[-1][1], 'last_t': records[-1][2],
                            'terminal_reason': int(data['terminal_reason'][t, env]),
                        })
                    else:
                        dropped += 1
                    buffers[env] = []
                    starts[env] = good[env] = True
                    episode_ids[env] += 1
        del archive, data
        gc.collect()
    return episodes, provenance, {'invalid_complete_episodes': dropped,
        'unfinished_tail_decisions': sum(len(x) for x in buffers)}


def episode_output(policy, episode):
    """Replay the actual fixed observation sequence with the candidate's GRU."""
    obs = episode['obs']
    encoded = policy.encode(obs)
    hidden = obs.new_zeros((4, 128))
    following = []
    for row in encoded:
        hidden = policy.gru(row, hidden)
        following.append(hidden)
    features = torch.stack(following)
    latent = policy.mean(features)
    return {'features': features,
            'beta': policy.beta_max * latent[..., 1].tanh(),
            'rho': (latent[..., 0].tanh() + 1) * policy.rho_max / 2,
            'hold_logit': policy.hold_logit(features).squeeze(-1)}


@torch.no_grad()
def annotate_reference(policy, episodes):
    for episode in episodes:
        episode['reference'] = {k: v.detach() for k, v in episode_output(policy, episode).items()}


def angle_error(beta, target):
    return torch.atan2((beta-target).sin(), (beta-target).cos())


def direction_loss(beta, angle, eligible, sector):
    error = angle_error(beta, angle)
    losses = []
    for index in range(3):
        mask = eligible & (sector == index)
        if mask.any():
            losses.append(F.smooth_l1_loss(error[mask], torch.zeros_like(error[mask]), beta=.5))
    if not losses:
        raise ValueError('No legal nonzero relative-neighbor center in the selected sample')
    return torch.stack(losses).mean()


@torch.no_grad()
def metrics(policy, episodes, *, cached_features=False):
    report = {}
    for partition in ('fit', 'validation'):
        rows = []
        for episode in episodes:
            if episode['partition'] != partition:
                continue
            ref = episode['reference']
            if cached_features:
                latent = policy.mean(ref['features'])
                output = {'beta': policy.beta_max * latent[..., 1].tanh(),
                          'rho': (latent[..., 0].tanh()+1)*policy.rho_max/2,
                          'hold_logit': ref['hold_logit']}
            else:
                output = episode_output(policy, episode)
            rows.append({
                'error': angle_error(output['beta'], episode['angle']).flatten(),
                'eligible': episode['eligible'].flatten(), 'sector': episode['sector'].flatten(),
                'hold_delta': (output['hold_logit'].sigmoid()-ref['hold_logit'].sigmoid()).flatten(),
                'rho_delta': (output['rho']-ref['rho']).flatten(),
                'gate_flip': ((output['hold_logit'] > 0) != (ref['hold_logit'] > 0)).flatten(),
            })
        if not rows:
            raise ValueError('Environment split has an empty partition')
        merged = {k: torch.cat([row[k] for row in rows]) for k in rows[0]}
        row = {'actor_states': len(merged['error']), 'sectors': {}}
        for name, index in (('all', -1), *zip(SECTORS, range(3))):
            mask = merged['eligible']
            if index >= 0:
                mask = mask & (merged['sector'] == index)
            error = merged['error'][mask]
            row['sectors'][name] = {'n': int(mask.sum()),
                'mean_abs_error_deg': float(error.abs().mean()*180/math.pi) if len(error) else None,
                'inward_fraction': float((error.cos() > 0).float().mean()) if len(error) else None}
        row.update(hold_probability_abs_delta_mean=float(merged['hold_delta'].abs().mean()),
                   hold_probability_abs_delta_max=float(merged['hold_delta'].abs().max()),
                   deterministic_gate_flips=int(merged['gate_flip'].sum()),
                   rho_abs_delta_mean_m=float(merged['rho_delta'].abs().mean()))
        report[partition] = row
    return report


def screen(report):
    row = report['validation']
    return {
        'all_sectors_have_heldout_examples': all(row['sectors'][s]['n'] > 0 for s in SECTORS),
        'each_sector_error_below_45deg': all(row['sectors'][s]['mean_abs_error_deg'] is not None
            and row['sectors'][s]['mean_abs_error_deg'] < 45 for s in SECTORS),
        'each_sector_inward_at_least_80pct': all(row['sectors'][s]['inward_fraction'] is not None
            and row['sectors'][s]['inward_fraction'] >= .8 for s in SECTORS),
        'hold_mean_change_below_1pct': row['hold_probability_abs_delta_mean'] < .01,
        'hold_max_change_below_3pct': row['hold_probability_abs_delta_max'] < .03,
        'no_deterministic_gate_flip': row['deterministic_gate_flips'] == 0,
        'rho_mean_change_below_10cm': row['rho_abs_delta_mean_m'] < .1,
    }


def configure_fit(policy, encoder):
    prefixes = ('mean.', 'neighbors.', 'trunk.', 'gru.') if encoder else ('mean.',)
    for name, param in policy.named_parameters():
        param.requires_grad_(name.startswith(prefixes))
    # Preserve the radial row exactly; shared representation can still change
    # its output, so the encoder stage separately distills radial and HOLD output.
    handles = [policy.mean.weight.register_hook(lambda grad: torch.stack((grad[0]*0, grad[1]))),
               policy.mean.bias.register_hook(lambda grad: torch.stack((grad[0]*0, grad[1])))]
    return handles


def fit(policy, episodes, args, destination):
    annotate_reference(policy, episodes)
    report = {'initial': metrics(policy, episodes, cached_features=True), 'stages': {}}
    fit_episodes = [e for e in episodes if e['partition'] == 'fit']
    features = torch.cat([e['reference']['features'].flatten(0, 1) for e in fit_episodes])
    label = {k: torch.cat([e[k].flatten() for e in fit_episodes])
             for k in ('angle', 'eligible', 'sector')}
    indices = [torch.where(label['eligible'] & (label['sector'] == i))[0] for i in range(3)]
    if any(len(x) == 0 for x in indices):
        raise ValueError('Fitting data must contain front, side, and rear legal examples')
    used = 0
    for stage, steps, lr, encoder in (
        ('angle_head', args.head_steps, args.head_lr, False),
        ('neighbor_recurrent', args.encoder_steps, args.encoder_lr, True),
    ):
        if not steps:
            continue
        handles = configure_fit(policy, encoder)
        optimizer = torch.optim.Adam([p for p in policy.parameters() if p.requires_grad], lr=lr)
        losses = []
        for step in range(steps):
            optimizer.zero_grad()
            if not encoder:
                chosen = torch.cat([ix[torch.randint(len(ix), (args.head_batch//3,))] for ix in indices])
                beta = policy.beta_max * policy.mean(features[chosen])[..., 1].tanh()
                loss = direction_loss(beta, label['angle'][chosen], label['eligible'][chosen],
                                      label['sector'][chosen])
            else:
                chosen = torch.randperm(len(fit_episodes))[:args.episode_batch].tolist()
                terms = []
                for index in chosen:
                    episode = fit_episodes[index]
                    output = episode_output(policy, episode)
                    reference = episode['reference']
                    terms.append(direction_loss(output['beta'], episode['angle'],
                        episode['eligible'], episode['sector'])
                        + args.hold_distill_weight * F.mse_loss(output['hold_logit'], reference['hold_logit'])
                        + args.rho_distill_weight * F.mse_loss(output['rho'], reference['rho']))
                loss = torch.stack(terms).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite auxiliary direction loss')
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(
                [p for p in policy.parameters() if p.requires_grad], 1., error_if_nonfinite=True)
            optimizer.step()
            used += 1
            losses.append({'step': step+1, 'loss': float(loss.detach()), 'raw_gradient_norm': float(grad)})
            if (step+1) % 25 == 0 or step+1 == steps:
                json_write(destination/'metrics'/f'fit_progress_{used:06d}.json',
                           {'stage': stage, 'stage_step': step+1, 'offline_optimizer_steps': used,
                            'new_environment_steps': 0, **losses[-1]})
                print(json.dumps({'stage': stage, 'offline_optimizer_steps': used,
                                  'loss': losses[-1]['loss']}), flush=True)
        result = metrics(policy, episodes, cached_features=not encoder)
        report['stages'][stage] = {'steps': steps, 'losses': losses, 'metrics': result,
                                   'screen': screen(result)}
        torch.save({'policy': policy.export_diagnostic_state(),
                    'scope': 'supervised_direction_capacity_diagnostic_not_ppo',
                    'offline_optimizer_steps': used, 'new_environment_steps': 0,
                    'seed': args.seed, 'source_checkpoint': str(args.checkpoint)},
                   destination/'checkpoints'/f'{stage}.pt')
        json_write(destination/'metrics'/f'{stage}.json', report['stages'][stage])
        for handle in handles:
            handle.remove()
        del optimizer
        # A head-only success already answers capacity without changing memory.
        if not encoder and all(report['stages'][stage]['screen'].values()):
            report['encoder_skipped'] = 'head_only_candidate_already_met_capacity_screen'
            break
    report.update(offline_optimizer_steps=used, new_environment_steps=0,
                  maximum_offline_optimizer_steps=args.head_steps+args.encoder_steps,
                  ppo_updates=0, production_defaults_modified=False)
    return report


def direction_cases():
    """Four shared spawns; desired center bearing is front/left/right/rear."""
    radii = np.array([12., 18., 24., 30.])
    relative_center_bearing = np.array([0., math.pi/2, -math.pi/2, math.pi])
    bearings = np.pi/4 + np.arange(4)*np.pi/2
    poses = np.zeros((4, 4, 3))
    for env, radius in enumerate(radii):
        poses[env, :, :2] = radius*np.stack((np.cos(bearings), np.sin(bearings)), -1)
        yaw = bearings+np.pi-relative_center_bearing[env]
        poses[env, :, 2] = (yaw+np.pi) % (2*np.pi)-np.pi
    return poses, {'radius_m': radii.tolist(),
        'desired_center_bearing_deg': np.degrees(relative_center_bearing).tolist(),
        'poses_world_xy_yaw': poses.tolist(), 'initial_linear_speed_mps': .1,
        'initial_angular_speed_radps': 0., 'deadline_seconds': 300.,
        'scope': 'fixed geometric cases; radius and relative direction vary together',
        'actor_input': 'unchanged legal observations; world poses used only for spawning/evaluation'}


def evaluate_candidate(args):
    """Explicitly separate off-policy no-learning checks from offline fitting."""
    from diagnose_upstream_environment import event_data
    from run_upstream_t11 import make_collector
    from upstream_curriculum_experiment import load_curriculum, timing_for
    from upstream_decision_evaluation import panel

    if args.candidate is None:
        raise ValueError('--execute-eval requires an explicit --candidate checkpoint')
    destination = args.output
    baseline = destination/'checkpoints'/'source_baseline.pt'
    candidate = args.candidate.resolve()
    fit_summary = json.loads((destination/'metrics'/'summary.json').read_text())
    protocol = json.loads((destination/'config'/'protocol.json').read_text())
    config = load_curriculum(args.curriculum_config)
    if (config['version'] != 'rendezvous_curriculum_1km_v2'
            or config['observation_scaling'] != 'log1p'
            or config['policy_decision_period_s'] != 5.):
        raise ValueError('This panel preserves the reviewed v2 observation/timing settings')
    if Path(protocol['arguments']['physical_config']).resolve() != Path(args.physical_config).resolve():
        raise ValueError('Use the same physical configuration as offline fitting')
    paths = {'baseline': baseline, 'candidate': candidate}
    policies = {name: load_policy(torch.load(path, map_location='cpu', weights_only=False)['policy'])
                for name, path in paths.items()}
    expected = cfg_from_experiment(args.physical_config).spec.fingerprint
    if any(policy.contract.profile_fingerprint != expected for policy in policies.values()):
        raise ValueError('Diagnostic policies do not match the physical profile')
    reservation = destination/'metrics'/'closed_loop_reservation.json'
    if reservation.exists():
        raise FileExistsError('Closed-loop budget already reserved; do not replay or extend it')
    poses, cases = direction_cases()
    environment_seed, action_seed = 104729, 104759
    record = {'maximum_base_steps': 1600, 'reserved_base_steps': 1600,
        'panels': 4, 'environments_per_panel': 4, 'seconds_per_panel': 20.,
        'base_dt': .2, 'optimizer_updates': 0, 'environment_seed': environment_seed,
        'action_seed': action_seed, 'policies': {name: {'file': str(path), 'sha256': file_sha256(path)}
                                               for name, path in paths.items()},
        'script_sha256': file_sha256(Path(__file__)), 'learning_time': timing_for(config).metadata(.2),
        'physical_config': args.physical_config, 'curriculum_config': args.curriculum_config,
        'physical_config_sha256': file_sha256(args.physical_config),
        'curriculum_config_sha256': file_sha256(args.curriculum_config),
        'MPPI_samples': 512, 'cases': cases,
        'offline_optimizer_steps': fit_summary['offline_optimizer_steps'],
        'scope': 'paired short-horizon behavior, not complete rendezvous success or PPO learning'}
    json_write(reservation, record)
    json_write(destination/'state'/'closed_loop_initial_cases.json', cases)
    results = {}
    actual = 0
    try:
        with torch.random.fork_rng(devices=[]):
            for name, policy in policies.items():
                for deterministic in (False, True):
                    if actual+400 > 1600:
                        raise ValueError('Insufficient remaining full-vector panel budget')
                    key = f'{name}_{"mean" if deterministic else "sample"}'
                    collector = make_collector(args.physical_config, 'A1', environment_seed,
                        samples=512, num_envs=4, episode_seconds=1200.,
                        policy=deepcopy(policy).cpu().eval(), learning_time=timing_for(config),
                        curriculum_config=deepcopy(config))
                    collector.env.curriculum.set_level(2)
                    collector.env.curriculum.episode_levels[:] = 2
                    collector.env.curriculum.deadlines[:] = 300.
                    collector.env.task.deadlines[:] = 300.
                    collector.env.reset(np.arange(4), poses.copy())
                    np.testing.assert_allclose(collector.env.core.poses, poses, atol=1e-12)
                    np.testing.assert_allclose(collector.env.core.actual_velocity[..., 0], .1)
                    np.testing.assert_allclose(collector.env.core.actual_velocity[..., 1], 0.)
                    obs, _ = collector.env.observe()
                    labels = legal_labels(obs, collector.env.spec)
                    if not (labels['neighbor_count'] == 3).all():
                        raise ValueError('Fixed initial cases require three legally visible neighbors')
                    np.testing.assert_allclose(labels['angle'].cos().numpy(),
                        np.repeat(np.cos(np.radians(cases['desired_center_bearing_deg']))[:, None], 4, axis=1),
                        atol=2e-6)
                    np.testing.assert_allclose(labels['angle'].sin().numpy(),
                        np.repeat(np.sin(np.radians(cases['desired_center_bearing_deg']))[:, None], 4, axis=1),
                        atol=2e-6)
                    torch.manual_seed(action_seed)
                    row = panel(collector, deterministic=deterministic, seconds=20.)
                    actual += row['actual_base_steps']
                    if actual > 1600:
                        raise AssertionError('Physical panel exceeded the explicitly reserved budget')
                    json_write(destination/'metrics'/f'closed_loop_trace_{key}.json', event_data(row.pop('trace')))
                    results[key] = row
                    json_write(destination/'metrics'/f'closed_loop_{key}.json', event_data(row))
                    json_write(destination/'metrics'/f'closed_loop_progress_{key}.json',
                        {'completed_panel': key, 'actual_base_steps': actual, 'optimizer_updates': 0})
                    print(json.dumps({'panel': key, 'actual_base_steps': actual,
                        'mean_dmax_reduction': float(np.mean(row['dmax_reduction']))}), flush=True)
                    del collector
                    gc.collect()
        comparison = {}
        for mode in ('sample', 'mean'):
            before, after = results[f'baseline_{mode}'], results[f'candidate_{mode}']
            comparison[mode] = {key: {
                'baseline': before[key], 'candidate': after[key],
                'candidate_minus_baseline': np.asarray(after[key])-np.asarray(before[key])}
                for key in ('dmax_reduction', 'dispersion_reduction', 'inward_move_fraction',
                            'hold_fraction', 'hold_probability')}
        summary = event_data({'actual_base_steps': actual, 'reserved_base_steps': 1600,
            'optimizer_updates': 0, 'full_episode_evaluation': False,
            'results': results, 'comparison': comparison,
            'labels_never_enter_actor': True, 'counterfactual_trajectories_never_enter_ppo': True})
        json_write(destination/'metrics'/'closed_loop_summary.json', summary)
        json_write(destination/'metrics'/'closed_loop_settlement.json',
            {'actual_base_steps': actual, 'charged_base_steps': actual, 'optimizer_updates': 0})
    except BaseException as error:
        # A partially advanced failing panel is not replayed for free.
        json_write(destination/'metrics'/'closed_loop_failure.json', {
            'error': repr(error), 'completed_actual_base_steps': actual,
            'charged_base_steps': 1600, 'reservation_retained': True})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--policy-state', type=Path,
                        help='Optional previous diagnostic checkpoint containing an explicit policy export')
    parser.add_argument('--physical-config', default='configs/experiment/upstream_1km_curriculum_v1.yaml')
    parser.add_argument('--curriculum-config', default='configs/training/upstream_1km_curriculum_v2.yaml')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--candidate', type=Path,
                        help='Explicit direction candidate for the independent 1600-step paired panel')
    parser.add_argument('--batches', type=int, default=20)
    parser.add_argument('--head-steps', type=int, default=150)
    parser.add_argument('--encoder-steps', type=int, default=350)
    parser.add_argument('--head-lr', type=float, default=1e-3)
    parser.add_argument('--encoder-lr', type=float, default=3e-4)
    parser.add_argument('--head-batch', type=int, default=768)
    parser.add_argument('--episode-batch', type=int, default=2)
    parser.add_argument('--hold-distill-weight', type=float, default=10.)
    parser.add_argument('--rho-distill-weight', type=float, default=1.)
    parser.add_argument('--seed', type=int, default=20261021)
    execution = parser.add_mutually_exclusive_group()
    execution.add_argument('--execute-fit', action='store_true')
    execution.add_argument('--execute-eval', action='store_true')
    args = parser.parse_args()
    if not (0 < args.head_steps+args.encoder_steps <= 500 and min(args.head_steps, args.encoder_steps) >= 0):
        raise ValueError('This isolated diagnostic has a hard total of at most 500 optimization steps')
    if not (1 <= args.batches <= 20 and args.head_batch >= 3 and args.episode_batch >= 1):
        raise ValueError('Invalid finite dataset/minibatch setting')
    args.source = args.source.resolve()
    args.output = args.output.resolve()
    args.checkpoint = (args.checkpoint or args.source/'checkpoints'/'update_000020.pt').resolve()
    if args.execute_eval:
        torch.set_num_threads(1)
        evaluate_candidate(args)
        return
    protocol = {'scope': 'offline_supervised_direction_capacity_not_ppo',
        'source_run': str(args.source), 'source_checkpoint': str(args.checkpoint),
        'maximum_offline_optimizer_steps': args.head_steps+args.encoder_steps,
        'new_environment_steps': 0, 'validation_envs': VALIDATION_ENVS,
        'split': 'whole_episodes_and_global_environment_ids; no sample-wise leakage',
        'recurrent_replay': 'current_candidate_from_zero_on_each_complete_episode',
        'labels': 'legal Actor relative-neighbor center only; no centralized state or world positions',
        'target': 'conditional MOVE heading; no reward, gate, radius, or safety changes',
        'limitation': 'supervised capacity diagnostic; closed-loop and PPO learning not established',
        'closed_loop_proposal': 'after review: original/candidate x 4 fixed front-side-rear cases '
            'x sampled/deterministic x 20 seconds = at most 1600 base steps; no optimizer updates',
        'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    }
    if not args.execute_fit:
        print(json.dumps(protocol, indent=2))
        return
    if args.output == args.source or args.output.is_relative_to(args.source):
        raise ValueError('Use an independent output run; source experiment must remain immutable')
    args.output.mkdir(parents=True, exist_ok=False)
    for name in ('config', 'metrics', 'checkpoints', 'state', 'logs'):
        (args.output/name).mkdir()
    protocol['command'] = [sys.executable, *sys.argv]
    protocol['script_sha256'] = file_sha256(Path(__file__))
    protocol['source_checkpoint_sha256'] = file_sha256(args.checkpoint)
    if args.policy_state:
        protocol['policy_state_sha256'] = file_sha256(args.policy_state)
    json_write(args.output/'run_manifest.json', protocol)
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    cfg = cfg_from_experiment(args.physical_config)
    saved = read_checkpoint(args.checkpoint, learner_only=True)
    state = saved['trainer']['collector']['policy']
    if args.policy_state:
        state = torch.load(args.policy_state, map_location='cpu', weights_only=False)['policy']
    policy = load_policy(state)
    if policy.contract.profile_fingerprint != cfg.spec.fingerprint:
        raise ValueError('Source policy and physical scaling profile differ')
    # The selected source is the explicit log1p v2 archive, not a legacy layout.
    source_config = json.loads((args.source/'config'/'effective_configuration.json').read_text())
    if source_config['curriculum']['observation_scaling'] != 'log1p':
        raise ValueError('Source configuration is not the explicit log1p protocol')
    torch.save({'policy': policy.export_diagnostic_state(),
                'scope': 'frozen_reference_before_auxiliary_direction_fitting',
                'source_checkpoint': str(args.checkpoint),
                'source_policy_state': str(args.policy_state) if args.policy_state else None},
               args.output/'checkpoints'/'source_baseline.pt')
    del saved
    episodes, provenance, dropped = complete_episodes(args.source, args.batches, cfg.spec)
    split = [{k: v for k, v in episode.items() if not torch.is_tensor(v)} | {
        'team_decisions': len(episode['obs']),
        'sector_counts': {name: int((episode['eligible'] & (episode['sector'] == i)).sum())
                          for i, name in enumerate(SECTORS)}} for episode in episodes]
    json_write(args.output/'state'/'dataset_split.json', {
        'source_files': provenance, 'episodes': split, 'dropped': dropped,
        'validation_envs': VALIDATION_ENVS})
    json_write(args.output/'config'/'protocol.json', protocol)
    original_frozen = {k: v.clone() for k, v in policy.state_dict().items()
        if k.startswith(('critic.', 'value_normalizer.', 'hold_logit.')) or k == 'log_std'}
    report = fit(policy, episodes, args, args.output)
    report['frozen_parameters_unchanged'] = all(torch.equal(value, policy.state_dict()[key])
                                               for key, value in original_frozen.items())
    if not report['frozen_parameters_unchanged']:
        raise AssertionError('Direction fitting changed a frozen gate/critic/noise parameter')
    json_write(args.output/'metrics'/'summary.json', report)
    print(json.dumps({k: v for k, v in report.items() if k not in ('initial', 'stages')}, indent=2))


if __name__ == '__main__':
    main()
