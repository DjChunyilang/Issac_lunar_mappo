"""Bounded, isolated learning autopsy; never resumes the production trainer."""
import argparse
from copy import deepcopy
from dataclasses import replace
import gc
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from types import MethodType

import torch
import numpy as np

from _common import ROOT, cfg_from_experiment
from planner_evidence import json_write
from run_upstream_t11 import make_collector
from upstream_checkpoint import read_checkpoint
from upstream_run_state import save_state, restore_state
from upstream_decision_evaluation import far_collector, panel
from diagnose_upstream_decision_credit import local_centroid_requests, geometry
from diagnose_upstream_environment import event_data
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.action_ablation import (
    AblationPolicy, ActionContract, ActionPayload, AblationSample,
    replay_action_sequence, clipped_actor_objective,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.learning import duration_gae
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_decision import LearningTimeConfig, step_decision, action_can_affect_transition
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_ablation_learning import renew_legal_bounds, UpstreamAblationRollout
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_environment import UpstreamEnvironment
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_training import TrainingOptions, recurrent_mappo_update

BASE = 'configs/experiment/upstream_scale_s04_braking_lease_cuda_v5.yaml'
SOURCE = ROOT / 'outputs/runs/upstream_scale_t11/decision_credit_seed23_200k_20261018_01'
RUN = ROOT / 'outputs/runs/upstream_scale_t11/decision_credit_repair_20261019_01'
MAX_BASE_STEPS = 12000
FAR = ROOT / 'outputs/runs/upstream_scale_t11/oom_fix_owner16_20261017_01_a1/checkpoints/update_000061.pt'


def prepare_run():
    for name in ('config', 'metrics', 'state', 'logs'):
        (RUN / name).mkdir(parents=True, exist_ok=True)
    if not (RUN / 'run_manifest.json').exists():
        json_write(RUN / 'run_manifest.json', {
            'purpose': 'ordered_learning_repair_diagnostics', 'source_suite': str(SOURCE),
            'maximum_base_steps': MAX_BASE_STEPS, 'production_training': False,
            'isolated_optimizer_updates': True, 'original_checkpoints_modified': False,
        })


def reserve(name, steps):
    path = RUN / 'metrics' / (name + '_reservation.json')
    used = 0
    for p in (RUN / 'metrics').glob('*_reservation.json'):
        settlement = p.with_name(p.name.replace('_reservation.json', '_settlement.json'))
        if settlement.exists():
            closed = json.loads(settlement.read_text())
            used += closed['charged_base_steps'] if 'charged_base_steps' in closed else closed['actual_base_steps']
        else:
            used += json.loads(p.read_text())['reserved_base_steps']
    if path.exists() or used + steps > MAX_BASE_STEPS:
        raise ValueError('Duplicate phase or diagnostic budget exceeded')
    json_write(path, {'reserved_base_steps': steps, 'already_reserved': used})


def policy_from(state):
    md = state['metadata']
    contract = ActionContract(**{k: md[k] for k in (
        'representation', 'profile_fingerprint', 'rho_max', 'speed_cap_max', 'beta_max', 'request_seconds')})
    policy = AblationPolicy(991, contract)
    policy.load_diagnostic_state(state)
    return policy


def checkpoint(arm, index):
    return Path(str(SOURCE) + '_' + arm.lower()) / 'checkpoints' / f'update_{index:06d}.pt'


def collect(arm):
    name = 'batch_' + arm.lower()
    reserve(name, 2500)
    saved = read_checkpoint(checkpoint(arm, 19))
    central = saved['trainer']['collector']
    timing = LearningTimeConfig(5., .2 if arm == 'B' else 1.)
    policy = policy_from(central['policy'])
    c = make_collector(BASE, 'A1', 23, num_envs=4, policy=policy, mode='signal', learning_time=timing)
    env_state = torch.load(central['owner_shards'][0]['file'], map_location='cpu', weights_only=False)
    c.env.load_state_dict(env_state)
    del env_state
    c.hidden = central['hidden'][:4].clone()
    torch.manual_seed(20261019)
    rng = torch.get_rng_state()
    parts = []; started = time.perf_counter()
    for i in range(25):
        batch, _, _ = c.collect(1)
        parts.append(batch)
        status = {'decisions_per_env': i + 1, 'actual_base_steps': c.base_steps,
                  'seconds': time.perf_counter() - started}
        json_write(RUN / 'metrics' / f'{name}_progress_{i+1:03d}.json', status)
        print(json.dumps({'arm': arm, **status}), flush=True)
    batch = {k: torch.cat([p[k] for p in parts]) for k in parts[0] if torch.is_tensor(parts[0][k])}
    batch['samples'] = [s for p in parts for s in p['samples']]
    batch['metadata'] = parts[0]['metadata']
    save_state(SimpleNamespace(path=RUN), name, {
        'batch': batch, 'policy': central['policy'], 'optimizer': saved['trainer']['optimizer'],
        'options': saved['identity']['options'], 'sampling_rng': rng, 'update_rng': torch.get_rng_state(),
        'actual_base_steps': c.base_steps, 'source': str(checkpoint(arm, 19)),
        'scope': 'owner_zero_new_on_policy_sample; not exact historical full-vector trajectory',
    })


def norm(parameters):
    return float(sum(p.grad.square().sum() for p in parameters if p.grad is not None).sqrt())


def quality(policy, batch, advantages, targets):
    with torch.no_grad(), torch.random.fork_rng():
        replay = replay_action_sequence(policy, batch['obs'], batch['hidden'][0], batch['samples'],
                                       batch['learning_valid'], batch['terminated'])
        log = replay['log_prob'] - torch.stack([s.log_prob for s in batch['samples']])
        values = policy.value(batch['state'])
        return {'pg_loss': float(clipped_actor_objective(replay['ratio'], advantages, batch['learning_valid'])),
                'kl': float((replay['ratio'] - 1 - log).mean()),
                'critic_mse': float((values - targets).square().mean()),
                'max_logprob_error': float(log.abs().max())}


def update(arm):
    data = restore_state(RUN / 'state' / ('batch_' + arm.lower() + '.pkl.gz'))
    batch = data['batch']; options = TrainingOptions(**data['options'])
    adv, targets = duration_gae(batch['reward'], batch['value'], batch['next_value'], batch['duration'],
        batch['terminated'], batch['truncated'], gamma=options.gamma, lam=options.lam,
        bootstrap_allowed=batch['bootstrap_allowed'], learning_valid=batch['learning_valid'])
    adv = (adv - adv.mean()) / adv.std(unbiased=False).clamp_min(1e-6)
    report = {'arm': arm, 'new_base_steps': 0, 'optimizer_steps': 10, 'variants': {}}
    for variant in ('global', 'separate'):
        policy = policy_from(data['policy'])
        optimizer = torch.optim.Adam(policy.parameters(), lr=options.learning_rate)
        optimizer.load_state_dict(deepcopy(data['optimizer']))
        actor_params = [p for n, p in policy.named_parameters() if not n.startswith('critic.')]
        critic_params = list(policy.critic.parameters())
        torch.set_rng_state(data['update_rng'])
        rows = []; initial = quality(policy, batch, adv, targets)
        if initial['max_logprob_error'] > 2e-5:
            raise ValueError('Full recurrent probability replay differs before updating')
        for _ in range(5):
            old = [p.detach().clone() for p in actor_params]
            optimizer.zero_grad()
            replay = replay_action_sequence(policy, batch['obs'], batch['hidden'][0], batch['samples'],
                                           batch['learning_valid'], batch['terminated'])
            pg = clipped_actor_objective(replay['ratio'], adv, batch['learning_valid'], options.clip)
            mse = (policy.value(batch['state']) - targets).square().mean()
            loss = pg + options.value_weight * mse - options.entropy_weight * replay['entropy'].mean()
            loss.backward()
            actor_norm, critic_norm = norm(actor_params), norm(critic_params)
            if variant == 'global':
                torch.nn.utils.clip_grad_norm_(policy.parameters(), options.gradient_limit)
            else:
                torch.nn.utils.clip_grad_norm_(actor_params, options.gradient_limit)
                torch.nn.utils.clip_grad_norm_(critic_params, options.gradient_limit)
            optimizer.step()
            rows.append({'actor_raw_norm': actor_norm, 'critic_raw_norm': critic_norm,
                'actor_parameter_delta': float(sum((p.detach()-q).square().sum() for p,q in zip(actor_params,old)).sqrt()),
                **quality(policy, batch, adv, targets)})
        report['variants'][variant] = {'initial': initial, 'epochs': rows}
        del policy, optimizer; gc.collect()
    # Fixed target regression, 3 environments for fitting, one held out.
    policy = policy_from(data['policy'])
    optimizer = torch.optim.Adam(policy.critic.parameters(), lr=1e-3)
    rows = []
    for step in range(301):
        if step % 25 == 0:
            with torch.no_grad():
                residual = policy.value(batch['state']) - targets
                rows.append({'step': step, 'fit_mse': float(residual[:, :3].square().mean()),
                             'heldout_mse': float(residual[:, 3:].square().mean()),
                             'fit_bias': float(residual[:, :3].mean()),
                             'heldout_bias': float(residual[:, 3:].mean())})
        if step == 300:
            break
        optimizer.zero_grad()
        loss = (policy.value(batch['state'][:, :3]) - targets[:, :3]).square().mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.critic.parameters(), 1.)
        optimizer.step()
    report['critic_fit'] = rows
    report['optimizer_steps'] += 300
    report['critic_fit_scope'] = 'fixed GAE targets, fresh critic-only Adam lr1e-3, not true-value calibration'
    json_write(RUN / 'metrics' / ('update_' + arm.lower() + '.json'), report)
    print(json.dumps(report), flush=True)


def intervention_sampler(gate, noise_scale):
    """Factorized diagnostic actions, never eligible for PPO training."""
    def sample(policy, obs, hidden, deterministic=False):
        distribution, following = policy.distribution(obs, hidden)
        # Always consume the same draws, irrespective of the forced gate.
        uniform = torch.rand_like(distribution.logits)
        noise = torch.randn_like(distribution.normal.loc)
        if gate == 'move':
            holds = torch.zeros_like(uniform, dtype=torch.bool)
        elif gate == 'hold':
            holds = torch.ones_like(uniform, dtype=torch.bool)
        elif gate == 'mean':
            holds = distribution.logits > 0
        else:
            holds = uniform < distribution.logits.sigmoid()
        latent = distribution.normal.loc + noise_scale * distribution.normal.scale * noise
        latent = torch.where(holds[..., None], 0., latent)
        physical = torch.where(holds[..., None], 0., distribution.transform(latent))
        action = ActionPayload(physical, holds, torch.full_like(holds, -1, dtype=torch.int64), policy.contract.version)
        return AblationSample(action, latent, distribution.log_prob(latent, holds), following)
    return sample


def verify(arm):
    """Check the production update against the isolated five-epoch comparison."""
    data = restore_state(RUN / 'state' / ('batch_' + arm.lower() + '.pkl.gz'))
    batch = data['batch']
    reference = json.loads((RUN / 'metrics' / ('update_' + arm.lower() + '.json')).read_text())
    results = {}
    for mode, label in [('global', 'global'), ('actor_critic', 'separate')]:
        options = TrainingOptions(**(data['options'] | {'gradient_clipping': mode}))
        policy = policy_from(data['policy'])
        optimizer = torch.optim.Adam(policy.parameters(), lr=options.learning_rate)
        optimizer.load_state_dict(deepcopy(data['optimizer']))
        adv, targets = duration_gae(batch['reward'], batch['value'], batch['next_value'], batch['duration'],
            batch['terminated'], batch['truncated'], gamma=options.gamma, lam=options.lam,
            bootstrap_allowed=batch['bootstrap_allowed'], learning_valid=batch['learning_valid'])
        adv = (adv - adv.mean()) / adv.std(unbiased=False).clamp_min(1e-6)
        torch.set_rng_state(data['update_rng'])
        metrics = recurrent_mappo_update(policy, optimizer, batch, options)
        actual = quality(policy, batch, adv, targets)
        expected = reference['variants'][label]['epochs'][-1]
        for key in ('pg_loss', 'kl', 'critic_mse'):
            np.testing.assert_allclose(actual[key], expected[key], rtol=1e-5, atol=1e-6)
        results[mode] = {'quality': actual, 'metrics': metrics, 'matches_isolated_update': True}
    json_write(RUN / 'metrics' / ('verify_' + arm.lower() + '.json'), results)
    print(json.dumps(results), flush=True)


def behavior():
    modes = [('initial_move_mean', 0, 'move', 0.),
             ('final_move_random', 20, 'move', 1.),
             ('final_random_mean', 20, 'random', 0.),
             ('final_random_random', 20, 'random', 1.),
             ('final_random_quarter_noise', 20, 'random', .25)]
    for name, index, gate, scale in modes:
        reserve(name, 400)
        saved = read_checkpoint(checkpoint('C', index), learner_only=True)
        policy = policy_from(saved['trainer']['collector']['policy'])
        c = far_collector(BASE, FAR, policy, LearningTimeConfig(5., 1.))
        c.policy.sample = MethodType(intervention_sampler(gate, scale), c.policy)
        torch.manual_seed(20261019)
        result = panel(c)
        json_write(RUN / 'metrics' / (name + '.json'), event_data({
            'gate': gate, 'continuous_noise_scale': scale, 'source_update': index,
            'intervention_only': True, 'optimizer_updates': 0, 'result': result}))
        print(json.dumps({'phase': name, 'base_steps': result['actual_base_steps'],
                          'dmax_reduction': result['dmax_reduction'].tolist()}), flush=True)
        del c, saved, policy; gc.collect()


@torch.no_grad()
def credit():
    """Change rover 1 for one decision; all following actions use common noise."""
    for mode in ('hold', 'toward', 'away'):
        name = 'credit_' + mode
        reserve(name, 400)
        saved = read_checkpoint(checkpoint('C', 20), learner_only=True)
        policy = policy_from(saved['trainer']['collector']['policy'])
        c = far_collector(BASE, FAR, policy, LearningTimeConfig(5., 1.))
        sampler = intervention_sampler('random', 1.)
        hidden = torch.zeros_like(c.hidden)
        torch.manual_seed(20261019)
        initial = geometry(c.env.core.poses)
        rows = []; trace = []; discounted = np.zeros(4); elapsed = np.zeros(4, dtype=np.int64)
        for step in range(4):
            renew_legal_bounds(c.env)
            obs, state = c.env.observe()
            sampled = sampler(c.policy, obs, hidden)
            hidden = sampled.hidden
            action = sampled.action
            if step == 0:
                physical, holds = action.continuous.clone(), action.hold.clone()
                targets = local_centroid_requests(c.env, hold=mode=='hold', away=mode=='away')
                for e in range(4):
                    pose = c.env.local_pose(e, 1)
                    delta = targets[e][1].goal_xy - pose[:2]
                    holds[e, 1] = targets[e][1].hold
                    angle = np.arctan2(delta[1], delta[0]) - pose[2]
                    physical[e, 1] = torch.tensor([min(2., np.linalg.norm(delta)), np.arctan2(np.sin(angle), np.cos(angle))])
                physical[holds] = 0.
                action = replace(action, continuous=physical, hold=holds)
            value = c.policy.value(state)
            transition = step_decision(c.bridge, action, c.learning_time, renew_legal_bounds)
            if not transition.learning_valid[transition.duration > 0].all():
                raise ValueError('Invalid single-decision credit trajectory')
            following = c.policy.value(transition.terminal_critic)
            following = torch.where(transition.bootstrap_allowed, following, 0.)
            row = {'reward': transition.reward, 'value': value, 'next_value': following,
                   'duration': transition.duration, 'terminated': transition.terminated,
                   'truncated': transition.truncated, 'learning_valid': transition.learning_valid,
                   'bootstrap_allowed': transition.bootstrap_allowed}
            rows.append(row)
            trace.append({'geometry': geometry(c.env.core.poses), 'action': action.continuous,
                          'hold': action.hold, 'reward_components': transition.reward_components})
            discounted += c.gamma_step**elapsed * transition.reward.numpy()
            elapsed += transition.duration.numpy()
            hidden = torch.where(transition.terminated[:, None, None], 0., hidden)
            c.env.discard_delivered_history()
        arrays = {k: torch.stack([r[k] for r in rows]) for k in rows[0]}
        adv, _ = duration_gae(arrays['reward'], arrays['value'], arrays['next_value'], arrays['duration'],
            arrays['terminated'], arrays['truncated'], gamma=c.gamma_step, lam=c.lambda_step,
            bootstrap_allowed=arrays['bootstrap_allowed'], learning_valid=arrays['learning_valid'])
        first_td = arrays['reward'][0] + c.gamma_step**arrays['duration'][0] * arrays['next_value'][0] - arrays['value'][0]
        result = {'mode': mode, 'changed_rover': 1, 'changed_decisions': 1, 'optimizer_updates': 0,
                  'actual_base_steps': int(elapsed.sum()), 'dmax_reduction': initial['dmax']-geometry(c.env.core.poses)['dmax'],
                  'dispersion_reduction': initial['dispersion']-geometry(c.env.core.poses)['dispersion'],
                  'discounted_reward': discounted, 'first_td': first_td, 'first_gae': adv[0],
                  'finite_window_residual': discounted+c.gamma_step**elapsed*following.numpy()-arrays['value'][0].numpy(),
                  'learning': arrays, 'trace': trace}
        json_write(RUN / 'metrics' / (name + '.json'), event_data(result))
        print(json.dumps(event_data(result)), flush=True)
        del c, policy, saved; gc.collect()


def diverse_near(policy):
    """Four distinct legal parking geometries, not rotations of one square."""
    timing = LearningTimeConfig(5., 1.)
    c = make_collector(BASE, 'A1', 23, num_envs=4, policy=policy, learning_time=timing)
    poses = np.zeros((4, 4, 3))
    poses[..., :2] = [
        [[-3.6,-3.6],[-3.6,3.6],[3.6,-3.6],[3.6,3.6]],
        [[-4.25,-3.5],[-4.25,3.5],[4.25,-3.5],[4.25,3.5]],
        [[-4.,-3.4],[-3.4,3.6],[3.6,-3.5],[4.1,3.7]],
        [[-3.8,-3.4],[-3.3,3.8],[3.5,-3.8],[4.2,3.5]],
    ]
    poses[..., 2] = [[0., 1.2, -2.1, 2.8], [.8,-.6,2.2,-2.6],
                      [-1.,2.5,.4,-2.], [2.9,-1.8,.9,-.2]]
    distance = np.linalg.norm(poses[:, :, None, :2]-poses[:, None, :, :2], axis=-1)
    minimum = np.min(np.where(np.eye(4, dtype=bool), np.inf, distance), axis=(1,2))
    radius = np.linalg.norm(poses[..., :2]-poses[..., :2].mean(1,keepdims=True), axis=-1).max(1)
    if (np.any(minimum < 2*c.env.spec.vehicle.effective_radius) or np.any(distance.max((1,2))>12.)
            or np.any(radius+c.env.spec.vehicle.effective_radius > c.env.spec.task.worksite_radius)):
        raise ValueError('Parking probe initial geometry does not satisfy task clearance/containment')
    env = UpstreamEnvironment(c.env.cfg, poses, reward_settings=c.env.reward_tracker.settings)
    physical = env.core.state_dict()
    physical['previous_control'][..., 0] = .1
    physical['actual_velocity'][..., 0] = .1
    env.core.load_state_dict(physical)
    return UpstreamAblationRollout(env, policy, timing)


def parking():
    # The invalid first attempt is charged separately. Reuse its valid far
    # baseline, shorten learning, and keep this retry inside the original cap.
    reserve('parking', 1800)
    saved = read_checkpoint(checkpoint('C', 20), learner_only=True)
    policy = policy_from(saved['trainer']['collector']['policy'])
    g, l = LearningTimeConfig(5., 1.).discounts(.2)
    options = TrainingOptions(rollout_macro_steps=2, gamma=g, lam=l, gradient_clipping='actor_critic')
    optimizer = torch.optim.Adam(policy.parameters(), lr=options.learning_rate)
    evaluations = {}; batches = []; actual = 0
    reference = ROOT / 'outputs/runs/upstream_scale_t11/decision_credit_repair_20261019_01/metrics/parking_attempt1_far_before.json'
    evaluations['far_before'] = json.loads(reference.read_text())
    for name in ('near_before', 'near_hold'):
        c = diverse_near(deepcopy(policy))
        if name == 'near_hold':
            c.policy.sample = MethodType(intervention_sampler('hold', 0.), c.policy)
        torch.manual_seed(20261019)
        evaluations[name] = panel(c, deterministic=True, seconds=10. if name=='near_hold' else 20.)
        actual += evaluations[name]['actual_base_steps']
        json_write(RUN / 'metrics' / ('parking_' + name + '.json'), event_data(evaluations[name]))
        del c; gc.collect()
    for index in range(2):
        c = diverse_near(policy)
        torch.manual_seed(20261019 + index)
        batch, episodes, records = c.collect(2)
        actual += c.base_steps
        save_state(SimpleNamespace(path=RUN), f'parking_batch_{index:02d}', {
            'batch': batch, 'policy': policy.export_diagnostic_state(), 'optimizer': optimizer.state_dict(),
            'options': options.metadata, 'update_rng': torch.get_rng_state(),
        })
        update_result = recurrent_mappo_update(policy, optimizer, batch, options)
        row = {'index': index, 'actual_base_steps': c.base_steps, 'episodes': episodes,
               'contained_steps': sum(int(np.count_nonzero(r['task']['contained'] & r['execution']['active_envs']))
                                      for r in records if 'execution' in r),
               'update': update_result}
        batches.append(row)
        json_write(RUN / 'metrics' / f'parking_update_{index:02d}.json', event_data(row))
        print(json.dumps({'parking_update': index, 'steps': actual, 'episodes': episodes}), flush=True)
        del c, batch, records; gc.collect()
    for name in ('near_after', 'far_after'):
        c = (far_collector(BASE, FAR, policy, LearningTimeConfig(5., 1.)) if name.startswith('far')
             else diverse_near(deepcopy(policy)))
        torch.manual_seed(20261019)
        evaluations[name] = panel(c, deterministic=True)
        actual += evaluations[name]['actual_base_steps']
        json_write(RUN / 'metrics' / ('parking_' + name + '.json'), event_data(evaluations[name]))
        del c; gc.collect()
    save_state(SimpleNamespace(path=RUN), 'parking_diagnostic_learner', {
        'policy': policy.export_diagnostic_state(), 'optimizer': optimizer.state_dict(), 'options': options.metadata,
        'scope': 'near-only bounded diagnostic, not production or transferable skill acceptance',
    })
    json_write(RUN / 'metrics' / 'parking_settlement.json', {'actual_base_steps': actual})
    json_write(RUN / 'metrics' / 'parking_summary.json', event_data({
        'actual_base_steps': actual, 'optimizer_steps': 10, 'batches': batches,
        'evaluations': evaluations, 'initial_state_scope': 'four near geometries, heterogeneous headings, 0.1m/s',
        'fresh_adam': True, 'maximum_learning_steps': 400, 'reused_far_reference': str(reference),
    }))


def terminal_credit():
    """Reuse the observed HOLD-to-unexecuted-MOVE terminal; no environment steps."""
    data=restore_state(RUN / 'state/parking_batch_01.pkl.gz')
    batch=data['batch'];cfg=cfg_from_experiment(BASE)
    batch['action_effective']=action_can_affect_transition(batch['terminated'],batch['duration'],
                                                          cfg.simulated_latency,cfg.spec.grid.dt)
    result={'new_base_steps':0,'latency':cfg.simulated_latency,'dt':cfg.spec.grid.dt,
            'action_effective':batch['action_effective'],'source':'state/parking_batch_01.pkl.gz','variants':{}}
    critics=[]
    for mode in ('all','effective'):
        policy=policy_from(data['policy'])
        optimizer=torch.optim.Adam(policy.parameters(),lr=data['options']['learning_rate'])
        optimizer.load_state_dict(deepcopy(data['optimizer']))
        options=TrainingOptions(**(data['options']|{'terminal_actor_credit':mode}))
        with torch.no_grad():
            distribution,_=policy.distribution(batch['obs'][0],batch['hidden'][0])
            before=distribution.logits.sigmoid().clone()
        torch.set_rng_state(data['update_rng'])
        metrics=recurrent_mappo_update(policy,optimizer,batch,options)
        with torch.no_grad():
            distribution,_=policy.distribution(batch['obs'][0],batch['hidden'][0])
            after=distribution.logits.sigmoid()
        result['variants'][mode]={'metrics':metrics,'hold_probability_before':before,
                                 'hold_probability_after':after,'hold_probability_change':after-before}
        critics.append(deepcopy(policy.critic.state_dict()))
    assert all(torch.equal(critics[0][key],critics[1][key]) for key in critics[0])
    result['critic_parameters_identical']=True
    json_write(RUN/'metrics/terminal_credit_fix.json',event_data(result))
    print(json.dumps(event_data(result)),flush=True)


def main():
    global RUN
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['collect', 'update', 'verify', 'behavior', 'credit', 'parking', 'terminal-credit'])
    parser.add_argument('--arm', choices=['B', 'C'])
    parser.add_argument('--run-id', default=RUN.name, help='Use a new run name to reproduce; existing phases are never overwritten')
    args = parser.parse_args()
    if Path(args.run_id).name != args.run_id or args.run_id in ('.', '..'):
        parser.error('Expected a single run directory name')
    RUN = ROOT / 'outputs/runs/upstream_scale_t11' / args.run_id
    torch.set_num_threads(2)
    prepare_run()
    json_write(RUN / 'config' / f'invocation_{time.time_ns()}.json', {'command': sys.argv,
               'torch_num_threads': 2, 'base_config': BASE, 'maximum_base_steps': MAX_BASE_STEPS})
    if args.phase == 'behavior':
        behavior()
    elif args.phase == 'credit':
        credit()
    elif args.phase == 'parking':
        parking()
    elif args.phase == 'terminal-credit':
        terminal_credit()
    elif args.arm is None:
        parser.error('--arm is required for collect/update')
    elif args.phase == 'collect':
        collect(args.arm)
    elif args.phase == 'verify':
        verify(args.arm)
    else:
        update(args.arm)


if __name__ == '__main__':
    main()
