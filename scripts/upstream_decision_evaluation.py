"""Fixed 20-second panels for the bounded decision-credit experiment.

Counterfactual trajectories never enter the learner. Evaluator RNG, recurrent
state and reward accumulators are independent of the training collector.
"""
from collections import Counter
from copy import deepcopy
import gc

import numpy as np
import torch

from diagnose_upstream_decision_credit import geometry, local_centroid_requests
from run_upstream_t11 import make_collector
from upstream_checkpoint import read_checkpoint
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_ablation_learning import UpstreamAblationRollout, renew_legal_bounds
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_decision import step_decision, renew_fixed_targets
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_environment import UpstreamEnvironment
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_reward import UpstreamReward, UpstreamRewardSettings


def far_collector(base, checkpoint, policy, timing, *, samples=512):
    saved = read_checkpoint(checkpoint)
    central = saved['trainer']['collector']
    if central['metadata']['policy']['representation'] != 'A1':
        raise ValueError('Far panel requires the historical A1 owner-0 snapshot')
    shard = central['owner_shards'][0]
    c = make_collector(base, 'A1', 23, samples=samples, num_envs=4, mode='signal',
                       policy=deepcopy(policy).cpu().eval())
    # Restore the original lifecycle first. Only reward accounting is migrated;
    # physical state, queues, terrain memories and legal messages stay identical.
    state = torch.load(shard['file'], map_location='cpu', weights_only=False)
    c.env.load_state_dict(state)
    del state, saved
    g, l = timing.discounts(c.env.spec.grid.dt)
    c.env.reward_tracker = UpstreamReward(c.env.spec, 4, UpstreamRewardSettings(gamma=g, lam=l))
    result = UpstreamAblationRollout(c.env, c.policy, timing)
    if result.env._ended.any():
        raise ValueError('Far panel starts in an ended environment')
    return result


def near_collector(base, policy, timing, *, samples=512):
    c = make_collector(base, 'A1', 23, samples=samples, num_envs=4, learning_time=timing,
                       policy=deepcopy(policy).cpu().eval())
    poses = np.zeros((4, 4, 3))
    square = np.array([[-3.6, -3.6], [-3.6, 3.6], [3.6, -3.6], [3.6, 3.6]])
    for e in range(4):
        yaw = e * np.pi / 2
        rotation = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        poses[e, :, :2] = square @ rotation.T
        poses[e, :, 2] = (yaw + np.pi) % (2*np.pi) - np.pi
    env = UpstreamEnvironment(c.env.cfg, poses, reward_settings=c.env.reward_tracker.settings)
    physical = env.core.state_dict()
    physical['previous_control'][..., 0] = .1
    physical['actual_velocity'][..., 0] = .1
    env.core.load_state_dict(physical)
    return UpstreamAblationRollout(env, c.policy, timing)


def inward_flags(env, targets):
    flags = np.zeros((env.e, 4), bool)
    eligible = np.zeros_like(flags)
    for e in range(env.e):
        if env._ended[e]:
            continue
        for a in range(4):
            pose = env.local_pose(e, a)
            snapshot = env.radio.snapshot(e, a, float(env.core.times[e]), pose)
            peers = [m.pose[:2] for i, m in enumerate(snapshot.messages) if snapshot.state_valid[i]]
            target = targets[e][a]
            if peers and not target.hold:
                delta = np.mean([pose[:2], *peers], axis=0)-pose[:2]
                direction = target.goal_xy-pose[:2]
                eligible[e,a] = np.linalg.norm(delta)>1e-8 and np.linalg.norm(direction)>1e-8
                flags[e,a] = eligible[e,a] and float(delta @ direction)>0
    return flags, eligible


@torch.no_grad()
def panel(c, *, deterministic=False, intervention=None, seconds=20):
    env = c.env
    initial = geometry(env.core.poses)
    hidden = torch.zeros_like(c.hidden)
    _, state = env.observe()
    initial_value = c.policy.value(state).numpy()
    elapsed = np.zeros(env.e, np.int64)
    discounted = np.zeros(env.e)
    hold_weight = np.zeros(env.e); probability_weight = np.zeros(env.e)
    action_weight = np.zeros(env.e); inward = np.zeros(env.e); eligible = np.zeros(env.e)
    longest_move = np.zeros((env.e,4)); moving = np.zeros_like(longest_move)
    jumps = np.zeros(env.e); jump_count = np.zeros(env.e); previous_targets = None
    target_changes = np.zeros(env.e)
    path_length = np.zeros((env.e,4)); speed_sum = np.zeros(env.e)
    overlap_steps = 0; clearance_steps = 0
    statuses = Counter(); episodes = []; trace = []
    step_seconds = c.learning_time.policy_decision_period_s if intervention is None else 1.
    scripted_targets = None
    for index in range(round(seconds/step_seconds)):
        if env._ended.all():
            break
        renew_legal_bounds(env)
        obs, state = env.observe()
        if intervention is None:
            sample = c.policy.sample(obs, hidden, deterministic=deterministic)
            targets, _ = c.bridge.decode(sample.action)
            probability = c.policy.hold_logit(sample.hidden).squeeze(-1).sigmoid().numpy()
            flags, legal = inward_flags(env, targets)
            transition = step_decision(c.bridge, sample.action, c.learning_time, renew_legal_bounds)
            hidden = torch.where(transition.terminated[:,None,None], 0., sample.hidden)
            holds = sample.action.hold.numpy()
        else:
            if index % 5 == 0:
                scripted_targets = local_centroid_requests(env, hold=intervention=='hold', away=intervention=='away')
            targets = renew_fixed_targets(scripted_targets, env.core.times, env.spec.grid.request_validity)
            flags, legal = inward_flags(env, targets)
            holds = np.array([[intent.hold for intent in row] for row in targets])
            probability = holds.astype(float)
            transition = env.step_intents(targets)
        duration = transition.duration.numpy()
        discounted += c.gamma_step**elapsed * transition.reward.numpy()
        elapsed += duration
        if not transition.learning_valid[transition.duration>0].all():
            raise ValueError('Invalid learning/execution state in evaluation panel')
        hold_weight += holds.mean(1)*duration
        probability_weight += probability.mean(1)*duration
        action_weight += duration
        inward += flags.sum(1)*duration; eligible += legal.sum(1)*duration
        moving = np.where(holds, 0., moving+duration[:,None]*env.spec.grid.dt)
        longest_move = np.maximum(longest_move, moving)
        target_array = np.array([[intent.goal_xy for intent in row] for row in targets])
        if previous_targets is not None:
            difference = np.linalg.norm(target_array-previous_targets, axis=-1)
            jumps += difference.mean(1)*(duration>0)
            target_changes += (difference>1e-8).mean(1)*(duration>0)
            jump_count += duration>0
        previous_targets = target_array
        for record in transition.records:
            if 'execution' not in record:
                continue
            ex = record['execution']
            path_length += np.linalg.norm(ex['after'][...,:2]-ex['before'][...,:2],axis=-1)
            speed_sum += np.abs(ex['applied_control'][...,0]).mean(1)
            overlap_steps += int(np.count_nonzero(ex['geometry_overlap']))
            clearance_steps += int(np.count_nonzero(ex['clearance_violation']))
            statuses.update(item['execution_status'] for item in record['decisions'])
        episodes.extend(transition.episodes)
        trace.append({'elapsed_steps':elapsed.copy(),'reward':transition.reward.numpy(),
            'duration':duration,'hold':holds,'hold_probability':probability,'targets':target_array,
            'geometry':geometry(env.core.poses),'terminated':transition.terminated.numpy(),
            'bootstrap_allowed':transition.bootstrap_allowed.numpy(),
            'reward_components':{k:v.numpy() for k,v in transition.reward_components.items()}})
        env.discard_delivered_history()
    _, state = env.observe()
    final_value = np.zeros(env.e)
    allowed = ~env._ended & env._valid
    if allowed.any():
        final_value[allowed] = c.policy.value(state[allowed]).numpy()
    final = geometry(env.core.poses)
    return {'initial':initial,'final':final,'dmax_reduction':initial['dmax']-final['dmax'],
        'dispersion_reduction':initial['dispersion']-final['dispersion'],
        'hold_fraction':hold_weight/np.maximum(action_weight,1),
        'hold_probability':probability_weight/np.maximum(action_weight,1),
        'inward_move_fraction':np.divide(inward,eligible,out=np.zeros_like(inward),where=eligible>0),
        'inward_eligible_rover_steps':eligible,'longest_move_seconds':longest_move,
        'mean_target_jump_m':jumps/np.maximum(jump_count,1),'path_length_m':path_length,
        'target_changes_per_second':target_changes/np.maximum(elapsed*env.spec.grid.dt,env.spec.grid.dt),
        'mean_applied_speed':speed_sum/np.maximum(elapsed,1),
        'discounted_reward':discounted,'critic_residual':discounted+c.gamma_step**elapsed*final_value-initial_value,
        'initial_value':initial_value,'final_value':final_value,
        'successes':sum(e['terminal_reason']=='success' for e in episodes),'episodes':episodes,
        'execution_status':dict(statuses),'actual_base_steps':int(elapsed.sum()),
        'geometry_overlap_steps':overlap_steps,'clearance_violation_steps':clearance_steps,
        'final_hold_count':env.last_task['hold_count'].copy(),
        'final_actual_stopped':env.last_task['actual_stopped'].copy(),
        'optimizer_updates':0,'trace':trace,
        'residual_scope':'finite-window critic residual; interventions are off-policy, never PPO samples'}


def evaluate_panels(base, checkpoint, policy, timing, *, final=False):
    results = {}
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        for geometry_name, constructor in (('far', far_collector), ('near', near_collector)):
            for deterministic in (False, True):
                kwargs = {'checkpoint':checkpoint} if geometry_name=='far' else {}
                c = constructor(base, policy=policy, timing=timing, **kwargs)
                torch.manual_seed(20261018)
                results[f'{geometry_name}_{"mean" if deterministic else "sample"}'] = panel(c, deterministic=deterministic)
                del c; gc.collect()
        if final:
            for name in ('hold','toward','away'):
                c = far_collector(base, checkpoint, policy, timing)
                torch.manual_seed(20261018)
                results[name] = panel(c, intervention=name)
                del c; gc.collect()
    return results


def compare_results(results):
    """Predeclared screening only; no statistical/significance claim."""
    baseline = results['A']
    screened = {}
    for arm in ('B','C'):
        by_mode = {}
        for mode in ('sample','mean'):
            a, b = baseline[f'far_{mode}'], results[arm][f'far_{mode}']
            progress = np.asarray(b['dmax_reduction'])
            checks = {'at_least_three_approach':int((progress>0).sum())>=3,
                'one_meter_better_than_A':float(progress.mean()-np.mean(a['dmax_reduction']))>=1.,
                'dispersion_not_increased':float(np.mean(b['dispersion_reduction']))>=0.,
                'remote_hold_not_higher':float(np.mean(b['hold_probability']))<=float(np.mean(a['hold_probability'])),
                'near_parking_not_worse':results[arm][f'near_{mode}']['successes']>=baseline[f'near_{mode}']['successes']}
            by_mode[mode] = {'checks':checks,'passed':all(checks.values())}
        screened[arm] = {'modes':by_mode,'passed':all(row['passed'] for row in by_mode.values())}
    c_further = all(np.mean(results['C'][f'far_{m}']['dmax_reduction'])>
                    np.mean(results['B'][f'far_{m}']['dmax_reduction']) and
                    np.mean(results['C'][f'far_{m}']['hold_probability'])<=np.mean(results['B'][f'far_{m}']['hold_probability']) and
                    results['C'][f'near_{m}']['successes']>=results['B'][f'near_{m}']['successes']
                    for m in ('sample','mean'))
    choice = ('C' if screened['C']['passed'] and (not screened['B']['passed'] or c_further)
              else 'B' if screened['B']['passed'] else None)
    return {'screening':screened,'candidate':choice,'C_further_improvement':bool(c_further),
        'next_step':'retain_candidate_for_followup' if choice else 'legal_local_direction_guidance',
        'full_gathering_acceptance':False,'physx_acceptance':False}
