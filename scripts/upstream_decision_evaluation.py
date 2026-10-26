"""Fixed 20-second panels for the bounded decision-credit experiment.

Counterfactual trajectories never enter the learner. Evaluator RNG, recurrent
state and reward accumulators are independent of the training collector.
"""
from collections import Counter
from copy import deepcopy
import gc
import json
from pathlib import Path

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


def bounded_outcome_metrics(elapsed_steps, ended, episodes, path_length_m, dt):
    """Observed survival times include live cases, whose later outcome is censored.

    Actual travel distance stays fixed after termination, unlike the archived
    applied-control statistic. Dividing by each case's own active time therefore
    excludes frozen velocities from already-ended environments.
    """
    elapsed = np.asarray(elapsed_steps) * dt
    censored = ~np.asarray(ended, bool)
    failures = Counter(e['terminal_reason'] for e in episodes if e['terminal_reason'] != 'success')
    speed = np.divide(np.asarray(path_length_m).mean(-1), elapsed,
                      out=np.zeros_like(elapsed, dtype=float), where=elapsed > 0)
    return {'observed_seconds': elapsed, 'mean_observed_seconds': float(elapsed.mean()),
            'censored': censored, 'censored_count': int(censored.sum()),
            'window_failure_count': sum(failures.values()),
            'window_failure_fraction': sum(failures.values()) / len(elapsed),
            'window_failure_reasons': dict(failures), 'mean_actual_speed': speed,
            'survival_scope': 'observed time to termination or window cut; live cases are censored, not successes'}


@torch.no_grad()
def panel(c, *, deterministic=False, intervention=None, seconds=20, progress=None, capture=None,
          bounded_metrics=False):
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
        if progress is not None:
            progress(elapsed)
        if env._ended.all():
            break
        renew_legal_bounds(env)
        obs, state = env.observe()
        if intervention is None:
            sample = c.policy.sample(obs, hidden, deterministic=deterministic)
            targets, _ = c.bridge.decode(sample.action)
            probability = c.policy.action_distribution(sample.hidden, observation=obs).branch.probs.numpy()
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
            if capture is not None:
                capture(ex)
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
    return {**(bounded_outcome_metrics(elapsed, env._ended, episodes, path_length, env.spec.grid.dt)
               if bounded_metrics else {}),
        'initial':initial,'final':final,'dmax_reduction':initial['dmax']-final['dmax'],
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


class EvaluationPoseCapture:
    """Keep actual world poses only while each independent episode is active."""

    def __init__(self, initial_poses, dt):
        self.dt = float(dt)
        self.histories = [[pose.copy()] for pose in np.asarray(initial_poses)]

    def __call__(self, execution):
        for env_id in np.flatnonzero(execution['active_envs']):
            history = self.histories[env_id]
            if not np.array_equal(history[-1], execution['before'][env_id]):
                raise ValueError('Evaluation capture lost an executed pose or crossed an episode reset')
            history.append(execution['after'][env_id].copy())

    def arrays(self):
        return [np.stack(history) for history in self.histories]


def _compare_evaluation_replay(expected, actual, *, atol=1e-6, rtol=1e-6):
    """Compare recorded fields recursively, retaining useful mismatch evidence."""
    mismatches = []
    compared = 0
    max_absolute_error = 0.

    def compare(reference, replay, path):
        nonlocal compared, max_absolute_error
        if isinstance(reference, dict):
            if not isinstance(replay, dict) or set(reference) != set(replay):
                mismatches.append({'field': path, 'reason': 'keys'})
                return
            for key in reference:
                compare(reference[key], replay[key], f'{path}.{key}')
        elif isinstance(reference, list) and reference and isinstance(reference[0], dict):
            if len(reference) != len(replay):
                mismatches.append({'field': path, 'reason': 'length'})
                return
            for index, (left, right) in enumerate(zip(reference, replay)):
                compare(left, right, f'{path}[{index}]')
        elif isinstance(reference, (int, float, list, np.ndarray)) and not isinstance(reference, bool):
            left, right = np.asarray(reference), np.asarray(replay)
            compared += 1
            if left.shape != right.shape:
                mismatches.append({'field': path, 'reason': 'shape'})
            elif left.dtype.kind in 'biu':
                if not np.array_equal(left, right):
                    mismatches.append({'field': path, 'reason': 'discrete_value'})
            elif left.dtype.kind == 'f':
                error = float(np.max(np.abs(left-right))) if left.size else 0.
                max_absolute_error = max(max_absolute_error, error)
                if not np.allclose(left, right, atol=atol, rtol=rtol):
                    mismatches.append({'field': path, 'reason': 'numeric_value', 'max_abs_error': error})
            elif not np.array_equal(left, right):
                mismatches.append({'field': path, 'reason': 'value'})
        else:
            compared += 1
            if reference != replay:
                mismatches.append({'field': path, 'reason': 'value'})

    compare(expected, actual, 'evaluation')
    return {'matched': not mismatches, 'atol': atol, 'rtol': rtol,
            'compared_fields': compared, 'max_abs_error': max_absolute_error,
            'mismatches': mismatches}


def replay_saved_evaluation(run_dir: Path, phase='final', *, progress=None):
    """Replay the frozen fixed-lunar panels for the existing animation renderer.

    No optimizer or original output is mutated. Returned ``panels`` contain
    world-frame [time, rover, x/y/yaw] arrays including t=0 and the terminal
    state, plus single-environment TerrainRuntime objects for terrain_viz.
    Validation is returned even on a mismatch so the caller can save evidence
    before refusing to label the render as the original evaluation.
    """
    from _common import ROOT
    from planner_evidence import file_sha256
    from upstream_curriculum_experiment import bounded_evaluation, evaluation_panels, is_fixed, timing_for
    from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.action_ablation import ActionContract, AblationPolicy
    from lunar_rover_tasks.tasks.multi_rover_gathering.terrain_features import make_terrain_runtime

    run_dir = Path(run_dir).resolve()
    evaluation_path = run_dir/'metrics'/f'evaluation_{phase}.json'
    original = json.loads(evaluation_path.read_text())
    candidates = []
    for header in sorted((run_dir/'checkpoints').glob('update_*.json')):
        if not header.name.endswith('.learner.json'):
            metadata = json.loads(header.read_text())
            if metadata.get('actual_base_steps') == original['training_base_steps']:
                candidates.append(header.with_suffix('.pt'))
    if len(candidates) != 1:
        raise ValueError('Evaluation replay requires one full checkpoint at the recorded training step')
    checkpoint = candidates[0]
    saved = read_checkpoint(checkpoint, learner_only=True)
    central = saved['trainer']['collector']
    config = deepcopy(saved['identity']['config'])
    if not is_fixed(config):
        raise ValueError('Saved animation replay currently requires the fixed lunar evaluation contract')
    base = Path(saved['identity']['base_config'])
    if file_sha256(base) != saved['identity']['base_config_sha256']:
        raise ValueError('Evaluation base configuration has changed since the checkpoint')
    policy_state = central['policy']
    metadata = policy_state['metadata']
    contract = ActionContract(**{key: metadata[key] for key in
        ('representation', 'profile_fingerprint', 'rho_max', 'speed_cap_max', 'beta_max', 'request_seconds')})
    timing = timing_for(config)
    results, panels, checks, traces = {}, {}, {}, {}
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.random.fork_rng(devices=[]):
            policy = AblationPolicy(metadata['critic_dim'], contract, terrain_channels=metadata['terrain_channels'])
            if 'value_normalization' in metadata:
                policy.enable_value_normalization()
            policy.load_diagnostic_state(policy_state)
            policy.cpu().eval()
            for prefix, level, seconds, complete in evaluation_panels(
                    config, original['curriculum_level'], phase, final=phase == 'final',
                    full=original['full_episode_evaluation']):
                for deterministic in (False, True):
                    name = f'{prefix}_{"mean" if deterministic else "sample"}'
                    eval_config = deepcopy(config)
                    seed, policy_seed = (130363, 130367) if prefix == 'unseen' else (104729, 104759)
                    eval_config['scene_role'] = 'unseen' if prefix == 'unseen' else 'fixed_eval'
                    # Match evaluate() exactly, including the second spawn draw
                    # after make_collector's construction-time spawn.
                    c = make_collector(base, 'A1', seed, samples=512, num_envs=4,
                        episode_seconds=config['stages'][level]['deadline_s'],
                        policy=deepcopy(policy), learning_time=timing, curriculum_config=eval_config)
                    c.env.curriculum.set_level(level)
                    poses = c.env.curriculum.sample(np.arange(4), replay=False)
                    c.env.task.deadlines[:] = c.env.curriculum.deadlines
                    c.env.reset(np.arange(4), poses)
                    capture = EvaluationPoseCapture(c.env.core.poses, c.env.spec.grid.dt)
                    torch.manual_seed(policy_seed)
                    row = panel(c, deterministic=deterministic, seconds=seconds, capture=capture,
                        progress=None if progress is None else lambda elapsed, name=name: progress(name, elapsed.copy()),
                        **({'bounded_metrics': True} if bounded_evaluation(config) else {}))
                    row['full_episode_evaluation'] = complete
                    row['observation_seconds'] = seconds
                    trace = row.pop('trace')
                    trace_path = run_dir/'metrics'/f'trace_{phase}_{name}.json'
                    trace_reference = json.loads(trace_path.read_text())
                    histories = capture.arrays()
                    ordered_episodes = sorted(row['episodes'], key=lambda episode: episode['env_id'])
                    capture_steps = [len(history)-1 for history in histories]
                    check = _compare_evaluation_replay(
                        {'result': original['results'][name], 'trace': trace_reference},
                        {'result': row, 'trace': trace})
                    if bounded_evaluation(config):
                        capture_matches = (np.array_equal(capture_steps, np.rint(np.asarray(row['observed_seconds']) / c.env.spec.grid.dt).astype(int))
                            and all(capture_steps[e['env_id']] == e['steps'] for e in ordered_episodes)
                            and sum(capture_steps) == row['actual_base_steps'])
                    else:
                        capture_matches = (len(ordered_episodes) == 4 and
                            capture_steps == [episode['steps'] for episode in ordered_episodes] and
                            sum(capture_steps) == row['actual_base_steps'])
                    check.update(capture_matches=capture_matches, capture_steps=capture_steps,
                                 matched=check['matched'] and capture_matches)
                    runtime = c.env._terrain_runtime
                    single_runtimes, scene_manifests = [], []
                    for env_id in range(c.env.e):
                        single = make_terrain_runtime(1, device='cpu', dtype=torch.float64)
                        single.scene_bank = runtime.scene_bank
                        single.scene_indices = runtime.scene_indices[env_id:env_id+1].clone()
                        single_runtimes.append(single)
                        scene_manifests.append(str(runtime.scene_bank.scenes[int(single.scene_indices[0])].path))
                    panels[name] = {'dt': capture.dt, 'histories': histories,
                        'times': [np.arange(len(history))*capture.dt for history in histories],
                        'episodes': ordered_episodes, 'terrain_cfg': deepcopy(c.env._terrain_cfg),
                        'terrain_runtimes': single_runtimes, 'scene_manifests': scene_manifests,
                        'vehicle': {'length': c.env.spec.vehicle.length, 'width': c.env.spec.vehicle.width},
                        'seed': seed, 'policy_seed': policy_seed, 'coordinate_frame': 'world_xy_m_yaw_rad'}
                    results[name], traces[name], checks[name] = row, trace, check
                    del c
                    gc.collect()
    finally:
        torch.set_num_threads(old_threads)
    actual = sum(row['actual_base_steps'] for row in results.values())
    replay = {**original, 'results': results, 'actual_base_steps': actual, 'optimizer_updates': 0}
    sources = [checkpoint, evaluation_path, base,
               *(run_dir/'metrics'/f'trace_{phase}_{name}.json' for name in results)]
    scenes = sorted({scene for panel_data in panels.values() for scene in panel_data['scene_manifests']})
    return {'evaluation': replay, 'traces': traces, 'panels': panels,
        'validation': {'matched': all(check['matched'] for check in checks.values()) and
                       actual == original['actual_base_steps'], 'panels': checks,
                       'original_actual_base_steps': original['actual_base_steps'],
                       'replay_actual_base_steps': actual, 'atol': 1e-6, 'rtol': 1e-6},
        'provenance': {'phase': phase, 'run_dir': str(run_dir), 'checkpoint': str(checkpoint),
            'evaluation': str(evaluation_path), 'training_base_steps': central['base_steps'],
            'additional_evaluation_base_steps': actual, 'optimizer_updates': 0,
            'sources': {str(path): file_sha256(path) for path in sources},
            'scenes': {scene: file_sha256(scene) for scene in scenes},
            'replay_source_sha256': file_sha256(ROOT/'scripts/upstream_decision_evaluation.py')}}


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
