"""Bounded prelaunch audit of the isolated 1km curriculum and reward scale."""
import argparse
from pathlib import Path

import numpy as np
import torch
from _common import cfg_from_experiment
from diagnose_upstream_environment import event_data
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_curriculum import (
    SpawnCurriculum,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_reward import (
    UpstreamReward,
    UpstreamRewardSettings, configured_reward_settings,
)
from planner_evidence import json_write, source_snapshot
from run_upstream_t11 import make_collector
from upstream_curriculum_experiment import load_curriculum, timing_for, is_fixed, initialize_fresh_policy, evaluation_budget_upper_bound
from upstream_decision_evaluation import panel
from training_tensorboard import TrainingTensorBoard


def reward_audit(spec, config, *, obstacle_input=None):
    g, l = timing_for(config).discounts(.2)
    reward = UpstreamReward(spec, 1, configured_reward_settings(g,l,config), obstacle_input=obstacle_input)
    rows = []
    for radius in (7., 12., 30., 75., 150., 300.):
        def facts(r):
            return {'dmax': np.array([2*r]), 'dispersion': np.array([r*r]),
                    'flatness_cost': np.zeros(1), 'hold_count': np.zeros(1)}
        before = facts(radius)
        phi = reward.potentials(before)['gather'][0]
        returns = {}
        for name, r in (('toward', radius-.1), ('hold', radius), ('away', radius+.1)):
            after = facts(r)
            d, v = reward.gather_excess(after)
            returns[name] = float(g*reward.potentials(after)['gather'][0]-phi
                                 -reward.settings.distance_level*d[0]-reward.settings.dispersion_level*v[0])
            if reward.settings.continuous_cost_scale != 1.:
                returns[name] = float(g*reward.potentials(after)['gather'][0]-phi
                    -reward.settings.continuous_cost_scale*(reward.settings.distance_level*d[0]
                                                           +reward.settings.dispersion_level*v[0]))
        assert returns['toward'] > returns['hold'] > returns['away']
        if reward.settings.continuous_cost_scale == 1.:
            assert returns['hold'] < 0
        # Discounted potential terms telescope, including a true terminal.
        states = [facts(radius-k*.1) for k in range(26)]
        potentials = [sum(reward.potentials(s).values())[0] for s in states]
        r = [g*potentials[k+1]-potentials[k] for k in range(25)]
        assert np.isclose(sum(g**k*r[k] for k in range(25)), g**25*potentials[-1]-potentials[0])
        r[-1] = -potentials[-2]
        assert np.isclose(sum(g**k*r[k] for k in range(25)), -potentials[0])
        rows.append({'radius_m': radius, 'potential': phi, 'one_step_gather_rewards': returns})
    return {'settings': reward.metadata, 'radial_equal_cost_reward_order': rows,
            'terminal_and_nonterminal_telescoping_passed': True,
            'scope': 'algebraic reward direction at equal execution costs; not a reachable-trajectory claim'}


def geometry_audit(spec, config):
    sampler = SpawnCurriculum(spec, config, 23, 128)
    rows = []
    for level in range(len(config['stages'])):
        sampler.set_level(level)
        poses = sampler.sample(range(128), replay=False)
        xy = poses[..., :2]
        pairs = np.linalg.norm(xy[:, :, None]-xy[:, None, :], axis=-1)
        pairs[:, np.arange(4), np.arange(4)] = np.inf
        minimum = float(pairs.min())
        assert minimum >= 2*spec.vehicle.effective_radius
        assert np.max(abs(xy[..., 0]))+spec.vehicle.effective_radius < spec.world_width/2
        assert np.max(abs(xy[..., 1]))+spec.vehicle.effective_radius < spec.world_height/2
        pairs[:, np.arange(4), np.arange(4)] = 0.
        assert pairs.max() <= spec.communication.sparse_radius
        rows.append({'level': level, 'sampled_teams': 128, 'minimum_spacing_m': minimum,
                     'maximum_pair_distance_m': float(pairs.max()), 'deadlines_s': np.unique(sampler.deadlines).tolist()})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--curriculum-config', '--training-config', dest='curriculum_config', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    path = Path(args.output)
    config = load_curriculum(args.curriculum_config)
    runtime_cfg = cfg_from_experiment(args.config)
    spec = runtime_cfg.spec
    json_write(path/'run_manifest.json', {'command': __import__('sys').argv, 'source_snapshot': source_snapshot(),
        'scope': 'prelaunch_fixed_lunar_runnable_audit' if is_fixed(config) else 'prelaunch_reward_geometry_parking_audit', 'maximum_physical_base_steps': 400 if is_fixed(config) else 1800, 'optimizer_updates': 0})
    json_write(path/'metrics/static_audit.json', {'curriculum_config': config,
                                                'geometry': geometry_audit(spec, config),
                                                'reward': reward_audit(spec, config, obstacle_input=runtime_cfg.obstacle_input)})
    if is_fixed(config):
        # No optimizer and no success threshold: this checks runnable, legal
        # starts and finite real transitions, not whether an untrained policy wins.
        results = {}; steps = 0
        tensorboard = TrainingTensorBoard(path)
        for role in ('training', 'fixed_eval', 'unseen'):
            seeds = (23,27) if role == 'training' else ((104729,) if role == 'fixed_eval' else (130363,))
            for seed in seeds:
                setup = dict(config, scene_role=role)
                c = make_collector(args.config, 'A1', seed, samples=512, num_envs=1, episode_seconds=1500.,
                    curriculum_config=setup, learning_time=timing_for(config))
                initialize_fresh_policy(c.policy, config)
                assert all(r.passed for row in c.env._reports(parking=False) for r in row)
                assert np.all(c.env.task.deadlines == 1500.)
                assert c.env.metadata['terrain_type'] == 'lunar_scene_pack'
                actor, critic = c.env.observe()
                assert torch.isfinite(actor).all() and torch.isfinite(critic).all()
                batch, episodes, records = c.collect(1)
                if not batch['learning_valid'][batch['duration'] > 0].all():
                    json_write(path/'metrics'/f'invalid_{role}_{seed}.json', event_data({
                        'actual_base_steps':c.base_steps,'episodes':episodes,'records':records,
                        'reports':[[r.__dict__ for r in row] for row in c.env._reports(parking=False)]}))
                    torch.save(c.env.state_dict(), path/f'invalid_{role}_{seed}.pt')
                assert batch['learning_valid'][batch['duration'] > 0].all()
                assert all(torch.isfinite(batch[k]).all() for k in ('obs','state','reward','value','next_value'))
                steps += c.base_steps
                tensorboard.writer.add_scalar('Audit/Base_steps', steps, steps)
                tensorboard.writer.add_scalar('Audit/Reward', float(batch['reward'].mean()), steps)
                tensorboard.writer.flush()
                results[f'{role}_{seed}'] = {'actual_base_steps':c.base_steps,
                    'duration':batch['duration'].tolist(), 'episodes':episodes,
                    'spawn_attempts':c.env.curriculum.spawn_attempts,'spawn_rejections':c.env.curriculum.spawn_rejections,
                    'map_identities':c.env.map_identities,'geometry':geometry_audit(spec,config),
                    'reward':batch['reward'].tolist()}
                json_write(path/'metrics'/f'{role}_{seed}.json',event_data(results[f'{role}_{seed}']))
                del batch, records, c
        assert steps <= 400
        tensorboard.close()
        expected_evaluation = (config['evaluation_max_base_steps']
                              if config.get('evaluation_mode') == 'bounded_fixed_window_v1' else 360000)
        assert evaluation_budget_upper_bound(config,2000000) == expected_evaluation
        json_write(path/'metrics/summary.json',event_data({'passed':True,'actual_base_steps':steps,
            'optimizer_updates':0,'scope':'fixed_lunar_finite_legal_transition_audit','maximum_physical_base_steps':400,
            'results':results,'source_snapshot':source_snapshot()}))
        return
    results = {}
    for level, interventions in ((0, ('hold',)), (3, ('hold', 'toward', 'away'))):
        for intervention in interventions:
            c = make_collector(args.config, 'A1', 23, samples=512, num_envs=4, episode_seconds=1200.,
                               curriculum_config=config, learning_time=timing_for(config))
            c.env.curriculum.set_level(level)
            poses = c.env.curriculum.sample(range(4), replay=False)
            c.env.task.deadlines[:] = c.env.curriculum.deadlines
            c.env.reset(np.arange(4), poses)
            result = panel(c, intervention=intervention, seconds=30 if level == 0 else 20)
            results[f'level_{level}_{intervention}'] = result
            json_write(path/'metrics'/f'level_{level}_{intervention}.json', event_data(result))
            if level == 0:
                assert result['successes'] == 4, 'Near parking must be physically achievable before learning'
            del c
    steps = sum(r['actual_base_steps'] for r in results.values())
    assert steps <= 1800
    comparison = {name: {'discounted_reward': row['discounted_reward'], 'dmax_reduction': row['dmax_reduction'],
                        'successes': row['successes']} for name, row in results.items()}
    json_write(path/'metrics/summary.json', event_data({'passed': True, 'actual_base_steps': steps,
        'optimizer_updates': 0, 'comparisons': comparison, 'source_snapshot': source_snapshot()}))


if __name__ == '__main__':
    main()
