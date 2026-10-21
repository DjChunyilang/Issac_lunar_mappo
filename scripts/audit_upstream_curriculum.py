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
    UpstreamRewardSettings,
)
from planner_evidence import json_write, source_snapshot
from run_upstream_t11 import make_collector
from upstream_curriculum_experiment import load_curriculum, timing_for
from upstream_decision_evaluation import panel


def reward_audit(spec, config):
    g, l = timing_for(config).discounts(.2)
    reward = UpstreamReward(spec, 1, UpstreamRewardSettings(gamma=g, lam=l, gather_scaling=config['reward_scaling']))
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
        assert returns['toward'] > returns['hold'] > returns['away']
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
    parser.add_argument('--curriculum-config', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    path = Path(args.output)
    config = load_curriculum(args.curriculum_config)
    spec = cfg_from_experiment(args.config).spec
    json_write(path/'run_manifest.json', {'command': __import__('sys').argv, 'source_snapshot': source_snapshot(),
        'scope': 'prelaunch_reward_geometry_parking_audit', 'maximum_physical_base_steps': 1800, 'optimizer_updates': 0})
    json_write(path/'metrics/static_audit.json', {'curriculum_config': config,
                                                'geometry': geometry_audit(spec, config),
                                                'reward': reward_audit(spec, config)})
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
