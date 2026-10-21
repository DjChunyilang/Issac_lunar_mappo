"""Seeded reset curriculum; promotions never interrupt an active episode."""
from copy import deepcopy

import numpy as np

from .upstream_spec import digest


class SpawnCurriculum:
    def __init__(self, spec, config, seed, num_envs):
        self.spec = spec
        self.config = deepcopy(config)
        self.rng = np.random.default_rng(seed)
        self.level = config['initial_level'] if config['version'] == 'rendezvous_curriculum_1km_v2' else 0
        self.episode_levels = np.zeros(num_envs, dtype=np.int64)
        self.deadlines = np.zeros(num_envs)

    @property
    def metadata(self):
        return {'version': self.config['version'], 'configuration': self.config,
                'fingerprint': digest(self.config)}

    def set_level(self, level):
        if type(level) is not int or not 0 <= level < len(self.config['stages']):
            raise ValueError('Invalid curriculum level')
        self.level = level

    def sample(self, ids, *, replay=True):
        result = []
        for env_id in ids:
            level = self.level
            if replay and level and self.rng.random() < self.config['easy_replay_fraction']:
                level = int(self.rng.integers(level))
            stage = self.config['stages'][level]
            angle = self.rng.uniform(-np.pi, np.pi)
            if level == 0:
                radius = self.spec.task.spacing / np.sqrt(2)
                radii = np.full(4, radius)
            else:
                radius = self.rng.uniform(*stage['radius_m'])
                radii = np.full(4, radius)
                # Independent radii break the exact-square prior at large ranges.
                if level >= 3:
                    radii *= self.rng.uniform(.9, 1., 4)
            bearings = angle + np.pi / 4 + np.arange(4) * np.pi / 2
            xy = radii[:, None] * np.stack([np.cos(bearings), np.sin(bearings)], -1)
            margin = max(self.spec.perception.forward, self.spec.perception.reverse,
                         self.spec.perception.lateral) + self.spec.vehicle.effective_radius + 5.
            limit = np.array([self.spec.world_width, self.spec.world_height]) / 2 - radii.max() - margin
            if (limit <= 0).any():
                raise ValueError('Curriculum spawn does not fit the physical world')
            xy += self.rng.uniform(-limit, limit)
            yaw = bearings + np.pi + self.rng.uniform(-stage['heading_noise_deg'], stage['heading_noise_deg'], 4) * np.pi / 180
            poses = np.column_stack([xy, (yaw + np.pi) % (2 * np.pi) - np.pi])
            result.append(poses[self.rng.permutation(4)])
            self.episode_levels[env_id] = level
            self.deadlines[env_id] = stage['deadline_s']
        return np.asarray(result)

    def state_dict(self):
        return deepcopy({'metadata': self.metadata, 'level': self.level, 'rng': self.rng.bit_generator.state,
                         'episode_levels': self.episode_levels, 'deadlines': self.deadlines})

    def load_state_dict(self, state):
        if state['metadata'] != self.metadata:
            raise ValueError('Cross-curriculum resume is forbidden')
        self.set_level(state['level'])
        levels = np.asarray(state['episode_levels'], dtype=np.int64)
        deadlines = np.asarray(state['deadlines'], dtype=float)
        if (levels.shape != self.episode_levels.shape or deadlines.shape != levels.shape
                or (levels < 0).any() or (levels >= len(self.config['stages'])).any()
                or not np.array_equal(deadlines, [self.config['stages'][i]['deadline_s'] for i in levels])):
            raise ValueError('Invalid curriculum episode state')
        self.episode_levels = levels.copy()
        self.deadlines = deadlines.copy()
        self.rng.bit_generator.state = deepcopy(state['rng'])


def update_progress(config, progress, episodes, base_steps):
    """Evaluate only episodes actually born at the current difficulty."""
    p = deepcopy(progress)
    window = config['promotion_window_episodes']
    p['outcomes'] = (p['outcomes'] + [e['terminal_reason'] == 'success' for e in episodes
                                    if e['curriculum_level'] == p['level']])[-window:]
    promoted = (len(p['outcomes']) == window
                and np.mean(p['outcomes']) >= config['promotion_success_fraction']
                and base_steps - p['stage_start_base_steps'] >= config['minimum_stage_base_steps']
                and p['level'] + 1 < len(config['stages']))
    if promoted:
        p['history'].append({'from_level': p['level'], 'base_steps': base_steps,
                             'success_fraction': float(np.mean(p['outcomes']))})
        p.update(level=p['level'] + 1, stage_start_base_steps=base_steps, outcomes=[])
    return p, bool(promoted)
