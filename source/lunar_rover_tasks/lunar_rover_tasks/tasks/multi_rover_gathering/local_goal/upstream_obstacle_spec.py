"""One immutable configuration for incremental perception, costs and rewards."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

OBSERVATION_VERSION = 'obstacle_ground_known_actor_v1'


@dataclass(frozen=True)
class ObstacleInputSpec:
    configuration_json: str
    root: str

    @classmethod
    def load(cls, path, root, vehicle_spec):
        data = yaml.safe_load(Path(path).read_text())
        graded = data.get('version') == 'local_obstacle_input_v2'
        expected = {
            'version': 'local_obstacle_input_v2' if graded else 'local_obstacle_input_v1',
            'labels': {'crater_region': 'mouth_disk', 'rock_region': 'projected_disk', 'rasterization': 'any_intersection'},
            'ground': {'aggregation': 'conservative_native_max_v1'},
            'perception': {'reveal_rule': 'full_cell_in_fov'},
            'memory': {'eviction': 'oldest_unseen_tile'},
            'observation': {'version': OBSERVATION_VERSION, 'aggregation': 'region_conservative'},
            'geometry': {'distance': 'oriented_rectangle_cover_disks_v1'},
            'cost': {'hard_obstacle_rejection': False},
            'reward': {'team_reduce': 'mean', 'observation_time': 'pre_step'},
        }
        numeric = {'labels': ['resolution_m'], 'ground': ['resolution_m', 'slope_score_scale', 'rms_score_scale_m'],
                   'perception': ['forward_m', 'reverse_m', 'lateral_m', 'update_period_s'],
                   'memory': ['tile_cells', 'capacity_cells'], 'observation': [], 'geometry': ['cover_spacing_m'],
                   'cost': ['soft_band_m', 'near_weight_per_s', 'overlap_weight_per_s'],
                   'reward': ['near_weight_per_s', 'overlap_weight_per_s']}
        extra = {'crater_classes'} if graded else set()
        if set(data) != set(expected) | {'atlas_root'} | extra or data['version'] != expected['version']:
            raise ValueError('Unsupported obstacle input schema')
        if graded:
            classes = data['crater_classes']
            semantics = {'semantics': 'wheel_radius_proxy_no_dynamics_v1',
                         'driving_surface': 'base_rocks_fractal_without_craters',
                         'parking_surface': 'original_terrain'}
            positive = {'medium_risk', 'medium_cost_per_s', 'medium_reward_per_s', 'medium_reward_cap'}
            if set(classes) != set(semantics) | positive | {'thresholds'}:
                raise ValueError('Unsupported crater class fields')
            if any(classes[k] != v for k, v in semantics.items()):
                raise ValueError('Unsupported crater class semantics')
            thresholds = classes['thresholds']
            if set(thresholds) != {'small_below_m', 'large_from_m'}:
                raise ValueError('Unsupported crater class thresholds')
            for key, value in list(thresholds.items()) + [(k, classes[k]) for k in positive]:
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
                    raise ValueError('Expected positive finite crater setting: ' + key)
            if thresholds['small_below_m'] >= thresholds['large_from_m'] or classes['medium_risk'] >= 1:
                raise ValueError('Crater thresholds must increase and medium risk must be below one')
        for section, keys in numeric.items():
            if set(data[section]) != set(keys) | set(expected[section]):
                raise ValueError('Unsupported obstacle input fields: ' + section)
            for key, value in expected[section].items():
                if data[section][key] != value:
                    raise ValueError('Unsupported obstacle input semantics: ' + section + '.' + key)
            for key in keys:
                value = data[section][key]
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
                    raise ValueError('Expected positive finite obstacle setting: ' + section + '.' + key)
        m, p, t = data['memory'], data['perception'], vehicle_spec.perception
        if any(type(m[k]) is not int for k in ('tile_cells', 'capacity_cells')) or m['capacity_cells'] % m['tile_cells']**2:
            raise ValueError('Grid capacity must contain whole tiles')
        if (p['forward_m'], p['reverse_m'], p['lateral_m'], p['update_period_s']) != (t.forward, t.reverse, t.lateral, vehicle_spec.grid.dt):
            raise ValueError('Obstacle input must preserve current FOV and control period')
        ratio = data['ground']['resolution_m'] / data['labels']['resolution_m']
        if ratio < 1 or not np.isclose(ratio, round(ratio)):
            raise ValueError('Ground cells must align with obstacle cells')
        halo=data['cost']['soft_band_m']+np.sqrt(2)*data['geometry']['cover_spacing_m']
        if halo>m['tile_cells']*data['labels']['resolution_m']:
            raise ValueError('Obstacle soft band exceeds the adjacent-tile halo')
        return cls(json.dumps(data, sort_keys=True, separators=(',', ':')), str(Path(root).resolve()))

    @property
    def data(self):
        return json.loads(self.configuration_json)

    @property
    def graded_craters(self):
        return self.data['version'] == 'local_obstacle_input_v2'

    @property
    def fingerprint(self):
        return hashlib.sha256(self.configuration_json.encode()).hexdigest()

    @property
    def atlas_root(self):
        return Path(self.root) / self.data['atlas_root']

    @property
    def metadata(self):
        return {'fingerprint': self.fingerprint, 'configuration': self.data}
