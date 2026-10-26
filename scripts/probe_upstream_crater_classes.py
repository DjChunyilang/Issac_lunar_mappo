#!/usr/bin/env python3
"""Bounded crater-class execution and MPPI checks; no policy or optimizer.

Single-feature fixtures reuse the production crater profile, terrain metrics,
ScenePack, sensing and actual execution loop. Scripted straight commands check
the environment contract; separate MOVE requests exercise the real planner.
"""
import argparse
from collections import Counter
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import shlex
import sys
import time
import traceback

import numpy as np
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

from _common import ROOT
from upstream_local_closure import runtime_config
from lunar_rover_tasks.terrain_data.batched import SceneBank
from lunar_rover_tasks.terrain_data.enhancement import crater_profile
from lunar_rover_tasks.terrain_data.metrics import local_metrics
from lunar_rover_tasks.terrain_data.obstacle_atlas import build_obstacle_atlas, ObstacleAtlas
from lunar_rover_tasks.terrain_data.scene_pack import ScenePack, SCHEMA, file_hash
from lunar_rover_tasks.tasks.multi_rover_gathering.gathering_env_cfg import TerrainCfg
from lunar_rover_tasks.tasks.multi_rover_gathering.terrain_features import make_terrain_runtime
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_environment import UpstreamEnvironment
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_reward import configured_reward_settings
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_planning import UpstreamIntent


def serial(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=serial, allow_nan=False) + '\n')


def make_scene(path, kind, diameter, extent=64.):
    """Flat base plus one production-shape synthetic feature, in metres."""
    path.mkdir(parents=True, exist_ok=False)
    spacing = .1
    axis = (np.arange(round(extent / spacing)) + .5) * spacing - extent / 2
    xx, yy = np.meshgrid(axis, axis[::-1])
    base = np.zeros(xx.shape, np.float32)
    craters = base.copy(); rocks = base.copy()
    catalog = {'craters': [], 'rocks': []}
    if kind != 'flat':
        radius = np.hypot(xx, yy) / (diameter / 2)
        item = {'id': 0, 'accepted': True, 'affects_target': True, 'centre_in_target': True,
                'local_xy_m': [0., 0.], 'x_m': extent / 2, 'y_m': extent / 2,
                'diameter_m': diameter}
        if kind == 'crater':
            craters = crater_profile(radius, depth_m=.08 * diameter,
                                      rim_height_m=.025 * diameter).astype(np.float32)
            item.update(depth_m=.08 * diameter, rim_height_m=.025 * diameter,
                        profile='c1_polynomial_hypothesis_v1')
            catalog['craters'].append(item)
        elif kind == 'rock':
            rocks = (.3 * diameter * np.sqrt(np.maximum(0., 1. - radius ** 2))).astype(np.float32)
            item.update(height_m=.3 * diameter, profile='half_ellipsoid')
            catalog['rocks'].append(item)
        else:
            raise ValueError(kind)
    height = base + craters + rocks
    valid = np.ones(height.shape, bool)
    metrics = local_metrics(height, valid, spacing, spacing, 4)
    arrays = {'features': np.stack([height, metrics['slope_x'], metrics['slope_y'], metrics['roughness']]).astype(np.float32),
              'base_height': base, 'layer_craters': craters, 'layer_rocks': rocks,
              'height_valid': valid, 'metrics_valid': metrics['valid']}
    files = {}
    for name, array in arrays.items():
        target = path / (name + '.npy'); np.save(target, array, allow_pickle=False)
        files[name] = {'file': target.name, 'sha256': file_hash(target),
                       'shape': list(array.shape), 'dtype': str(array.dtype)}
    meta = {'schema': SCHEMA, 'source_id': 'synthetic_single_feature_validation',
            'geographic_group': 'synthetic_validation', 'role': 'diagnostic',
            'classification': 'hypothesis', 'extent_xy_m': [extent, extent],
            'shape': list(height.shape), 'processing_spacing_xy_m': [spacing, spacing],
            'metric_baseline_xy_m': [.8, .8], 'native_spacing_xy_m': [spacing, spacing],
            'transform': [spacing, 0., -extent / 2, 0., -spacing, extent / 2],
            'origin_projected_xy_m': [0., 0.], 'files': files, 'feature_catalog': catalog,
            'enhancement': {'method': 'nasa_sfd_v2', 'craters': {'depth_to_diameter': .08, 'rim_to_diameter': .025}},
            'provenance': 'Production crater_profile; flat background; no learned behavior or real lunar observation.'}
    write(path / 'scene.json', meta)
    return ScenePack(path / 'scene.json')


def configuration(base, arm, run, samples):
    cfg, effective = runtime_config(base, name='crater_classes_' + arm, seed=23,
                                   device='cpu', samples=samples, num_envs=1, episode_seconds=120.)
    data = cfg.obstacle_input.data
    data['atlas_root'] = str((run / 'config' / 'atlases' / arm).relative_to(ROOT))
    settings = replace(cfg.obstacle_input, configuration_json=json.dumps(data, sort_keys=True, separators=(',', ':')))
    cfg = replace(cfg, obstacle_input=settings)
    write(run / 'config' / (arm + '_effective.json'), {'experiment': effective, 'obstacle_input': settings.metadata,
        'reward': reward_settings(cfg).metadata, 'overrides': {'terrain_failure_terminal': True,
        'purpose': 'Match trained branch terminal/reward semantics; no optimizer'}})
    return cfg


def reward_settings(cfg):
    data = yaml.safe_load((ROOT / 'configs/training/upstream_1km_far_obstacle_input_v1.yaml').read_text())
    power = cfg.spec.grid.dt / data['discount_reference_period_s']
    return configured_reward_settings(data['gamma'] ** power, data['lambda'] ** power, data)


def make_environment(cfg, scene, poses=None):
    atlas_path = cfg.obstacle_input.atlas_root / file_hash(scene.path) / 'obstacle.json'
    if not atlas_path.exists():
        build_obstacle_atlas(scene, atlas_path.parent, cfg.obstacle_input, cfg.spec)
    runtime = make_terrain_runtime(1, device='cpu', dtype=torch.float64)
    runtime.scene_bank = SceneBank([scene.path], storage='mmap')
    runtime.scene_indices = torch.zeros(1, dtype=torch.long)
    runtime.scene_bank.prefetch(runtime.scene_indices)
    if poses is None:
        poses = np.array([[[-6., 0., 0.], [-20., -20., 0.], [20., -20., 0.], [20., 20., 0.]]])
    terrain_cfg = TerrainCfg(type='lunar_scene_pack', raster_risk_slope_scale=.6, raster_risk_rms_scale_m=.05)
    env = UpstreamEnvironment(cfg, poses, terrain_cfg=terrain_cfg, terrain_runtime=runtime,
                              map_identities=(file_hash(scene.path),),
                              reward_settings=reward_settings(cfg))
    # The archived v1 enables this through its curriculum; this small fixture
    # preserves the same terminal contract without importing the far-spawn sampler.
    env.task.terrain_failure_terminal = True
    env.cpu_workers = 0
    return env


def run_case(cfg, scene, name, mode, run, ledger, writer, max_steps=120):
    env = make_environment(cfg, scene)
    if mode == 'scripted_straight':
        env._nominal = lambda e, a: (np.array([.6, 0.]) if a == 0 else np.zeros(2),
                                    'diagnostic_scripted_straight_not_learned')
    totals = Counter(); trace = []; phases = Counter(); elapsed = 0
    initial = env.core.poses.copy(); started = time.perf_counter()
    while elapsed < max_steps and not env._ended[0] and env.core.poses[0, 0, 0] < 6.:
        if mode == 'mppi_move':
            now = float(env.core.times[0]); requests = []
            for a in range(4):
                pose = env.local_pose(0, a)
                target = pose[:2].copy()
                if a == 0:
                    delta = np.array([12., 0.]) - pose[:2]
                    target += delta * min(1., cfg.spec.rho_max / max(np.linalg.norm(delta), 1e-12))
                requests.append(UpstreamIntent(target, a != 0, now, now + cfg.spec.grid.request_validity))
            if (0, 0) not in env.pending:
                env.enqueue([requests])
        slices = min(5 if mode == 'mppi_move' else 1, max_steps - elapsed)
        for _ in range(slices):
            if env._ended[0]:
                break
            if ledger['actual_base_steps'] >= ledger['base_step_budget']:
                raise RuntimeError('Bounded validation exhausted its explicit step budget')
            _, terms, record = env.tick()
            elapsed += 1; ledger['actual_base_steps'] += 1
            for key, value in terms.items():
                totals[key] += float(value[0])
            report = env._reports(parking=False)[0][0]
            parking = env._reports(parking=True)[0][0]
            phases[report.reason or 'passed'] += 1
            row = {'case_step': elapsed, 'total_base_steps': ledger['actual_base_steps'],
                   'time_s': float(env.core.times[0]), 'pose': env.core.poses[0, 0].copy(),
                   'velocity': env.core.actual_velocity[0, 0].copy(),
                   'driving_reason': report.reason, 'parking_reason': parking.reason,
                   'terminal_reason': str(env.last_task['terminal_reason'][0]),
                   'terrain_failure_reasons': record.get('terrain_failure_reasons'),
                   'reward_components': {k: float(v[0]) for k, v in terms.items()}}
            trace.append(row)
            writer.add_scalar(name + '/ego_x_m', row['pose'][0], ledger['actual_base_steps'])
            writer.add_scalar(name + '/crater_medium', float(terms.get('crater_medium', np.zeros(1))[0]), ledger['actual_base_steps'])
            if elapsed % 25 == 0:
                writer.flush(); write(run / 'metrics' / 'progress.json', ledger)
        if mode == 'mppi_move':
            for plan in env.plans[0]:
                if plan is not None:
                    phases['planner_valid' if plan.valid else 'planner_invalid:' + plan.fallback_reason] += 1
        env.discard_delivered_history()
    result = {'name': name, 'mode': mode, 'actual_base_steps': elapsed, 'optimizer_updates': 0,
              'passed_across_feature': bool(env.core.poses[0, 0, 0] >= 6.), 'ended': bool(env._ended[0]),
              'terminal_reason': str(env.last_task['terminal_reason'][0]) if env.last_task is not None else 'none',
              'failure_reasons': trace[-1]['terrain_failure_reasons'] if trace else None,
              'initial_poses': initial, 'final_poses': env.core.poses.copy(), 'reward_components': dict(totals),
              'driving_and_planner_counts': dict(phases), 'wall_seconds': time.perf_counter() - started,
              'configuration': cfg.metadata, 'terrain_manifest': str(scene.path)}
    write(run / 'metrics' / (name + '.json'), result)
    with (run / 'metrics' / (name + '_trace.jsonl')).open('x') as stream:
        for row in trace:
            stream.write(json.dumps(row, default=serial, allow_nan=False) + '\n')
    writer.flush()
    print(json.dumps({k: result[k] for k in ['name', 'actual_base_steps', 'passed_across_feature', 'terminal_reason', 'reward_components']}), flush=True)
    return result


def static_checks(cfgs, scenes, run):
    """Boundary and parking witnesses consume no environment steps."""
    cfg = cfgs['graded']; rows = []
    for label, diameter, expected_bit in [('below_small', .4999, 1), ('at_small', .5, 4),
                                          ('below_large', 1.7999, 4), ('at_large', 1.8, 8)]:
        scene = make_scene(run / 'config' / 'scenes' / label, 'crater', diameter, extent=16.)
        path = build_obstacle_atlas(scene, cfg.obstacle_input.atlas_root / file_hash(scene.path), cfg.obstacle_input, cfg.spec)
        atlas = ObstacleAtlas(path, cfg.obstacle_input, scene)
        cell = np.floor(atlas.extent[None] / (2 * atlas.res)).astype(int)
        _, kind, known = atlas.read(cell)
        rows.append({'diameter_m': diameter, 'observed_bit': int(kind[0]), 'expected_bit': expected_bit,
                     'passed': bool(known[0] and kind[0] == expected_bit)})
    scene = scenes['large']
    path = cfg.obstacle_input.atlas_root / file_hash(scene.path) / 'obstacle.json'
    if not path.exists():
        build_obstacle_atlas(scene, path.parent, cfg.obstacle_input, cfg.spec)
    atlas = ObstacleAtlas(path, cfg.obstacle_input, scene)
    length, width, radius = cfg.spec.vehicle.length, cfg.spec.vehicle.width, .95
    geometry = []
    for yaw in [0., .73]:
        poses = []
        for gap in [.1, 1e-4, 0., -1e-4]:
            reach = length / 2 + radius + gap
            poses.append([-reach * np.cos(yaw), -reach * np.sin(yaw), yaw])
        hits = atlas.exact_large_crater_overlap(np.asarray(poses), length, width)
        geometry.append({'yaw': yaw, 'gaps_m': [.1, 1e-4, 0., -1e-4], 'overlap': hits,
                         'expected': [False, False, False, True],
                         'passed': bool(np.array_equal(hits, [False, False, False, True]))})
    poses = np.array([[[0., 0., 0.], [-20., -20., 0.], [20., -20., 0.], [20., 20., 0.]]])
    parking = {}
    for arm, local in cfgs.items():
        env = make_environment(local, scenes['medium'], poses=poses)
        parking[arm] = {'parking': asdict(env._reports(parking=True)[0][0]),
                        'driving': asdict(env._reports(parking=False)[0][0])}
    checks = {'diameter_boundaries': all(row['passed'] for row in rows),
              'exact_body_positive_area_overlap': all(row['passed'] for row in geometry),
              'parking_report_unchanged': parking['legacy']['parking'] == parking['graded']['parking'],
              'medium_parking_still_rejected': not parking['graded']['parking']['passed'],
              'medium_driving_allowed': parking['graded']['driving']['passed']}
    result = {'checks': checks, 'diameter_classes': rows, 'body_geometry': geometry, 'parking': parking,
              'actual_base_steps': 0, 'optimizer_updates': 0}
    write(run / 'metrics' / 'static_contracts.json', result)
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', default='crater_classes_validation_20261023_01')
    parser.add_argument('--budget', type=int, default=2000)
    parser.add_argument('--samples', type=int, default=64)
    args = parser.parse_args()
    if not 1 <= args.budget <= 2000:
        raise ValueError('Diagnostic budget must be 1..2000 actual base steps')
    torch.set_num_threads(1); torch.manual_seed(23); np.random.seed(23)
    run = ROOT / 'outputs/runs/local_obstacle_input' / args.run_id
    run.mkdir(parents=True, exist_ok=True)
    if (run / 'metrics' / 'crater_classes_summary.json').exists():
        raise FileExistsError('Do not overwrite completed validation evidence')
    for name in ['config', 'metrics', 'logs', 'tensorboard']:
        (run / name).mkdir(exist_ok=True)
    command = shlex.join([sys.executable, *sys.argv])
    (run / 'config' / 'validation_command.txt').write_text(command + '\n')
    ledger = {'schema_version': 1, 'experiment': 'local_obstacle_input', 'run_id': args.run_id,
              'role': 'bounded_crater_class_validation', 'status': 'running', 'command': command,
              'base_step_budget': args.budget, 'actual_base_steps': 0, 'learning_base_steps': 0,
              'optimizer_updates': 0, 'axis': 'actual_base_environment_steps',
              'tensorboard': {'directory': str(run / 'tensorboard'), 'purpose': 'diagnostics_no_learning'},
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'scope': 'Synthetic Proxy contract and short real-MPPI loop, not learned navigation or PhysX'}
    write(run / 'run_manifest.json', ledger)
    writer = SummaryWriter(str(run / 'tensorboard'), flush_secs=1)
    writer.add_text('run/configuration', json.dumps(ledger, ensure_ascii=False), 0); writer.flush()
    results = {}
    try:
        cfgs = {'legacy': configuration('configs/experiment/upstream_1km_obstacle_input_v1.yaml', 'legacy', run, args.samples),
                'graded': configuration('configs/experiment/upstream_1km_crater_classes_v1.yaml', 'graded', run, args.samples)}
        specs = {'flat': ('flat', 0.), 'small': ('crater', .4), 'medium': ('crater', 1.2),
                 'large': ('crater', 1.9), 'rock': ('rock', 1.2)}
        scenes = {name: make_scene(run / 'config' / 'scenes' / name, *params) for name, params in specs.items()}
        contracts = static_checks(cfgs, scenes, run)
        for arm, cfg in cfgs.items():
            for feature in ['flat', 'small', 'medium', 'large', 'rock']:
                key = arm + '_' + feature
                results[key] = run_case(cfg, scenes[feature], key, 'scripted_straight', run, ledger, writer)
        for feature in ['medium', 'large']:
            key = 'graded_mppi_' + feature
            results[key] = run_case(cfgs['graded'], scenes[feature], key, 'mppi_move', run, ledger, writer, max_steps=150)
        checks = {**contracts,
            'small_crosses': results['graded_small']['passed_across_feature'],
            'medium_crosses': results['graded_medium']['passed_across_feature'],
            'large_fails': results['graded_large']['terminal_reason'] == 'terrain_failure',
            'medium_light_penalty': -2. <= results['graded_medium']['reward_components'].get('crater_medium', 0.) < 0.,
            'small_no_crater_penalty': results['graded_small']['reward_components'].get('crater_medium', 0.) == 0.,
            'small_matches_flat_terrain_penalty': all(np.isclose(results['graded_small']['reward_components'].get(k, 0.),
                results['graded_flat']['reward_components'].get(k, 0.), atol=1e-9) for k in ['terrain', 'flatness', 'obstacle_near', 'obstacle_overlap']),
            'rock_terminal_preserved': results['graded_rock']['terminal_reason'] == results['legacy_rock']['terminal_reason'] == 'terrain_failure',
            'zero_learning': ledger['learning_base_steps'] == ledger['optimizer_updates'] == 0,
        }
        summary = {'checks': checks, 'all_checks_passed': all(checks.values()), 'cases': results,
                   'actual_base_steps': ledger['actual_base_steps'], 'optimizer_updates': 0,
                   'interpretation': 'Contract assertions are separate from measured MPPI navigation outcomes.'}
        write(run / 'metrics' / 'crater_classes_summary.json', summary)
        ledger['status'] = 'complete' if all(checks.values()) else 'validation_failed'
        writer.add_scalar('Progress/base_steps', ledger['actual_base_steps'], ledger['actual_base_steps'])
        writer.add_scalar('Progress/optimizer_updates', 0, ledger['actual_base_steps'])
        if not all(checks.values()):
            raise AssertionError({k: v for k, v in checks.items() if not v})
    except BaseException:
        if ledger['status'] == 'running':
            ledger['status'] = 'error'
        (run / 'logs' / 'validation_failure.log').write_text(traceback.format_exc())
        raise
    finally:
        writer.close(); write(run / 'run_manifest.json', ledger)


if __name__ == '__main__':
    main()
