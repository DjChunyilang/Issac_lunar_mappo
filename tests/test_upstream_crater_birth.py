"""Spawn admission uses the same crater classes and driving surface as execution."""
from copy import copy
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from _common import cfg_from_experiment
from upstream_curriculum_experiment import load_curriculum
from test_upstream_crater_atlas import synthetic_scene
from lunar_rover_tasks.terrain_data.batched import SceneBank
from lunar_rover_tasks.terrain_data.metrics import local_metrics
from lunar_rover_tasks.terrain_data.obstacle_atlas import build_obstacle_atlas,ObstacleAtlas
from lunar_rover_tasks.tasks.multi_rover_gathering.gathering_env_cfg import TerrainCfg
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_curriculum import SpawnCurriculum
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_flat_review import create_reviewer
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_sensing import DenseSensor


@pytest.fixture
def setup(tmp_path):
    cfg=cfg_from_experiment('configs/experiment/upstream_1km_crater_classes_v1.yaml')
    scene=synthetic_scene(tmp_path/'scene.json')
    # Deliberately steep medium pit reproduces the old geometric rejection;
    # the class contract now permits its derived driving surface.
    scene.arrays['layer_craters']*=2
    height=sum(scene.arrays[key] for key in ('base_height','layer_rocks','layer_fractal','layer_craters'))
    metrics=local_metrics(height,scene.arrays['height_valid'],.1,.1,4)
    scene.arrays['features']=np.stack([height,metrics['slope_x'],metrics['slope_y'],metrics['roughness']]).astype(np.float32)
    scene.arrays['metrics_valid']=metrics['valid']
    scene.meta['feature_catalog']['craters'][1]['diameter_m']=1.6
    scene.path.write_text(json.dumps(scene.meta))
    path=build_obstacle_atlas(scene,tmp_path/'atlas',cfg.obstacle_input,cfg.spec)
    atlas=ObstacleAtlas(path,cfg.obstacle_input,scene)
    bank=object.__new__(SceneBank)
    bank.storage='mmap';bank.scenes=[scene];bank.resident={0:scene.arrays}
    runtime=SimpleNamespace(scene_bank=bank,scene_indices=torch.zeros(1,dtype=torch.long))
    sensor=DenseSensor(cfg.spec,TerrainCfg(type='lunar_scene_pack',raster_risk_slope_scale=.6,
                                         raster_risk_rms_scale_m=.05),runtime=runtime)
    reviewer=create_reviewer(cfg.spec,'flat_frame_pose_cache_v3')
    config=load_curriculum('configs/training/upstream_1km_far_soft_hazard_v1.yaml')
    return cfg.spec,config,sensor,reviewer,atlas


def test_medium_crater_birth_uses_derived_surface_and_legacy_stays_strict(setup):
    spec,config,sensor,reviewer,atlas=setup
    legacy=SpawnCurriculum(spec,config,23,1);legacy.bind_terrain(sensor,reviewer)
    graded=SpawnCurriculum(spec,config,23,1);graded.bind_terrain(sensor,reviewer,driving_atlas=atlas)
    poses=np.tile([-2.,3.,0.],(4,1))
    assert not atlas.exact_large_crater_overlap(poses,spec.vehicle.length,spec.vehicle.width).any()
    assert not legacy._legal_birth(poses)
    assert graded._legal_birth(poses)
    assert 'birth_driving_atlas' not in legacy.metadata
    assert graded.metadata['birth_driving_atlas']==atlas.identity


def test_large_crater_birth_rejects_before_sampling_any_rover(setup):
    spec,config,sensor,reviewer,atlas=setup
    sampler=SpawnCurriculum(spec,config,23,1);sampler.bind_terrain(sensor,reviewer,driving_atlas=atlas)
    poses=np.tile([-2.,3.,0.],(4,1));poses[3]=[4.,3.,0.]
    queries=sensor.queried_points
    assert not sampler._legal_birth(poses)
    assert sensor.queried_points==queries


def test_birth_restore_rejects_different_atlas_or_legacy_semantics(setup):
    spec,config,sensor,reviewer,atlas=setup
    sampler=SpawnCurriculum(spec,config,23,1);sampler.bind_terrain(sensor,reviewer,driving_atlas=atlas)
    sampler._sample_geometry([0])
    state=sampler.state_dict()
    restored=SpawnCurriculum(spec,config,23,1);restored.bind_terrain(sensor,reviewer,driving_atlas=atlas)
    restored.load_state_dict(state)
    assert restored.metadata==sampler.metadata
    different=copy(atlas);different.identity='different-derived-atlas'
    restored.bind_terrain(sensor,reviewer,driving_atlas=different)
    with pytest.raises(ValueError,match='Cross-curriculum'):
        restored.load_state_dict(state)
    legacy=SpawnCurriculum(spec,config,23,1);legacy.bind_terrain(sensor,reviewer)
    with pytest.raises(ValueError,match='Cross-curriculum'):
        legacy.load_state_dict(state)
