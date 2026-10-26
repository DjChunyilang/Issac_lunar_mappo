"""Birth-only sparse sensing must reproduce full direct-memory admission."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from _common import cfg_from_experiment
from upstream_curriculum_experiment import load_curriculum
from lunar_rover_tasks.tasks.multi_rover_gathering.gathering_env_cfg import TerrainCfg
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_curriculum import SpawnCurriculum
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_flat_review import create_reviewer
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_sensing import DenseSensor, DenseTerrainMemory


@pytest.fixture(scope='module', autouse=True)
def single_thread():
    previous = torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


class TerrainBank:
    def __init__(self, case, spec):
        self.case, self.spec = case, spec
        self.queries = []

    def query(self, xy, ids):
        self.queries.append(xy.numel() // 2)
        f = torch.zeros((*xy.shape[:-1], 4), dtype=xy.dtype, device=xy.device)
        slope = {'slope_at_limit': self.spec.terrain.driving_slope + 1e-8,
                 'slope_over_limit': self.spec.terrain.driving_slope + 2e-8}.get(self.case, .03)
        f[..., 0] = slope * xy[..., 0]
        f[..., 1] = slope
        if self.case not in ('slope_at_limit', 'slope_over_limit'):
            f[..., 2] = -.02
            f[..., 0] -= .02 * xy[..., 1]
        patch = (xy[..., 0].abs() < .35) & (xy[..., 1].abs() < .35)
        if self.case == 'rock':f[..., 0] += patch * .7
        if self.case == 'rough':f[..., 0] += .1 * torch.sin(7 * xy[..., 0]) * torch.cos(5 * xy[..., 1])
        if self.case == 'steep':f[..., 1] = .4
        good = ~patch if self.case == 'hole' else torch.ones(xy.shape[:-1], dtype=torch.bool)
        return SimpleNamespace(features=f, metrics_valid=good)


def setup(case='plane'):
    spec = cfg_from_experiment('configs/experiment/upstream_1km_soft_hazard_v1.yaml').spec
    bank = TerrainBank(case, spec)
    runtime = SimpleNamespace(scene_bank=bank, scene_indices=torch.zeros(1, dtype=torch.long))
    sensor = DenseSensor(spec, TerrainCfg(type='lunar_scene_pack', raster_risk_slope_scale=.6,
                                        raster_risk_rms_scale_m=.05), runtime=runtime)
    reviewer = create_reviewer(spec, 'flat_frame_pose_cache_v3')
    sensor.prepare_birth(reviewer.points)
    return spec, bank, sensor, reviewer


def full_legal(sampler, poses):
    sensor, reviewer = sampler._birth_sensor, sampler._birth_reviewer
    anchors = poses[None].copy(); anchors[..., 2] = 0.
    frames = sensor.sample(poses[None], anchors, np.zeros(1))[0]
    for a in range(4):
        memory = DenseTerrainMemory(sampler.spec); memory.observe(frames[a])
        if not reviewer.pose([0., 0., poses[a, 2]], memory.view()).passed:
            return False
    return True


@pytest.mark.parametrize('case', ['plane', 'rock', 'rough', 'hole', 'steep', 'slope_at_limit', 'slope_over_limit'])
@pytest.mark.parametrize('yaw', [0., .73, -2.8])
def test_sparse_birth_preserves_observed_features_support_and_footprint(case, yaw):
    spec, bank, sensor, reviewer = setup(case)
    pose = np.array([.017, -.011, yaw])
    poses = np.tile(pose, (1, 4, 1)); anchors = poses.copy(); anchors[..., 2] = 0.
    full = sensor.sample(poses, anchors, [0.])[0][0]
    memory = DenseTerrainMemory(spec); memory.observe(full)
    cropped = sensor.birth_terrain(pose)
    snap = cropped.snapshot()
    # Every sparse measurement is an original frame point, with its original
    # complete 3x3 neighborhood, including unknown-mask propagation.
    distance, ids = memory.view()._tree.query(snap.xy)
    np.testing.assert_array_equal(distance, 0.)
    np.testing.assert_allclose(snap.features, memory.view()._features[ids], atol=1e-15, rtol=0.)
    reference = reviewer.pose([0., 0., yaw], memory.view())
    actual = reviewer.pose([0., 0., yaw], cropped)
    assert (actual.passed, actual.reason, actual.checked_poses) == (reference.passed, reference.reason, reference.checked_poses)
    for name in ('slope', 'roughness_rms', 'maximum_step', 'minimum_body_clearance'):
        a, b = getattr(actual, name), getattr(reference, name)
        if b is None:assert a is None
        else:assert abs(a - b) <= 1e-12
    assert bank.queries[-1] < spec.perception.sample_count // 4


def test_workspace_unknown_and_early_rejection_do_not_scan_later_rovers():
    spec, bank, sensor, reviewer = setup('steep')
    sampler = SpawnCurriculum(spec, load_curriculum('configs/training/upstream_1km_far_soft_hazard_v1.yaml'), 23, 4)
    sampler.bind_terrain(sensor, reviewer)
    poses = np.zeros((4, 3))
    assert not sampler._legal_birth(poses)
    assert len(bank.queries) == 1  # First rejected rover stops the candidate.
    edge = np.array([spec.world_width / 2, 0., .73])
    assert sensor.birth_terrain(edge).lookup([[0., 0.]])[1].shape == (1,)
    poses[:] = edge
    assert sampler._legal_birth(poses) == full_legal(sampler, poses) == False


def test_candidate_order_rng_counters_and_restore_remain_identical():
    spec, _, sensor, reviewer = setup('plane')
    config = load_curriculum('configs/training/upstream_1km_far_soft_hazard_v1.yaml')
    fast = SpawnCurriculum(spec, config, 23, 4); fast.bind_terrain(sensor, reviewer)
    reference = deepcopy(fast)
    reference._legal_birth = lambda poses: full_legal(reference, poses)
    np.testing.assert_array_equal(fast.sample([2, 0, 1, 3]), reference.sample([2, 0, 1, 3]))
    np.testing.assert_array_equal(fast.deadlines, reference.deadlines)
    np.testing.assert_array_equal(fast.episode_levels, reference.episode_levels)
    assert fast.rng.bit_generator.state == reference.rng.bit_generator.state
    assert (fast.spawn_attempts, fast.spawn_rejections) == (reference.spawn_attempts, reference.spawn_rejections)
    saved = fast.state_dict(); expected = fast.sample([1, 3]); fast.load_state_dict(saved)
    np.testing.assert_array_equal(fast.sample([1, 3]), expected)
