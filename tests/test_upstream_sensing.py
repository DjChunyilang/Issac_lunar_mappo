"""Direct dense observation, local information boundary and bounded memory."""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from _common import cfg_from_experiment
from lunar_rover_tasks.tasks.multi_rover_gathering.gathering_env_cfg import TerrainCfg
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_sensing import (
    DenseSensor, DenseTerrainMemory, SensorFrame, world_to_odometry,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_mppi import UpstreamMPPIAdapter, UpstreamMPPIKernel, UpstreamMPPISettings
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_planning import UpstreamIntent, UpstreamPlannerInput
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_footprint import FootprintReviewer


@pytest.fixture(scope="module", autouse=True)
def single_thread():
    old = torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@pytest.fixture(scope="module")
def cfg():
    return cfg_from_experiment("configs/experiment/upstream_scale_s02.yaml")


@pytest.fixture(scope="module")
def frame(cfg):
    return DenseSensor(cfg.spec, None).sample(np.zeros((1, 4, 3)), np.zeros((1, 4, 3)), [0.])[0][0]


def sparse(frame, *, time=0., offset=0., slope=.0, traversability=1., roughness=0., height=0.):
    valid = np.zeros(len(frame.xy), dtype=bool)
    center = np.linalg.norm(frame.xy, axis=-1).argmin(); valid[center] = True
    features = np.zeros_like(frame.features)
    features[center] = [height, slope, 0., roughness, traversability]
    return replace(frame, timestamp=time, xy=frame.xy + [offset, 0.],
                   sensor_pose=frame.sensor_pose + [offset, 0., 0.], features=features,
                   valid=valid, roughness_valid=valid)


def test_frame_is_dense_readonly_and_not_an_actor_grid(cfg, frame):
    assert frame.xy.shape == (12231, 2) and frame.features.shape == (12231, 5)
    assert frame.valid.all() and frame.roughness_valid.all()
    assert not frame.xy.flags.writeable and not frame.features.flags.writeable
    np.testing.assert_array_equal(frame.features[:, 4], 1.)
    np.testing.assert_allclose(frame.features[:, 3], 0., atol=1e-12)
    assert not hasattr(frame, "runtime") and not hasattr(frame, "scene_bank")
    snap = frame.snapshot()
    assert snap.xy.shape == frame.xy.shape and snap.support_radius == cfg.spec.perception.support_radius


def test_world_and_fixed_odometry_transform_agree(cfg):
    poses = np.zeros((1, 4, 3)); poses[0, 0] = [10., 20., np.pi / 2]
    anchors = np.zeros_like(poses); anchors[0, 0] = [7., 18., np.pi / 2]
    f = DenseSensor(cfg.spec, None).sample(poses, anchors, [1.])[0][0]
    np.testing.assert_allclose(f.sensor_pose, [2., -3., 0.], atol=1e-12)
    np.testing.assert_allclose(f.xy.min(0), [-13., -11.], atol=1e-12)
    np.testing.assert_allclose(f.xy.max(0), [17., 5.], atol=1e-12)
    np.testing.assert_allclose(world_to_odometry([[10., 20.]], anchors[0, 0]), [[2., -3.]], atol=1e-12)
    with pytest.raises(ValueError):world_to_odometry([[np.inf, 0.]], [0., 0., 0.])


class LocalPlaneBank:
    """Sensor backend test double; never passed into a planner input."""
    def __init__(self, hole=False):self.hole = hole
    def query(self, xy, ids):
        features = torch.zeros((*xy.shape[:-1], 4), device=xy.device, dtype=xy.dtype)
        features[..., 0] = 2. + .05 * xy[..., 0] - .03 * xy[..., 1]
        features[..., 1] = .05; features[..., 2] = -.03
        valid = torch.ones(xy.shape[:-1], device=xy.device, dtype=torch.bool)
        if self.hole:valid &= (xy[..., 0].abs() > .11) | (xy[..., 1].abs() > .11)
        return SimpleNamespace(features=features, metrics_valid=valid)


def plane_sensor(cfg, device="cpu", hole=False):
    runtime = SimpleNamespace(scene_bank=LocalPlaneBank(hole), scene_indices=torch.zeros(1, dtype=torch.long))
    terrain_cfg = TerrainCfg(type="lunar_scene_pack", raster_risk_slope_scale=.8, raster_risk_rms_scale_m=.05)
    return DenseSensor(cfg.spec, terrain_cfg, runtime=runtime, map_identities=("local_plane_sensor_fixture",), device=device)


def test_plane_is_detrended_and_gradient_rotates_to_odometry(cfg):
    poses = np.zeros((1, 4, 3)); anchors = poses.copy(); anchors[..., 2] = np.pi / 2
    f = plane_sensor(cfg).sample(poses, anchors, [0.])[0][0]
    assert f.valid.all()
    np.testing.assert_allclose(f.features[:, 1:3], np.tile([-.03, -.05], (12231, 1)), atol=1e-12)
    assert f.features[:, 3].max() < 1e-12
    assert f.features[:, 0].min() != f.features[:, 0].max()


def test_finite_raster_placeholder_does_not_fill_unknown_hole(cfg):
    f = plane_sensor(cfg, hole=True).sample(np.zeros((1, 4, 3)), np.zeros((1, 4, 3)), [0.])[0][0]
    center = np.linalg.norm(f.xy, axis=-1).argmin()
    assert not f.valid[center] and not f.roughness_valid[center]
    np.testing.assert_array_equal(f.features[center], np.zeros(5))
    memory = DenseTerrainMemory(cfg.spec);memory.observe(f)
    _, known, direct = memory.lookup([[0., 0.]])
    assert not known[0] and not direct[0]


def test_sensor_workspace_boundary_remains_unknown(cfg):
    poses = np.zeros((1, 4, 3));poses[..., 0] = 99.
    f = DenseSensor(cfg.spec, None).sample(poses, np.zeros_like(poses), [0.])[0][0]
    assert not f.valid.all() and not f.valid[f.xy[:, 0] > 100.].any()
    np.testing.assert_array_equal(f.features[~f.valid], np.zeros_like(f.features[~f.valid]))


@pytest.mark.parametrize("bad", ["grid", "version", "mask", "unknown_value", "negative_time", "map_identity"])
def test_malformed_sensor_frame_rejected(cfg, frame, bad):
    changes = {"grid": {"xy": frame.xy + [100., 0.]}, "version": {"version": "legacy"},
               "mask": {"valid": frame.valid.astype(float)},
               "unknown_value": {"valid": np.zeros(len(frame.xy), bool), "roughness_valid": np.zeros(len(frame.xy), bool)},
               "negative_time": {"timestamp": -1.}, "map_identity": {"map_identity": ""}}[bad]
    with pytest.raises(ValueError):replace(frame, **changes)


@pytest.mark.parametrize("slope,trav,rms", [(.2, 1., 0.), (0., .2, 0.), (0., 1., .06)])
def test_new_scale_hazards_survive_safe_reobservation_in_same_cell(cfg, frame, slope, trav, rms):
    memory = DenseTerrainMemory(cfg.spec)
    first = sparse(frame, slope=slope, traversability=trav, roughness=rms)
    memory.observe(first)
    memory.observe(sparse(frame, time=.2, offset=.01))
    snap = memory.view().snapshot()
    assert len(snap.xy) == 2  # indexing does not move or merge direct observations
    feature, known, direct = memory.lookup([[0., 0.]])
    assert known[0] and direct[0]
    assert feature[0, 1] == slope and feature[0, 4] == trav and feature[0, 3] == rms
    assert cfg.spec.terrain.driving_slope < .2 < .8


def test_complete_dense_frames_capacity_eviction_and_detached_old_view(cfg):
    memory = DenseTerrainMemory(cfg.spec)
    sensor = DenseSensor(cfg.spec, None)
    frames = np.zeros((1, 4, 3)); first_view = None
    for step in range(8):
        poses = np.zeros_like(frames);poses[..., 0] = step * 5. + step * .013
        f = sensor.sample(poses, frames, [step * .2])[0][0]
        memory.observe(f)
        if step == 0:first_view = memory.view()
        snap = memory.view().snapshot()
        assert len(snap.xy) <= 65536
        # The entire last direct batch survives bounded LRU eviction.
        features, known, direct = memory.lookup(f.xy[f.valid])
        assert known.all() and direct.all() and np.isfinite(features).all()
    assert len(memory.view().snapshot().xy) > len(first_view.snapshot().xy)
    assert first_view.snapshot().xy[:, 0].max() <= 15.


def test_memory_identity_time_and_restore_validation_are_atomic(cfg, frame):
    memory = DenseTerrainMemory(cfg.spec);memory.observe(frame)
    saved = memory.state_dict()
    with pytest.raises(ValueError, match="another rover"):
        memory.observe(replace(frame, agent_id=1))
    with pytest.raises(ValueError, match="map episode"):
        memory.observe(replace(frame, map_identity="other_map"))
    memory.observe(replace(frame, timestamp=.2))
    with pytest.raises(ValueError, match="backwards"):memory.observe(frame)
    restored = DenseTerrainMemory(cfg.spec);restored.load_state_dict(saved)
    np.testing.assert_array_equal(restored.view().snapshot().xy, saved["xy"])
    for name, value in (("version", "old"), ("profile_fingerprint", "bad"), ("last_time", -.1), ("observations", True)):
        bad = deepcopy(saved);bad[name] = value
        with pytest.raises(ValueError):restored.load_state_dict(bad)
        np.testing.assert_array_equal(restored.view().snapshot().features, saved["features"])
    bad = deepcopy(saved);bad["observed_at"][0] = 10.
    with pytest.raises(ValueError):restored.load_state_dict(bad)
    memory.reset();memory.observe(replace(frame, agent_id=1))
    assert memory.state_dict()["identity"] == (0, 1)


def test_other_environment_state_does_not_change_legal_local_frame(cfg):
    sensor = DenseSensor(cfg.spec, None)
    poses = np.zeros((2, 4, 3));anchors = poses.copy()
    a = sensor.sample(poses, anchors, [0., 0.])[0][0]
    poses[1] = [50., 60., 2.]
    b = sensor.sample(poses, anchors, [0., 4.])[0][0]
    np.testing.assert_array_equal(a.xy, b.xy)
    np.testing.assert_array_equal(a.features, b.features)


def test_actual_dense_sensor_snapshot_drives_real_mppi(cfg, frame):
    memory = DenseTerrainMemory(cfg.spec);memory.observe(frame)
    planner = UpstreamMPPIAdapter(cfg, UpstreamMPPIKernel(cfg, UpstreamMPPISettings(samples=8)))
    context = UpstreamPlannerInput(0, 0, np.zeros(3), np.zeros(2), UpstreamIntent([4., 0.], False, 0., 1.),
        memory.view(), (), np.zeros(2), .6, .6 + np.arange(41) * .2, (-100., -100., 100., 100.), cfg.spec,
        information_pose=np.zeros(3), information_time=0.)
    output = planner.plan(context)
    assert output.valid and output.trajectory.states[-1, 0] > .5
    assert output.diagnostics["information_reserve_verified"]
    assert not hasattr(context.terrain, "memory") and not hasattr(context.terrain, "runtime")


def test_required_actual_cuda_sensor_pair(cfg):
    assert torch.cuda.is_available(), "S03 sensor CUDA pairing requires a real device"
    poses = np.zeros((1, 4, 3));anchors = np.zeros_like(poses)
    poses[..., 2] = .31;anchors[..., 2] = -.47
    a = plane_sensor(cfg).sample(poses, anchors, [0.])[0][0]
    b = plane_sensor(cfg, "cuda").sample(poses, anchors, [0.])[0][0]
    np.testing.assert_allclose(a.xy, b.xy, atol=1e-10, rtol=1e-8)
    np.testing.assert_allclose(a.features, b.features, atol=1e-10, rtol=1e-8)
    np.testing.assert_array_equal(a.valid, b.valid)


def test_real_scene_bank_sensor_memory_and_unsafe_plane_classification(cfg, tmp_path):
    from test_lunar_training_maps import source, pack
    from lunar_rover_tasks.terrain_data.batched import SceneBank
    file = source.__wrapped__(tmp_path)
    manifest = pack(file, tmp_path)
    bank = SceneBank([manifest]);bank.prefetch([0])
    runtime = SimpleNamespace(scene_bank=bank, scene_indices=torch.zeros(1, dtype=torch.long))
    terrain_cfg = TerrainCfg(type="lunar_scene_pack", raster_risk_slope_scale=.8, raster_risk_rms_scale_m=.05)
    sensor = DenseSensor(cfg.spec, terrain_cfg, runtime=runtime, map_identities=("generated_plane_scene",))
    frame = sensor.sample(np.zeros((1, 4, 3)), np.zeros((1, 4, 3)), [0.])[0][0]
    assert frame.valid.all()
    np.testing.assert_allclose(frame.features[:, 1:3], np.tile([.1, .2], (12231, 1)), atol=2e-5)
    assert frame.features[:, 3].max() < 2e-5
    memory = DenseTerrainMemory(cfg.spec);memory.observe(frame)
    report = FootprintReviewer(cfg.spec).pose(np.zeros(3), memory.view())
    assert not report.passed and report.reason == "driving_slope"
    assert not hasattr(memory.view(), "scene_bank")


def test_invalid_raster_sensor_configuration_is_not_recast_as_unknown(cfg):
    with pytest.raises(ValueError, match="risk scales"):
        DenseSensor(cfg.spec, TerrainCfg(type="lunar_scene_pack"))


@pytest.mark.parametrize("offset", [0., .01])
def test_reobservation_cannot_erase_conflicting_height_or_subcell_step(cfg, frame, offset):
    memory = DenseTerrainMemory(cfg.spec)
    memory.observe(sparse(frame, height=.2))
    memory.observe(sparse(frame, time=.2, offset=offset))
    feature, known, direct = memory.lookup([[0., 0.]])
    assert known[0] and direct[0] and feature[0, 0] == .2
    if offset == 0.:
        assert feature[0, 4] == 0.  # conflicting direct measurements remain uncertified
    else:
        assert len(memory.view().snapshot().xy) == 2
