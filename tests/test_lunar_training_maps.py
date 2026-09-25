"""Geometry, validity, reproducibility and real-data runtime contracts."""

from pathlib import Path

import numpy as np
import pytest
import rasterio
import torch
from lunar_rover_tasks.terrain_data.batched import SceneBank
from lunar_rover_tasks.terrain_data.enhancement import crater_profile, enhance
from lunar_rover_tasks.terrain_data.scene_pack import SCHEMA, ScenePack, build_scene
from rasterio.transform import from_origin


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "plane.tif"
    yy, xx = np.meshgrid(np.arange(40) + 0.5, np.arange(60) + 0.5, indexing="ij")
    z = 100 + 0.1 * (xx * 2) + 0.2 * (120 - yy * 3)
    crs = "+proj=stere +lat_0=-90 +lon_0=0 +R=1737400 +units=m +no_defs"
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=60,
        height=40,
        count=1,
        dtype="float32",
        crs=crs,
        transform=from_origin(10000, 20000, 2, 3),
        nodata=-9999,
    ) as dst:
        dst.write(z.astype(np.float32), 1)
    return path


def pack(source, tmp_path, name="scene", **kwargs):
    return build_scene(
        source,
        tmp_path / name,
        source_id="plane",
        geographic_group="test_group",
        pixel_window=[5, 5, 40, 25],
        processing_spacing_m=1,
        metric_baseline_m=4,
        source_half_width=2,
        **kwargs,
    )


def test_plane_rectangular_geometry_and_units(source, tmp_path):
    scene = ScenePack(pack(source, tmp_path))
    assert scene.meta["extent_xy_m"] == [80, 75]
    f = scene.arrays["features"]
    good = scene.arrays["metrics_valid"]
    np.testing.assert_allclose(f[1][good], 0.1, atol=2e-5)
    np.testing.assert_allclose(f[2][good], 0.2, atol=2e-5)
    assert np.max(f[3][good]) < 1e-4
    xy = np.array([[0, 0], [-15, 22]])
    np.testing.assert_allclose(scene.projected_to_local(scene.local_to_projected(xy)), xy)
    with rasterio.open(scene.path.parent / "height.tif") as ds:
        np.testing.assert_allclose(ds.read(1), f[0], equal_nan=True)
        assert float(ds.tags()["height_origin_m"]) == scene.meta["height_origin_m"]


def test_void_and_edges_are_never_supported(source, tmp_path):
    with rasterio.open(source, "r+") as ds:
        z = ds.read(1)
        z[18:21, 24:27] = -9999
        ds.write(z, 1)
    manifest = pack(source, tmp_path)
    scene = ScenePack(manifest)
    assert not scene.arrays["height_valid"].all()
    bank = SceneBank([manifest])
    bank.prefetch([0])
    p = np.array(scene.transform * (20, 0.5)) - scene.meta["origin_projected_xy_m"]
    q = bank.query(torch.tensor([[p.tolist(), [1000, 0]]], dtype=torch.float32), torch.tensor([0]))
    assert not q.metrics_valid.any()
    assert not q.height_valid[0, 1]
    assert torch.isnan(q.features[0, 1]).all()


def test_quality_is_optional_not_fake_zero_error(source, tmp_path):
    bank = SceneBank([pack(source, tmp_path)])
    bank.prefetch([0])
    q = bank.query(torch.zeros(1, 1, 2), torch.tensor([0]))
    assert q.metrics_valid.all()
    assert not q.error_valid.any()
    assert torch.isnan(q.height_error).all()


def test_pack_checksum_and_immutability(source, tmp_path):
    p = pack(source, tmp_path)
    with pytest.raises(FileExistsError):
        pack(source, tmp_path)
    file = p.parent / "features.npy"
    with file.open("ab") as f:
        f.write(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        ScenePack(p)


def test_cache_budget_and_reset_only_loading(source, tmp_path):
    p = pack(source, tmp_path)
    bank = SceneBank([p], max_bytes=100)
    with pytest.raises(MemoryError):
        bank.prefetch([0])
    bank = SceneBank([p])
    with pytest.raises(RuntimeError, match="prefetched"):
        bank.query(torch.zeros(1, 2), torch.tensor([0]))
    bank.prefetch([0])
    before = bank.bytes
    bank.query(torch.zeros(16, 4, 112, 2), torch.zeros(16, dtype=torch.long))
    assert bank.bytes == before


def test_cpu_cuda_match(source, tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable in this environment")
    p = pack(source, tmp_path)
    cpu = SceneBank([p])
    gpu = SceneBank([p], device="cuda")
    cpu.prefetch([0])
    gpu.prefetch([0])
    torch.manual_seed(11)
    xy = (torch.rand(8, 4, 112, 2) - 0.5) * 60
    ids = torch.zeros(8, dtype=torch.long)
    a, b = cpu.query(xy, ids), gpu.query(xy.cuda(), ids.cuda())
    torch.testing.assert_close(a.features, b.features.cpu(), atol=1e-4, rtol=1e-5, equal_nan=True)
    assert torch.equal(a.metrics_valid, b.metrics_valid.cpu())


def test_crater_centre_and_rim_are_finite_continuous():
    r = np.array([0, 1 - 1e-6, 1, 1 + 1e-6, 2, 3])
    z = crater_profile(r, depth_m=0.4, rim_height_m=0.1)
    np.testing.assert_allclose(z, [-0.4, 0.1, 0.1, 0.1, 0, 0], atol=1e-6)
    assert np.isfinite(z).all()


def test_enhancement_reproducible_and_preserves_mask():
    import yaml

    spec = yaml.safe_load(Path("configs/terrain/nasa_enhancement_v1.yaml").read_text())
    z = np.zeros((100, 140), np.float32)
    valid = np.ones_like(z, bool)
    valid[0, :] = False
    z[~valid] = np.nan
    a, layers, catalog = enhance(z, valid, (0.2, 0.3), spec)
    b, _, other = enhance(z, valid, (0.2, 0.3), spec)
    np.testing.assert_array_equal(a, b)
    assert catalog == other and np.isnan(a[~valid]).all() and np.isfinite(a[valid]).all()
    assert set(layers) == {"fractal", "craters", "rocks"}
    assert all(c["diameter_m"] <= 2 for c in catalog["craters"])


def test_enhancement_requires_declared_hypotheses():
    with pytest.raises(ValueError, match="hypothesis"):
        enhance(np.zeros((10, 10)), np.ones((10, 10), bool), (1, 1), {})


def raster_cfg(manifest, device="cpu"):
    from lunar_rover_tasks.tasks.multi_rover_gathering.gathering_env_cfg import make_debug_cfg

    c = make_debug_cfg(num_envs=2, device=device)
    c.terrain.type = "lunar_scene_pack"
    c.terrain.scene_manifests = [str(manifest)]
    c.terrain.feature_schema = SCHEMA
    c.terrain.dynamics_enabled = True
    c.terrain.raster_risk_slope_scale = 0.6
    c.terrain.raster_risk_rms_scale_m = 0.15
    c.terrain.raster_task_basis = "unit test only; not vehicle safety requirements"
    c.observation.schema_version = "ego_v12_lunar_multiscale"
    c.gather_point.search_method = "geometric_median"
    return c


def test_runtime_and_actor_validity_channel(source, tmp_path):
    from lunar_rover_tasks.tasks.multi_rover_gathering.gathering_env import MultiRoverGatheringCore

    c = raster_cfg(pack(source, tmp_path))
    env = MultiRoverGatheringCore(c)
    obs, _state = env.reset()
    assert obs.shape == (2, 4, 407)
    assert torch.isfinite(obs).all()
    assert env.terrain_runtime.clone().scene_bank is env.terrain_runtime.scene_bank
    out = env.step(torch.zeros(2, 4, 2))
    assert torch.isfinite(out.actor_obs).all() and torch.isfinite(out.rewards).all()
    # Third channel distinguishes missing support from a physically high risk.
    terrain = obs[..., 66:402].reshape(2, 4, 112, 3)
    assert torch.isin(terrain[..., 2], torch.tensor([0.0, 1.0])).all()


def test_invalid_support_never_passes_flatness(source, tmp_path):
    from lunar_rover_tasks.tasks.multi_rover_gathering.gathering_env import MultiRoverGatheringCore
    from lunar_rover_tasks.tasks.multi_rover_gathering.terrain_features import (
        evaluate_gather_point_flatness,
    )

    env = MultiRoverGatheringCore(raster_cfg(pack(source, tmp_path)))
    flat = evaluate_gather_point_flatness(
        torch.full((2, 2), 1000.0),
        env.cfg.terrain,
        env.terrain_runtime,
        radius=1,
        rings=1,
        samples_per_ring=8,
        max_height_range=1,
        max_slope=1,
    )
    assert not flat.is_flat.any()
    assert torch.isfinite(flat.height_range).all()


def test_enhanced_pack_serializes_and_records_native_scale_change(source, tmp_path):
    import yaml

    spec = yaml.safe_load(Path("configs/terrain/nasa_enhancement_v1.yaml").read_text())
    scene = ScenePack(pack(source, tmp_path, enhancement=spec))
    assert scene.meta["classification"] == "real_plus_hypothesis"
    assert np.isfinite(scene.meta["enhancement_diagnostics"]["native_area_mean_residual_rms_m"])


def test_local_goal_new_channels_checkpoint_and_unknown_memory(source, tmp_path):
    import io

    from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.environment import (
        LocalGoalEnvironment,
    )
    from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.learning import RecurrentPolicy
    from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.memory import LocalTerrainMemory

    class FailureInjection:
        def reset(self):
            pass

        def set(self, *args):
            pass

        def constraints_set(self, *args):
            pass

        def solve(self):
            return 4

    c = raster_cfg(pack(source, tmp_path))
    c.safety.collision_termination_enabled = True
    for k in ("oracle", "motion", "consistency", "active_dstc"):
        setattr(c.reward_weights, k, 0)
    c.terrain.local_memory_resolution_m = 0.2
    c.terrain.local_memory_support_radius_m = 0.155
    env = LocalGoalEnvironment(c, tmp_path, solver_factory=lambda *a, **kw: FailureInjection())
    try:
        obs, state = env.observe()
        assert obs.shape == (2, 4, 522)
        policy = RecurrentPolicy(state.shape[-1], terrain_channels=3)
        action, _, _, _ = policy.sample(obs, torch.zeros(2, 4, 128), deterministic=True)
        out = env.step(action, max_low_steps=1)
        assert torch.isfinite(out.reward).all()
        saved = env.state_dict()
        assert saved["core"]["terrain_runtime"].scene_bank is None
        buffer = io.BytesIO()
        torch.save(saved, buffer)
        buffer.seek(0)
        env.load_state_dict(torch.load(buffer, weights_only=False))
        assert env.core.terrain_runtime.scene_bank is not None
        saved["terrain_contract"]["observation"] = "legacy"
        with pytest.raises(ValueError, match="contract"):
            env.load_state_dict(saved)
        # Invalid samples cannot become direct planner observations.
        for memory in env.memories:
            memory.reset()
        base, _ = env.core.get_observations()
        base[..., 66:402].reshape(2, 4, 112, 3)[..., 2] = 0
        env._sensors(base, env.poses())
        assert all(len(m.samples) == 0 for m in env.memories)
        m = LocalTerrainMemory()
        m.observe(np.empty((0, 2)), np.empty((0, 5)), 0)
        assert not m.lookup([[0, 0]])[1].any()
    finally:
        env.close()


def test_crossing_unsupported_terrain_truncates(source, tmp_path):
    from lunar_rover_tasks.tasks.multi_rover_gathering.gathering_env import MultiRoverGatheringCore
    from lunar_rover_tasks.tasks.multi_rover_gathering.metrics import compute_team_metrics
    from lunar_rover_tasks.tasks.multi_rover_gathering.simple_controller import ControlCommand
    from lunar_rover_tasks.tasks.multi_rover_gathering.termination import compute_done

    env = MultiRoverGatheringCore(raster_cfg(pack(source, tmp_path)))
    env.positions[..., 0] = 39.0
    env.yaws.zero_()
    before = env.positions.clone()
    env._integrate(ControlCommand(torch.ones(2, 4), torch.zeros(2, 4)))
    assert env.last_terrain_unknown.all()
    torch.testing.assert_close(env.positions[..., :2], before[..., :2])
    done, _ = compute_done(
        env.positions,
        env.velocities_xy,
        compute_team_metrics(env.positions, env.velocities_xy),
        env.success_hold_count,
        env.step_count,
        env.max_episode_steps,
        env.cfg.success_thresholds,
        env.cfg.safety,
        flatness_ok=torch.zeros(2, dtype=torch.bool),
        terrain_unknown=env.last_terrain_unknown.any(-1),
    )
    assert done.invalid_terrain.all() and not done.success.any()
