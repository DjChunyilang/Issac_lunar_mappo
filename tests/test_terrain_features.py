from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from _common import cfg_from_experiment
from check_terrain_profile import sample_terrain_profile
from lunar_rover_tasks.tasks.multi_rover_gathering.gathering_env import MultiRoverGatheringCore
from lunar_rover_tasks.tasks.multi_rover_gathering.gathering_env_cfg import make_debug_cfg
from lunar_rover_tasks.tasks.multi_rover_gathering.terrain_features import (
    LOCAL_TERRAIN_GRID_CHANNELS,
    LOCAL_TERRAIN_GRID_X,
    LOCAL_TERRAIN_GRID_Y,
    TERRAIN_FEATURE_NAMES,
    build_local_terrain_grid,
    build_global_terrain_state,
    build_terrain_features,
    flatten_local_terrain_grid,
    local_terrain_grid_world_points,
    make_terrain_runtime,
    query_height,
    query_terrain_features,
    randomize_terrain_runtime,
    sample_path_terrain_risk,
    sample_trajectory_terrain_risk,
    summarize_local_terrain_grid,
)
from terrain_viz import height_grid_for_extent


def _terrain_slice(cfg):
    ego_end = cfg.observation.ego_dim
    neighbor_end = ego_end + cfg.observation.max_neighbors * cfg.observation.neighbor_dim
    return neighbor_end, neighbor_end + cfg.observation.terrain_dim


def test_flat_terrain_features_remain_zero() -> None:
    cfg = make_debug_cfg(num_envs=1, device="cpu")
    positions = torch.tensor([[[0.0, 0.0, 0.0], [1.0, -1.0, 0.0]]])

    local = build_terrain_features(positions, cfg.observation, cfg.terrain)
    grid = build_local_terrain_grid(
        positions,
        torch.zeros(positions.shape[:-1]),
        cfg.terrain,
    )
    global_state = build_global_terrain_state(
        positions,
        cfg.state.terrain_state_dim,
        positions.device,
        cfg.terrain,
    )

    assert TERRAIN_FEATURE_NAMES == ("height", "slope_x", "slope_y", "roughness", "traversability")
    assert local.shape == (1, 2, 5)
    assert grid.shape == (1, 2, 5, 5, 2)
    assert global_state.shape == (1, cfg.state.terrain_state_dim)
    assert torch.allclose(local, torch.zeros_like(local))
    assert torch.allclose(grid, torch.zeros_like(grid))
    assert torch.allclose(global_state, torch.zeros_like(global_state))


def test_procedural_terrain_features_are_finite_and_structured() -> None:
    cfg = make_debug_cfg(num_envs=1, device="cpu")
    cfg.terrain.type = "lunar_heightfield_proxy"
    cfg.terrain.amplitude = 0.2
    cfg.terrain.wavelength = 3.0
    positions = torch.tensor([[[0.25, 0.4, 0.0], [1.5, -0.75, 0.0], [-1.0, 2.0, 0.0]]])

    local = build_terrain_features(positions, cfg.observation, cfg.terrain)
    grid = build_local_terrain_grid(
        positions,
        torch.zeros(positions.shape[:-1]),
        cfg.terrain,
    )
    height = query_height(positions[..., :2], cfg.terrain)
    global_state = build_global_terrain_state(
        positions,
        cfg.state.terrain_state_dim,
        positions.device,
        cfg.terrain,
    )

    assert local.shape == (1, 3, 5)
    assert grid.shape == (1, 3, 5, 5, 2)
    assert height.shape == (1, 3, 1)
    assert global_state.shape == (1, cfg.state.terrain_state_dim)
    assert torch.isfinite(local).all()
    assert torch.isfinite(height).all()
    assert torch.isfinite(global_state).all()
    assert not torch.allclose(local, torch.zeros_like(local))
    assert torch.isfinite(grid).all()
    assert torch.all((grid[..., 1] >= 0.0) & (grid[..., 1] <= 1.0))
    assert torch.all((local[..., 4] >= 0.0) & (local[..., 4] <= 1.0))


def test_randomized_terrain_runtime_changes_maps_per_environment_and_reset() -> None:
    cfg = make_debug_cfg(num_envs=3, device="cpu")
    cfg.terrain.type = "lunar_crater_proxy"
    cfg.terrain.amplitude = 0.1
    cfg.terrain.crater_count = 5
    cfg.terrain.randomize_per_reset = True
    cfg.terrain.random_translation_m = 1.5
    cfg.terrain.random_yaw_rad = torch.pi
    cfg.terrain.amplitude_scale_min = 0.9
    cfg.terrain.amplitude_scale_max = 1.1
    cfg.terrain.crater_radius_scale_min = 0.85
    cfg.terrain.crater_radius_scale_max = 1.15
    cfg.terrain.crater_depth_scale_min = 0.8
    cfg.terrain.crater_depth_scale_max = 1.2
    runtime = make_terrain_runtime(3, device="cpu")
    generator = torch.Generator().manual_seed(123)
    env_ids = torch.arange(3)

    randomize_terrain_runtime(runtime, env_ids, cfg.terrain, generator=generator)
    before = runtime.clone()
    xy = torch.zeros(3, 4, 2)
    features = query_terrain_features(xy, cfg.terrain, runtime)

    assert features.shape == (3, 4, 5)
    assert torch.isfinite(features).all()
    assert not torch.allclose(features[0], features[1])
    randomize_terrain_runtime(
        runtime,
        torch.tensor([1]),
        cfg.terrain,
        generator=generator,
    )
    assert torch.allclose(runtime.phase[0], before.phase[0])
    assert not torch.allclose(runtime.phase[1], before.phase[1])
    assert torch.allclose(runtime.phase[2], before.phase[2])


def test_environment_randomizes_terrain_only_for_reset_episodes() -> None:
    cfg = make_debug_cfg(num_envs=2, device="cpu")
    cfg.terrain.type = "lunar_crater_proxy"
    cfg.terrain.dynamics_enabled = True
    cfg.terrain.amplitude = 0.1
    cfg.terrain.crater_count = 5
    cfg.terrain.randomize_per_reset = True
    cfg.terrain.random_translation_m = 1.0
    cfg.terrain.random_yaw_rad = torch.pi
    env = MultiRoverGatheringCore(cfg)
    before = env.terrain_runtime.clone()

    env.reset(torch.tensor([0]))

    assert not torch.allclose(env.terrain_runtime.phase[0], before.phase[0])
    assert torch.allclose(env.terrain_runtime.phase[1], before.phase[1])
    expected_height = query_height(
        env.positions[..., :2],
        cfg.terrain,
        env.terrain_runtime,
    ).squeeze(-1)
    assert torch.allclose(env.positions[..., 2], expected_height)


def test_local_terrain_grid_rotates_from_body_to_world_frame() -> None:
    positions = torch.tensor([[[10.0, 20.0, 0.0]]])
    yaws = torch.tensor([[torch.pi / 2.0]])

    world = local_terrain_grid_world_points(positions, yaws)

    assert LOCAL_TERRAIN_GRID_X == (-0.4, 0.0, 0.4, 0.8, 1.2)
    assert LOCAL_TERRAIN_GRID_Y == (-0.8, -0.4, 0.0, 0.4, 0.8)
    assert LOCAL_TERRAIN_GRID_CHANNELS == ("relative_height", "risk")
    assert world.shape == (1, 1, 5, 5, 2)
    assert torch.allclose(world[0, 0, 4, 4], torch.tensor([9.2, 21.2]), atol=1.0e-6)


def test_local_grid_distinguishes_same_underfoot_height_with_different_forward_terrain() -> None:
    cfg = make_debug_cfg(num_envs=1, device="cpu")
    cfg.terrain.type = "lunar_crater_proxy"
    cfg.terrain.crater_count = 1
    cfg.terrain.crater_min_radius = 1.0
    cfg.terrain.crater_max_radius = 1.0
    cfg.terrain.crater_depth_to_diameter = 0.08
    cfg.terrain.crater_rim_height_to_diameter = 0.02
    positions = torch.tensor([[[-1.2, 0.0, 0.0], [1.2, 0.0, 0.0]]])
    yaws = torch.zeros(1, 2)

    underfoot_height = query_height(positions[..., :2], cfg.terrain)
    grid = build_local_terrain_grid(positions, yaws, cfg.terrain)

    assert torch.allclose(underfoot_height[:, 0], underfoot_height[:, 1], atol=1.0e-6)
    assert not torch.allclose(grid[:, 0], grid[:, 1])
    assert grid[0, 0, ..., 0].amin() < grid[0, 1, ..., 0].amin()


def test_local_grid_flatten_order_and_critic_summary() -> None:
    grid = torch.zeros(1, 1, 5, 5, 2)
    grid[0, 0, 0, 0] = torch.tensor([-0.3, 0.2])
    grid[0, 0, 4, 4] = torch.tensor([0.5, 0.8])

    flat = flatten_local_terrain_grid(grid)
    summary = summarize_local_terrain_grid(grid)

    assert flat.shape == (1, 1, 50)
    assert torch.allclose(flat[0, 0, :2], torch.tensor([-0.3, 0.2]))
    assert torch.allclose(flat[0, 0, -2:], torch.tensor([0.5, 0.8]))
    assert torch.allclose(
        summary,
        torch.tensor([[(0.3 + 0.5) / 25.0, 0.5, 0.3, 1.0 / 25.0, 0.8]]),
    )


def test_lunar_crater_proxy_has_depressed_bowl_and_raised_rim() -> None:
    cfg = make_debug_cfg(num_envs=1, device="cpu")
    cfg.terrain.type = "lunar_crater_proxy"
    cfg.terrain.amplitude = 0.0
    cfg.terrain.crater_count = 1
    cfg.terrain.crater_min_radius = 1.0
    cfg.terrain.crater_max_radius = 1.0
    cfg.terrain.crater_depth_to_diameter = 0.08
    cfg.terrain.crater_rim_height_to_diameter = 0.02
    positions = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]])

    local = build_terrain_features(positions, cfg.observation, cfg.terrain)

    center_height = local[0, 0, 0]
    rim_height = local[0, 1, 0]
    outside_height = local[0, 2, 0]
    assert torch.isfinite(local).all()
    assert center_height < -0.10
    assert rim_height > 0.0
    assert abs(float(outside_height)) < 0.01
    assert torch.all((local[..., 4] >= 0.0) & (local[..., 4] <= 1.0))


def test_path_terrain_risk_is_zero_on_flat_terrain() -> None:
    cfg = make_debug_cfg(num_envs=1, device="cpu")
    start = torch.tensor([[[0.0, 0.0, 0.0]]])
    target = torch.tensor([[[1.0, 0.0, 0.0]]])

    risk = sample_path_terrain_risk(start, target, cfg.terrain, num_samples=5)

    assert torch.allclose(risk["risk_mean"], torch.zeros(1, 1))
    assert torch.allclose(risk["risk_max"], torch.zeros(1, 1))
    assert torch.allclose(risk["height_change_mean"], torch.zeros(1, 1))


def test_path_terrain_risk_distinguishes_crater_crossing_from_bypass() -> None:
    cfg = make_debug_cfg(num_envs=1, device="cpu")
    cfg.terrain.type = "lunar_crater_proxy"
    cfg.terrain.amplitude = 0.0
    cfg.terrain.crater_count = 1
    cfg.terrain.crater_min_radius = 1.0
    cfg.terrain.crater_max_radius = 1.0
    cfg.terrain.crater_depth_to_diameter = 0.12
    cfg.terrain.crater_rim_height_to_diameter = 0.025
    cfg.terrain.traversability_slope_scale = 0.45
    start = torch.tensor([[[-1.5, 0.0, 0.0], [-1.5, 1.8, 0.0]]])
    target = torch.tensor([[[1.5, 0.0, 0.0], [1.5, 1.8, 0.0]]])

    risk = sample_path_terrain_risk(start, target, cfg.terrain, num_samples=7)

    assert risk["risk_mean"][0, 0] > risk["risk_mean"][0, 1]
    assert risk["risk_max"][0, 0] > risk["risk_max"][0, 1]
    assert risk["height_change_mean"][0, 0] > risk["height_change_mean"][0, 1]


def test_trajectory_terrain_risk_uses_supplied_curve_samples() -> None:
    cfg = make_debug_cfg(num_envs=1, device="cpu")
    cfg.terrain.type = "lunar_crater_proxy"
    cfg.terrain.amplitude = 0.0
    cfg.terrain.crater_count = 1
    cfg.terrain.crater_min_radius = 1.0
    cfg.terrain.crater_max_radius = 1.0
    cfg.terrain.crater_depth_to_diameter = 0.12
    cfg.terrain.crater_rim_height_to_diameter = 0.025
    cfg.terrain.traversability_slope_scale = 0.45
    x = torch.linspace(-1.5, 1.5, 9)
    crossing = torch.stack((x, torch.zeros_like(x), torch.zeros_like(x)), dim=-1)
    bypass = torch.stack((x, torch.full_like(x, 1.8), torch.zeros_like(x)), dim=-1)
    points = torch.stack((crossing, bypass), dim=0).unsqueeze(0)

    risk = sample_trajectory_terrain_risk(points, cfg.terrain)

    assert risk["risk_mean"].shape == (1, 2)
    assert risk["risk_mean"][0, 0] > risk["risk_mean"][0, 1]
    assert risk["risk_max"][0, 0] > risk["risk_max"][0, 1]


def test_height_grid_for_extent_samples_nonflat_terrain() -> None:
    cfg = make_debug_cfg(num_envs=1, device="cpu")
    cfg.terrain.type = "lunar_crater_proxy"
    cfg.terrain.amplitude = 0.04
    cfg.terrain.crater_count = 1
    cfg.terrain.crater_min_radius = 1.0
    cfg.terrain.crater_max_radius = 1.0

    height, extent, value_range = height_grid_for_extent(
        cfg.terrain,
        torch.tensor([-1.5, -1.5]).numpy(),
        torch.tensor([1.5, 1.5]).numpy(),
        resolution=32,
    )

    assert height.shape == (32, 32)
    assert len(extent) == 4
    assert value_range[1] > value_range[0]
    assert abs(float(height.max() - height.min())) > 1.0e-3


def test_exp009_strong_terrain_profile_meets_target_range_and_slows_more_than_exp008() -> None:
    exp008 = sample_terrain_profile(
        "configs/experiment/exp_008_terrain3d_weak_warmstart_select.yaml",
        resolution=64,
    )
    exp009 = sample_terrain_profile(
        "configs/experiment/exp_009_terrain3d_strong_weak_warmstart.yaml",
        resolution=64,
    )
    cfg = cfg_from_experiment("configs/experiment/exp_009_terrain3d_strong_weak_warmstart.yaml")

    assert cfg.terrain.dynamics_enabled is True
    assert cfg.terrain.slope_speed_scale == 1.25
    assert cfg.reward_weights.terrain == 0.45
    assert 0.6 <= exp009["height_range"] <= 1.0
    assert exp009["mean_terrain_speed_scale"] < exp008["mean_terrain_speed_scale"]


def test_actor_and_critic_include_structured_terrain_without_shape_changes() -> None:
    cfg = make_debug_cfg(num_envs=1, device="cpu")
    cfg.terrain.type = "lunar_heightfield_proxy"
    cfg.terrain.amplitude = 0.15
    env = MultiRoverGatheringCore(cfg)
    env.positions.copy_(
        torch.tensor(
            [[[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [-1.0, 1.0, 0.0], [1.0, 1.0, 0.0]]],
            device=env.device,
        )
    )

    actor_obs, critic_state = env.get_observations()
    terrain_start, terrain_end = _terrain_slice(cfg)
    terrain = actor_obs[..., terrain_start:terrain_end]

    assert actor_obs.shape == (1, 4, cfg.actor_obs_dim)
    assert critic_state.shape == (1, cfg.critic_state_dim)
    assert cfg.actor_obs_dim == 86
    assert cfg.critic_state_dim == 54
    assert torch.isfinite(actor_obs).all()
    assert torch.isfinite(critic_state).all()
    assert not torch.allclose(terrain, torch.zeros_like(terrain))
