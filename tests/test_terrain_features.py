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
    TERRAIN_FEATURE_NAMES,
    build_global_terrain_state,
    build_terrain_features,
    query_height,
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
    global_state = build_global_terrain_state(
        positions,
        cfg.state.terrain_state_dim,
        positions.device,
        cfg.terrain,
    )

    assert TERRAIN_FEATURE_NAMES == ("height", "slope_x", "slope_y", "roughness", "traversability")
    assert local.shape == (1, 2, cfg.observation.terrain_dim)
    assert global_state.shape == (1, cfg.state.terrain_state_dim)
    assert torch.allclose(local, torch.zeros_like(local))
    assert torch.allclose(global_state, torch.zeros_like(global_state))


def test_procedural_terrain_features_are_finite_and_structured() -> None:
    cfg = make_debug_cfg(num_envs=1, device="cpu")
    cfg.terrain.type = "lunar_heightfield_proxy"
    cfg.terrain.amplitude = 0.2
    cfg.terrain.wavelength = 3.0
    positions = torch.tensor([[[0.25, 0.4, 0.0], [1.5, -0.75, 0.0], [-1.0, 2.0, 0.0]]])

    local = build_terrain_features(positions, cfg.observation, cfg.terrain)
    height = query_height(positions[..., :2], cfg.terrain)
    global_state = build_global_terrain_state(
        positions,
        cfg.state.terrain_state_dim,
        positions.device,
        cfg.terrain,
    )

    assert local.shape == (1, 3, cfg.observation.terrain_dim)
    assert height.shape == (1, 3, 1)
    assert global_state.shape == (1, cfg.state.terrain_state_dim)
    assert torch.isfinite(local).all()
    assert torch.isfinite(height).all()
    assert torch.isfinite(global_state).all()
    assert not torch.allclose(local, torch.zeros_like(local))
    assert torch.all((local[..., 4] >= 0.0) & (local[..., 4] <= 1.0))


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
    assert torch.isfinite(actor_obs).all()
    assert torch.isfinite(critic_state).all()
    assert not torch.allclose(terrain, torch.zeros_like(terrain))
