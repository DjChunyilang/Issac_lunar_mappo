from __future__ import annotations

import torch

from lunar_rover_tasks.tasks.multi_rover_gathering.gathering_env import MultiRoverGatheringCore
from lunar_rover_tasks.tasks.multi_rover_gathering.gathering_env_cfg import make_debug_cfg
from lunar_rover_tasks.tasks.multi_rover_gathering.terrain_features import (
    TERRAIN_FEATURE_NAMES,
    build_global_terrain_state,
    build_terrain_features,
    query_height,
)


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
