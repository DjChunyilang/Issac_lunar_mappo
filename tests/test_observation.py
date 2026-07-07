from __future__ import annotations

import torch

from lunar_rover_tasks.tasks.multi_rover_gathering.gathering_env import MultiRoverGatheringCore
from lunar_rover_tasks.tasks.multi_rover_gathering.gathering_env_cfg import make_debug_cfg


def test_actor_observation_does_not_change_when_oracle_changes() -> None:
    cfg = make_debug_cfg(num_envs=2, device="cpu")
    env = MultiRoverGatheringCore(cfg)
    actor_obs, critic_state = env.get_observations()
    assert actor_obs.shape == (2, 4, cfg.actor_obs_dim)
    assert critic_state.shape == (2, cfg.critic_state_dim)
    baseline = actor_obs.clone()
    env.oracle_point += 123.0
    changed_actor, _ = env.get_observations()
    assert torch.allclose(changed_actor, baseline)


def test_critic_state_changes_when_oracle_changes() -> None:
    cfg = make_debug_cfg(num_envs=2, device="cpu")
    env = MultiRoverGatheringCore(cfg)
    _, critic_state = env.get_observations()
    env.oracle_point += 123.0
    _, changed_state = env.get_observations()
    assert not torch.allclose(changed_state, critic_state)


def test_ego_schema_replaces_zero_placeholders_with_motion_features() -> None:
    cfg = make_debug_cfg(num_envs=1, device="cpu")
    env = MultiRoverGatheringCore(cfg)
    env.positions.zero_()
    env.yaws.zero_()
    env.velocities_xy.copy_(
        torch.tensor([[[3.0, 4.0], [0.0, 2.0], [5.0, 12.0], [8.0, 15.0]]])
    )
    env.angular_velocities.copy_(torch.tensor([[1.5, -2.5, 0.25, -0.75]]))

    actor_obs, _ = env.get_observations()
    ego = actor_obs[..., : cfg.observation.ego_dim]

    expected_speed = torch.linalg.norm(env.velocities_xy, dim=-1)
    expected_abs_angular = env.angular_velocities.abs()
    assert cfg.observation.schema_version == "ego_v3_local_terrain_grid"
    assert cfg.actor_obs_dim == 86
    assert cfg.critic_state_dim == 54
    assert torch.allclose(ego[..., -2], expected_speed)
    assert torch.allclose(ego[..., -1], expected_abs_angular)
    assert not torch.allclose(ego[..., -2:], torch.zeros_like(ego[..., -2:]))


def test_communication_radius_changes_neighbor_visibility_and_aggregation() -> None:
    cfg = make_debug_cfg(num_envs=1, device="cpu")
    cfg.observation.communication_radius = 1.0
    env = MultiRoverGatheringCore(cfg)
    env.positions.copy_(
        torch.tensor(
            [[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0], [6.0, 0.0, 0.0]]],
            device=env.device,
        )
    )
    env.yaws.zero_()
    env.velocities_xy.zero_()
    env.angular_velocities.zero_()

    small_radius_actor, _ = env.get_observations()
    cfg.observation.communication_radius = 2.1
    large_radius_actor, _ = env.get_observations()

    ego_end = cfg.observation.ego_dim
    neighbor_end = ego_end + cfg.observation.max_neighbors * cfg.observation.neighbor_dim
    small_neighbor = small_radius_actor[..., ego_end:neighbor_end]
    large_neighbor = large_radius_actor[..., ego_end:neighbor_end]
    small_aggregation = small_radius_actor[..., neighbor_end + cfg.observation.terrain_dim :]
    large_aggregation = large_radius_actor[..., neighbor_end + cfg.observation.terrain_dim :]

    assert env.communication_radius == 2.1
    assert torch.all(small_neighbor.reshape(1, 4, 3, 7)[..., 6] == 0.0)
    assert torch.any(large_neighbor.reshape(1, 4, 3, 7)[..., 6] == 1.0)
    assert not torch.allclose(small_aggregation, large_aggregation)


def test_zero_communication_radius_means_unlimited_visibility() -> None:
    cfg = make_debug_cfg(num_envs=1, device="cpu")
    env = MultiRoverGatheringCore(cfg)
    env.positions.copy_(
        torch.tensor(
            [[[0.0, 0.0, 0.0], [50.0, 0.0, 0.0], [0.0, 50.0, 0.0], [-50.0, 0.0, 0.0]]],
            device=env.device,
        )
    )
    env.yaws.zero_()
    env.velocities_xy.zero_()
    env.angular_velocities.zero_()

    cfg.observation.communication_radius = 1.0
    finite_radius_actor, _ = env.get_observations()
    cfg.observation.communication_radius = 0.0
    unlimited_actor, _ = env.get_observations()
    cfg.observation.communication_radius = 1.0e6
    huge_radius_actor, _ = env.get_observations()

    ego_end = cfg.observation.ego_dim
    neighbor_end = ego_end + cfg.observation.max_neighbors * cfg.observation.neighbor_dim
    finite_neighbor_mask = finite_radius_actor[..., ego_end:neighbor_end].reshape(1, 4, 3, 7)[..., 6]
    unlimited_neighbor_mask = unlimited_actor[..., ego_end:neighbor_end].reshape(1, 4, 3, 7)[..., 6]

    assert torch.all(finite_neighbor_mask == 0.0)
    assert torch.all(unlimited_neighbor_mask == 1.0)
    assert torch.allclose(unlimited_actor, huge_radius_actor)


def test_initial_state_cfg_controls_reset_distribution() -> None:
    cfg = make_debug_cfg(num_envs=1, device="cpu")
    cfg.initial_state.spawn_radius_min = 6.0
    cfg.initial_state.spawn_radius_max = 6.0
    cfg.initial_state.center_xy_range = 0.0
    cfg.initial_state.jitter_std = 0.0

    env = MultiRoverGatheringCore(cfg)
    env.reset()

    radii = torch.linalg.norm(env.positions[0, :, :2], dim=-1)
    assert torch.allclose(radii, torch.full_like(radii, 6.0), atol=1.0e-5)


def test_initial_state_curriculum_interpolates_only_when_progress_is_set() -> None:
    cfg = make_debug_cfg(num_envs=1, device="cpu")
    cfg.initial_state.curriculum_enabled = True
    cfg.initial_state.curriculum_start_spawn_radius_min = 3.0
    cfg.initial_state.curriculum_start_spawn_radius_max = 3.0
    cfg.initial_state.curriculum_start_center_xy_range = 0.0
    cfg.initial_state.curriculum_start_jitter_std = 0.0
    cfg.initial_state.spawn_radius_min = 6.0
    cfg.initial_state.spawn_radius_max = 6.0
    cfg.initial_state.center_xy_range = 0.0
    cfg.initial_state.jitter_std = 0.0
    cfg.initial_state.curriculum_warmup_timesteps = 0
    cfg.initial_state.curriculum_ramp_timesteps = 100

    env = MultiRoverGatheringCore(cfg)

    cfg.initial_state.progress_timestep_override = 0
    env.reset()
    start_radii = torch.linalg.norm(env.positions[0, :, :2], dim=-1)
    assert torch.allclose(start_radii, torch.full_like(start_radii, 3.0), atol=1.0e-5)

    cfg.initial_state.progress_timestep_override = 100
    env.reset()
    target_radii = torch.linalg.norm(env.positions[0, :, :2], dim=-1)
    assert torch.allclose(target_radii, torch.full_like(target_radii, 6.0), atol=1.0e-5)

    cfg.initial_state.progress_timestep_override = -1
    env.reset()
    eval_radii = torch.linalg.norm(env.positions[0, :, :2], dim=-1)
    assert torch.allclose(eval_radii, torch.full_like(eval_radii, 6.0), atol=1.0e-5)
