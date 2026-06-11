from __future__ import annotations

import torch

from lunar_rover_tasks.tasks.multi_rover_gathering.gathering_env import MultiRoverGatheringCore
from lunar_rover_tasks.tasks.multi_rover_gathering.gathering_env_cfg import make_debug_cfg
from lunar_rover_tasks.tasks.multi_rover_gathering.metrics import compute_team_metrics
from lunar_rover_tasks.tasks.multi_rover_gathering.oracle import (
    compute_geometric_median,
    compute_mean_oracle_distance,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.simple_controller import ControlCommand
from lunar_rover_tasks.tasks.multi_rover_gathering.terrain_features import query_height, query_terrain_features


def _set_square_state(env: MultiRoverGatheringCore) -> None:
    positions = torch.tensor(
        [[[-3.0, -3.0, 0.0], [-3.0, 3.0, 0.0], [3.0, -3.0, 0.0], [3.0, 3.0, 0.0]]],
        device=env.device,
    )
    env.positions.copy_(positions)
    env.yaws.zero_()
    env.velocities_xy.zero_()
    env.angular_velocities.zero_()
    env.previous_physical_action.zero_()
    env.step_count.zero_()
    env.success_hold_count.zero_()
    env.oracle_point.copy_(compute_geometric_median(env.positions))
    env.metrics = compute_team_metrics(env.positions, env.velocities_xy)
    env.prev_metrics = env.metrics
    env.prev_mean_oracle_distance = compute_mean_oracle_distance(
        env.positions,
        env.oracle_point,
    )


def test_proxy_rover_integrates_simplified_velocity_model() -> None:
    cfg = make_debug_cfg(num_envs=1, device="cpu")
    env = MultiRoverGatheringCore(cfg)
    _set_square_state(env)

    old_positions = env.positions.clone()
    old_yaws = env.yaws.clone()
    action = torch.zeros(1, cfg.task.n_agents, cfg.planner.action_dim, device=env.device)
    output = env.step(action)
    control = output.info["control"]

    assert int(env.step_count.item()) == 1
    assert torch.isfinite(output.actor_obs).all()
    assert torch.isfinite(output.critic_state).all()
    assert torch.isfinite(output.rewards).all()
    assert torch.allclose(env.positions[..., 2], torch.zeros_like(env.positions[..., 2]))

    assert torch.all(control.linear > 0.0)
    assert torch.all(control.linear <= cfg.low_level_control.max_linear_speed)
    assert torch.allclose(control.angular, torch.zeros_like(control.angular), atol=1.0e-6)
    assert torch.allclose(env.yaws, old_yaws, atol=1.0e-6)

    expected_delta_x = control.linear * cfg.simulation.planning_dt
    actual_delta_xy = env.positions[..., :2] - old_positions[..., :2]
    assert torch.allclose(actual_delta_xy[..., 0], expected_delta_x, atol=1.0e-6)
    assert torch.allclose(actual_delta_xy[..., 1], torch.zeros_like(expected_delta_x), atol=1.0e-6)
    assert torch.allclose(env.velocities_xy[..., 0], control.linear, atol=1.0e-6)
    assert torch.allclose(env.velocities_xy[..., 1], torch.zeros_like(control.linear), atol=1.0e-6)


def test_terrain_dynamics_reset_places_rovers_on_heightfield() -> None:
    cfg = make_debug_cfg(num_envs=2, device="cpu")
    cfg.terrain.type = "lunar_crater_proxy"
    cfg.terrain.dynamics_enabled = True
    cfg.terrain.amplitude = 0.05
    cfg.terrain.crater_count = 1
    cfg.terrain.crater_min_radius = 1.0
    cfg.terrain.crater_max_radius = 1.0

    env = MultiRoverGatheringCore(cfg)
    expected_z = query_height(env.positions[..., :2], cfg.terrain).squeeze(-1)

    assert torch.isfinite(env.positions).all()
    assert torch.allclose(env.positions[..., 2], expected_z, atol=1.0e-6)
    assert not torch.allclose(env.positions[..., 2], torch.zeros_like(env.positions[..., 2]))


def test_terrain_dynamics_updates_z_and_reduces_speed_on_slope() -> None:
    cfg = make_debug_cfg(num_envs=1, device="cpu")
    cfg.terrain.type = "lunar_heightfield_proxy"
    cfg.terrain.dynamics_enabled = True
    cfg.terrain.amplitude = 0.30
    cfg.terrain.wavelength = 3.0
    cfg.terrain.slope_speed_scale = 2.0
    cfg.terrain.min_speed_scale = 0.05
    env = MultiRoverGatheringCore(cfg)
    xy = torch.tensor(
        [[[0.25, 0.40], [1.50, -0.75], [-1.00, 2.00], [2.20, 1.10]]],
        dtype=torch.float32,
        device=env.device,
    )
    env.positions[..., :2] = xy
    env.positions[..., 2] = query_height(xy, cfg.terrain).squeeze(-1)
    env.last_terrain_features = query_terrain_features(xy, cfg.terrain)
    env.yaws.zero_()
    old_positions = env.positions.clone()

    control = ControlCommand(
        linear=torch.ones(1, cfg.task.n_agents, device=env.device),
        angular=torch.zeros(1, cfg.task.n_agents, device=env.device),
    )
    env._integrate(control)

    expected_unscaled_dx = cfg.simulation.planning_dt
    actual_dx = env.positions[..., 0] - old_positions[..., 0]
    expected_z = query_height(env.positions[..., :2], cfg.terrain).squeeze(-1)
    assert torch.all(env.last_terrain_speed_scale < 1.0)
    assert torch.all(actual_dx < expected_unscaled_dx)
    assert torch.allclose(env.positions[..., 2], expected_z, atol=1.0e-6)
    assert torch.isfinite(env.last_height_delta).all()
