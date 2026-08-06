from __future__ import annotations

import torch

from lunar_rover_tasks.tasks.multi_rover_gathering.action_interpreter import decode_action
from lunar_rover_tasks.tasks.multi_rover_gathering.gathering_env_cfg import make_debug_cfg
from lunar_rover_tasks.tasks.multi_rover_gathering.simple_controller import (
    compute_control,
    interpolate_trajectory_point,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.trajectory_generator import (
    Trajectory,
    generate_trajectory,
)


def test_local_subgoal_trajectory_and_velocity_control_chain() -> None:
    cfg = make_debug_cfg(num_envs=1, device="cpu")
    cfg.trajectory_generator.n_trajectory_points = 5
    positions = torch.tensor(
        [[[-3.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]]],
        dtype=torch.float32,
    )
    yaws = torch.zeros(1, cfg.task.n_agents)
    normalized_action = torch.tensor(
        [[[1.0, 0.0], [1.0, 0.5], [1.0, -0.5], [-1.0, 0.0]]],
        dtype=torch.float32,
    )

    decoded = decode_action(normalized_action, positions, yaws, cfg.planner)
    trajectory = generate_trajectory(
        positions,
        decoded.world_subgoal,
        cfg.trajectory_generator,
        cfg.simulation.planning_dt,
        current_yaws=yaws,
    )
    control = compute_control(positions, yaws, trajectory, cfg.low_level_control)

    assert trajectory.packed.shape == (1, cfg.task.n_agents, 5, 6)
    assert torch.allclose(trajectory.points[:, :, 0], positions)
    assert torch.allclose(trajectory.points[:, :, -1], decoded.world_subgoal)
    assert torch.all(trajectory.timestamps[..., 1:] >= trajectory.timestamps[..., :-1])

    assert torch.all(control.linear >= 0.0)
    assert torch.all(control.linear <= cfg.low_level_control.max_linear_speed)
    assert torch.all(control.angular.abs() <= cfg.low_level_control.max_angular_speed)
    assert torch.isclose(control.angular[0, 0], torch.tensor(0.0), atol=1.0e-6)
    assert control.angular[0, 1] > 0.0
    assert control.angular[0, 2] < 0.0
    assert torch.isclose(control.linear[0, 3], torch.tensor(0.0), atol=1.0e-6)


def test_planning_time_tracking_interpolates_at_physical_lookahead() -> None:
    points = torch.tensor([[[[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0]]]])
    timestamps = torch.tensor([[[0.0, 0.5, 1.0]]])
    trajectory = Trajectory(
        points=points,
        headings=torch.zeros(1, 1, 3),
        timestamps=timestamps,
        reference_speed=torch.ones(1, 1, 3),
    )
    target = interpolate_trajectory_point(trajectory, 0.2)
    assert torch.allclose(target, torch.tensor([[[0.2, 0.0, 0.0]]]))


def test_planning_time_control_requires_explicit_planning_dt() -> None:
    cfg = make_debug_cfg(num_envs=1, device="cpu")
    cfg.low_level_control.tracking_point_mode = "planning_time"
    positions = torch.zeros(1, cfg.task.n_agents, 3)
    yaws = torch.zeros(1, cfg.task.n_agents)
    trajectory = generate_trajectory(
        positions,
        positions + torch.tensor([1.0, 0.0, 0.0]),
        cfg.trajectory_generator,
        cfg.simulation.planning_dt,
        current_yaws=yaws,
    )
    try:
        compute_control(positions, yaws, trajectory, cfg.low_level_control)
    except ValueError as error:
        assert "positive planning_dt" in str(error)
    else:
        raise AssertionError("planning_time tracking accepted a missing planning_dt")
