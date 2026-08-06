from __future__ import annotations

import torch

from lunar_rover_tasks.tasks.multi_rover_gathering.gathering_env_cfg import TrajectoryGeneratorCfg
from lunar_rover_tasks.tasks.multi_rover_gathering.trajectory_generator import generate_trajectory


def test_trajectory_shape_and_timestamps() -> None:
    cfg = TrajectoryGeneratorCfg(n_trajectory_points=6)
    positions = torch.zeros(2, 4, 3)
    subgoals = torch.ones(2, 4, 3)
    trajectory = generate_trajectory(positions, subgoals, cfg, dt=0.2)
    assert trajectory.packed.shape == (2, 4, 6, 6)
    assert torch.all(trajectory.timestamps[..., 1:] >= trajectory.timestamps[..., :-1])
    assert torch.allclose(trajectory.points[:, :, 0], positions)
    assert torch.allclose(trajectory.points[:, :, -1], subgoals)


def test_line_and_quintic_generation_share_interface() -> None:
    positions = torch.zeros(2, 4, 3)
    subgoals = torch.ones(2, 4, 3)
    current_yaws = torch.zeros(2, 4)

    for method in ("line", "quintic"):
        cfg = TrajectoryGeneratorCfg(n_trajectory_points=8, geometry_method=method)
        trajectory = generate_trajectory(
            positions,
            subgoals,
            cfg,
            dt=0.2,
            current_yaws=current_yaws,
        )

        assert trajectory.packed.shape == (2, 4, 8, 6)
        assert torch.isfinite(trajectory.packed).all()
        assert torch.all(trajectory.timestamps[..., 1:] >= trajectory.timestamps[..., :-1])
        assert torch.allclose(trajectory.points[:, :, 0], positions, atol=1.0e-6)
        assert torch.allclose(trajectory.points[:, :, -1], subgoals, atol=1.0e-6)


def test_quintic_path_hits_endpoint_and_heading_constraints() -> None:
    cfg = TrajectoryGeneratorCfg(
        n_trajectory_points=9,
        geometry_method="quintic",
        quintic_tangent_scale=0.5,
    )
    positions = torch.tensor([[[0.0, 0.0, 0.0]]], dtype=torch.float32)
    subgoals = torch.tensor([[[1.0, 1.0, 0.2]]], dtype=torch.float32)
    current_yaws = torch.zeros(1, 1)

    trajectory = generate_trajectory(
        positions,
        subgoals,
        cfg,
        dt=0.2,
        current_yaws=current_yaws,
    )

    assert torch.allclose(trajectory.points[:, :, 0], positions, atol=1.0e-6)
    assert torch.allclose(trajectory.points[:, :, -1], subgoals, atol=1.0e-6)
    assert torch.allclose(trajectory.headings[:, :, 0], current_yaws, atol=1.0e-4)
    expected_end_heading = torch.atan2(
        subgoals[..., 1] - positions[..., 1],
        subgoals[..., 0] - positions[..., 0],
    )
    assert torch.allclose(trajectory.headings[:, :, -1], expected_end_heading, atol=1.0e-4)


def test_arc_length_timing_preserves_geometry_and_uses_reference_speed() -> None:
    positions = torch.zeros(1, 1, 3)
    subgoals = torch.tensor([[[1.0, 0.0, 0.0]]])
    legacy_cfg = TrajectoryGeneratorCfg(
        n_trajectory_points=6,
        geometry_method="quintic",
        reference_speed=0.5,
        time_parameterization="planning_step",
    )
    physical_cfg = TrajectoryGeneratorCfg(
        n_trajectory_points=6,
        geometry_method="quintic",
        reference_speed=0.5,
        time_parameterization="arc_length_reference_speed",
    )
    legacy = generate_trajectory(positions, subgoals, legacy_cfg, dt=0.2)
    physical = generate_trajectory(positions, subgoals, physical_cfg, dt=0.2)

    assert torch.equal(legacy.points, physical.points)
    arc_length = torch.linalg.vector_norm(
        physical.points[..., 1:, :2] - physical.points[..., :-1, :2], dim=-1
    ).sum(dim=-1)
    assert torch.allclose(physical.timestamps[..., -1], arc_length / 0.5)
    assert torch.allclose(legacy.timestamps[..., -1], torch.tensor([[0.2]]))


def test_arc_length_timing_gives_zero_path_one_planning_step() -> None:
    cfg = TrajectoryGeneratorCfg(
        geometry_method="quintic",
        time_parameterization="arc_length_reference_speed",
    )
    positions = torch.zeros(2, 4, 3)
    trajectory = generate_trajectory(positions, positions.clone(), cfg, dt=0.2)
    assert torch.isfinite(trajectory.packed).all()
    assert torch.allclose(trajectory.timestamps[..., -1], torch.full((2, 4), 0.2))
