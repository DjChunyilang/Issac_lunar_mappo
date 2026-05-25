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

