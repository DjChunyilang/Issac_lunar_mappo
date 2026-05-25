"""Placeholder visualization utilities for first-stage trajectory debugging."""

from __future__ import annotations

import torch


def trajectory_to_xy_list(trajectory: torch.Tensor) -> list[list[tuple[float, float]]]:
    data = trajectory.detach().cpu()
    return [
        [(float(point[0]), float(point[1])) for point in agent_traj]
        for agent_traj in data.reshape(-1, data.shape[-2], data.shape[-1])
    ]

