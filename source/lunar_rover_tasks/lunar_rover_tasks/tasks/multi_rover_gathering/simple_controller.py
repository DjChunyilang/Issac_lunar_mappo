"""Simplified velocity tracking controller."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from lunar_rover_tasks.tasks.multi_rover_gathering.gathering_env_cfg import LowLevelControlCfg
from lunar_rover_tasks.tasks.multi_rover_gathering.trajectory_generator import Trajectory
from lunar_rover_tasks.utils.math_utils import wrap_to_pi


@dataclass(slots=True)
class ControlCommand:
    linear: torch.Tensor
    angular: torch.Tensor

    @property
    def packed(self) -> torch.Tensor:
        return torch.stack((self.linear, self.angular), dim=-1)


def select_tracking_point(trajectory: Trajectory, index: int = 1) -> torch.Tensor:
    index = min(index, trajectory.points.shape[-2] - 1)
    return trajectory.points[:, :, index, :]


def compute_control(
    positions: torch.Tensor,
    yaws: torch.Tensor,
    trajectory: Trajectory,
    cfg: LowLevelControlCfg,
) -> ControlCommand:
    target = select_tracking_point(trajectory)
    delta = target[..., :2] - positions[..., :2]
    distance = torch.linalg.norm(delta, dim=-1)
    desired_yaw = torch.atan2(delta[..., 1], delta[..., 0])
    heading_error = wrap_to_pi(desired_yaw - yaws)
    linear = torch.clamp(cfg.k_linear * distance, 0.0, cfg.max_linear_speed)
    angular = torch.clamp(
        cfg.k_angular * heading_error,
        -cfg.max_angular_speed,
        cfg.max_angular_speed,
    )
    return ControlCommand(linear=linear, angular=angular)

