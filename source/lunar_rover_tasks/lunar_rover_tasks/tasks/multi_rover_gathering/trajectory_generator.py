"""Deterministic first-stage trajectory generator."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from lunar_rover_tasks.tasks.multi_rover_gathering.gathering_env_cfg import TrajectoryGeneratorCfg
from lunar_rover_tasks.utils.math_utils import heading_from_delta


@dataclass(slots=True)
class Trajectory:
    points: torch.Tensor
    headings: torch.Tensor
    timestamps: torch.Tensor
    reference_speed: torch.Tensor

    @property
    def packed(self) -> torch.Tensor:
        return torch.cat(
            (
                self.points,
                self.headings.unsqueeze(-1),
                self.timestamps.unsqueeze(-1),
                self.reference_speed.unsqueeze(-1),
            ),
            dim=-1,
        )


def generate_line_path(
    positions: torch.Tensor,
    subgoals: torch.Tensor,
    cfg: TrajectoryGeneratorCfg,
    dt: float,
    current_yaws: torch.Tensor | None = None,
) -> Trajectory:
    n_points = cfg.n_trajectory_points
    fractions = torch.linspace(0.0, 1.0, n_points, dtype=positions.dtype, device=positions.device)
    points = positions[:, :, None, :] + fractions[None, None, :, None] * (
        subgoals - positions
    )[:, :, None, :]
    delta = subgoals[:, :, :2] - positions[:, :, :2]
    fallback = current_yaws if current_yaws is not None else torch.zeros_like(delta[..., 0])
    heading = heading_from_delta(delta, fallback=fallback)
    headings = heading[:, :, None].expand(-1, -1, n_points)
    timestamps = fractions[None, None, :].expand(positions.shape[0], positions.shape[1], -1) * dt
    reference_speed = torch.full_like(timestamps, cfg.reference_speed)
    return Trajectory(
        points=points,
        headings=headings,
        timestamps=timestamps,
        reference_speed=reference_speed,
    )


def generate_trajectory(
    positions: torch.Tensor,
    subgoals: torch.Tensor,
    cfg: TrajectoryGeneratorCfg,
    dt: float,
    current_yaws: torch.Tensor | None = None,
) -> Trajectory:
    if cfg.geometry_method != "line":
        raise ValueError(f"Unsupported first-stage geometry_method: {cfg.geometry_method}")
    return generate_line_path(positions, subgoals, cfg, dt, current_yaws=current_yaws)

