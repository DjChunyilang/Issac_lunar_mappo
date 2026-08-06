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


def _trajectory_timing(
    points: torch.Tensor,
    fractions: torch.Tensor,
    cfg: TrajectoryGeneratorCfg,
    dt: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-rover physical timestamps and reference-speed samples."""

    if dt <= 0.0:
        raise ValueError("Trajectory planning dt must be positive.")
    reference_speed_value = float(cfg.reference_speed)
    if reference_speed_value <= 0.0:
        raise ValueError("Trajectory reference speed must be positive.")
    if cfg.time_parameterization == "planning_step":
        duration = torch.full(
            points.shape[:2],
            float(dt),
            device=points.device,
            dtype=points.dtype,
        )
    elif cfg.time_parameterization == "arc_length_reference_speed":
        segment_length = torch.linalg.vector_norm(
            points[..., 1:, :2] - points[..., :-1, :2],
            dim=-1,
        )
        arc_length = segment_length.sum(dim=-1)
        duration = (arc_length / reference_speed_value).clamp_min(float(dt))
    else:
        raise ValueError(
            "Unsupported trajectory time_parameterization: "
            f"{cfg.time_parameterization}"
        )
    timestamps = fractions[None, None, :] * duration[..., None]
    reference_speed = torch.full_like(timestamps, reference_speed_value)
    return timestamps, reference_speed


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
    timestamps, reference_speed = _trajectory_timing(points, fractions, cfg, dt)
    return Trajectory(
        points=points,
        headings=headings,
        timestamps=timestamps,
        reference_speed=reference_speed,
    )


def generate_quintic_path(
    positions: torch.Tensor,
    subgoals: torch.Tensor,
    cfg: TrajectoryGeneratorCfg,
    dt: float,
    current_yaws: torch.Tensor | None = None,
) -> Trajectory:
    n_points = cfg.n_trajectory_points
    fractions = torch.linspace(0.0, 1.0, n_points, dtype=positions.dtype, device=positions.device)
    s = fractions[None, None, :, None]
    p0_xy = positions[..., :2]
    p1_xy = subgoals[..., :2]
    delta = p1_xy - p0_xy
    distance = torch.linalg.norm(delta, dim=-1, keepdim=True)
    eps = torch.finfo(positions.dtype).eps
    end_unit = delta / distance.clamp_min(eps)

    fallback = current_yaws if current_yaws is not None else torch.zeros_like(delta[..., 0])
    start_unit = torch.stack((torch.cos(fallback), torch.sin(fallback)), dim=-1)
    tangent_scale = float(cfg.quintic_tangent_scale)
    v0 = start_unit * distance * tangent_scale
    v1 = end_unit * distance * tangent_scale
    v0 = torch.where(distance > eps, v0, torch.zeros_like(v0))
    v1 = torch.where(distance > eps, v1, torch.zeros_like(v1))

    d = p1_xy - p0_xy - v0
    e = v1 - v0
    a0 = p0_xy
    a1 = v0
    a3 = 10.0 * d - 4.0 * e
    a4 = 7.0 * e - 15.0 * d
    a5 = 6.0 * d - 3.0 * e

    s2 = s * s
    s3 = s2 * s
    s4 = s3 * s
    s5 = s4 * s
    xy = (
        a0[:, :, None, :]
        + a1[:, :, None, :] * s
        + a3[:, :, None, :] * s3
        + a4[:, :, None, :] * s4
        + a5[:, :, None, :] * s5
    )
    derivative = (
        a1[:, :, None, :]
        + 3.0 * a3[:, :, None, :] * s2
        + 4.0 * a4[:, :, None, :] * s3
        + 5.0 * a5[:, :, None, :] * s4
    )
    z = positions[..., 2:3][:, :, None, :] + fractions[
        None, None, :, None
    ] * (subgoals[..., 2:3] - positions[..., 2:3])[:, :, None, :]
    points = torch.cat((xy, z), dim=-1)
    headings = heading_from_delta(
        derivative,
        fallback=fallback[:, :, None].expand(-1, -1, n_points),
    )
    timestamps, reference_speed = _trajectory_timing(points, fractions, cfg, dt)
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
    if cfg.geometry_method == "line":
        return generate_line_path(positions, subgoals, cfg, dt, current_yaws=current_yaws)
    if cfg.geometry_method == "quintic":
        return generate_quintic_path(positions, subgoals, cfg, dt, current_yaws=current_yaws)
    raise ValueError(f"Unsupported first-stage geometry_method: {cfg.geometry_method}")
