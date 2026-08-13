"""Deterministic first-stage trajectory generator."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from lunar_rover_tasks.tasks.multi_rover_gathering.gathering_env_cfg import TrajectoryGeneratorCfg
from lunar_rover_tasks.tasks.multi_rover_gathering.action_interpreter import (
    PRIMITIVE_HOLD,
    PRIMITIVE_REVERSE,
    PRIMITIVE_SPIN,
    PRIMITIVE_YIELD,
)
from lunar_rover_tasks.utils.math_utils import heading_from_delta, wrap_to_pi


@dataclass(slots=True)
class Trajectory:
    points: torch.Tensor
    headings: torch.Tensor
    timestamps: torch.Tensor
    reference_speed: torch.Tensor
    motion_direction: torch.Tensor | None = None
    planned_yaw_delta: torch.Tensor | None = None
    primitive_type: torch.Tensor | None = None

    @property
    def packed(self) -> torch.Tensor:
        motion_direction = (
            self.motion_direction
            if self.motion_direction is not None
            else torch.sign(self.reference_speed)
        )
        planned_yaw_delta = (
            self.planned_yaw_delta
            if self.planned_yaw_delta is not None
            else torch.zeros_like(self.reference_speed)
        )
        primitive_type = (
            self.primitive_type.to(dtype=self.points.dtype)
            if self.primitive_type is not None
            else torch.zeros_like(self.reference_speed)
        )
        return torch.cat(
            (
                self.points,
                self.headings.unsqueeze(-1),
                self.timestamps.unsqueeze(-1),
                self.reference_speed.unsqueeze(-1),
                motion_direction.unsqueeze(-1),
                planned_yaw_delta.unsqueeze(-1),
                primitive_type.unsqueeze(-1),
            ),
            dim=-1,
        )


def _trajectory_timing(
    points: torch.Tensor,
    fractions: torch.Tensor,
    cfg: TrajectoryGeneratorCfg,
    dt: float,
    reference_speed_override: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-rover physical timestamps and reference-speed samples."""

    if dt <= 0.0:
        raise ValueError("Trajectory planning dt must be positive.")
    reference_speed_value = float(cfg.reference_speed)
    if reference_speed_value <= 0.0 and reference_speed_override is None:
        raise ValueError("Trajectory reference speed must be positive.")
    if reference_speed_override is None:
        speed = torch.full(
            points.shape[:2],
            reference_speed_value,
            device=points.device,
            dtype=points.dtype,
        )
    else:
        if reference_speed_override.shape != points.shape[:2]:
            raise ValueError(
                "reference_speed_override must match trajectory batch/agent dimensions."
            )
        speed = reference_speed_override.to(device=points.device, dtype=points.dtype)
    speed_magnitude = speed.abs()
    safe_speed = speed_magnitude.clamp_min(torch.finfo(points.dtype).eps)
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
        duration = torch.where(
            speed_magnitude > 0.0,
            (arc_length / safe_speed).clamp_min(float(dt)),
            torch.full_like(arc_length, float(dt)),
        )
    else:
        raise ValueError(
            "Unsupported trajectory time_parameterization: "
            f"{cfg.time_parameterization}"
        )
    timestamps = fractions[None, None, :] * duration[..., None]
    reference_speed = speed[..., None].expand_as(timestamps)
    return timestamps, reference_speed


def generate_line_path(
    positions: torch.Tensor,
    subgoals: torch.Tensor,
    cfg: TrajectoryGeneratorCfg,
    dt: float,
    current_yaws: torch.Tensor | None = None,
    reference_speed: torch.Tensor | None = None,
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
    timestamps, reference_speed = _trajectory_timing(
        points, fractions, cfg, dt, reference_speed
    )
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
    reference_speed: torch.Tensor | None = None,
    motion_direction: torch.Tensor | None = None,
    planned_yaw_delta: torch.Tensor | None = None,
    primitive_type: torch.Tensor | None = None,
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
    if motion_direction is None:
        motion_direction = torch.ones_like(fallback)
    if planned_yaw_delta is None:
        planned_yaw_delta = torch.zeros_like(fallback)
    if primitive_type is None:
        primitive_type = torch.ones_like(fallback, dtype=torch.long)
    reverse = primitive_type == PRIMITIVE_REVERSE
    yielding = primitive_type == PRIMITIVE_YIELD
    path_start_unit = torch.where(
        reverse.unsqueeze(-1),
        -start_unit,
        start_unit,
    )
    path_end_unit = torch.where(yielding.unsqueeze(-1), start_unit, end_unit)
    tangent_scale = float(cfg.quintic_tangent_scale)
    v0 = path_start_unit * distance * tangent_scale
    v1 = path_end_unit * distance * tangent_scale
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
    headings = torch.where(
        reverse[..., None],
        wrap_to_pi(headings + torch.pi),
        headings,
    )
    timestamps, reference_speed = _trajectory_timing(
        points, fractions, cfg, dt, reference_speed
    )
    return Trajectory(
        points=points,
        headings=headings,
        timestamps=timestamps,
        reference_speed=reference_speed,
        motion_direction=motion_direction[..., None].expand_as(reference_speed),
        planned_yaw_delta=planned_yaw_delta[..., None].expand_as(reference_speed),
        primitive_type=primitive_type[..., None].expand_as(reference_speed),
    )


def generate_stationary_pose_trajectory(
    positions: torch.Tensor,
    current_yaws: torch.Tensor,
    cfg: TrajectoryGeneratorCfg,
    dt: float,
    planned_yaw_delta: torch.Tensor,
    primitive_type: torch.Tensor,
) -> Trajectory:
    """Generate true hold or zero-radius differential-drive rotation."""

    n_points = cfg.n_trajectory_points
    fractions = torch.linspace(
        0.0,
        1.0,
        n_points,
        dtype=positions.dtype,
        device=positions.device,
    )
    points = positions[:, :, None, :].expand(-1, -1, n_points, -1).clone()
    headings = wrap_to_pi(
        current_yaws[..., None] + planned_yaw_delta[..., None] * fractions
    )
    spin = primitive_type == PRIMITIVE_SPIN
    duration = torch.where(
        spin,
        (
            planned_yaw_delta.abs()
            / max(float(cfg.rotation_reference_speed), 1.0e-6)
        ).clamp_min(float(dt)),
        torch.full_like(planned_yaw_delta, float(dt)),
    )
    timestamps = fractions[None, None, :] * duration[..., None]
    zeros = torch.zeros_like(timestamps)
    return Trajectory(
        points=points,
        headings=headings,
        timestamps=timestamps,
        reference_speed=zeros,
        motion_direction=zeros,
        planned_yaw_delta=planned_yaw_delta[..., None].expand_as(zeros),
        primitive_type=primitive_type[..., None].expand_as(zeros),
    )


def generate_trajectory(
    positions: torch.Tensor,
    subgoals: torch.Tensor,
    cfg: TrajectoryGeneratorCfg,
    dt: float,
    current_yaws: torch.Tensor | None = None,
    reference_speed: torch.Tensor | None = None,
    motion_direction: torch.Tensor | None = None,
    planned_yaw_delta: torch.Tensor | None = None,
    primitive_type: torch.Tensor | None = None,
) -> Trajectory:
    if primitive_type is not None:
        if current_yaws is None or planned_yaw_delta is None:
            raise ValueError(
                "Differential primitives require current_yaws and planned_yaw_delta."
            )
        stationary = (primitive_type == PRIMITIVE_HOLD) | (
            primitive_type == PRIMITIVE_SPIN
        )
        if bool(stationary.all()):
            return generate_stationary_pose_trajectory(
                positions,
                current_yaws,
                cfg,
                dt,
                planned_yaw_delta,
                primitive_type,
            )
        if bool(stationary.any()):
            moving_type = torch.where(
                stationary,
                torch.ones_like(primitive_type),
                primitive_type,
            )
            moving_speed = (
                torch.where(stationary, torch.ones_like(reference_speed), reference_speed)
                if reference_speed is not None
                else None
            )
            moving_direction = (
                torch.where(stationary, torch.ones_like(motion_direction), motion_direction)
                if motion_direction is not None
                else None
            )
            moving = generate_quintic_path(
                positions,
                subgoals,
                cfg,
                dt,
                current_yaws=current_yaws,
                reference_speed=moving_speed,
                motion_direction=moving_direction,
                planned_yaw_delta=planned_yaw_delta,
                primitive_type=moving_type,
            )
            stopped = generate_stationary_pose_trajectory(
                positions,
                current_yaws,
                cfg,
                dt,
                planned_yaw_delta,
                primitive_type,
            )
            mask_points = stationary[..., None, None]
            mask_samples = stationary[..., None]
            return Trajectory(
                points=torch.where(mask_points, stopped.points, moving.points),
                headings=torch.where(mask_samples, stopped.headings, moving.headings),
                timestamps=torch.where(mask_samples, stopped.timestamps, moving.timestamps),
                reference_speed=torch.where(
                    mask_samples,
                    stopped.reference_speed,
                    moving.reference_speed,
                ),
                motion_direction=torch.where(
                    mask_samples,
                    stopped.motion_direction,
                    moving.motion_direction,
                ),
                planned_yaw_delta=torch.where(
                    mask_samples,
                    stopped.planned_yaw_delta,
                    moving.planned_yaw_delta,
                ),
                primitive_type=torch.where(
                    mask_samples,
                    stopped.primitive_type,
                    moving.primitive_type,
                ),
            )
    if cfg.geometry_method == "line":
        if primitive_type is not None:
            raise ValueError(
                "Differential trajectory primitives require quintic geometry."
            )
        return generate_line_path(
            positions,
            subgoals,
            cfg,
            dt,
            current_yaws=current_yaws,
            reference_speed=reference_speed,
        )
    if cfg.geometry_method == "quintic":
        return generate_quintic_path(
            positions,
            subgoals,
            cfg,
            dt,
            current_yaws=current_yaws,
            reference_speed=reference_speed,
            motion_direction=motion_direction,
            planned_yaw_delta=planned_yaw_delta,
            primitive_type=primitive_type,
        )
    raise ValueError(f"Unsupported first-stage geometry_method: {cfg.geometry_method}")
