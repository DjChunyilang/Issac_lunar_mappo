"""Simplified velocity tracking controller."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from lunar_rover_tasks.tasks.multi_rover_gathering.gathering_env_cfg import (
    LowLevelControlCfg,
    SuccessThresholdsCfg,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.metrics import TeamMetrics
from lunar_rover_tasks.tasks.multi_rover_gathering.trajectory_generator import Trajectory
from lunar_rover_tasks.utils.math_utils import wrap_to_pi


@dataclass(slots=True)
class ControlCommand:
    linear: torch.Tensor
    angular: torch.Tensor

    @property
    def packed(self) -> torch.Tensor:
        return torch.stack((self.linear, self.angular), dim=-1)


@dataclass(slots=True)
class ControlSafetyProjection:
    control: ControlCommand
    info: dict[str, Any]


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


def _empty_control_safety_info(
    control: ControlCommand,
    *,
    enabled: bool,
) -> dict[str, Any]:
    ones = torch.ones_like(control.linear)
    zeros = torch.zeros_like(control.linear)
    env_zeros = torch.zeros(control.linear.shape[0], device=control.linear.device)
    return {
        "enabled": enabled,
        "linear_scale": ones,
        "raw_linear": control.linear,
        "projected_linear": control.linear,
        "applied": torch.zeros_like(control.linear, dtype=torch.bool),
        "pairwise_risk": zeros,
        "predicted_nearest_distance": zeros,
        "success_zone_active": torch.zeros_like(env_zeros, dtype=torch.bool),
        "linear_scale_mean": float(1.0),
        "linear_scale_min": float(1.0),
        "applied_fraction": float(0.0),
        "pairwise_risk_mean": float(0.0),
        "success_zone_fraction": float(0.0),
    }


def apply_control_safety_projection(
    control: ControlCommand,
    positions: torch.Tensor,
    yaws: torch.Tensor,
    metrics: TeamMetrics,
    cfg: LowLevelControlCfg,
    thresholds: SuccessThresholdsCfg,
    planning_dt: float,
    *,
    communication_radius: float | None = None,
) -> ControlSafetyProjection:
    """Damp one-step control commands that would close unsafe pairwise gaps.

    This is a low-level execution guard for the proxy model. It does not change
    the actor action, subgoal, trajectory, or reward target; it only scales the
    linear speed that will be integrated in the next proxy step.
    """

    enabled = bool(cfg.safety_projection_enabled or cfg.success_zone_damping_enabled)
    if not enabled:
        return ControlSafetyProjection(
            control=control,
            info=_empty_control_safety_info(control, enabled=False),
        )

    linear_scale = torch.ones_like(control.linear)
    pairwise_risk = torch.zeros_like(control.linear)
    predicted_nearest = torch.zeros_like(control.linear)
    activation = float(cfg.projection_activation_distance)
    stop = float(cfg.projection_stop_distance)
    horizon = max(float(cfg.projection_horizon_s), float(planning_dt))
    eps = torch.finfo(control.linear.dtype).eps

    if bool(cfg.safety_projection_enabled) and activation > 0.0 and activation > stop:
        yaw_after = wrap_to_pi(yaws + control.angular * float(planning_dt))
        direction = torch.stack((torch.cos(yaw_after), torch.sin(yaw_after)), dim=-1)
        desired_velocity = direction * control.linear.unsqueeze(-1)
        delta = positions[:, :, None, :2] - positions[:, None, :, :2]
        distance = torch.linalg.norm(delta, dim=-1)
        relative_velocity = desired_velocity[:, :, None, :] - desired_velocity[:, None, :, :]
        predicted_delta = delta + relative_velocity * horizon
        predicted_distance = torch.linalg.norm(predicted_delta, dim=-1)
        closing = (delta * relative_velocity).sum(dim=-1) < 0.0

        n_agents = positions.shape[1]
        eye = torch.eye(n_agents, dtype=torch.bool, device=positions.device).unsqueeze(0)
        visible = ~eye
        if communication_radius is not None and communication_radius > 0.0:
            visible = visible & (distance <= float(communication_radius))

        denom = max(activation - stop, 1.0e-6)
        predicted_violation = ((activation - predicted_distance) / denom).clamp(0.0, 1.0)
        current_violation = ((activation - distance) / denom).clamp(0.0, 1.0)
        nonclosing_near_threshold = (
            activation if bool(cfg.projection_damp_nonclosing_near) else stop
        )
        closing_or_too_close = closing | (distance < nonclosing_near_threshold)
        pairwise_violation = torch.where(
            closing_or_too_close,
            torch.maximum(predicted_violation, current_violation),
            torch.zeros_like(predicted_violation),
        )
        pairwise_violation = pairwise_violation.masked_fill(~visible, 0.0)
        if bool(cfg.projection_directional_agent_scale):
            # Attribute a pairwise closing risk to the rover that is actually
            # moving inward toward its neighbor.  The previous symmetric
            # projection slowed both rovers in a closing pair; that is safe but
            # can stall the rover that is already escaping the near contact.
            safe_distance = distance.clamp_min(1.0e-6)
            direction_to_neighbor = -delta / safe_distance.unsqueeze(-1)
            inward_speed = (desired_velocity[:, :, None, :] * direction_to_neighbor).sum(dim=-1)
            if str(cfg.projection_directional_agent_scale_mode) == "mask":
                inward_fraction = (inward_speed > 1.0e-6).to(pairwise_violation.dtype)
            else:
                inward_fraction = (
                    inward_speed.clamp_min(0.0) / max(float(cfg.max_linear_speed), 1.0e-6)
                ).clamp(0.0, 1.0)
            pairwise_violation = pairwise_violation * inward_fraction
        pairwise_risk = pairwise_violation.amax(dim=-1)

        predicted_nearest_raw = predicted_distance.masked_fill(~visible, float("inf")).amin(dim=-1)
        predicted_nearest = torch.where(
            torch.isinf(predicted_nearest_raw),
            torch.zeros_like(predicted_nearest_raw),
            predicted_nearest_raw,
        )
        projection_scale = 1.0 - float(cfg.projection_strength) * pairwise_risk
        projection_scale = projection_scale.clamp(
            min=float(cfg.projection_min_linear_scale),
            max=1.0,
        )
        linear_scale = torch.minimum(linear_scale, projection_scale)

    success_zone_active = torch.zeros(positions.shape[0], dtype=torch.bool, device=positions.device)
    if bool(cfg.success_zone_damping_enabled):
        dmax_limit = float(thresholds.dmax) * float(cfg.success_zone_dmax_multiplier)
        dispersion_limit = float(thresholds.dispersion) * float(
            cfg.success_zone_dispersion_multiplier
        )
        success_zone_active = (metrics.dmax <= dmax_limit) & (
            metrics.dispersion <= dispersion_limit
        )
        if float(thresholds.min_pairwise_distance) > 0.0:
            nearest = metrics.nearest_neighbor_distance.amin(dim=-1)
            success_zone_active = success_zone_active & (
                nearest >= float(thresholds.min_pairwise_distance)
            )
        damping_scale = torch.where(
            success_zone_active[:, None],
            torch.full_like(linear_scale, float(cfg.success_zone_linear_scale)),
            torch.ones_like(linear_scale),
        )
        linear_scale = torch.minimum(linear_scale, damping_scale)

    linear_scale = torch.nan_to_num(linear_scale, nan=1.0, posinf=1.0, neginf=0.0).clamp(
        min=0.0,
        max=1.0,
    )
    projected = ControlCommand(
        linear=control.linear * linear_scale,
        angular=control.angular,
    )
    applied = linear_scale < (1.0 - 10.0 * eps)
    info = {
        "enabled": True,
        "linear_scale": linear_scale,
        "raw_linear": control.linear,
        "projected_linear": projected.linear,
        "applied": applied,
        "pairwise_risk": pairwise_risk,
        "predicted_nearest_distance": predicted_nearest,
        "success_zone_active": success_zone_active,
        "linear_scale_mean": float(linear_scale.detach().float().mean().cpu()),
        "linear_scale_min": float(linear_scale.detach().float().amin().cpu()),
        "applied_fraction": float(applied.detach().float().mean().cpu()),
        "pairwise_risk_mean": float(pairwise_risk.detach().float().mean().cpu()),
        "success_zone_fraction": float(success_zone_active.detach().float().mean().cpu()),
    }
    return ControlSafetyProjection(control=projected, info=info)
