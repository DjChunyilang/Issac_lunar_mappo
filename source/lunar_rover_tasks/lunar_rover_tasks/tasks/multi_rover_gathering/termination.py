"""Success and failure predicates."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from lunar_rover_tasks.tasks.multi_rover_gathering.gathering_env_cfg import (
    SafetyCfg,
    SuccessThresholdsCfg,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.metrics import TeamMetrics
from lunar_rover_tasks.utils.geometry_utils import pairwise_distances_xy


@dataclass(slots=True)
class DoneFlags:
    success: torch.Tensor
    collision: torch.Tensor
    out_of_bounds: torch.Tensor
    timeout: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    done: torch.Tensor


@dataclass(slots=True)
class SuccessGateDiagnostics:
    dmax_ok: torch.Tensor
    dispersion_ok: torch.Tensor
    speed_ok: torch.Tensor
    instant_success: torch.Tensor


def compute_success_gates(
    metrics: TeamMetrics,
    velocities_xy: torch.Tensor,
    thresholds: SuccessThresholdsCfg,
) -> SuccessGateDiagnostics:
    dmax_ok = metrics.dmax <= thresholds.dmax
    dispersion_ok = metrics.dispersion <= thresholds.dispersion
    speed_ok = (torch.linalg.norm(velocities_xy, dim=-1) <= thresholds.speed).all(dim=-1)
    instant_success = dmax_ok & dispersion_ok & speed_ok
    return SuccessGateDiagnostics(
        dmax_ok=dmax_ok,
        dispersion_ok=dispersion_ok,
        speed_ok=speed_ok,
        instant_success=instant_success,
    )


def check_collision(positions: torch.Tensor, safety: SafetyCfg) -> torch.Tensor:
    pairwise = pairwise_distances_xy(positions)
    n_agents = positions.shape[1]
    eye = torch.eye(n_agents, dtype=torch.bool, device=positions.device).unsqueeze(0)
    return (pairwise.masked_fill(eye, float("inf")) < safety.collision_distance).any(dim=(-1, -2))


def check_out_of_bounds(positions: torch.Tensor, safety: SafetyCfg) -> torch.Tensor:
    return (positions[..., :2].abs() > safety.world_xy_limit).any(dim=(-1, -2))


def update_success_hold_count(
    hold_count: torch.Tensor,
    metrics: TeamMetrics,
    velocities_xy: torch.Tensor,
    thresholds: SuccessThresholdsCfg,
) -> torch.Tensor:
    gates = compute_success_gates(metrics, velocities_xy, thresholds)
    return torch.where(gates.instant_success, hold_count + 1, torch.zeros_like(hold_count))


def compute_done(
    positions: torch.Tensor,
    velocities_xy: torch.Tensor,
    metrics: TeamMetrics,
    hold_count: torch.Tensor,
    step_count: torch.Tensor,
    max_episode_steps: int,
    thresholds: SuccessThresholdsCfg,
    safety: SafetyCfg,
) -> tuple[DoneFlags, torch.Tensor]:
    next_hold = update_success_hold_count(hold_count, metrics, velocities_xy, thresholds)
    success = next_hold >= thresholds.hold_steps
    collision = check_collision(positions, safety)
    out_of_bounds = check_out_of_bounds(positions, safety)
    timeout = step_count >= max_episode_steps
    terminated = success | collision | out_of_bounds
    truncated = timeout & ~terminated
    done = terminated | truncated
    return (
        DoneFlags(
            success=success,
            collision=collision,
            out_of_bounds=out_of_bounds,
            timeout=timeout,
            terminated=terminated,
            truncated=truncated,
            done=done,
        ),
        next_hold,
    )
