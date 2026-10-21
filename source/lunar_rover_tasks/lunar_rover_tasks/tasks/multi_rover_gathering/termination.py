"""Success and failure predicates."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import torch

from lunar_rover_tasks.tasks.multi_rover_gathering.gathering_env_cfg import (
    SafetyCfg,
    SuccessThresholdsCfg,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.metrics import TeamMetrics
from lunar_rover_tasks.utils.geometry_utils import pairwise_distances_xy


class TerminalReason(IntEnum):
    NONE = 0
    SUCCESS = 1
    COLLISION = 2
    OUT_OF_BOUNDS = 3
    TASK_DEADLINE = 4
    INVALID_TERRAIN = 5


@dataclass(slots=True)
class DoneFlags:
    success: torch.Tensor
    collision: torch.Tensor
    out_of_bounds: torch.Tensor
    timeout: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    done: torch.Tensor
    invalid_terrain: torch.Tensor | None = None
    terminal_reason: torch.Tensor | None = None
    bootstrap_allowed: torch.Tensor | None = None
    learning_valid: torch.Tensor | None = None


@dataclass(slots=True)
class SuccessGateDiagnostics:
    dmax_ok: torch.Tensor
    dispersion_ok: torch.Tensor
    speed_ok: torch.Tensor
    min_pairwise_ok: torch.Tensor
    flatness_ok: torch.Tensor
    instant_success: torch.Tensor
    angular_speed_ok: torch.Tensor


def compute_success_gates(
    metrics: TeamMetrics,
    velocities_xy: torch.Tensor,
    thresholds: SuccessThresholdsCfg,
    *,
    flatness_ok: torch.Tensor | None = None,
    angular_velocities: torch.Tensor | None = None,
) -> SuccessGateDiagnostics:
    dmax_ok = metrics.dmax <= thresholds.dmax
    dispersion_ok = metrics.dispersion <= thresholds.dispersion
    speed_ok = (torch.linalg.norm(velocities_xy, dim=-1) <= thresholds.speed).all(dim=-1)
    if thresholds.min_pairwise_distance > 0.0:
        min_pairwise = metrics.nearest_neighbor_distance.amin(dim=-1)
        min_pairwise_ok = min_pairwise >= thresholds.min_pairwise_distance
    else:
        min_pairwise_ok = torch.ones_like(dmax_ok, dtype=torch.bool)
    if flatness_ok is None:
        flatness_ok = torch.ones_like(dmax_ok, dtype=torch.bool)
    else:
        if flatness_ok.shape != dmax_ok.shape:
            raise ValueError(
                f"flatness_ok must have shape {tuple(dmax_ok.shape)}, "
                f"got {tuple(flatness_ok.shape)}."
            )
        flatness_ok = flatness_ok.to(device=dmax_ok.device, dtype=torch.bool)
    angular_speed_ok = torch.ones_like(speed_ok)
    if angular_velocities is not None:
        angular_speed_ok = (angular_velocities.abs() <= thresholds.angular_speed).all(dim=-1)
    instant_success = (
        dmax_ok & dispersion_ok & speed_ok & min_pairwise_ok & flatness_ok & angular_speed_ok
    )
    return SuccessGateDiagnostics(
        dmax_ok=dmax_ok,
        dispersion_ok=dispersion_ok,
        speed_ok=speed_ok,
        min_pairwise_ok=min_pairwise_ok,
        flatness_ok=flatness_ok,
        instant_success=instant_success,
        angular_speed_ok=angular_speed_ok,
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
    *,
    flatness_ok: torch.Tensor | None = None,
    angular_velocities: torch.Tensor | None = None,
) -> torch.Tensor:
    gates = compute_success_gates(
        metrics,
        velocities_xy,
        thresholds,
        flatness_ok=flatness_ok,
        angular_velocities=angular_velocities,
    )
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
    *,
    flatness_ok: torch.Tensor | None = None,
    terrain_unknown: torch.Tensor | None = None,
    task_contract_version: str = "legacy_v1",
    angular_velocities: torch.Tensor | None = None,
) -> tuple[DoneFlags, torch.Tensor]:
    if task_contract_version not in ("legacy_v1", "finite_task_v2"):
        raise ValueError("Unsupported task contract")
    finite = task_contract_version == "finite_task_v2"
    if finite and angular_velocities is None:
        raise ValueError("Finite task success requires actual angular velocities")
    next_hold = update_success_hold_count(
        hold_count,
        metrics,
        velocities_xy,
        thresholds,
        flatness_ok=flatness_ok,
        angular_velocities=angular_velocities if finite else None,
    )
    invalid = (
        torch.zeros_like(next_hold, dtype=torch.bool)
        if terrain_unknown is None
        else terrain_unknown
    )
    next_hold = torch.where(invalid, 0, next_hold)
    success = next_hold >= thresholds.hold_steps
    collision = check_collision(positions, safety)
    out_of_bounds = check_out_of_bounds(positions, safety)
    timeout = step_count >= max_episode_steps
    collision_terminal = (
        collision if safety.collision_termination_enabled else torch.zeros_like(collision)
    )
    terminated = success | collision_terminal | out_of_bounds
    truncated = (timeout | invalid) & ~terminated
    reason = torch.zeros_like(next_hold)
    # Ascending precedence: invalid > collision > bounds > success > deadline.
    for mask, value in (
        (timeout, TerminalReason.TASK_DEADLINE),
        (success, TerminalReason.SUCCESS),
        (out_of_bounds, TerminalReason.OUT_OF_BOUNDS),
        (collision_terminal, TerminalReason.COLLISION),
        (invalid, TerminalReason.INVALID_TERRAIN),
    ):
        reason = torch.where(mask, int(value), reason)
    if finite:
        success = reason == TerminalReason.SUCCESS
        timeout = reason == TerminalReason.TASK_DEADLINE
        terminated = (reason != TerminalReason.NONE) & ~invalid
        truncated = invalid
        next_hold = torch.where(invalid | collision_terminal | out_of_bounds, 0, next_hold)
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
            invalid_terrain=invalid,
            terminal_reason=reason,
            bootstrap_allowed=~terminated & ~invalid if finite else ~terminated,
            learning_valid=~invalid if finite else torch.ones_like(terminated),
        ),
        next_hold,
    )
