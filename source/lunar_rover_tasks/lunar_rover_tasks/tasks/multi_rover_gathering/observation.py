"""Actor observation construction.

The actor observation deliberately excludes all oracle fields.
"""

from __future__ import annotations

import torch

from lunar_rover_tasks.tasks.multi_rover_gathering.communication import (
    build_neighbor_features,
    compute_visibility_mask,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.gathering_env_cfg import MultiRoverGatheringEnvCfg
from lunar_rover_tasks.tasks.multi_rover_gathering.metrics import TeamMetrics
from lunar_rover_tasks.tasks.multi_rover_gathering.terrain_features import (
    build_local_terrain_grid,
    flatten_local_terrain_grid,
)


def build_ego_features(
    positions: torch.Tensor,
    yaws: torch.Tensor,
    velocities_xy: torch.Tensor,
    angular_velocities: torch.Tensor,
) -> torch.Tensor:
    z = positions[..., 2:3]
    speed_xy = torch.linalg.norm(velocities_xy, dim=-1, keepdim=True)
    abs_angular_velocity = angular_velocities.abs().unsqueeze(-1)
    return torch.cat(
        (
            positions[..., :2],
            z,
            torch.cos(yaws).unsqueeze(-1),
            torch.sin(yaws).unsqueeze(-1),
            velocities_xy,
            angular_velocities.unsqueeze(-1),
            speed_xy,
            abs_angular_velocity,
        ),
        dim=-1,
    )


def build_aggregation_features(
    positions: torch.Tensor,
    velocities_xy: torch.Tensor,
    communication_radius: float,
) -> torch.Tensor:
    del velocities_xy
    delta = positions[:, None, :, :2] - positions[:, :, None, :2]
    dist = torch.linalg.norm(delta, dim=-1)
    visible = compute_visibility_mask(positions, communication_radius)
    visible_f = visible.float()
    count = visible_f.sum(dim=-1).clamp_min(1.0)
    centroid_rel = (delta * visible_f[..., None]).sum(dim=2) / count[..., None]
    avg_dist = (dist * visible_f).sum(dim=-1, keepdim=True) / count.unsqueeze(-1)
    nearest = dist.masked_fill(~visible, float("inf")).amin(dim=-1, keepdim=True)
    nearest = torch.where(torch.isinf(nearest), torch.zeros_like(nearest), nearest)
    dispersion = (
        torch.sum((delta - centroid_rel[:, :, None, :]).square(), dim=-1) * visible_f
    ).sum(dim=-1, keepdim=True) / count.unsqueeze(-1)
    return torch.cat((centroid_rel, avg_dist, nearest, dispersion), dim=-1)


def build_terminal_gate_features(
    metrics: TeamMetrics,
    velocities_xy: torch.Tensor,
    success_hold_count: torch.Tensor,
    cfg: MultiRoverGatheringEnvCfg,
) -> torch.Tensor:
    thresholds = cfg.success_thresholds

    def normalized_margin(threshold: float, value: torch.Tensor) -> torch.Tensor:
        scale = max(float(threshold), 1.0e-6)
        return torch.clamp((float(threshold) - value) / scale, -2.0, 2.0)

    dmax_margin = normalized_margin(thresholds.dmax, metrics.dmax)
    dispersion_margin = normalized_margin(thresholds.dispersion, metrics.dispersion)
    max_speed = torch.linalg.norm(velocities_xy, dim=-1).amax(dim=-1)
    speed_margin = normalized_margin(thresholds.speed, max_speed)
    if thresholds.min_pairwise_distance > 0.0:
        pairwise_scale = float(thresholds.min_pairwise_distance)
        pairwise_margin = torch.clamp(
            (metrics.nearest_neighbor_distance - pairwise_scale) / pairwise_scale,
            -2.0,
            2.0,
        )
    else:
        pairwise_margin = torch.ones_like(metrics.nearest_neighbor_distance)
    hold_ratio = (
        success_hold_count.float() / float(max(thresholds.hold_steps, 1))
    ).clamp(0.0, 1.0)
    team_features = torch.stack(
        (dmax_margin, dispersion_margin, speed_margin, hold_ratio),
        dim=-1,
    )
    return torch.cat(
        (
            team_features[:, None, :].expand(-1, velocities_xy.shape[1], -1),
            pairwise_margin.unsqueeze(-1),
        ),
        dim=-1,
    )


def build_actor_observation(
    positions: torch.Tensor,
    yaws: torch.Tensor,
    velocities_xy: torch.Tensor,
    angular_velocities: torch.Tensor,
    communication_radius: float,
    cfg: MultiRoverGatheringEnvCfg,
    terrain_grid: torch.Tensor | None = None,
    metrics: TeamMetrics | None = None,
    success_hold_count: torch.Tensor | None = None,
) -> torch.Tensor:
    ego = build_ego_features(positions, yaws, velocities_xy, angular_velocities)
    neighbor, _ = build_neighbor_features(
        positions,
        velocities_xy,
        yaws,
        communication_radius,
        cfg.observation,
    )
    if terrain_grid is None:
        terrain_grid = build_local_terrain_grid(positions, yaws, cfg.terrain)
    terrain = flatten_local_terrain_grid(terrain_grid)
    if terrain.shape[-1] != cfg.observation.terrain_dim:
        raise ValueError(
            f"Local terrain grid has dim {terrain.shape[-1]}, "
            f"expected {cfg.observation.terrain_dim}."
        )
    aggregation = build_aggregation_features(positions, velocities_xy, communication_radius)
    parts = [ego, neighbor, terrain, aggregation]
    if cfg.observation.terminal_gate_dim > 0:
        if metrics is None or success_hold_count is None:
            raise ValueError("terminal gate observation requires metrics and success_hold_count.")
        terminal_gate = build_terminal_gate_features(
            metrics,
            velocities_xy,
            success_hold_count,
            cfg,
        )
        if terminal_gate.shape[-1] != cfg.observation.terminal_gate_dim:
            raise ValueError(
                f"Terminal gate observation has dim {terminal_gate.shape[-1]}, "
                f"expected {cfg.observation.terminal_gate_dim}."
            )
        parts.append(terminal_gate)
    return torch.cat(parts, dim=-1)
