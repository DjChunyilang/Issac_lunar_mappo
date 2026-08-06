"""Actor observation construction.

The default actor schemas exclude all oracle fields.  The opt-in
``ego_v5_gather_site_goal`` and ``ego_v6_gather_slot_goal`` schemas instead
receive a rover-frame execution target produced by the terrain-aware
gather-site planner.  They are broadcast-goal contracts, not hidden critic
features: they contain neither global coordinates nor any search diagnostics.
"""

from __future__ import annotations

import torch

from lunar_rover_tasks.tasks.multi_rover_gathering.communication import (
    CommunicationSnapshot,
    build_cached_aggregation_features,
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


def build_gather_site_goal_features(
    positions: torch.Tensor,
    yaws: torch.Tensor,
    gather_site_point: torch.Tensor,
    cfg: MultiRoverGatheringEnvCfg,
) -> torch.Tensor:
    """Encode the planned gathering point in each rover's body frame.

    The normalized local vector preserves the direction required for local
    control while avoiding a global-map coordinate shortcut.  The separate
    radial feature keeps the scale informative when either vector component is
    small.  The local search envelope, rather than the full world bound, keeps
    the signal numerically comparable to the other actor inputs during the
    normal gathering curriculum.
    """
    shared_shape = (*positions.shape[:1], 3)
    per_rover_shape = positions.shape
    if gather_site_point.shape == shared_shape:
        target_xy = gather_site_point[:, None, :2]
    elif gather_site_point.shape == per_rover_shape:
        target_xy = gather_site_point[..., :2]
    else:
        raise ValueError(
            "gather_site_point must have shared shape "
            f"{shared_shape} or per-rover shape {per_rover_shape}, got "
            f"{tuple(gather_site_point.shape)}."
        )
    delta_world = target_xy - positions[..., :2]
    cos_yaw = torch.cos(yaws)
    sin_yaw = torch.sin(yaws)
    local_x = cos_yaw * delta_world[..., 0] + sin_yaw * delta_world[..., 1]
    local_y = -sin_yaw * delta_world[..., 0] + cos_yaw * delta_world[..., 1]
    scale = max(
        float(cfg.success_thresholds.dmax),
        float(cfg.initial_state.spawn_radius_max) + float(cfg.gather_point.search_margin),
        1.0e-6,
    )
    local_delta = torch.stack((local_x, local_y), dim=-1).div(scale).clamp(-2.0, 2.0)
    distance = torch.linalg.norm(delta_world, dim=-1, keepdim=True).div(scale).clamp(0.0, 2.0)
    return torch.cat((local_delta, distance), dim=-1)


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
    gather_site_point: torch.Tensor | None = None,
    gather_slot_point: torch.Tensor | None = None,
    communication_snapshot: CommunicationSnapshot | None = None,
) -> torch.Tensor:
    ego = build_ego_features(positions, yaws, velocities_xy, angular_velocities)
    if cfg.observation.schema_version == "ego_v8_decentralized_tiered":
        if communication_snapshot is None:
            raise ValueError(
                "ego_v8_decentralized_tiered requires an explicit communication cache snapshot."
            )
        neighbor = communication_snapshot.features
        expected_neighbor_dim = (
            cfg.observation.max_neighbors * cfg.observation.effective_neighbor_dim
        )
        if neighbor.shape[-1] != expected_neighbor_dim:
            raise ValueError(
                f"Tiered neighbor messages have dim {neighbor.shape[-1]}, "
                f"expected {expected_neighbor_dim}."
            )
    else:
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
    if communication_snapshot is not None and (
        cfg.observation.schema_version == "ego_v8_decentralized_tiered"
    ):
        aggregation = build_cached_aggregation_features(
            communication_snapshot,
            map_max_distance_m=2.0 * float(cfg.safety.world_xy_limit) * (2.0**0.5),
        )
    else:
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
    if cfg.observation.gather_site_goal_dim > 0:
        if gather_site_point is None:
            raise ValueError(
                "the gather-site goal schema requires a planned execution target."
            )
        gather_site_goal = build_gather_site_goal_features(
            positions,
            yaws,
            gather_site_point,
            cfg,
        )
        if cfg.observation.schema_version == "ego_v7_gather_site_and_slot_goal":
            if gather_slot_point is None:
                raise ValueError(
                    "ego_v7_gather_site_and_slot_goal requires assigned gather slots."
                )
            gather_slot_goal = build_gather_site_goal_features(
                positions,
                yaws,
                gather_slot_point,
                cfg,
            )
            gather_site_goal = torch.cat((gather_site_goal, gather_slot_goal), dim=-1)
        if gather_site_goal.shape[-1] != cfg.observation.gather_site_goal_dim:
            raise ValueError(
                "gather-site goal feature has unexpected dim "
                f"{gather_site_goal.shape[-1]}."
            )
        parts.append(gather_site_goal)
    return torch.cat(parts, dim=-1)
