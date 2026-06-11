"""Actor observation construction.

The actor observation deliberately excludes all oracle fields.
"""

from __future__ import annotations

import torch

from lunar_rover_tasks.tasks.multi_rover_gathering.communication import build_neighbor_features
from lunar_rover_tasks.tasks.multi_rover_gathering.gathering_env_cfg import MultiRoverGatheringEnvCfg
from lunar_rover_tasks.tasks.multi_rover_gathering.terrain_features import build_terrain_features


def _fit_terrain_dim(features: torch.Tensor, dim: int) -> torch.Tensor:
    if features.shape[-1] == dim:
        return features
    if features.shape[-1] > dim:
        return features[..., :dim]
    pad = torch.zeros(*features.shape[:-1], dim - features.shape[-1], dtype=features.dtype, device=features.device)
    return torch.cat((features, pad), dim=-1)


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
    n_agents = positions.shape[1]
    self_mask = torch.eye(n_agents, dtype=torch.bool, device=positions.device).unsqueeze(0)
    visible = (dist <= communication_radius) & ~self_mask
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


def build_actor_observation(
    positions: torch.Tensor,
    yaws: torch.Tensor,
    velocities_xy: torch.Tensor,
    angular_velocities: torch.Tensor,
    communication_radius: float,
    cfg: MultiRoverGatheringEnvCfg,
    terrain_features: torch.Tensor | None = None,
) -> torch.Tensor:
    ego = build_ego_features(positions, yaws, velocities_xy, angular_velocities)
    neighbor, _ = build_neighbor_features(
        positions,
        velocities_xy,
        yaws,
        communication_radius,
        cfg.observation,
    )
    terrain = (
        _fit_terrain_dim(terrain_features, cfg.observation.terrain_dim)
        if terrain_features is not None
        else build_terrain_features(positions, cfg.observation, cfg.terrain)
    )
    aggregation = build_aggregation_features(positions, velocities_xy, communication_radius)
    return torch.cat((ego, neighbor, terrain, aggregation), dim=-1)
