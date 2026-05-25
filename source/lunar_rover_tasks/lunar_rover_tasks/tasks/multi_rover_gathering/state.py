"""Centralized critic state construction."""

from __future__ import annotations

import torch

from lunar_rover_tasks.tasks.multi_rover_gathering.gathering_env_cfg import MultiRoverGatheringEnvCfg
from lunar_rover_tasks.tasks.multi_rover_gathering.metrics import TeamMetrics
from lunar_rover_tasks.tasks.multi_rover_gathering.oracle import build_oracle_features
from lunar_rover_tasks.tasks.multi_rover_gathering.terrain_features import build_global_terrain_state


def build_agent_global_state(
    positions: torch.Tensor,
    yaws: torch.Tensor,
    velocities_xy: torch.Tensor,
    angular_velocities: torch.Tensor,
) -> torch.Tensor:
    per_agent = torch.cat(
        (
            positions,
            torch.cos(yaws).unsqueeze(-1),
            torch.sin(yaws).unsqueeze(-1),
            velocities_xy,
            angular_velocities.unsqueeze(-1),
        ),
        dim=-1,
    )
    return per_agent.flatten(start_dim=1)


def build_team_state(metrics: TeamMetrics, success_hold_count: torch.Tensor) -> torch.Tensor:
    return torch.cat(
        (
            metrics.dmax.unsqueeze(-1),
            metrics.dispersion.unsqueeze(-1),
            metrics.centroid,
            metrics.mean_pairwise_distance.unsqueeze(-1),
            metrics.mean_speed.unsqueeze(-1),
            success_hold_count.float().unsqueeze(-1),
        ),
        dim=-1,
    )


def build_critic_state(
    positions: torch.Tensor,
    yaws: torch.Tensor,
    velocities_xy: torch.Tensor,
    angular_velocities: torch.Tensor,
    metrics: TeamMetrics,
    oracle_point: torch.Tensor,
    success_hold_count: torch.Tensor,
    cfg: MultiRoverGatheringEnvCfg,
) -> torch.Tensor:
    agent = build_agent_global_state(positions, yaws, velocities_xy, angular_velocities)
    team = build_team_state(metrics, success_hold_count)
    terrain = build_global_terrain_state(
        positions,
        cfg.state.terrain_state_dim,
        positions.device,
        cfg.terrain,
    )
    oracle = build_oracle_features(positions, metrics.centroid, oracle_point)
    return torch.cat((agent, team, terrain, oracle), dim=-1)
