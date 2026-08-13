"""Centralized critic state construction."""

from __future__ import annotations

import torch

from lunar_rover_tasks.tasks.multi_rover_gathering.gathering_env_cfg import MultiRoverGatheringEnvCfg
from lunar_rover_tasks.tasks.multi_rover_gathering.metrics import TeamMetrics
from lunar_rover_tasks.tasks.multi_rover_gathering.oracle import build_oracle_features
from lunar_rover_tasks.tasks.multi_rover_gathering.terrain_features import (
    build_local_terrain_grid,
    summarize_local_terrain_grid,
)


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


def build_team_state(
    metrics: TeamMetrics,
    success_hold_count: torch.Tensor,
    *,
    include_terminal_min_pairwise: bool = False,
) -> torch.Tensor:
    parts = [
        metrics.dmax.unsqueeze(-1),
        metrics.dispersion.unsqueeze(-1),
        metrics.centroid,
        metrics.mean_pairwise_distance.unsqueeze(-1),
        metrics.mean_speed.unsqueeze(-1),
        success_hold_count.float().unsqueeze(-1),
    ]
    if include_terminal_min_pairwise:
        parts.append(metrics.nearest_neighbor_distance.amin(dim=-1, keepdim=True))
    return torch.cat(parts, dim=-1)


def build_critic_state(
    positions: torch.Tensor,
    yaws: torch.Tensor,
    velocities_xy: torch.Tensor,
    angular_velocities: torch.Tensor,
    metrics: TeamMetrics,
    oracle_point: torch.Tensor,
    success_hold_count: torch.Tensor,
    cfg: MultiRoverGatheringEnvCfg,
    terrain_grid: torch.Tensor | None = None,
    multiscale_agent_terrain: torch.Tensor | None = None,
) -> torch.Tensor:
    agent = build_agent_global_state(positions, yaws, velocities_xy, angular_velocities)
    include_terminal_min_pairwise = (
        cfg.state.include_terminal_min_pairwise
        or cfg.observation.schema_version == "ego_v4_terminal_gate"
    )
    team = build_team_state(
        metrics,
        success_hold_count,
        include_terminal_min_pairwise=include_terminal_min_pairwise,
    )
    expected_team_dim = cfg.state.team_state_dim + (1 if include_terminal_min_pairwise else 0)
    if team.shape[-1] != expected_team_dim:
        raise ValueError(
            f"Team state has dim {team.shape[-1]}, expected {expected_team_dim}."
        )
    if terrain_grid is None:
        terrain_grid = build_local_terrain_grid(positions, yaws, cfg.terrain)
    terrain = summarize_local_terrain_grid(terrain_grid)
    if terrain.shape[-1] != cfg.state.terrain_state_dim:
        raise ValueError(
            f"Terrain state summary has dim {terrain.shape[-1]}, "
            f"expected {cfg.state.terrain_state_dim}."
        )
    oracle = build_oracle_features(positions, metrics.centroid, oracle_point)
    parts = [agent, team, terrain, oracle]
    if cfg.state.include_multiscale_agent_terrain:
        if multiscale_agent_terrain is None:
            raise ValueError(
                "The configured 950-dim critic state requires each rover's "
                "224-dim multiscale terrain observation."
            )
        expected_shape = (*positions.shape[:2], 224)
        if multiscale_agent_terrain.shape != expected_shape:
            raise ValueError(
                "Multiscale critic terrain has shape "
                f"{tuple(multiscale_agent_terrain.shape)}, expected {expected_shape}."
            )
        parts.append(multiscale_agent_terrain.flatten(start_dim=1))
    state = torch.cat(parts, dim=-1)
    if state.shape[-1] != cfg.critic_state_dim:
        raise ValueError(
            f"Critic state has dim {state.shape[-1]}, expected {cfg.critic_state_dim}."
        )
    return state
