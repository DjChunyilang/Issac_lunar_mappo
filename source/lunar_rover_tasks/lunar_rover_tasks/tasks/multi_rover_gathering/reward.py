"""Reward computation for first-stage gathering."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from lunar_rover_tasks.tasks.multi_rover_gathering.gathering_env_cfg import (
    MultiRoverGatheringEnvCfg,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.metrics import TeamMetrics
from lunar_rover_tasks.tasks.multi_rover_gathering.oracle import compute_mean_oracle_distance
from lunar_rover_tasks.tasks.multi_rover_gathering.termination import DoneFlags
from lunar_rover_tasks.utils.geometry_utils import pairwise_distances_xy


@dataclass(slots=True)
class RewardTerms:
    gather: torch.Tensor
    oracle: torch.Tensor
    energy: torch.Tensor
    safety: torch.Tensor
    terrain: torch.Tensor
    motion: torch.Tensor
    consistency: torch.Tensor
    success_hold: torch.Tensor
    terminal: torch.Tensor
    total: torch.Tensor


def compute_gather_reward(
    prev_metrics: TeamMetrics,
    metrics: TeamMetrics,
    cfg: MultiRoverGatheringEnvCfg,
) -> torch.Tensor:
    coeff = cfg.reward_coefficients
    return (
        coeff.dmax_progress * (prev_metrics.dmax - metrics.dmax)
        + coeff.dispersion_progress * (prev_metrics.dispersion - metrics.dispersion)
        - coeff.dmax_level * metrics.dmax
        - coeff.dispersion_level * metrics.dispersion
    )


def compute_oracle_reward(
    positions: torch.Tensor,
    oracle_point: torch.Tensor,
    prev_mean_oracle_distance: torch.Tensor,
    cfg: MultiRoverGatheringEnvCfg,
) -> tuple[torch.Tensor, torch.Tensor]:
    mean_distance = compute_mean_oracle_distance(positions, oracle_point)
    reward = cfg.reward_coefficients.oracle_mean_distance_progress * (
        prev_mean_oracle_distance - mean_distance
    )
    return reward, mean_distance


def compute_energy_reward(physical_action: torch.Tensor, cfg: MultiRoverGatheringEnvCfg) -> torch.Tensor:
    coeff = cfg.reward_coefficients
    path_cost = coeff.path_length * physical_action[..., 0].mean(dim=-1)
    turn_cost = coeff.turn_cost * physical_action[..., 1].abs().mean(dim=-1)
    return -(path_cost + turn_cost)


def compute_safety_reward(
    positions: torch.Tensor,
    done: DoneFlags,
    cfg: MultiRoverGatheringEnvCfg,
) -> torch.Tensor:
    coeff = cfg.reward_coefficients
    pairwise = pairwise_distances_xy(positions)
    n_agents = positions.shape[1]
    eye = torch.eye(n_agents, dtype=torch.bool, device=positions.device).unsqueeze(0)
    nearest = pairwise.masked_fill(eye, float("inf")).amin(dim=-1)
    near_penalty = torch.relu(cfg.safety.near_distance - nearest).mean(dim=-1)
    collision_penalty = done.collision.float() * coeff.inter_agent_collision
    return -(coeff.near_distance * near_penalty + collision_penalty)


def compute_terrain_reward(
    terrain_features: torch.Tensor | None,
    cfg: MultiRoverGatheringEnvCfg,
    positions: torch.Tensor,
    *,
    subgoal_terrain_features: torch.Tensor | None = None,
    terrain_speed_scale: torch.Tensor | None = None,
    height_delta: torch.Tensor | None = None,
    path_terrain_risk_mean: torch.Tensor | None = None,
    path_terrain_risk_max: torch.Tensor | None = None,
    path_height_change_mean: torch.Tensor | None = None,
) -> torch.Tensor:
    coeff = cfg.reward_coefficients
    if terrain_features is None:
        return torch.zeros(positions.shape[0], dtype=positions.dtype, device=positions.device)
    roughness = terrain_features[..., 3]
    traversability = terrain_features[..., 4]
    cost = (
        coeff.slope_cost * roughness
        + coeff.terrain_cost * (1.0 - traversability)
    ).mean(dim=-1)
    if subgoal_terrain_features is not None and coeff.subgoal_terrain_cost != 0.0:
        subgoal_risk = 1.0 - subgoal_terrain_features[..., 4]
        cost = cost + coeff.subgoal_terrain_cost * subgoal_risk.mean(dim=-1)
    if terrain_speed_scale is not None and coeff.terrain_speed_loss_cost != 0.0:
        cost = cost + coeff.terrain_speed_loss_cost * (1.0 - terrain_speed_scale).mean(dim=-1)
    if height_delta is not None and coeff.terrain_height_change_cost != 0.0:
        cost = cost + coeff.terrain_height_change_cost * height_delta.abs().mean(dim=-1)
    if path_terrain_risk_mean is not None and coeff.path_terrain_mean_cost != 0.0:
        cost = cost + coeff.path_terrain_mean_cost * path_terrain_risk_mean.mean(dim=-1)
    if path_terrain_risk_max is not None and coeff.path_terrain_max_cost != 0.0:
        cost = cost + coeff.path_terrain_max_cost * path_terrain_risk_max.amax(dim=-1)
    if path_height_change_mean is not None and coeff.path_height_change_cost != 0.0:
        cost = cost + coeff.path_height_change_cost * path_height_change_mean.mean(dim=-1)
    return -cost


def compute_motion_reward(physical_action: torch.Tensor, cfg: MultiRoverGatheringEnvCfg) -> torch.Tensor:
    coeff = cfg.reward_coefficients
    turn = physical_action[..., 1].square().mean(dim=-1) * coeff.subgoal_turn
    stagnation = torch.relu(0.15 - physical_action[..., 0]).mean(dim=-1) * coeff.subgoal_stagnation
    return -(turn + stagnation)


def compute_consistency_reward(
    physical_action: torch.Tensor,
    previous_physical_action: torch.Tensor,
    cfg: MultiRoverGatheringEnvCfg,
) -> torch.Tensor:
    delta = physical_action - previous_physical_action
    return -cfg.reward_coefficients.action_consistency * delta.square().sum(dim=-1).mean(dim=-1)


def compute_success_hold_reward(
    success_hold_count: torch.Tensor,
    cfg: MultiRoverGatheringEnvCfg,
) -> torch.Tensor:
    coeff = cfg.reward_coefficients.success_hold_step
    if coeff == 0.0:
        return torch.zeros_like(success_hold_count, dtype=torch.float32)
    hold_ratio = success_hold_count.float() / float(max(cfg.success_thresholds.hold_steps, 1))
    return coeff * hold_ratio.clamp(max=1.0)


def compute_terminal_reward(done: DoneFlags, cfg: MultiRoverGatheringEnvCfg) -> torch.Tensor:
    coeff = cfg.reward_coefficients
    reward = torch.zeros_like(done.success, dtype=torch.float32)
    reward = torch.where(done.success, reward + coeff.success_bonus, reward)
    fail = done.collision | done.out_of_bounds
    reward = torch.where(fail, reward - coeff.failure_penalty, reward)
    reward = torch.where(done.truncated, reward - coeff.timeout_penalty, reward)
    return reward


def compute_reward(
    positions: torch.Tensor,
    oracle_point: torch.Tensor,
    prev_metrics: TeamMetrics,
    metrics: TeamMetrics,
    prev_mean_oracle_distance: torch.Tensor,
    physical_action: torch.Tensor,
    previous_physical_action: torch.Tensor,
    done: DoneFlags,
    success_hold_count: torch.Tensor,
    terrain_features: torch.Tensor | None,
    cfg: MultiRoverGatheringEnvCfg,
    *,
    subgoal_terrain_features: torch.Tensor | None = None,
    terrain_speed_scale: torch.Tensor | None = None,
    height_delta: torch.Tensor | None = None,
    path_terrain_risk_mean: torch.Tensor | None = None,
    path_terrain_risk_max: torch.Tensor | None = None,
    path_height_change_mean: torch.Tensor | None = None,
) -> tuple[RewardTerms, torch.Tensor]:
    weights = cfg.reward_weights
    gather = compute_gather_reward(prev_metrics, metrics, cfg)
    oracle, mean_oracle_distance = compute_oracle_reward(
        positions,
        oracle_point,
        prev_mean_oracle_distance,
        cfg,
    )
    energy = compute_energy_reward(physical_action, cfg)
    safety = compute_safety_reward(positions, done, cfg)
    terrain = compute_terrain_reward(
        terrain_features,
        cfg,
        positions,
        subgoal_terrain_features=subgoal_terrain_features,
        terrain_speed_scale=terrain_speed_scale,
        height_delta=height_delta,
        path_terrain_risk_mean=path_terrain_risk_mean,
        path_terrain_risk_max=path_terrain_risk_max,
        path_height_change_mean=path_height_change_mean,
    )
    motion = compute_motion_reward(physical_action, cfg)
    consistency = compute_consistency_reward(physical_action, previous_physical_action, cfg)
    success_hold = compute_success_hold_reward(success_hold_count, cfg)
    terminal = compute_terminal_reward(done, cfg)
    total = (
        weights.gather * gather
        + weights.oracle * oracle
        + weights.energy * energy
        + weights.safety * safety
        + weights.terrain * terrain
        + weights.motion * motion
        + weights.consistency * consistency
        + success_hold
        + weights.terminal * terminal
    )
    return (
        RewardTerms(
            gather=gather,
            oracle=oracle,
            energy=energy,
            safety=safety,
            terrain=terrain,
            motion=motion,
            consistency=consistency,
            success_hold=success_hold,
            terminal=terminal,
            total=total,
        ),
        mean_oracle_distance,
    )
