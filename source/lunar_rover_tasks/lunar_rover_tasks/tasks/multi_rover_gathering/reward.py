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
    motion: torch.Tensor
    consistency: torch.Tensor
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


def compute_terminal_reward(done: DoneFlags) -> torch.Tensor:
    reward = torch.zeros_like(done.success, dtype=torch.float32)
    reward = torch.where(done.success, reward + 10.0, reward)
    fail = done.collision | done.out_of_bounds
    reward = torch.where(fail, reward - 10.0, reward)
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
    cfg: MultiRoverGatheringEnvCfg,
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
    motion = compute_motion_reward(physical_action, cfg)
    consistency = compute_consistency_reward(physical_action, previous_physical_action, cfg)
    terminal = compute_terminal_reward(done)
    total = (
        weights.gather * gather
        + weights.oracle * oracle
        + weights.energy * energy
        + weights.safety * safety
        + weights.motion * motion
        + weights.consistency * consistency
        + weights.terminal * terminal
    )
    return (
        RewardTerms(
            gather=gather,
            oracle=oracle,
            energy=energy,
            safety=safety,
            motion=motion,
            consistency=consistency,
            terminal=terminal,
            total=total,
        ),
        mean_oracle_distance,
    )

