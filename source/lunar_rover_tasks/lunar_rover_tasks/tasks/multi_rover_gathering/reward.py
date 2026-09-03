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
    flatness: torch.Tensor
    motion: torch.Tensor
    consistency: torch.Tensor
    success_hold: torch.Tensor
    terminal: torch.Tensor
    active_dstc: torch.Tensor
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
    *,
    oracle_feasible: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    mean_distance = compute_mean_oracle_distance(positions, oracle_point)
    reward = cfg.reward_coefficients.oracle_mean_distance_progress * (
        prev_mean_oracle_distance - mean_distance
    )
    if oracle_feasible is not None:
        if oracle_feasible.shape != reward.shape:
            raise ValueError(
                f"oracle_feasible must have shape {tuple(reward.shape)}, "
                f"got {tuple(oracle_feasible.shape)}."
            )
        reward = torch.where(
            oracle_feasible.to(device=reward.device, dtype=torch.bool),
            reward,
            torch.zeros_like(reward),
        )
    return reward, mean_distance


def compute_energy_reward(physical_action: torch.Tensor, cfg: MultiRoverGatheringEnvCfg) -> torch.Tensor:
    coeff = cfg.reward_coefficients
    path_cost = coeff.path_length * physical_action[..., 0].mean(dim=-1)
    turn_cost = coeff.turn_cost * physical_action[..., 1].abs().mean(dim=-1)
    return -(path_cost + turn_cost)


def compute_safety_reward(
    positions: torch.Tensor,
    metrics: TeamMetrics,
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
    terminal_pairwise_penalty = torch.zeros_like(near_penalty)
    if coeff.terminal_pairwise_gap != 0.0 and cfg.success_thresholds.min_pairwise_distance > 0.0:
        target_distance = float(cfg.success_thresholds.min_pairwise_distance)
        dmax_limit = float(cfg.success_thresholds.dmax) * float(
            coeff.terminal_pairwise_dmax_multiplier
        )
        dispersion_limit = float(cfg.success_thresholds.dispersion) * float(
            coeff.terminal_pairwise_dispersion_multiplier
        )
        near_success_zone = (metrics.dmax <= dmax_limit) & (metrics.dispersion <= dispersion_limit)
        pairwise_gap = torch.relu(target_distance - nearest).mean(dim=-1)
        terminal_pairwise_penalty = near_success_zone.float() * pairwise_gap
    return -(
        coeff.near_distance * near_penalty
        + collision_penalty
        + coeff.terminal_pairwise_gap * terminal_pairwise_penalty
    )


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
    path_terrain_reference_risk_mean: torch.Tensor | None = None,
    path_height_change_mean: torch.Tensor | None = None,
    filter_raw_path_risk_mean: torch.Tensor | None = None,
    filter_deviation: torch.Tensor | None = None,
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
    if coeff.path_terrain_relative_cost != 0.0:
        if path_terrain_risk_mean is None or path_terrain_reference_risk_mean is None:
            raise ValueError(
                "Relative trajectory risk requires selected and reference risk tensors."
            )
        relative_risk = path_terrain_risk_mean - path_terrain_reference_risk_mean
        cost = cost + coeff.path_terrain_relative_cost * relative_risk.mean(dim=-1)
    if path_height_change_mean is not None and coeff.path_height_change_cost != 0.0:
        cost = cost + coeff.path_height_change_cost * path_height_change_mean.mean(dim=-1)
    if filter_raw_path_risk_mean is not None and coeff.filter_raw_path_risk_cost != 0.0:
        cost = cost + coeff.filter_raw_path_risk_cost * filter_raw_path_risk_mean.mean(dim=-1)
    if filter_deviation is not None and coeff.filter_deviation_cost != 0.0:
        cost = cost + coeff.filter_deviation_cost * filter_deviation.mean(dim=-1)
    return -cost


def compute_centroid_flatness_cost(
    height_range: torch.Tensor,
    max_slope: torch.Tensor,
    cfg: MultiRoverGatheringEnvCfg,
) -> torch.Tensor:
    """Return the normalized actual-centroid footprint cost used by the hard gate.

    A cost at or below one is exactly equivalent to satisfying both configured
    flatness thresholds. Capping extreme violations keeps the shaping signal
    bounded without changing that acceptance boundary.
    """
    if height_range.shape != max_slope.shape:
        raise ValueError(
            "height_range and max_slope must have the same shape, got "
            f"{tuple(height_range.shape)} and {tuple(max_slope.shape)}."
        )
    normalized_height = height_range / max(
        float(cfg.gather_point.max_height_range),
        1.0e-6,
    )
    normalized_slope = max_slope / max(
        float(cfg.gather_point.max_slope),
        1.0e-6,
    )
    return torch.maximum(normalized_height, normalized_slope).clamp(0.0, 3.0)


def compute_centroid_flatness_activation(
    dmax: torch.Tensor,
    cfg: MultiRoverGatheringEnvCfg,
) -> torch.Tensor:
    """Ramp actual-centroid flatness shaping into the geometric success zone."""
    coeff = cfg.reward_coefficients
    threshold = max(float(cfg.success_thresholds.dmax), 1.0e-6)
    multiplier = float(coeff.centroid_flatness_dmax_multiplier)
    ramp_width = max((multiplier - 1.0) * threshold, 1.0e-6)
    return ((multiplier * threshold - dmax) / ramp_width).clamp(0.0, 1.0)


def compute_centroid_flatness_reward(
    previous_cost: torch.Tensor,
    current_cost: torch.Tensor,
    previous_dmax: torch.Tensor,
    current_dmax: torch.Tensor,
    cfg: MultiRoverGatheringEnvCfg,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Shape flatness only once the team is close enough to form a footprint.

    The activation ramps from zero at ``multiplier * dmax_threshold`` to one at
    the geometric success threshold. The progress term rewards movement toward
    a flatter actual centroid, while the small excess penalty removes the
    incentive once the hard flatness gate is satisfied.
    """
    if not (
        previous_cost.shape
        == current_cost.shape
        == previous_dmax.shape
        == current_dmax.shape
    ):
        raise ValueError(
            "previous_cost, current_cost, previous_dmax, and current_dmax must "
            "have the same shape, got "
            f"{tuple(previous_cost.shape)}, {tuple(current_cost.shape)}, "
            f"{tuple(previous_dmax.shape)}, and {tuple(current_dmax.shape)}."
        )
    coeff = cfg.reward_coefficients
    previous_activation = compute_centroid_flatness_activation(
        previous_dmax,
        cfg,
    )
    activation = compute_centroid_flatness_activation(current_dmax, cfg)
    progress = (
        previous_activation * previous_cost
        - activation * current_cost
    )
    excess = torch.relu(current_cost - 1.0)
    reward = activation * (
        -float(coeff.centroid_flatness_excess) * excess
    ) + float(coeff.centroid_flatness_progress) * progress
    return reward, progress, activation


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
    oracle_feasible: torch.Tensor | None = None,
    subgoal_terrain_features: torch.Tensor | None = None,
    terrain_speed_scale: torch.Tensor | None = None,
    height_delta: torch.Tensor | None = None,
    path_terrain_risk_mean: torch.Tensor | None = None,
    path_terrain_risk_max: torch.Tensor | None = None,
    path_terrain_reference_risk_mean: torch.Tensor | None = None,
    path_height_change_mean: torch.Tensor | None = None,
    filter_raw_path_risk_mean: torch.Tensor | None = None,
    filter_deviation: torch.Tensor | None = None,
    centroid_flatness_reward: torch.Tensor | None = None,
    active_dstc_reward: torch.Tensor | None = None,
) -> tuple[RewardTerms, torch.Tensor]:
    weights = cfg.reward_weights
    gather = compute_gather_reward(prev_metrics, metrics, cfg)
    oracle, mean_oracle_distance = compute_oracle_reward(
        positions,
        oracle_point,
        prev_mean_oracle_distance,
        cfg,
        oracle_feasible=oracle_feasible,
    )
    energy = compute_energy_reward(physical_action, cfg)
    safety = compute_safety_reward(positions, metrics, done, cfg)
    terrain = compute_terrain_reward(
        terrain_features,
        cfg,
        positions,
        subgoal_terrain_features=subgoal_terrain_features,
        terrain_speed_scale=terrain_speed_scale,
        height_delta=height_delta,
        path_terrain_risk_mean=path_terrain_risk_mean,
        path_terrain_risk_max=path_terrain_risk_max,
        path_terrain_reference_risk_mean=path_terrain_reference_risk_mean,
        path_height_change_mean=path_height_change_mean,
        filter_raw_path_risk_mean=filter_raw_path_risk_mean,
        filter_deviation=filter_deviation,
    )
    flatness = (
        centroid_flatness_reward
        if centroid_flatness_reward is not None
        else torch.zeros_like(gather)
    )
    if flatness.shape != gather.shape:
        raise ValueError(
            f"centroid_flatness_reward must have shape {tuple(gather.shape)}, "
            f"got {tuple(flatness.shape)}."
        )
    motion = compute_motion_reward(physical_action, cfg)
    consistency = compute_consistency_reward(physical_action, previous_physical_action, cfg)
    success_hold = compute_success_hold_reward(success_hold_count, cfg)
    terminal = compute_terminal_reward(done, cfg)
    active_dstc = (
        active_dstc_reward
        if active_dstc_reward is not None
        else torch.zeros_like(gather)
    )
    if active_dstc.shape != gather.shape:
        raise ValueError(
            f"active_dstc_reward must have shape {tuple(gather.shape)}, "
            f"got {tuple(active_dstc.shape)}."
        )
    total = (
        weights.gather * gather
        + weights.oracle * oracle
        + weights.energy * energy
        + weights.safety * safety
        + weights.terrain * terrain
        + weights.flatness * flatness
        + weights.motion * motion
        + weights.consistency * consistency
        + success_hold
        + weights.terminal * terminal
        + weights.active_dstc * active_dstc
    )
    return (
        RewardTerms(
            gather=gather,
            oracle=oracle,
            energy=energy,
            safety=safety,
            terrain=terrain,
            flatness=flatness,
            motion=motion,
            consistency=consistency,
            success_hold=success_hold,
            terminal=terminal,
            active_dstc=active_dstc,
            total=total,
        ),
        mean_oracle_distance,
    )
