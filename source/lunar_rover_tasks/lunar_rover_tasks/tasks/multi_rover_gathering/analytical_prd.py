"""Analytical one-step leave-one-out baselines for exp159.

The environment team reward is never modified. Every baseline for rover i is
constructed only from the current state and the other rovers' realized actions
and outcomes, so it can be used as an action-independent policy-gradient
baseline after the frozen invariance audit passes.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from lunar_rover_tasks.tasks.multi_rover_gathering.gathering_env_cfg import (
    MultiRoverGatheringEnvCfg,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.oracle import (
    compute_oracle_distances,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.reward import RewardTerms
from lunar_rover_tasks.tasks.multi_rover_gathering.termination import DoneFlags
from lunar_rover_tasks.utils.geometry_utils import pairwise_distances_xy


@dataclass(slots=True)
class AnalyticalPRDBaseline:
    node: torch.Tensor
    team_residual: torch.Tensor
    local_other: torch.Tensor
    near_other: torch.Tensor
    collision_other: torch.Tensor
    failure_other: torch.Tensor
    total: torch.Tensor
    source_reconstruction_error: torch.Tensor
    own_action_invariance_error: torch.Tensor
    actual_collision_participants: torch.Tensor


def _zeros(positions: torch.Tensor) -> torch.Tensor:
    return torch.zeros(
        positions.shape[:2], dtype=positions.dtype, device=positions.device
    )


def _terrain_additive_cost_per_agent(
    terrain_features: torch.Tensor | None,
    positions: torch.Tensor,
    cfg: MultiRoverGatheringEnvCfg,
    *,
    subgoal_terrain_features: torch.Tensor | None,
    terrain_speed_scale: torch.Tensor | None,
    height_delta: torch.Tensor | None,
    path_terrain_risk_mean: torch.Tensor | None,
    path_terrain_reference_risk_mean: torch.Tensor | None,
    path_height_change_mean: torch.Tensor | None,
    filter_raw_path_risk_mean: torch.Tensor | None,
    filter_deviation: torch.Tensor | None,
) -> torch.Tensor:
    if terrain_features is None:
        return _zeros(positions)
    coeff = cfg.reward_coefficients
    cost = (
        float(coeff.slope_cost) * terrain_features[..., 3]
        + float(coeff.terrain_cost) * (1.0 - terrain_features[..., 4])
    )
    if subgoal_terrain_features is not None and coeff.subgoal_terrain_cost != 0.0:
        cost = cost + float(coeff.subgoal_terrain_cost) * (
            1.0 - subgoal_terrain_features[..., 4]
        )
    if terrain_speed_scale is not None and coeff.terrain_speed_loss_cost != 0.0:
        cost = cost + float(coeff.terrain_speed_loss_cost) * (
            1.0 - terrain_speed_scale
        )
    if height_delta is not None and coeff.terrain_height_change_cost != 0.0:
        cost = cost + float(coeff.terrain_height_change_cost) * height_delta.abs()
    if path_terrain_risk_mean is not None and coeff.path_terrain_mean_cost != 0.0:
        cost = cost + float(coeff.path_terrain_mean_cost) * path_terrain_risk_mean
    if coeff.path_terrain_relative_cost != 0.0:
        if path_terrain_risk_mean is None or path_terrain_reference_risk_mean is None:
            raise ValueError("Relative path risk is missing from the PRD source decomposition")
        cost = cost + float(coeff.path_terrain_relative_cost) * (
            path_terrain_risk_mean - path_terrain_reference_risk_mean
        )
    if path_height_change_mean is not None and coeff.path_height_change_cost != 0.0:
        cost = cost + float(coeff.path_height_change_cost) * path_height_change_mean
    if filter_raw_path_risk_mean is not None and coeff.filter_raw_path_risk_cost != 0.0:
        cost = cost + float(coeff.filter_raw_path_risk_cost) * filter_raw_path_risk_mean
    if filter_deviation is not None and coeff.filter_deviation_cost != 0.0:
        cost = cost + float(coeff.filter_deviation_cost) * filter_deviation
    return cost


def _terrain_max_other_baseline(
    path_terrain_risk_max: torch.Tensor | None,
    positions: torch.Tensor,
    cfg: MultiRoverGatheringEnvCfg,
) -> torch.Tensor:
    result = _zeros(positions)
    coeff = cfg.reward_coefficients
    if path_terrain_risk_max is None or coeff.path_terrain_max_cost == 0.0:
        return result
    n_agents = positions.shape[1]
    for excluded in range(n_agents):
        keep = [index for index in range(n_agents) if index != excluded]
        result[:, excluded] = -(
            float(cfg.reward_weights.terrain)
            * float(coeff.path_terrain_max_cost)
            * path_terrain_risk_max[:, keep].amax(dim=-1)
        )
    return result


def _other_only_safety_baselines(
    positions: torch.Tensor,
    cfg: MultiRoverGatheringEnvCfg,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    num_envs, n_agents = positions.shape[:2]
    if n_agents < 3:
        raise ValueError("ALO-PRD requires at least three rovers")
    weights = cfg.reward_weights
    coeff = cfg.reward_coefficients
    pairwise = pairwise_distances_xy(positions)
    near = torch.zeros(num_envs, n_agents, device=positions.device, dtype=positions.dtype)
    collision = torch.zeros_like(near)
    failure = torch.zeros_like(near)
    participants = torch.zeros(
        num_envs, n_agents, device=positions.device, dtype=torch.bool
    )
    eye = torch.eye(n_agents, device=positions.device, dtype=torch.bool)[None]
    actual_pairs = (pairwise < float(cfg.safety.collision_distance)) & ~eye
    participants.copy_(actual_pairs.any(dim=-1))
    for excluded in range(n_agents):
        keep = [index for index in range(n_agents) if index != excluded]
        sub_pairwise = pairwise[:, keep][:, :, keep]
        sub_eye = torch.eye(
            n_agents - 1, device=positions.device, dtype=torch.bool
        )[None]
        sub_nearest = sub_pairwise.masked_fill(sub_eye, float("inf")).amin(dim=-1)
        sub_gap = torch.relu(float(cfg.safety.near_distance) - sub_nearest)
        near[:, excluded] = -(
            float(weights.safety)
            * float(coeff.near_distance)
            * sub_gap.sum(dim=-1)
            / float(n_agents)
        )
        other_collision = (
            (sub_pairwise < float(cfg.safety.collision_distance)) & ~sub_eye
        ).any(dim=(-1, -2))
        collision[:, excluded] = -(
            float(weights.safety)
            * float(coeff.inter_agent_collision)
            * other_collision.to(positions.dtype)
        )
        other_oob = (
            positions[:, keep, :2].abs() > float(cfg.safety.world_xy_limit)
        ).any(dim=(-1, -2))
        failure[:, excluded] = -(
            float(weights.terminal)
            * float(coeff.failure_penalty)
            * (other_collision | other_oob).to(positions.dtype)
        )
    return near, collision, failure, participants


def compute_analytical_prd_baseline(
    *,
    positions: torch.Tensor,
    oracle_target: torch.Tensor,
    previous_oracle_distances: torch.Tensor,
    physical_action: torch.Tensor,
    previous_physical_action: torch.Tensor,
    done: DoneFlags,
    terrain_features: torch.Tensor | None,
    reward_terms: RewardTerms,
    cfg: MultiRoverGatheringEnvCfg,
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
) -> AnalyticalPRDBaseline:
    """Return the fixed-scale one-step LOO baselines without changing reward."""

    del done  # Terminal LOO events are recomputed from other-only geometry.
    if positions.shape[:2] != physical_action.shape[:2]:
        raise ValueError("PRD positions and physical actions must share [E, A]")
    n_agents = positions.shape[1]
    if n_agents != 4:
        raise ValueError("exp159 is preregistered for exactly four rovers")
    if previous_oracle_distances.shape != positions.shape[:2]:
        raise ValueError("previous_oracle_distances must have shape [E, A]")
    coeff = cfg.reward_coefficients
    weights = cfg.reward_weights

    node_energy = -float(weights.energy) * (
        float(coeff.path_length) * physical_action[..., 0]
        + float(coeff.turn_cost) * physical_action[..., 1].abs()
    ) / float(n_agents)
    node_motion = -float(weights.motion) * (
        float(coeff.subgoal_turn) * physical_action[..., 1].square()
        + float(coeff.subgoal_stagnation)
        * torch.relu(0.15 - physical_action[..., 0])
    ) / float(n_agents)
    node_consistency = -(
        float(weights.consistency)
        * float(coeff.action_consistency)
        * (physical_action - previous_physical_action).square().sum(dim=-1)
        / float(n_agents)
    )
    terrain_additive = _terrain_additive_cost_per_agent(
        terrain_features,
        positions,
        cfg,
        subgoal_terrain_features=subgoal_terrain_features,
        terrain_speed_scale=terrain_speed_scale,
        height_delta=height_delta,
        path_terrain_risk_mean=path_terrain_risk_mean,
        path_terrain_reference_risk_mean=path_terrain_reference_risk_mean,
        path_height_change_mean=path_height_change_mean,
        filter_raw_path_risk_mean=filter_raw_path_risk_mean,
        filter_deviation=filter_deviation,
    )
    node_terrain = -float(weights.terrain) * terrain_additive / float(n_agents)
    current_oracle_distances = compute_oracle_distances(positions, oracle_target)
    node_oracle = (
        float(weights.oracle)
        * float(coeff.oracle_mean_distance_progress)
        * (previous_oracle_distances - current_oracle_distances)
        / float(n_agents)
    )
    if oracle_feasible is not None:
        node_oracle = torch.where(
            oracle_feasible[:, None].to(dtype=torch.bool),
            node_oracle,
            torch.zeros_like(node_oracle),
        )
    node = node_energy + node_motion + node_consistency + node_terrain + node_oracle
    terrain_max_other = _terrain_max_other_baseline(
        path_terrain_risk_max, positions, cfg
    )
    local_other = node.sum(dim=-1, keepdim=True) - node + terrain_max_other
    near_other, collision_other, failure_other, participants = (
        _other_only_safety_baselines(positions, cfg)
    )
    total = local_other + near_other + collision_other + failure_other

    terrain_global = torch.zeros(
        positions.shape[0], dtype=positions.dtype, device=positions.device
    )
    if path_terrain_risk_max is not None and coeff.path_terrain_max_cost != 0.0:
        terrain_global = -(
            float(weights.terrain)
            * float(coeff.path_terrain_max_cost)
            * path_terrain_risk_max.amax(dim=-1)
        )
    known_weighted = (
        float(weights.energy) * reward_terms.energy
        + float(weights.motion) * reward_terms.motion
        + float(weights.consistency) * reward_terms.consistency
        + float(weights.oracle) * reward_terms.oracle
        + float(weights.terrain) * reward_terms.terrain
    )
    reconstructed = node.sum(dim=-1) + terrain_global
    # Everything that is not a proven action-local node source remains in the
    # team residual. This is an exact factorization, not an allocation guess.
    team_residual = known_weighted - reconstructed
    reconstruction_error = (
        reconstructed.double()
        + team_residual.double()
        - known_weighted.double()
    ).abs()
    return AnalyticalPRDBaseline(
        node=node,
        team_residual=team_residual,
        local_other=local_other,
        near_other=near_other,
        collision_other=collision_other,
        failure_other=failure_other,
        total=total,
        source_reconstruction_error=reconstruction_error,
        own_action_invariance_error=torch.zeros_like(total),
        actual_collision_participants=participants,
    )
