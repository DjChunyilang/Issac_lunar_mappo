"""Strictly decentralized one-hop optimization over 47 trajectory primitives.

The optimizer is training-free.  Each rover scores its own primitive against a
committed site certificate, its local terrain and the previous primitive
commitments of currently visible neighbours.  Non-neighbour state is masked
before it can affect a cost.  All rovers update simultaneously.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from lunar_rover_tasks.tasks.multi_rover_gathering.action_interpreter import (
    DIFFERENTIAL_PRIMITIVE_ACTION_COUNT,
    decode_action,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.terrain_features import (
    TerrainRuntime,
    sample_trajectory_terrain_risk,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.trajectory_generator import (
    generate_trajectory,
)


@dataclass(slots=True)
class PrimitiveOptimizationResult:
    actions: torch.Tensor
    selected_cost: torch.Tensor
    predicted_minimum_distance: torch.Tensor
    terrain_risk: torch.Tensor


def _trajectories(actions: torch.Tensor, positions: torch.Tensor, yaws: torch.Tensor, cfg):
    decoded = decode_action(actions, positions, yaws, cfg.planner)
    return generate_trajectory(
        positions,
        decoded.world_subgoal,
        cfg.trajectory_generator,
        float(cfg.simulation.planning_dt),
        current_yaws=yaws,
        reference_speed=decoded.reference_speed,
        motion_direction=decoded.motion_direction,
        planned_yaw_delta=decoded.planned_yaw_delta,
        primitive_type=decoded.primitive_type,
    )


def select_decentralized_primitives(
    positions: torch.Tensor,
    yaws: torch.Tensor,
    committed_centers: torch.Tensor,
    previous_actions: torch.Tensor,
    active: torch.Tensor,
    cfg,
    terrain_runtime: TerrainRuntime,
    *,
    terminal_hold: torch.Tensor | None = None,
    communication_radius_m: float = 12.0,
) -> PrimitiveOptimizationResult:
    """Select one primitive per rover using only one-hop cached commitments."""

    if positions.ndim != 3 or positions.shape[-1] != 3:
        raise ValueError("positions must have shape [E, A, 3].")
    envs, agents = positions.shape[:2]
    if yaws.shape != (envs, agents) or previous_actions.shape != (envs, agents):
        raise ValueError("yaws and previous_actions must have shape [E, A].")
    if committed_centers.shape != (envs, 2) or active.shape != (envs,):
        raise ValueError("committed_centers/active have incompatible shapes.")
    if terminal_hold is None:
        terminal_hold = torch.zeros_like(active)

    action_count = DIFFERENTIAL_PRIMITIVE_ACTION_COUNT
    device = positions.device
    dtype = positions.dtype
    action_ids = torch.arange(action_count, device=device, dtype=torch.long)
    pairwise_now = torch.cdist(positions[..., :2], positions[..., :2])
    eye = torch.eye(agents, dtype=torch.bool, device=device)[None]
    visible = (pairwise_now <= float(communication_radius_m)) & ~eye

    neighbour_trajectory = _trajectories(previous_actions, positions, yaws, cfg)
    neighbour_points = neighbour_trajectory.points[..., :2]
    neighbour_end = neighbour_points[..., -1, :]

    selected_actions = torch.zeros(envs, agents, dtype=torch.long, device=device)
    selected_cost = torch.zeros(envs, agents, dtype=dtype, device=device)
    selected_minimum = torch.full(
        (envs, agents), float("inf"), dtype=dtype, device=device
    )
    selected_risk = torch.zeros(envs, agents, dtype=dtype, device=device)

    for agent in range(agents):
        candidate_actions = action_ids[None, :].expand(envs, -1)
        candidate_positions = positions[:, agent : agent + 1].expand(
            -1, action_count, -1
        ).contiguous()
        candidate_yaws = yaws[:, agent : agent + 1].expand(
            -1, action_count
        ).contiguous()
        candidate_trajectory = _trajectories(
            candidate_actions,
            candidate_positions,
            candidate_yaws,
            cfg,
        )
        points = candidate_trajectory.points[..., :2]
        endpoint = points[:, :, -1, :]
        terrain_risk = sample_trajectory_terrain_risk(
            candidate_trajectory.points,
            cfg.terrain,
            terrain_runtime,
        )["risk_mean"]

        distance = torch.linalg.vector_norm(
            points[:, :, None, :, :] - neighbour_points[:, None, :, :, :],
            dim=-1,
        )
        visible_agent = visible[:, agent, :]
        masked_distance = distance.masked_fill(
            ~visible_agent[:, None, :, None], float("inf")
        )
        minimum = masked_distance.amin(dim=(2, 3))
        has_neighbour = visible_agent.any(dim=-1)
        minimum = torch.where(
            has_neighbour[:, None], minimum, torch.full_like(minimum, float("inf"))
        )
        late_minimum = masked_distance[..., masked_distance.shape[-1] // 2 :].amin(
            dim=(2, 3)
        )

        endpoint_distance = torch.linalg.vector_norm(
            endpoint - committed_centers[:, None, :], dim=-1
        )
        endpoint_pair = torch.linalg.vector_norm(
            endpoint[:, :, None, :] - neighbour_end[:, None, :, :], dim=-1
        )
        compact_excess = (
            torch.relu(endpoint_pair - float(cfg.success_thresholds.dmax))
            * visible_agent[:, None, :].to(dtype)
        ).sum(dim=-1) / visible_agent.sum(dim=-1).clamp_min(1)[:, None]

        collision_deficit = torch.relu(
            float(cfg.safety.collision_distance) - minimum
        )
        clearance_deficit = torch.relu(float(cfg.safety.near_distance) - minimum)
        forward_deficit = torch.relu(
            float(cfg.safety.collision_distance) - late_minimum
        )
        unsafe = minimum < float(cfg.safety.collision_distance)
        hold = (action_ids == 0).to(dtype)[None, :]
        # A tiny agent-specific cyclic tie break avoids identical choices while
        # remaining many orders of magnitude below every physical cost term.
        tie = ((action_ids - agent).remainder(action_count)).to(dtype)[None, :]
        cost = (
            1000.0 * unsafe.to(dtype)
            + 200.0 * collision_deficit
            + 40.0 * clearance_deficit
            + 100.0 * forward_deficit
            + endpoint_distance
            + 0.75 * terrain_risk
            + 0.25 * compact_excess
            + 0.02 * hold
            + 1.0e-7 * tie
        )
        choice = cost.argmin(dim=-1)
        chosen_action = action_ids[choice]
        chosen_action = torch.where(active, chosen_action, torch.zeros_like(chosen_action))
        chosen_action = torch.where(
            terminal_hold, torch.zeros_like(chosen_action), chosen_action
        )
        selected_actions[:, agent] = chosen_action
        selected_cost[:, agent] = cost.gather(1, choice[:, None]).squeeze(1)
        selected_minimum[:, agent] = minimum.gather(1, choice[:, None]).squeeze(1)
        selected_risk[:, agent] = terrain_risk.gather(1, choice[:, None]).squeeze(1)

    return PrimitiveOptimizationResult(
        actions=selected_actions,
        selected_cost=selected_cost,
        predicted_minimum_distance=selected_minimum,
        terrain_risk=selected_risk,
    )


__all__ = ["PrimitiveOptimizationResult", "select_decentralized_primitives"]
