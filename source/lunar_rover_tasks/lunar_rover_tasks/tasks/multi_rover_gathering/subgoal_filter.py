"""Terrain- and safety-aware post-processing for decoded rover subgoals."""

from __future__ import annotations

from dataclasses import dataclass
from math import pi
from typing import Any

import torch

from lunar_rover_tasks.tasks.multi_rover_gathering.action_interpreter import DecodedAction
from lunar_rover_tasks.tasks.multi_rover_gathering.gathering_env_cfg import (
    MultiRoverGatheringEnvCfg,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.terrain_features import (
    TerrainRuntime,
    query_terrain_features,
    sample_path_terrain_risk,
)


@dataclass(slots=True)
class SubgoalFilterResult:
    decoded: DecodedAction
    info: dict[str, Any]


def _disabled_filter_result(decoded: DecodedAction) -> SubgoalFilterResult:
    shape = decoded.physical.shape[:-1]
    zeros = torch.zeros(shape, dtype=decoded.physical.dtype, device=decoded.physical.device)
    candidate_index = torch.zeros(shape, dtype=torch.long, device=decoded.physical.device)
    histogram = torch.ones(1, dtype=decoded.physical.dtype, device=decoded.physical.device)
    return SubgoalFilterResult(
        decoded=decoded,
        info={
            "enabled": False,
            "candidate_count": 1,
            "raw_path_terrain_risk_mean": zeros,
            "filtered_path_terrain_risk_mean": zeros,
            "raw_path_terrain_risk_max": zeros,
            "filtered_path_terrain_risk_max": zeros,
            "raw_path_height_change_mean": zeros,
            "filtered_path_height_change_mean": zeros,
            "path_terrain_risk_reduction": zeros,
            "subgoal_deviation": zeros,
            "suggested_subgoal_deviation": zeros,
            "endpoint_near_violation": zeros,
            "endpoint_collision_violation": zeros,
            "path_near_violation": zeros,
            "path_collision_violation": zeros,
            "mutual_path_near_violation": zeros,
            "mutual_path_collision_violation": zeros,
            "raw_endpoint_near_violation": zeros,
            "raw_endpoint_collision_violation": zeros,
            "raw_path_near_violation": zeros,
            "raw_path_collision_violation": zeros,
            "raw_mutual_path_near_violation": zeros,
            "raw_mutual_path_collision_violation": zeros,
            "candidate_feasible": zeros.bool(),
            "feasible_fraction": 1.0,
            "safety_override": zeros.bool(),
            "safety_override_fraction": 0.0,
            "collision_override": zeros.bool(),
            "collision_override_fraction": 0.0,
            "raw_visible_center_cost": zeros,
            "filtered_visible_center_cost": zeros,
            "suggested_visible_center_cost": zeros,
            "center_progress_regression": zeros,
            "suggested_center_progress_regression": zeros,
            "hold_zone_activation": zeros,
            "hold_zone_rho_cost": zeros,
            "hold_zone_spacing_violation": zeros,
            "raw_hold_zone_rho_cost": zeros,
            "raw_hold_zone_spacing_violation": zeros,
            "raw_score": zeros,
            "filtered_score": zeros,
            "score_margin": zeros,
            "applied": zeros.bool(),
            "deterministic_applied": zeros.bool(),
            "candidate_index": candidate_index,
            "suggested_candidate_index": candidate_index,
            "candidate_index_histogram": histogram,
            "schedule_progress_step": 0,
            "apply_probability": 0.0,
            "score_scale": 0.0,
            "deterministic_applied_fraction": 0.0,
        },
    )


def _candidate_physical_actions(
    raw_physical: torch.Tensor,
    cfg: MultiRoverGatheringEnvCfg,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    filter_cfg = cfg.planner.subgoal_filter
    device = raw_physical.device
    dtype = raw_physical.dtype
    rho_scales = torch.tensor(filter_cfg.rho_scales, dtype=dtype, device=device)
    beta_offsets = torch.tensor(filter_cfg.beta_offsets_deg, dtype=dtype, device=device)
    beta_offsets = beta_offsets * (pi / 180.0)
    rho = raw_physical[..., 0]
    beta = raw_physical[..., 1]
    candidate_rho = rho[..., None, None] * rho_scales.view(1, 1, -1, 1)
    candidate_beta = beta[..., None, None] + beta_offsets.view(1, 1, 1, -1)
    candidate_rho = candidate_rho.expand(*rho.shape, rho_scales.numel(), beta_offsets.numel())
    candidate_beta = candidate_beta.expand(*beta.shape, rho_scales.numel(), beta_offsets.numel())
    candidate_rho = candidate_rho.reshape(*rho.shape, -1).clamp(0.0, float(cfg.planner.rho_max))
    candidate_beta = candidate_beta.reshape(*beta.shape, -1).clamp(
        -float(cfg.planner.beta_max),
        float(cfg.planner.beta_max),
    )
    physical = torch.stack((candidate_rho, candidate_beta), dim=-1)
    deviation = (
        (candidate_rho - rho[..., None]).abs() / max(float(cfg.planner.rho_max), 1.0e-6)
        + (candidate_beta - beta[..., None]).abs() / max(float(cfg.planner.beta_max), 1.0e-6)
    )
    raw_scale_index = int(torch.argmin((rho_scales - 1.0).abs()).detach().cpu())
    raw_beta_index = int(torch.argmin(beta_offsets.abs()).detach().cpu())
    raw_candidate_index = raw_scale_index * int(beta_offsets.numel()) + raw_beta_index
    return physical, deviation, raw_candidate_index


def _local_to_world_candidates(
    positions: torch.Tensor,
    yaws: torch.Tensor,
    physical_candidates: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    rho = physical_candidates[..., 0]
    beta = physical_candidates[..., 1]
    local_xy = torch.stack((rho * torch.cos(beta), rho * torch.sin(beta)), dim=-1)
    cos_yaw = torch.cos(yaws)[..., None]
    sin_yaw = torch.sin(yaws)[..., None]
    local_x = local_xy[..., 0]
    local_y = local_xy[..., 1]
    world_x = positions[..., 0, None] + cos_yaw * local_x - sin_yaw * local_y
    world_y = positions[..., 1, None] + sin_yaw * local_x + cos_yaw * local_y
    z = torch.zeros_like(world_x)
    world = torch.stack((world_x, world_y, z), dim=-1)
    return local_xy, world


def _endpoint_safety_violations(
    positions: torch.Tensor,
    candidate_world_subgoals: torch.Tensor,
    cfg: MultiRoverGatheringEnvCfg,
) -> tuple[torch.Tensor, torch.Tensor]:
    current_delta = positions[:, :, None, :2] - positions[:, None, :, :2]
    current_distance = torch.linalg.norm(current_delta, dim=-1)
    n_agents = positions.shape[1]
    eye = torch.eye(n_agents, dtype=torch.bool, device=positions.device).unsqueeze(0)
    visible = (current_distance <= float(cfg.observation.communication_radius)) & ~eye
    candidate_delta = (
        candidate_world_subgoals[..., None, :2] - positions[:, None, None, :, :2]
    )
    candidate_distance = torch.linalg.norm(candidate_delta, dim=-1)
    nearest_visible = candidate_distance.masked_fill(~visible[:, :, None, :], float("inf")).amin(dim=-1)
    near_threshold = float(cfg.planner.subgoal_filter.endpoint_safe_distance)
    if near_threshold <= 0.0:
        near_threshold = float(cfg.success_thresholds.min_pairwise_distance)
    near_threshold = max(near_threshold, 1.0e-6)
    collision_threshold = max(float(cfg.safety.collision_distance), 1.0e-6)
    near_violation = torch.relu(near_threshold - nearest_visible) / near_threshold
    collision_violation = torch.relu(collision_threshold - nearest_visible) / collision_threshold
    return near_violation, collision_violation


def _path_safety_violations(
    positions: torch.Tensor,
    candidate_world_subgoals: torch.Tensor,
    cfg: MultiRoverGatheringEnvCfg,
    *,
    num_samples: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    n_agents = positions.shape[1]
    if n_agents <= 1:
        zeros = torch.zeros(
            candidate_world_subgoals.shape[:-1],
            dtype=candidate_world_subgoals.dtype,
            device=candidate_world_subgoals.device,
        )
        return zeros, zeros
    current_delta = positions[:, :, None, :2] - positions[:, None, :, :2]
    current_distance = torch.linalg.norm(current_delta, dim=-1)
    eye = torch.eye(n_agents, dtype=torch.bool, device=positions.device).unsqueeze(0)
    visible = (current_distance <= float(cfg.observation.communication_radius)) & ~eye

    samples = max(2, int(num_samples))
    t = torch.linspace(
        1.0 / float(samples),
        1.0,
        samples,
        dtype=positions.dtype,
        device=positions.device,
    )
    start = positions[:, :, None, None, :2]
    end = candidate_world_subgoals[..., None, :2]
    path_xy = start + (end - start) * t.view(1, 1, 1, samples, 1)
    delta = path_xy[..., None, :] - positions[:, None, None, None, :, :2]
    distance = torch.linalg.norm(delta, dim=-1)
    visible_mask = visible[:, :, None, None, :]
    nearest_visible = distance.masked_fill(~visible_mask, float("inf")).amin(dim=(-1, -2))

    near_threshold = float(cfg.planner.subgoal_filter.path_safe_distance)
    if near_threshold <= 0.0:
        near_threshold = float(cfg.success_thresholds.min_pairwise_distance)
    near_threshold = max(near_threshold, 1.0e-6)
    collision_threshold = max(float(cfg.safety.collision_distance), 1.0e-6)
    near_violation = torch.relu(near_threshold - nearest_visible) / near_threshold
    collision_violation = torch.relu(collision_threshold - nearest_visible) / collision_threshold
    return near_violation, collision_violation


def _mutual_path_safety_violations(
    positions: torch.Tensor,
    candidate_world_subgoals: torch.Tensor,
    raw_world_subgoals: torch.Tensor,
    cfg: MultiRoverGatheringEnvCfg,
    *,
    num_samples: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    n_agents = positions.shape[1]
    if n_agents <= 1:
        zeros = torch.zeros(
            candidate_world_subgoals.shape[:-1],
            dtype=candidate_world_subgoals.dtype,
            device=candidate_world_subgoals.device,
        )
        return zeros, zeros
    current_delta = positions[:, :, None, :2] - positions[:, None, :, :2]
    current_distance = torch.linalg.norm(current_delta, dim=-1)
    eye = torch.eye(n_agents, dtype=torch.bool, device=positions.device).unsqueeze(0)
    visible = (current_distance <= float(cfg.observation.communication_radius)) & ~eye

    samples = max(2, int(num_samples))
    t = torch.linspace(
        1.0 / float(samples),
        1.0,
        samples,
        dtype=positions.dtype,
        device=positions.device,
    )
    candidate_start = positions[:, :, None, None, :2]
    candidate_end = candidate_world_subgoals[..., None, :2]
    candidate_path = candidate_start + (candidate_end - candidate_start) * t.view(
        1,
        1,
        1,
        samples,
        1,
    )

    raw_start = positions[:, :, None, :2]
    raw_end = raw_world_subgoals[:, :, None, :2]
    raw_path = raw_start + (raw_end - raw_start) * t.view(1, 1, samples, 1)
    raw_path_by_sample = raw_path.permute(0, 2, 1, 3)
    delta = candidate_path[..., None, :] - raw_path_by_sample[:, None, None, :, :, :]
    distance = torch.linalg.norm(delta, dim=-1)
    visible_mask = visible[:, :, None, None, :]
    nearest_visible = distance.masked_fill(~visible_mask, float("inf")).amin(dim=(-1, -2))

    near_threshold = float(cfg.planner.subgoal_filter.path_safe_distance)
    if near_threshold <= 0.0:
        near_threshold = float(cfg.success_thresholds.min_pairwise_distance)
    near_threshold = max(near_threshold, 1.0e-6)
    collision_threshold = max(float(cfg.safety.collision_distance), 1.0e-6)
    near_violation = torch.relu(near_threshold - nearest_visible) / near_threshold
    collision_violation = torch.relu(collision_threshold - nearest_visible) / collision_threshold
    return near_violation, collision_violation


def _visible_neighbor_center_cost(
    positions: torch.Tensor,
    candidate_world_subgoals: torch.Tensor,
    cfg: MultiRoverGatheringEnvCfg,
) -> torch.Tensor:
    n_agents = positions.shape[1]
    if n_agents <= 1:
        return torch.zeros(
            candidate_world_subgoals.shape[:-1],
            dtype=candidate_world_subgoals.dtype,
            device=candidate_world_subgoals.device,
        )
    current_delta = positions[:, :, None, :2] - positions[:, None, :, :2]
    current_distance = torch.linalg.norm(current_delta, dim=-1)
    eye = torch.eye(n_agents, dtype=torch.bool, device=positions.device).unsqueeze(0)
    visible = (current_distance <= float(cfg.observation.communication_radius)) & ~eye
    weights = visible.to(dtype=positions.dtype)
    count = weights.sum(dim=-1, keepdim=True)
    center = torch.matmul(weights, positions[..., :2]) / count.clamp_min(1.0)
    distance = torch.linalg.norm(candidate_world_subgoals[..., :2] - center[:, :, None, :], dim=-1)
    normalized = distance / max(float(cfg.success_thresholds.dmax), 1.0e-6)
    return torch.where(count > 0.0, normalized, torch.zeros_like(normalized))


def _hold_zone_costs(
    positions: torch.Tensor,
    candidate_world_subgoals: torch.Tensor,
    physical_candidates: torch.Tensor,
    cfg: MultiRoverGatheringEnvCfg,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    filter_cfg = cfg.planner.subgoal_filter
    candidate_shape = candidate_world_subgoals.shape[:-1]
    zeros = torch.zeros(
        candidate_shape,
        dtype=candidate_world_subgoals.dtype,
        device=candidate_world_subgoals.device,
    )
    if (
        filter_cfg.hold_zone_dmax_multiplier <= 0.0
        or (
            filter_cfg.hold_zone_rho_weight <= 0.0
            and filter_cfg.hold_zone_spacing_weight <= 0.0
        )
    ):
        return zeros, zeros, zeros

    n_agents = positions.shape[1]
    current_delta = positions[:, :, None, :2] - positions[:, None, :, :2]
    current_distance = torch.linalg.norm(current_delta, dim=-1)
    eye = torch.eye(n_agents, dtype=torch.bool, device=positions.device).unsqueeze(0)
    current_dmax = current_distance.masked_fill(eye, 0.0).amax(dim=(-1, -2))
    centered = positions[..., :2] - positions[..., :2].mean(dim=1, keepdim=True)
    current_dispersion = torch.mean(torch.sum(centered.square(), dim=-1), dim=-1)
    activation = current_dmax <= (
        float(cfg.success_thresholds.dmax) * float(filter_cfg.hold_zone_dmax_multiplier)
    )
    if filter_cfg.hold_zone_dispersion_multiplier > 0.0:
        activation = activation & (
            current_dispersion
            <= (
                float(cfg.success_thresholds.dispersion)
                * float(filter_cfg.hold_zone_dispersion_multiplier)
            )
        )
    activation_f = activation.to(dtype=positions.dtype)[:, None, None]

    rho_cost = (
        physical_candidates[..., 0] / max(float(cfg.planner.rho_max), 1.0e-6)
    ) * activation_f

    if n_agents <= 1:
        return activation_f.expand_as(rho_cost), rho_cost, torch.zeros_like(rho_cost)

    visible = (current_distance <= float(cfg.observation.communication_radius)) & ~eye
    candidate_delta = (
        candidate_world_subgoals[..., None, :2] - positions[:, None, None, :, :2]
    )
    candidate_distance = torch.linalg.norm(candidate_delta, dim=-1)
    nearest_visible = candidate_distance.masked_fill(~visible[:, :, None, :], float("inf")).amin(dim=-1)
    target_distance = float(filter_cfg.hold_zone_pairwise_distance)
    if target_distance <= 0.0:
        target_distance = float(cfg.success_thresholds.min_pairwise_distance)
    if target_distance <= 0.0:
        target_distance = float(cfg.safety.near_distance)
    target_distance = max(target_distance, 1.0e-6)
    spacing_violation = torch.relu(target_distance - nearest_visible) / target_distance
    spacing_violation = spacing_violation * activation_f
    return activation_f.expand_as(rho_cost), rho_cost, spacing_violation


def _schedule_values(
    cfg: MultiRoverGatheringEnvCfg,
    progress_timestep: int | None,
) -> tuple[int, float, float]:
    filter_cfg = cfg.planner.subgoal_filter
    step = max(0, int(progress_timestep or 0))
    if filter_cfg.mode not in {
        "terrain_safe_candidate_curriculum",
        "terrain_safe_candidate_constrained_curriculum",
        "terrain_safe_candidate_soft_progress_curriculum",
        "terrain_safe_candidate_mutual_progress_curriculum",
        "terrain_safe_candidate_hold_progress_curriculum",
    }:
        return step, 1.0, 1.0

    warmup = max(0, int(filter_cfg.warmup_timesteps))
    ramp = max(1, int(filter_cfg.ramp_timesteps))
    if step < warmup:
        alpha = 0.0
        apply_probability = 0.0
    else:
        alpha = min(1.0, max(0.0, (step - warmup) / float(ramp)))
        apply_probability = alpha * float(filter_cfg.apply_probability_end)
    score_scale = float(filter_cfg.score_scale_start) + alpha * (
        float(filter_cfg.score_scale_end) - float(filter_cfg.score_scale_start)
    )
    return (
        step,
        max(0.0, min(1.0, apply_probability)),
        max(0.0, score_scale),
    )


def _gather_candidate(values: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    return torch.gather(
        values,
        dim=2,
        index=index[..., None, None].expand(*index.shape, 1, values.shape[-1]),
    ).squeeze(2)


def _gather_scalar(values: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    return torch.gather(values, dim=2, index=index[..., None]).squeeze(2)


def apply_subgoal_filter(
    decoded: DecodedAction,
    positions: torch.Tensor,
    yaws: torch.Tensor,
    cfg: MultiRoverGatheringEnvCfg,
    runtime: TerrainRuntime | None = None,
    *,
    progress_timestep: int | None = None,
    deterministic: bool | None = None,
    generator: torch.Generator | None = None,
) -> SubgoalFilterResult:
    filter_cfg = cfg.planner.subgoal_filter
    if not filter_cfg.enabled:
        return _disabled_filter_result(decoded)
    supported_modes = {
        "terrain_safe_candidate",
        "terrain_safe_candidate_curriculum",
        "terrain_safe_candidate_constrained_curriculum",
        "terrain_safe_candidate_soft_progress_curriculum",
        "terrain_safe_candidate_mutual_progress_curriculum",
        "terrain_safe_candidate_hold_progress_curriculum",
    }
    if filter_cfg.mode not in supported_modes:
        raise ValueError(f"Unsupported subgoal filter mode: {filter_cfg.mode}")
    if filter_cfg.path_samples <= 0:
        raise ValueError("planner.subgoal_filter.path_samples must be positive")
    schedule_step, apply_probability, score_scale = _schedule_values(cfg, progress_timestep)
    physical_candidates, intent_deviation, raw_candidate_index = _candidate_physical_actions(
        decoded.physical,
        cfg,
    )
    raw_index = torch.full(
        decoded.physical.shape[:-1],
        int(raw_candidate_index),
        dtype=torch.long,
        device=decoded.physical.device,
    )
    local_candidates, world_candidates = _local_to_world_candidates(
        positions,
        yaws,
        physical_candidates,
    )
    start_candidates = positions[:, :, None, :].expand_as(world_candidates)
    path = sample_path_terrain_risk(
        start_candidates,
        world_candidates,
        cfg.terrain,
        runtime,
        num_samples=int(filter_cfg.path_samples),
    )
    subgoal_features = query_terrain_features(world_candidates[..., :2], cfg.terrain, runtime)
    subgoal_risk = (1.0 - subgoal_features[..., 4]).clamp(0.0, 1.0)
    near_violation, collision_violation = _endpoint_safety_violations(
        positions,
        world_candidates,
        cfg,
    )
    path_near_violation, path_collision_violation = _path_safety_violations(
        positions,
        world_candidates,
        cfg,
        num_samples=int(filter_cfg.path_samples),
    )
    mutual_path_near_violation, mutual_path_collision_violation = _mutual_path_safety_violations(
        positions,
        world_candidates,
        decoded.world_subgoal,
        cfg,
        num_samples=int(filter_cfg.path_samples),
    )
    visible_center_cost = _visible_neighbor_center_cost(positions, world_candidates, cfg)
    current_center_cost = _visible_neighbor_center_cost(
        positions,
        positions[:, :, None, :],
        cfg,
    ).squeeze(-1)
    center_progress_regression = torch.relu(
        visible_center_cost
        - current_center_cost[..., None]
        + float(filter_cfg.center_progress_margin)
    )
    hold_zone_activation, hold_zone_rho_cost, hold_zone_spacing_violation = _hold_zone_costs(
        positions,
        world_candidates,
        physical_candidates,
        cfg,
    )
    auxiliary_score = (
        float(filter_cfg.path_terrain_mean_weight) * path["risk_mean"]
        + float(filter_cfg.path_terrain_max_weight) * path["risk_max"]
        + float(filter_cfg.path_height_change_weight) * path["height_change_mean"]
        + float(filter_cfg.subgoal_terrain_weight) * subgoal_risk
        + float(filter_cfg.endpoint_near_weight) * near_violation
        + float(filter_cfg.endpoint_collision_weight) * collision_violation
        + float(filter_cfg.path_near_weight) * path_near_violation
        + float(filter_cfg.path_collision_weight) * path_collision_violation
        + float(filter_cfg.mutual_path_near_weight) * mutual_path_near_violation
        + float(filter_cfg.mutual_path_collision_weight) * mutual_path_collision_violation
        + float(filter_cfg.visible_neighbor_center_weight) * visible_center_cost
        + float(filter_cfg.center_progress_weight) * center_progress_regression
        + float(filter_cfg.hold_zone_rho_weight) * hold_zone_rho_cost
        + float(filter_cfg.hold_zone_spacing_weight) * hold_zone_spacing_violation
    )
    score = (
        float(filter_cfg.intent_deviation_weight) * intent_deviation
        + float(score_scale) * auxiliary_score
    )
    raw_center_cost = _gather_scalar(
        visible_center_cost,
        torch.full(
            decoded.physical.shape[:-1],
            int(raw_candidate_index),
            dtype=torch.long,
            device=decoded.physical.device,
        ),
    )
    feasible = torch.ones_like(score, dtype=torch.bool)
    if filter_cfg.hard_endpoint_near_filter:
        feasible = feasible & (near_violation <= 1.0e-6)
    if filter_cfg.hard_path_collision_filter:
        feasible = feasible & (path_collision_violation <= 1.0e-6)
    if filter_cfg.hard_center_progress_filter:
        feasible = feasible & (
            visible_center_cost
            <= raw_center_cost[..., None] + float(filter_cfg.center_progress_slack)
        )
    constrained_score = torch.where(
        feasible,
        score,
        score + float(filter_cfg.hard_constraint_penalty),
    )
    constrained_selected = constrained_score.argmin(dim=-1)
    has_feasible = feasible.any(dim=-1)
    selected = torch.where(has_feasible, constrained_selected, score.argmin(dim=-1))
    raw_score = _gather_scalar(score, raw_index)
    selected_score = _gather_scalar(score, selected)
    score_margin = raw_score - selected_score
    raw_endpoint_near_violation = _gather_scalar(near_violation, raw_index)
    raw_endpoint_collision_violation = _gather_scalar(collision_violation, raw_index)
    raw_path_near_violation = _gather_scalar(path_near_violation, raw_index)
    raw_path_collision_violation = _gather_scalar(path_collision_violation, raw_index)
    raw_mutual_path_near_violation = _gather_scalar(mutual_path_near_violation, raw_index)
    raw_mutual_path_collision_violation = _gather_scalar(
        mutual_path_collision_violation,
        raw_index,
    )
    raw_hold_zone_rho_cost = _gather_scalar(hold_zone_rho_cost, raw_index)
    raw_hold_zone_spacing_violation = _gather_scalar(hold_zone_spacing_violation, raw_index)
    selected_endpoint_collision_violation = _gather_scalar(collision_violation, selected)
    selected_path_collision_violation_for_override = _gather_scalar(
        path_collision_violation,
        selected,
    )
    selected_mutual_collision_violation_for_override = _gather_scalar(
        mutual_path_collision_violation,
        selected,
    )
    selected_hold_zone_spacing_violation_for_override = _gather_scalar(
        hold_zone_spacing_violation,
        selected,
    )
    selected_hold_zone_activation_for_override = _gather_scalar(
        hold_zone_activation,
        selected,
    )
    raw_unsafe = (
        raw_endpoint_near_violation > 1.0e-6
        if filter_cfg.hard_endpoint_near_filter
        else raw_endpoint_collision_violation > 1.0e-6
    ) | (raw_path_collision_violation > 1.0e-6)
    safety_override = (
        bool(filter_cfg.safety_override_after_warmup)
        and schedule_step >= int(filter_cfg.warmup_timesteps)
    )
    safety_override_mask = (
        raw_unsafe & (selected != raw_index) & has_feasible
        if safety_override
        else torch.zeros_like(raw_unsafe)
    )
    raw_collision_unsafe = (
        raw_endpoint_collision_violation > 1.0e-6
    ) | (raw_path_collision_violation > 1.0e-6) | (
        raw_mutual_path_collision_violation > 1.0e-6
    )
    selected_collision_violation_total = (
        selected_endpoint_collision_violation
        + selected_path_collision_violation_for_override
        + selected_mutual_collision_violation_for_override
    )
    raw_collision_violation_total = (
        raw_endpoint_collision_violation
        + raw_path_collision_violation
        + raw_mutual_path_collision_violation
    )
    collision_override = (
        bool(filter_cfg.collision_override_after_warmup)
        and schedule_step >= int(filter_cfg.warmup_timesteps)
    )
    collision_override_mask = (
        raw_collision_unsafe
        & (selected != raw_index)
        & (selected_collision_violation_total < raw_collision_violation_total)
        if collision_override
        else torch.zeros_like(raw_collision_unsafe)
    )
    hold_zone_override = (
        bool(filter_cfg.hold_zone_override_after_warmup)
        and schedule_step >= int(filter_cfg.warmup_timesteps)
    )
    hold_zone_override_mask = (
        (raw_hold_zone_spacing_violation > 1.0e-6)
        & (selected != raw_index)
        & (selected_hold_zone_activation_for_override > 0.0)
        & (selected_hold_zone_spacing_violation_for_override < raw_hold_zone_spacing_violation)
        if hold_zone_override
        else torch.zeros_like(raw_hold_zone_spacing_violation, dtype=torch.bool)
    )
    deterministic_filter = (
        bool(filter_cfg.deterministic_eval) if deterministic is None else bool(deterministic)
    )
    deterministic_applied = (
        (selected != raw_index)
        & (score_margin >= float(filter_cfg.deterministic_improvement_margin))
        & (apply_probability > 0.0)
    )
    if deterministic_filter:
        applied = deterministic_applied
    elif apply_probability <= 0.0:
        applied = torch.zeros_like(deterministic_applied)
    elif apply_probability >= 1.0:
        applied = selected != raw_index
    else:
        random_values = torch.rand(
            selected.shape,
            dtype=decoded.physical.dtype,
            device=decoded.physical.device,
            generator=generator,
        )
        applied = (selected != raw_index) & (random_values < float(apply_probability))
    applied = applied | safety_override_mask | collision_override_mask | hold_zone_override_mask
    execution_index = torch.where(applied, selected, raw_index)
    filtered_physical = _gather_candidate(physical_candidates, execution_index)
    filtered_local_xy = _gather_candidate(local_candidates, execution_index)
    filtered_world = _gather_candidate(world_candidates, execution_index)
    filtered_normalized = torch.stack(
        (
            2.0 * filtered_physical[..., 0] / max(float(cfg.planner.rho_max), 1.0e-6) - 1.0,
            filtered_physical[..., 1] / max(float(cfg.planner.beta_max), 1.0e-6),
        ),
        dim=-1,
    ).clamp(-1.0, 1.0)
    raw_path = sample_path_terrain_risk(
        positions,
        decoded.world_subgoal,
        cfg.terrain,
        runtime,
        num_samples=int(filter_cfg.path_samples),
    )
    filtered_risk_mean = _gather_scalar(path["risk_mean"], execution_index)
    filtered_risk_max = _gather_scalar(path["risk_max"], execution_index)
    filtered_height_change = _gather_scalar(path["height_change_mean"], execution_index)
    selected_near_violation = _gather_scalar(near_violation, execution_index)
    selected_collision_violation = _gather_scalar(collision_violation, execution_index)
    selected_path_near_violation = _gather_scalar(path_near_violation, execution_index)
    selected_path_collision_violation = _gather_scalar(path_collision_violation, execution_index)
    selected_mutual_path_near_violation = _gather_scalar(
        mutual_path_near_violation,
        execution_index,
    )
    selected_mutual_path_collision_violation = _gather_scalar(
        mutual_path_collision_violation,
        execution_index,
    )
    selected_deviation = _gather_scalar(intent_deviation, execution_index)
    suggested_deviation = _gather_scalar(intent_deviation, selected)
    executed_score = _gather_scalar(score, execution_index)
    suggested_risk_mean = _gather_scalar(path["risk_mean"], selected)
    suggested_risk_max = _gather_scalar(path["risk_max"], selected)
    suggested_height_change = _gather_scalar(path["height_change_mean"], selected)
    suggested_near_violation = _gather_scalar(near_violation, selected)
    suggested_collision_violation = _gather_scalar(collision_violation, selected)
    suggested_path_near_violation = _gather_scalar(path_near_violation, selected)
    suggested_path_collision_violation = _gather_scalar(path_collision_violation, selected)
    suggested_mutual_path_near_violation = _gather_scalar(
        mutual_path_near_violation,
        selected,
    )
    suggested_mutual_path_collision_violation = _gather_scalar(
        mutual_path_collision_violation,
        selected,
    )
    selected_visible_center_cost = _gather_scalar(visible_center_cost, execution_index)
    suggested_visible_center_cost = _gather_scalar(visible_center_cost, selected)
    selected_center_regression = _gather_scalar(center_progress_regression, execution_index)
    suggested_center_regression = _gather_scalar(center_progress_regression, selected)
    selected_hold_zone_activation = _gather_scalar(hold_zone_activation, execution_index)
    selected_hold_zone_rho_cost = _gather_scalar(hold_zone_rho_cost, execution_index)
    selected_hold_zone_spacing_violation = _gather_scalar(
        hold_zone_spacing_violation,
        execution_index,
    )
    suggested_hold_zone_rho_cost = _gather_scalar(hold_zone_rho_cost, selected)
    suggested_hold_zone_spacing_violation = _gather_scalar(
        hold_zone_spacing_violation,
        selected,
    )
    candidate_count = physical_candidates.shape[2]
    histogram = torch.bincount(
        execution_index.reshape(-1),
        minlength=candidate_count,
    ).to(dtype=decoded.physical.dtype)
    histogram = histogram / histogram.sum().clamp_min(1.0)
    filtered = DecodedAction(
        clipped_normalized=filtered_normalized,
        physical=filtered_physical,
        local_subgoal_xy=filtered_local_xy,
        world_subgoal=filtered_world,
    )
    return SubgoalFilterResult(
        decoded=filtered,
        info={
            "enabled": True,
            "candidate_count": candidate_count,
            "raw_candidate_index": raw_candidate_index,
            "raw_path_terrain_risk_mean": raw_path["risk_mean"],
            "filtered_path_terrain_risk_mean": filtered_risk_mean,
            "suggested_path_terrain_risk_mean": suggested_risk_mean,
            "raw_path_terrain_risk_max": raw_path["risk_max"],
            "filtered_path_terrain_risk_max": filtered_risk_max,
            "suggested_path_terrain_risk_max": suggested_risk_max,
            "raw_path_height_change_mean": raw_path["height_change_mean"],
            "filtered_path_height_change_mean": filtered_height_change,
            "suggested_path_height_change_mean": suggested_height_change,
            "path_terrain_risk_reduction": raw_path["risk_mean"] - filtered_risk_mean,
            "suggested_path_terrain_risk_reduction": raw_path["risk_mean"] - suggested_risk_mean,
            "subgoal_deviation": selected_deviation,
            "suggested_subgoal_deviation": suggested_deviation,
            "endpoint_near_violation": selected_near_violation,
            "endpoint_collision_violation": selected_collision_violation,
            "path_near_violation": selected_path_near_violation,
            "path_collision_violation": selected_path_collision_violation,
            "mutual_path_near_violation": selected_mutual_path_near_violation,
            "mutual_path_collision_violation": selected_mutual_path_collision_violation,
            "raw_endpoint_near_violation": raw_endpoint_near_violation,
            "raw_endpoint_collision_violation": raw_endpoint_collision_violation,
            "raw_path_near_violation": raw_path_near_violation,
            "raw_path_collision_violation": raw_path_collision_violation,
            "raw_mutual_path_near_violation": raw_mutual_path_near_violation,
            "raw_mutual_path_collision_violation": raw_mutual_path_collision_violation,
            "suggested_endpoint_near_violation": suggested_near_violation,
            "suggested_endpoint_collision_violation": suggested_collision_violation,
            "suggested_path_near_violation": suggested_path_near_violation,
            "suggested_path_collision_violation": suggested_path_collision_violation,
            "suggested_mutual_path_near_violation": suggested_mutual_path_near_violation,
            "suggested_mutual_path_collision_violation": suggested_mutual_path_collision_violation,
            "candidate_feasible": _gather_scalar(feasible.float(), execution_index).bool(),
            "feasible_fraction": float(feasible.float().mean().detach().cpu()),
            "safety_override": safety_override_mask,
            "safety_override_fraction": float(
                safety_override_mask.float().mean().detach().cpu()
            ),
            "collision_override": collision_override_mask,
            "collision_override_fraction": float(
                collision_override_mask.float().mean().detach().cpu()
            ),
            "hold_zone_override": hold_zone_override_mask,
            "hold_zone_override_fraction": float(
                hold_zone_override_mask.float().mean().detach().cpu()
            ),
            "raw_visible_center_cost": raw_center_cost,
            "filtered_visible_center_cost": selected_visible_center_cost,
            "suggested_visible_center_cost": suggested_visible_center_cost,
            "center_progress_regression": selected_center_regression,
            "suggested_center_progress_regression": suggested_center_regression,
            "hold_zone_activation": selected_hold_zone_activation,
            "hold_zone_rho_cost": selected_hold_zone_rho_cost,
            "hold_zone_spacing_violation": selected_hold_zone_spacing_violation,
            "raw_hold_zone_rho_cost": raw_hold_zone_rho_cost,
            "raw_hold_zone_spacing_violation": raw_hold_zone_spacing_violation,
            "suggested_hold_zone_rho_cost": suggested_hold_zone_rho_cost,
            "suggested_hold_zone_spacing_violation": suggested_hold_zone_spacing_violation,
            "raw_score": raw_score,
            "filtered_score": executed_score,
            "suggested_score": selected_score,
            "score_margin": score_margin,
            "applied": applied,
            "deterministic_applied": deterministic_applied,
            "candidate_index": execution_index,
            "suggested_candidate_index": selected,
            "candidate_index_histogram": histogram,
            "schedule_progress_step": schedule_step,
            "apply_probability": float(apply_probability),
            "score_scale": float(score_scale),
            "deterministic_applied_fraction": float(
                deterministic_applied.float().mean().detach().cpu()
            ),
        },
    )
