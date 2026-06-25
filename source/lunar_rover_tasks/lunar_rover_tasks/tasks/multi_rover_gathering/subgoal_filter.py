"""Terrain- and safety-aware post-processing for decoded rover subgoals."""

from __future__ import annotations

from dataclasses import dataclass
from math import pi

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
    info: dict[str, torch.Tensor | bool | int]


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
            "endpoint_near_violation": zeros,
            "endpoint_collision_violation": zeros,
            "applied": zeros.bool(),
            "candidate_index": candidate_index,
            "candidate_index_histogram": histogram,
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
    near_threshold = max(float(cfg.success_thresholds.min_pairwise_distance), 1.0e-6)
    collision_threshold = max(float(cfg.safety.collision_distance), 1.0e-6)
    near_violation = torch.relu(near_threshold - nearest_visible) / near_threshold
    collision_violation = torch.relu(collision_threshold - nearest_visible) / collision_threshold
    return near_violation, collision_violation


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
) -> SubgoalFilterResult:
    filter_cfg = cfg.planner.subgoal_filter
    if not filter_cfg.enabled:
        return _disabled_filter_result(decoded)
    if filter_cfg.mode != "terrain_safe_candidate":
        raise ValueError(f"Unsupported subgoal filter mode: {filter_cfg.mode}")
    if filter_cfg.path_samples <= 0:
        raise ValueError("planner.subgoal_filter.path_samples must be positive")
    physical_candidates, intent_deviation, raw_candidate_index = _candidate_physical_actions(
        decoded.physical,
        cfg,
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
    score = (
        float(filter_cfg.intent_deviation_weight) * intent_deviation
        + float(filter_cfg.path_terrain_mean_weight) * path["risk_mean"]
        + float(filter_cfg.path_terrain_max_weight) * path["risk_max"]
        + float(filter_cfg.path_height_change_weight) * path["height_change_mean"]
        + float(filter_cfg.subgoal_terrain_weight) * subgoal_risk
        + float(filter_cfg.endpoint_near_weight) * near_violation
        + float(filter_cfg.endpoint_collision_weight) * collision_violation
    )
    selected = score.argmin(dim=-1)
    filtered_physical = _gather_candidate(physical_candidates, selected)
    filtered_local_xy = _gather_candidate(local_candidates, selected)
    filtered_world = _gather_candidate(world_candidates, selected)
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
    filtered_risk_mean = _gather_scalar(path["risk_mean"], selected)
    filtered_risk_max = _gather_scalar(path["risk_max"], selected)
    filtered_height_change = _gather_scalar(path["height_change_mean"], selected)
    selected_near_violation = _gather_scalar(near_violation, selected)
    selected_collision_violation = _gather_scalar(collision_violation, selected)
    selected_deviation = _gather_scalar(intent_deviation, selected)
    applied = selected != raw_candidate_index
    candidate_count = physical_candidates.shape[2]
    histogram = torch.bincount(
        selected.reshape(-1),
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
            "raw_path_terrain_risk_max": raw_path["risk_max"],
            "filtered_path_terrain_risk_max": filtered_risk_max,
            "raw_path_height_change_mean": raw_path["height_change_mean"],
            "filtered_path_height_change_mean": filtered_height_change,
            "path_terrain_risk_reduction": raw_path["risk_mean"] - filtered_risk_mean,
            "subgoal_deviation": selected_deviation,
            "endpoint_near_violation": selected_near_violation,
            "endpoint_collision_violation": selected_collision_violation,
            "applied": applied,
            "candidate_index": selected,
            "candidate_index_histogram": histogram,
        },
    )
