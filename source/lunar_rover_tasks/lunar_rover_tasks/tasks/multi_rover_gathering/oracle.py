"""Training-only oracle helpers."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from lunar_rover_tasks.tasks.multi_rover_gathering.gathering_env_cfg import (
    GatherPointCfg,
    TerrainCfg,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.terrain_features import (
    GatherPointFlatness,
    TerrainRuntime,
    evaluate_gather_point_flatness,
    is_flat_terrain,
    query_height,
    query_terrain_features,
)


@dataclass(slots=True)
class OptimalGatherPointResult:
    """Selected point and diagnostics for the constrained oracle search."""

    point: torch.Tensor
    objective: torch.Tensor
    feasible: torch.Tensor
    mean_distance: torch.Tensor
    max_distance: torch.Tensor
    path_risk: torch.Tensor
    path_height_change: torch.Tensor
    flatness: GatherPointFlatness


@dataclass(slots=True)
class GatherPointCandidateEvaluation:
    objective: torch.Tensor
    feasible: torch.Tensor
    violation: torch.Tensor
    mean_distance: torch.Tensor
    max_distance: torch.Tensor
    path_risk: torch.Tensor
    path_height_change: torch.Tensor
    flatness: GatherPointFlatness


def compute_geometric_median(
    points: torch.Tensor,
    iterations: int = 32,
    eps: float = 1.0e-6,
) -> torch.Tensor:
    """Compute a batched Weiszfeld geometric median for points shaped ``[E, N, D]``."""
    estimate = points.mean(dim=1)
    for _ in range(iterations):
        diff = points - estimate[:, None, :]
        dist = torch.linalg.norm(diff, dim=-1).clamp_min(eps)
        weights = 1.0 / dist
        estimate = (points * weights[..., None]).sum(dim=1) / weights.sum(dim=1, keepdim=True)
    return estimate


def _grid_candidates(lower: torch.Tensor, upper: torch.Tensor, grid_size: int) -> torch.Tensor:
    fractions = torch.linspace(
        0.0,
        1.0,
        grid_size,
        dtype=lower.dtype,
        device=lower.device,
    )
    x = lower[:, 0, None] + (upper[:, 0] - lower[:, 0])[:, None] * fractions[None, :]
    y = lower[:, 1, None] + (upper[:, 1] - lower[:, 1])[:, None] * fractions[None, :]
    grid_x = x[:, :, None].expand(-1, -1, grid_size)
    grid_y = y[:, None, :].expand(-1, grid_size, -1)
    return torch.stack((grid_x, grid_y), dim=-1).flatten(start_dim=1, end_dim=2)


def _candidate_path_metrics(
    positions: torch.Tensor,
    candidates_xy: torch.Tensor,
    terrain_cfg: TerrainCfg,
    runtime: TerrainRuntime | None,
    *,
    num_samples: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    shape = candidates_xy.shape[:-1]
    zeros = torch.zeros(shape, dtype=candidates_xy.dtype, device=candidates_xy.device)
    if is_flat_terrain(terrain_cfg):
        return zeros, zeros.clone()

    fractions = torch.linspace(
        1.0 / float(num_samples),
        1.0,
        num_samples,
        dtype=candidates_xy.dtype,
        device=candidates_xy.device,
    )
    start_xy = positions[:, None, :, None, :2]
    delta = candidates_xy[:, :, None, None, :] - start_xy
    sample_xy = start_xy + delta * fractions.view(1, 1, 1, num_samples, 1)
    features = query_terrain_features(sample_xy, terrain_cfg, runtime)
    risk = (1.0 - features[..., 4]).clamp(0.0, 1.0)
    start_height = query_height(positions[..., :2], terrain_cfg, runtime)[..., 0]
    height_change = (features[..., 0] - start_height[:, None, :, None]).abs()
    return risk.mean(dim=(-1, -2)), height_change.mean(dim=(-1, -2))


def _evaluate_search_flatness(
    candidates_xy: torch.Tensor,
    terrain_cfg: TerrainCfg,
    gather_cfg: GatherPointCfg,
    runtime: TerrainRuntime | None,
) -> GatherPointFlatness:
    """Evaluate the search footprint, optionally over an execution envelope.

    The terminal criterion always evaluates the actual team centroid.  A
    positive robustness radius makes oracle search conservative: every sampled
    centroid displacement around a candidate must pass the same footprint
    criterion.  The default zero radius exactly preserves single-center search
    behavior for existing configurations.
    """
    kwargs = {
        "radius": float(gather_cfg.flatness_radius),
        "rings": int(gather_cfg.flatness_rings),
        "samples_per_ring": int(gather_cfg.flatness_samples_per_ring),
        "max_height_range": float(gather_cfg.max_height_range),
        "max_slope": float(gather_cfg.max_slope),
    }
    if float(gather_cfg.robustness_radius) <= 0.0:
        return evaluate_gather_point_flatness(
            candidates_xy,
            terrain_cfg,
            runtime,
            **kwargs,
        )

    angles = torch.arange(
        int(gather_cfg.robustness_samples),
        dtype=candidates_xy.dtype,
        device=candidates_xy.device,
    ) * (2.0 * torch.pi / float(gather_cfg.robustness_samples))
    ring_offsets = float(gather_cfg.robustness_radius) * torch.stack(
        (torch.cos(angles), torch.sin(angles)),
        dim=-1,
    )
    offsets = torch.cat((torch.zeros_like(ring_offsets[:1]), ring_offsets), dim=0)
    offset_flatness = evaluate_gather_point_flatness(
        candidates_xy[:, :, None, :] + offsets[None, None, :, :],
        terrain_cfg,
        runtime,
        **kwargs,
    )
    return GatherPointFlatness(
        height_range=offset_flatness.height_range.amax(dim=-1),
        max_slope=offset_flatness.max_slope.amax(dim=-1),
        mean_slope=offset_flatness.mean_slope.mean(dim=-1),
        is_flat=offset_flatness.is_flat.all(dim=-1),
    )


def evaluate_gather_point_candidates(
    positions: torch.Tensor,
    candidates_xy: torch.Tensor,
    terrain_cfg: TerrainCfg,
    gather_cfg: GatherPointCfg,
    runtime: TerrainRuntime | None = None,
) -> GatherPointCandidateEvaluation:
    """Evaluate the configured objective for every candidate in a finite set."""
    if positions.ndim != 3 or positions.shape[-1] < 2:
        raise ValueError("positions must have shape [num_envs, num_agents, >=2].")
    if candidates_xy.ndim != 3 or candidates_xy.shape[-1] != 2:
        raise ValueError("candidates_xy must have shape [num_envs, num_candidates, 2].")
    if positions.shape[0] != candidates_xy.shape[0]:
        raise ValueError("positions and candidates_xy must have the same num_envs.")
    if gather_cfg.path_samples <= 0:
        raise ValueError("gather_point.path_samples must be positive.")

    distances = torch.linalg.norm(
        positions[:, None, :, :2] - candidates_xy[:, :, None, :],
        dim=-1,
    )
    mean_distance = distances.mean(dim=-1)
    max_distance = distances.amax(dim=-1)
    flatness = _evaluate_search_flatness(
        candidates_xy,
        terrain_cfg,
        gather_cfg,
        runtime,
    )
    if (
        gather_cfg.path_risk_weight == 0.0
        and gather_cfg.path_height_change_weight == 0.0
    ):
        path_risk = torch.zeros_like(mean_distance)
        path_height_change = torch.zeros_like(mean_distance)
    else:
        path_risk, path_height_change = _candidate_path_metrics(
            positions,
            candidates_xy,
            terrain_cfg,
            runtime,
            num_samples=int(gather_cfg.path_samples),
        )
    height_scale = max(float(gather_cfg.max_height_range), 1.0e-6)
    slope_scale = max(float(gather_cfg.max_slope), 1.0e-6)
    normalized_height = flatness.height_range / height_scale
    normalized_slope = flatness.max_slope / slope_scale
    flatness_cost = 0.5 * (normalized_height + normalized_slope)
    violation = torch.relu(normalized_height - 1.0) + torch.relu(normalized_slope - 1.0)
    objective = (
        float(gather_cfg.mean_distance_weight) * mean_distance
        + float(gather_cfg.max_distance_weight) * max_distance
        + float(gather_cfg.path_risk_weight) * path_risk
        + float(gather_cfg.path_height_change_weight) * path_height_change
        + float(gather_cfg.flatness_weight) * flatness_cost
    )
    return GatherPointCandidateEvaluation(
        objective=objective,
        feasible=flatness.is_flat,
        violation=violation,
        mean_distance=mean_distance,
        max_distance=max_distance,
        path_risk=path_risk,
        path_height_change=path_height_change,
        flatness=flatness,
    )


def _selected_indices(
    evaluation: GatherPointCandidateEvaluation,
    infeasible_penalty: float,
) -> torch.Tensor:
    feasible_score = evaluation.objective.masked_fill(~evaluation.feasible, float("inf"))
    fallback_score = evaluation.objective + float(infeasible_penalty) * evaluation.violation
    return torch.where(
        evaluation.feasible.any(dim=-1),
        feasible_score.argmin(dim=-1),
        fallback_score.argmin(dim=-1),
    )


def _select(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    return torch.gather(values, dim=1, index=indices[:, None]).squeeze(1)


def _result_from_selection(
    candidates: torch.Tensor,
    evaluation: GatherPointCandidateEvaluation,
    selected: torch.Tensor,
    terrain_cfg: TerrainCfg,
    runtime: TerrainRuntime | None,
) -> OptimalGatherPointResult:
    selected_xy = torch.gather(
        candidates,
        dim=1,
        index=selected[:, None, None].expand(-1, 1, 2),
    ).squeeze(1)
    selected_height = query_height(selected_xy, terrain_cfg, runtime)
    selected_flatness = GatherPointFlatness(
        height_range=_select(evaluation.flatness.height_range, selected),
        max_slope=_select(evaluation.flatness.max_slope, selected),
        mean_slope=_select(evaluation.flatness.mean_slope, selected),
        is_flat=_select(evaluation.flatness.is_flat, selected),
    )
    return OptimalGatherPointResult(
        point=torch.cat((selected_xy, selected_height), dim=-1),
        objective=_select(evaluation.objective, selected),
        feasible=_select(evaluation.feasible, selected),
        mean_distance=_select(evaluation.mean_distance, selected),
        max_distance=_select(evaluation.max_distance, selected),
        path_risk=_select(evaluation.path_risk, selected),
        path_height_change=_select(evaluation.path_height_change, selected),
        flatness=selected_flatness,
    )


def _beam_indices(
    evaluation: GatherPointCandidateEvaluation,
    width: int,
    infeasible_penalty: float,
) -> torch.Tensor:
    feasible_score = evaluation.objective.masked_fill(~evaluation.feasible, float("inf"))
    fallback_score = evaluation.objective + float(infeasible_penalty) * evaluation.violation
    ranking = torch.where(
        evaluation.feasible.any(dim=-1, keepdim=True),
        feasible_score,
        fallback_score,
    )
    return ranking.topk(
        min(int(width), ranking.shape[1]),
        dim=-1,
        largest=False,
    ).indices


def _fallback_score(
    objective: torch.Tensor,
    height_range: torch.Tensor,
    max_slope: torch.Tensor,
    gather_cfg: GatherPointCfg,
) -> torch.Tensor:
    normalized_height = height_range / max(
        float(gather_cfg.max_height_range),
        1.0e-6,
    )
    normalized_slope = max_slope / max(
        float(gather_cfg.max_slope),
        1.0e-6,
    )
    violation = torch.relu(normalized_height - 1.0) + torch.relu(
        normalized_slope - 1.0
    )
    return objective + float(gather_cfg.infeasible_penalty) * violation


def _global_search_batch(
    positions: torch.Tensor,
    terrain_cfg: TerrainCfg,
    gather_cfg: GatherPointCfg,
    runtime: TerrainRuntime | None,
    *,
    world_xy_limit: float,
) -> OptimalGatherPointResult:
    """Search the full safe world with a coarse objective grid and a local beam."""
    usable_limit = float(world_xy_limit) - float(gather_cfg.flatness_radius)
    lower = torch.full(
        (positions.shape[0], 2),
        -usable_limit,
        dtype=positions.dtype,
        device=positions.device,
    )
    upper = torch.full_like(lower, usable_limit)
    candidates = _grid_candidates(lower, upper, int(gather_cfg.global_grid_size))
    cell_size = (upper - lower) / float(int(gather_cfg.global_grid_size) - 1)
    evaluation = evaluate_gather_point_candidates(
        positions,
        candidates,
        terrain_cfg,
        gather_cfg,
        runtime,
    )
    beam_indices = _beam_indices(
        evaluation,
        gather_cfg.global_beam_width,
        gather_cfg.infeasible_penalty,
    )
    beam = torch.gather(
        candidates,
        dim=1,
        index=beam_indices[..., None].expand(-1, -1, 2),
    )

    for _ in range(int(gather_cfg.global_refinement_levels)):
        fractions = torch.linspace(
            -1.0,
            1.0,
            int(gather_cfg.refinement_grid_size),
            dtype=positions.dtype,
            device=positions.device,
        )
        offset_x, offset_y = torch.meshgrid(fractions, fractions, indexing="ij")
        unit_offsets = torch.stack((offset_x, offset_y), dim=-1).flatten(0, 1)
        unit_offsets = unit_offsets[unit_offsets.abs().amax(dim=-1) > 0.0]
        offsets = unit_offsets.view(1, 1, -1, 2) * cell_size[:, None, None, :]
        refined = (beam[:, :, None, :] + offsets).clamp(
            min=-usable_limit,
            max=usable_limit,
        )
        candidates = torch.cat((beam, refined.flatten(start_dim=1, end_dim=2)), dim=1)
        evaluation = evaluate_gather_point_candidates(
            positions,
            candidates,
            terrain_cfg,
            gather_cfg,
            runtime,
        )
        beam_indices = _beam_indices(
            evaluation,
            gather_cfg.global_beam_width,
            gather_cfg.infeasible_penalty,
        )
        beam = torch.gather(
            candidates,
            dim=1,
            index=beam_indices[..., None].expand(-1, -1, 2),
        )
        cell_size = (
            2.0
            * cell_size
            / float(max(int(gather_cfg.refinement_grid_size) - 1, 1))
        )

    selected = _selected_indices(evaluation, gather_cfg.infeasible_penalty)
    return _result_from_selection(
        candidates,
        evaluation,
        selected,
        terrain_cfg,
        runtime,
    )


def _replace_result_rows(
    destination: OptimalGatherPointResult,
    env_ids: torch.Tensor,
    source: OptimalGatherPointResult,
    source_ids: torch.Tensor | None = None,
) -> None:
    if source_ids is None:
        source_ids = torch.arange(
            env_ids.numel(),
            dtype=torch.long,
            device=env_ids.device,
        )
    destination.point[env_ids] = source.point[source_ids]
    destination.objective[env_ids] = source.objective[source_ids]
    destination.feasible[env_ids] = source.feasible[source_ids]
    destination.mean_distance[env_ids] = source.mean_distance[source_ids]
    destination.max_distance[env_ids] = source.max_distance[source_ids]
    destination.path_risk[env_ids] = source.path_risk[source_ids]
    destination.path_height_change[env_ids] = source.path_height_change[source_ids]
    destination.flatness.height_range[env_ids] = source.flatness.height_range[source_ids]
    destination.flatness.max_slope[env_ids] = source.flatness.max_slope[source_ids]
    destination.flatness.mean_slope[env_ids] = source.flatness.mean_slope[source_ids]
    destination.flatness.is_flat[env_ids] = source.flatness.is_flat[source_ids]


def _search_optimal_gather_point_batch(
    positions: torch.Tensor,
    terrain_cfg: TerrainCfg,
    gather_cfg: GatherPointCfg,
    runtime: TerrainRuntime | None,
    *,
    world_xy_limit: float,
) -> OptimalGatherPointResult:
    usable_limit = max(
        float(world_xy_limit) - float(gather_cfg.flatness_radius),
        0.0,
    )
    geometric_median = compute_geometric_median(positions[..., :2]).clamp(
        min=-usable_limit,
        max=usable_limit,
    )
    centroid = positions[..., :2].mean(dim=1).clamp(
        min=-usable_limit,
        max=usable_limit,
    )

    if gather_cfg.search_method == "geometric_median":
        candidates = geometric_median[:, None, :]
        evaluation = evaluate_gather_point_candidates(
            positions,
            candidates,
            terrain_cfg,
            gather_cfg,
            runtime,
        )
        selected = torch.zeros(positions.shape[0], dtype=torch.long, device=positions.device)
    else:
        lower = positions[..., :2].amin(dim=1) - float(gather_cfg.search_margin)
        upper = positions[..., :2].amax(dim=1) + float(gather_cfg.search_margin)
        lower = lower.clamp(min=-usable_limit, max=usable_limit)
        upper = upper.clamp(min=-usable_limit, max=usable_limit)
        domain_lower = lower
        domain_upper = upper
        candidates = _grid_candidates(lower, upper, int(gather_cfg.coarse_grid_size))
        seeds = torch.stack((geometric_median, centroid), dim=1)
        candidates = torch.cat((candidates, seeds), dim=1)
        evaluation = evaluate_gather_point_candidates(
            positions,
            candidates,
            terrain_cfg,
            gather_cfg,
            runtime,
        )
        selected = _selected_indices(evaluation, gather_cfg.infeasible_penalty)
        best_xy = torch.gather(
            candidates,
            dim=1,
            index=selected[:, None, None].expand(-1, 1, 2),
        ).squeeze(1)
        cell_size = (upper - lower) / float(max(int(gather_cfg.coarse_grid_size) - 1, 1))

        for _ in range(int(gather_cfg.refinement_levels)):
            lower = torch.maximum(best_xy - cell_size, domain_lower)
            upper = torch.minimum(best_xy + cell_size, domain_upper)
            refined = _grid_candidates(
                lower,
                upper,
                int(gather_cfg.refinement_grid_size),
            )
            candidates = torch.cat((best_xy[:, None, :], refined), dim=1)
            evaluation = evaluate_gather_point_candidates(
                positions,
                candidates,
                terrain_cfg,
                gather_cfg,
                runtime,
            )
            selected = _selected_indices(evaluation, gather_cfg.infeasible_penalty)
            best_xy = torch.gather(
                candidates,
                dim=1,
                index=selected[:, None, None].expand(-1, 1, 2),
            ).squeeze(1)
            cell_size = (upper - lower) / float(
                max(int(gather_cfg.refinement_grid_size) - 1, 1)
            )

    result = _result_from_selection(
        candidates,
        evaluation,
        selected,
        terrain_cfg,
        runtime,
    )
    if (
        gather_cfg.search_method == "terrain_aware_multiresolution"
        and gather_cfg.global_fallback_enabled
        and not bool(result.feasible.all())
    ):
        missing_ids = torch.nonzero(~result.feasible, as_tuple=False).flatten()
        global_batch_size = int(gather_cfg.global_max_envs_per_batch)
        for start in range(0, int(missing_ids.numel()), global_batch_size):
            chunk_ids = missing_ids[start : start + global_batch_size]
            chunk_runtime = runtime.subset(chunk_ids) if runtime is not None else None
            global_result = _global_search_batch(
                positions[chunk_ids],
                terrain_cfg,
                gather_cfg,
                chunk_runtime,
                world_xy_limit=world_xy_limit,
            )
            local_score = _fallback_score(
                result.objective[chunk_ids],
                result.flatness.height_range[chunk_ids],
                result.flatness.max_slope[chunk_ids],
                gather_cfg,
            )
            global_score = _fallback_score(
                global_result.objective,
                global_result.flatness.height_range,
                global_result.flatness.max_slope,
                gather_cfg,
            )
            use_global = global_result.feasible | (global_score < local_score)
            source_ids = torch.nonzero(use_global, as_tuple=False).flatten()
            _replace_result_rows(
                result,
                chunk_ids[source_ids],
                global_result,
                source_ids,
            )
    return result


def _concat_search_results(results: list[OptimalGatherPointResult]) -> OptimalGatherPointResult:
    return OptimalGatherPointResult(
        point=torch.cat([result.point for result in results], dim=0),
        objective=torch.cat([result.objective for result in results], dim=0),
        feasible=torch.cat([result.feasible for result in results], dim=0),
        mean_distance=torch.cat([result.mean_distance for result in results], dim=0),
        max_distance=torch.cat([result.max_distance for result in results], dim=0),
        path_risk=torch.cat([result.path_risk for result in results], dim=0),
        path_height_change=torch.cat(
            [result.path_height_change for result in results],
            dim=0,
        ),
        flatness=GatherPointFlatness(
            height_range=torch.cat(
                [result.flatness.height_range for result in results],
                dim=0,
            ),
            max_slope=torch.cat(
                [result.flatness.max_slope for result in results],
                dim=0,
            ),
            mean_slope=torch.cat(
                [result.flatness.mean_slope for result in results],
                dim=0,
            ),
            is_flat=torch.cat(
                [result.flatness.is_flat for result in results],
                dim=0,
            ),
        ),
    )


@torch.no_grad()
def search_optimal_gather_point(
    positions: torch.Tensor,
    terrain_cfg: TerrainCfg,
    gather_cfg: GatherPointCfg,
    runtime: TerrainRuntime | None = None,
    *,
    world_xy_limit: float,
) -> OptimalGatherPointResult:
    """Run deterministic constrained multi-resolution search for the oracle point.

    Feasible candidates must pass the same footprint flatness criteria used by
    the task success gate. Among feasible candidates, the configured combination
    of team travel distance, worst-agent distance, path terrain risk, path height
    change and local flatness is minimized. A failed local search optionally
    expands to a full-world coarse grid with beam refinement. If the explored
    domains still contain no feasible candidate, the minimum configured
    penalty-augmented fallback is returned with ``feasible=False``.
    """
    if positions.ndim != 3 or positions.shape[1] <= 0 or positions.shape[-1] < 2:
        raise ValueError("positions must have shape [num_envs, num_agents, >=2].")
    if gather_cfg.search_method not in {
        "terrain_aware_multiresolution",
        "geometric_median",
    }:
        raise ValueError(f"Unsupported gather-point search method: {gather_cfg.search_method}")
    if gather_cfg.coarse_grid_size < 3 or gather_cfg.coarse_grid_size % 2 == 0:
        raise ValueError("coarse_grid_size must be an odd integer >= 3.")
    if gather_cfg.refinement_grid_size < 3 or gather_cfg.refinement_grid_size % 2 == 0:
        raise ValueError("refinement_grid_size must be an odd integer >= 3.")
    if gather_cfg.refinement_levels < 0:
        raise ValueError("refinement_levels must be non-negative.")
    if gather_cfg.search_margin < 0.0:
        raise ValueError("search_margin must be non-negative.")
    if gather_cfg.global_grid_size < 3 or gather_cfg.global_grid_size % 2 == 0:
        raise ValueError("global_grid_size must be an odd integer >= 3.")
    if gather_cfg.global_beam_width <= 0:
        raise ValueError("global_beam_width must be positive.")
    if gather_cfg.global_refinement_levels < 0:
        raise ValueError("global_refinement_levels must be non-negative.")
    if gather_cfg.global_max_envs_per_batch <= 0:
        raise ValueError("global_max_envs_per_batch must be positive.")
    if gather_cfg.flatness_radius <= 0.0:
        raise ValueError("flatness_radius must be positive.")
    if gather_cfg.flatness_rings <= 0:
        raise ValueError("flatness_rings must be positive.")
    if gather_cfg.flatness_samples_per_ring < 4:
        raise ValueError("flatness_samples_per_ring must be at least 4.")
    if gather_cfg.max_height_range <= 0.0 or gather_cfg.max_slope <= 0.0:
        raise ValueError("Flatness height-range and slope thresholds must be positive.")
    if gather_cfg.robustness_radius < 0.0:
        raise ValueError("robustness_radius must be non-negative.")
    if gather_cfg.robustness_radius > 0.0 and gather_cfg.robustness_samples < 4:
        raise ValueError(
            "robustness_samples must be at least 4 when robustness_radius is positive."
        )
    objective_weights = (
        gather_cfg.mean_distance_weight,
        gather_cfg.max_distance_weight,
        gather_cfg.path_risk_weight,
        gather_cfg.path_height_change_weight,
        gather_cfg.flatness_weight,
    )
    if any(weight < 0.0 for weight in objective_weights):
        raise ValueError("Gather-point objective weights must be non-negative.")
    if not any(weight > 0.0 for weight in objective_weights):
        raise ValueError("At least one gather-point objective weight must be positive.")
    if gather_cfg.path_samples <= 0:
        raise ValueError("path_samples must be positive.")
    if gather_cfg.infeasible_penalty <= 0.0:
        raise ValueError("infeasible_penalty must be positive.")
    if gather_cfg.max_envs_per_batch <= 0:
        raise ValueError("max_envs_per_batch must be positive.")
    if world_xy_limit <= gather_cfg.flatness_radius:
        raise ValueError("world_xy_limit must be larger than flatness_radius.")

    batch_size = positions.shape[0]
    if batch_size <= 0:
        raise ValueError("positions must contain at least one environment.")
    if runtime is not None and runtime.yaw.shape[0] != batch_size:
        raise ValueError(
            f"Terrain runtime has {runtime.yaw.shape[0]} environments, "
            f"but positions has {batch_size}."
        )
    chunk_size = int(gather_cfg.max_envs_per_batch)
    if batch_size <= chunk_size:
        return _search_optimal_gather_point_batch(
            positions,
            terrain_cfg,
            gather_cfg,
            runtime,
            world_xy_limit=world_xy_limit,
        )

    results = []
    for start in range(0, batch_size, chunk_size):
        stop = min(start + chunk_size, batch_size)
        chunk_runtime = (
            runtime.subset(torch.arange(start, stop, device=positions.device))
            if runtime is not None
            else None
        )
        results.append(
            _search_optimal_gather_point_batch(
                positions[start:stop],
                terrain_cfg,
                gather_cfg,
                chunk_runtime,
                world_xy_limit=world_xy_limit,
            )
        )
    return _concat_search_results(results)


def compute_oracle_distances(positions: torch.Tensor, oracle_point: torch.Tensor) -> torch.Tensor:
    """Return rover-to-target distances for a shared point or assigned slots.

    ``oracle_point`` keeps its historical ``[E, 3]`` representation for a
    shared site, but can also be ``[E, A, 3]`` when each rover is assigned a
    fixed execution slot around that site.
    """
    if positions.ndim != 3 or positions.shape[-1] < 2:
        raise ValueError("positions must have shape [num_envs, num_agents, >=2].")
    if oracle_point.ndim == 2 and oracle_point.shape == (positions.shape[0], 3):
        target_xy = oracle_point[:, None, :2]
    elif oracle_point.ndim == 3 and oracle_point.shape == positions.shape:
        target_xy = oracle_point[..., :2]
    else:
        raise ValueError(
            "oracle_point must have shape [num_envs, 3] or "
            "[num_envs, num_agents, 3]."
        )
    return torch.linalg.norm(positions[..., :2] - target_xy, dim=-1)


def compute_mean_oracle_distance(positions: torch.Tensor, oracle_point: torch.Tensor) -> torch.Tensor:
    return compute_oracle_distances(positions, oracle_point).mean(dim=-1)


def build_oracle_features(
    positions: torch.Tensor,
    centroid: torch.Tensor,
    oracle_point: torch.Tensor,
) -> torch.Tensor:
    distances = compute_oracle_distances(positions, oracle_point)
    mean_distance = distances.mean(dim=-1, keepdim=True)
    centroid_gap = torch.linalg.norm(centroid[:, :2] - oracle_point[:, :2], dim=-1, keepdim=True)
    return torch.cat((oracle_point, distances, mean_distance, centroid_gap), dim=-1)
