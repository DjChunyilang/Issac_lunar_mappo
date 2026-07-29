#!/usr/bin/env python
"""Evaluate a proxy gathering policy without rendering."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from _common import ROOT, cfg_from_experiment, load_yaml
from lunar_rover_tasks.tasks.multi_rover_gathering.gathering_env import MultiRoverGatheringCore
from play import _load_policy_players


STRICT_PROXY_THRESHOLDS = {
    "dmax_reduction_ratio": 0.2,
    "success_rate": 0.9,
    "collision_rate": 0.02,
    "timeout_rate": 0.0,
}


def proxy_acceptance(metrics: dict, thresholds: dict | None = None) -> dict:
    thresholds = thresholds or STRICT_PROXY_THRESHOLDS
    checks = {
        "dmax_reduction_ratio": metrics["dmax_reduction_ratio"] <= thresholds["dmax_reduction_ratio"],
        "success_rate": metrics["success_rate"] >= thresholds["success_rate"],
        "collision_rate": metrics["collision_rate"] <= thresholds["collision_rate"],
        "timeout_rate": metrics["timeout_rate"] <= thresholds["timeout_rate"],
    }
    return {"passed": all(checks.values()), "checks": checks, "thresholds": thresholds}


def _resolve_path(path: str | Path | None) -> Path | None:
    if path is None:
        return None
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = ROOT / resolved
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def evaluate_checkpoint(
    config: str | Path,
    checkpoint: str | Path,
    device: str | None = None,
    num_envs: int = 256,
    steps: int = 100,
    seed: int | None = None,
    output: str | Path | None = None,
    run_dir: str | Path | None = None,
    filter_progress_override: int | None = None,
) -> dict:
    if run_dir is not None and output is None:
        output = Path(run_dir) / "metrics" / "final_eval_proxy.json"

    raw_cfg = load_yaml(config)
    cfg = cfg_from_experiment(config)
    cfg.simulation.num_envs = num_envs
    if device is not None:
        cfg.simulation.device = device
    if seed is not None:
        cfg.seed = seed

    map_location = torch.device(cfg.simulation.device)
    if map_location.type == "cuda" and not torch.cuda.is_available():
        map_location = torch.device("cpu")
    checkpoint_data = torch.load(checkpoint, map_location=map_location)
    metadata = checkpoint_data.get("metadata", {}) if isinstance(checkpoint_data, dict) else {}
    curriculum_filter_modes = {
        "terrain_safe_candidate_curriculum",
        "terrain_safe_candidate_constrained_curriculum",
        "terrain_safe_candidate_soft_progress_curriculum",
        "terrain_safe_candidate_mutual_progress_curriculum",
        "terrain_safe_candidate_hold_progress_curriculum",
    }
    if filter_progress_override is not None and int(filter_progress_override) < 0:
        raise ValueError("--filter-progress-override must be nonnegative.")
    if filter_progress_override is not None and cfg.planner.subgoal_filter.mode not in curriculum_filter_modes:
        raise ValueError(
            "--filter-progress-override requires a curriculum subgoal-filter mode."
        )
    if cfg.planner.subgoal_filter.mode in curriculum_filter_modes:
        checkpoint_progress = int(metadata.get("timesteps", 0))
        cfg.planner.subgoal_filter.progress_timestep_override = (
            checkpoint_progress
            if filter_progress_override is None
            else int(filter_progress_override)
        )
        cfg.planner.subgoal_filter.deterministic_eval = True

    env = MultiRoverGatheringCore(cfg)
    act, backend = _load_policy_players(checkpoint_data, cfg, env.device, raw_cfg=raw_cfg)
    actor_obs, _ = env.get_observations()

    initial_dmax = env.metrics.dmax.detach().clone()
    initial_dispersion = env.metrics.dispersion.detach().clone()
    final_dmax = initial_dmax.clone()
    final_dispersion = initial_dispersion.clone()
    final_mean_speed = env.metrics.mean_speed.detach().clone()
    final_nearest = env.metrics.nearest_neighbor_distance.amin(dim=-1).detach().clone()
    final_success_hold_count = env.success_hold_count.detach().clone()
    max_success_hold_count = env.success_hold_count.detach().clone()
    final_terrain_speed_scale = torch.ones(env.num_envs, device=env.device)
    initial_gather_point_flatness = env.evaluate_current_gather_point_flatness(env.metrics)
    initial_flatness_ok = (
        initial_gather_point_flatness.is_flat
        if env.cfg.gather_point.require_flat_for_success
        else torch.ones_like(initial_gather_point_flatness.is_flat)
    )
    final_flatness_ok = initial_flatness_ok.detach().clone()
    final_gather_point_is_flat = initial_gather_point_flatness.is_flat.detach().clone()
    final_gather_point_height_range = (
        initial_gather_point_flatness.height_range.detach().clone()
    )
    final_gather_point_max_slope = initial_gather_point_flatness.max_slope.detach().clone()
    final_gather_point_mean_slope = initial_gather_point_flatness.mean_slope.detach().clone()
    oracle_search_feasible = env.oracle_search_feasible.detach().clone()
    oracle_search_objective = env.oracle_search_objective.detach().clone()
    oracle_search_mean_distance = env.oracle_search_mean_distance.detach().clone()
    oracle_search_max_distance = env.oracle_search_max_distance.detach().clone()
    oracle_search_path_risk = env.oracle_search_path_risk.detach().clone()
    oracle_search_path_height_change = (
        env.oracle_search_path_height_change.detach().clone()
    )
    oracle_search_height_range = env.oracle_search_height_range.detach().clone()
    oracle_search_max_slope = env.oracle_search_max_slope.detach().clone()
    active = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    success_seen = torch.zeros_like(active)
    collision_seen = torch.zeros_like(active)
    timeout_seen = torch.zeros_like(active)
    done_step = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)
    first_collision_step = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)
    reward_sum = torch.tensor(0.0, device=env.device)
    reward_count = torch.tensor(0.0, device=env.device)
    nearest_sum = torch.tensor(0.0, device=env.device)
    nearest_count = torch.tensor(0.0, device=env.device)
    near_violation_count = torch.tensor(0.0, device=env.device)
    global_min_nearest = torch.tensor(float("inf"), device=env.device)
    terrain_height_sum = torch.tensor(0.0, device=env.device)
    terrain_height_count = torch.tensor(0.0, device=env.device)
    terrain_height_min = torch.tensor(float("inf"), device=env.device)
    terrain_height_max = torch.tensor(float("-inf"), device=env.device)
    terrain_roughness_sum = torch.tensor(0.0, device=env.device)
    terrain_roughness_max = torch.tensor(0.0, device=env.device)
    terrain_traversability_min = torch.tensor(float("inf"), device=env.device)
    terrain_speed_scale_sum = torch.tensor(0.0, device=env.device)
    terrain_speed_scale_count = torch.tensor(0.0, device=env.device)
    dmax_ok_sum = torch.tensor(0.0, device=env.device)
    dispersion_ok_sum = torch.tensor(0.0, device=env.device)
    speed_ok_sum = torch.tensor(0.0, device=env.device)
    min_pairwise_ok_sum = torch.tensor(0.0, device=env.device)
    flatness_ok_sum = torch.tensor(0.0, device=env.device)
    gather_point_is_flat_sum = torch.tensor(0.0, device=env.device)
    instant_success_sum = torch.tensor(0.0, device=env.device)
    gate_sample_count = torch.tensor(0.0, device=env.device)
    path_risk_mean_sum = torch.tensor(0.0, device=env.device)
    path_risk_count = torch.tensor(0.0, device=env.device)
    path_risk_max = torch.tensor(0.0, device=env.device)
    path_height_change_sum = torch.tensor(0.0, device=env.device)
    path_height_change_count = torch.tensor(0.0, device=env.device)
    filter_sample_count = torch.tensor(0.0, device=env.device)
    filter_applied_sum = torch.tensor(0.0, device=env.device)
    filter_raw_risk_sum = torch.tensor(0.0, device=env.device)
    filter_filtered_risk_sum = torch.tensor(0.0, device=env.device)
    filter_risk_reduction_sum = torch.tensor(0.0, device=env.device)
    filter_subgoal_deviation_sum = torch.tensor(0.0, device=env.device)
    filter_suggested_subgoal_deviation_sum = torch.tensor(0.0, device=env.device)
    filter_endpoint_near_violation_sum = torch.tensor(0.0, device=env.device)
    filter_endpoint_collision_violation_sum = torch.tensor(0.0, device=env.device)
    filter_endpoint_collision_violation_count = torch.tensor(0.0, device=env.device)
    filter_path_near_violation_sum = torch.tensor(0.0, device=env.device)
    filter_path_collision_violation_sum = torch.tensor(0.0, device=env.device)
    filter_path_collision_violation_count = torch.tensor(0.0, device=env.device)
    filter_mutual_path_near_violation_sum = torch.tensor(0.0, device=env.device)
    filter_mutual_path_collision_violation_sum = torch.tensor(0.0, device=env.device)
    filter_mutual_path_collision_violation_count = torch.tensor(0.0, device=env.device)
    filter_raw_endpoint_near_violation_sum = torch.tensor(0.0, device=env.device)
    filter_raw_endpoint_collision_violation_sum = torch.tensor(0.0, device=env.device)
    filter_raw_path_near_violation_sum = torch.tensor(0.0, device=env.device)
    filter_raw_path_collision_violation_sum = torch.tensor(0.0, device=env.device)
    filter_raw_mutual_path_near_violation_sum = torch.tensor(0.0, device=env.device)
    filter_raw_mutual_path_collision_violation_sum = torch.tensor(0.0, device=env.device)
    filter_candidate_feasible_sum = torch.tensor(0.0, device=env.device)
    filter_feasible_fraction_sum = torch.tensor(0.0, device=env.device)
    filter_safety_override_sum = torch.tensor(0.0, device=env.device)
    filter_collision_override_sum = torch.tensor(0.0, device=env.device)
    filter_raw_center_cost_sum = torch.tensor(0.0, device=env.device)
    filter_filtered_center_cost_sum = torch.tensor(0.0, device=env.device)
    filter_suggested_center_cost_sum = torch.tensor(0.0, device=env.device)
    filter_center_progress_regression_sum = torch.tensor(0.0, device=env.device)
    filter_hold_zone_activation_sum = torch.tensor(0.0, device=env.device)
    filter_hold_zone_rho_cost_sum = torch.tensor(0.0, device=env.device)
    filter_hold_zone_spacing_violation_sum = torch.tensor(0.0, device=env.device)
    filter_raw_hold_zone_rho_cost_sum = torch.tensor(0.0, device=env.device)
    filter_raw_hold_zone_spacing_violation_sum = torch.tensor(0.0, device=env.device)
    filter_candidate_index_sum = torch.tensor(0.0, device=env.device)
    filter_deterministic_applied_sum = torch.tensor(0.0, device=env.device)
    filter_raw_score_sum = torch.tensor(0.0, device=env.device)
    filter_filtered_score_sum = torch.tensor(0.0, device=env.device)
    filter_score_margin_sum = torch.tensor(0.0, device=env.device)
    filter_apply_probability_sum = torch.tensor(0.0, device=env.device)
    filter_score_scale_sum = torch.tensor(0.0, device=env.device)
    filter_schedule_progress_step = 0
    filter_candidate_histogram: torch.Tensor | None = None
    filter_candidate_count = 0
    control_safety_sample_count = torch.tensor(0.0, device=env.device)
    control_safety_env_count = torch.tensor(0.0, device=env.device)
    control_safety_applied_sum = torch.tensor(0.0, device=env.device)
    control_safety_linear_scale_sum = torch.tensor(0.0, device=env.device)
    control_safety_linear_scale_min = torch.tensor(float("inf"), device=env.device)
    control_safety_pairwise_risk_sum = torch.tensor(0.0, device=env.device)
    control_safety_predicted_nearest_sum = torch.tensor(0.0, device=env.device)
    control_safety_success_zone_sum = torch.tensor(0.0, device=env.device)
    control_safety_enabled = False
    formation_center_correction_env_count = torch.tensor(0.0, device=env.device)
    formation_center_correction_active_sum = torch.tensor(0.0, device=env.device)
    formation_center_correction_offset_sum = torch.tensor(0.0, device=env.device)
    formation_center_correction_offset_max = torch.tensor(0.0, device=env.device)
    formation_center_local_flatness_search_active_sum = torch.tensor(0.0, device=env.device)
    terminal_slot_capture_env_count = torch.tensor(0.0, device=env.device)
    terminal_slot_capture_active_sum = torch.tensor(0.0, device=env.device)

    for step_id in range(steps):
        active_before = active.clone()
        with torch.no_grad():
            action = act(actor_obs)
        step_output = env.step(action)
        actor_obs = step_output.actor_obs
        metrics = step_output.info["metrics"]
        done = step_output.info["done"]

        final_dmax = torch.where(active_before, metrics.dmax, final_dmax)
        final_dispersion = torch.where(active_before, metrics.dispersion, final_dispersion)
        final_mean_speed = torch.where(active_before, metrics.mean_speed, final_mean_speed)
        nearest = metrics.nearest_neighbor_distance.amin(dim=-1)
        final_nearest = torch.where(active_before, nearest, final_nearest)
        success_hold_count = step_output.info["success_hold_count"]
        final_success_hold_count = torch.where(active_before, success_hold_count, final_success_hold_count)
        max_success_hold_count = torch.maximum(
            max_success_hold_count,
            torch.where(active_before, success_hold_count, torch.zeros_like(success_hold_count)),
        )
        per_env_reward = step_output.rewards.mean(dim=-1)
        reward_sum = reward_sum + per_env_reward[active_before].sum()
        reward_count = reward_count + active_before.float().sum()
        success_gates = step_output.info["success_gates"]
        active_gate_count = active_before.float().sum()
        gate_sample_count = gate_sample_count + active_gate_count
        dmax_ok_sum = dmax_ok_sum + success_gates.dmax_ok[active_before].float().sum()
        dispersion_ok_sum = dispersion_ok_sum + success_gates.dispersion_ok[active_before].float().sum()
        speed_ok_sum = speed_ok_sum + success_gates.speed_ok[active_before].float().sum()
        min_pairwise_ok_sum = min_pairwise_ok_sum + success_gates.min_pairwise_ok[active_before].float().sum()
        flatness_ok_sum = (
            flatness_ok_sum + success_gates.flatness_ok[active_before].float().sum()
        )
        instant_success_sum = instant_success_sum + success_gates.instant_success[active_before].float().sum()
        gather_point_flatness = step_output.info["gather_point_flatness"]
        final_flatness_ok = torch.where(
            active_before,
            success_gates.flatness_ok,
            final_flatness_ok,
        )
        final_gather_point_is_flat = torch.where(
            active_before,
            gather_point_flatness.is_flat,
            final_gather_point_is_flat,
        )
        final_gather_point_height_range = torch.where(
            active_before,
            gather_point_flatness.height_range,
            final_gather_point_height_range,
        )
        final_gather_point_max_slope = torch.where(
            active_before,
            gather_point_flatness.max_slope,
            final_gather_point_max_slope,
        )
        final_gather_point_mean_slope = torch.where(
            active_before,
            gather_point_flatness.mean_slope,
            final_gather_point_mean_slope,
        )
        gather_point_is_flat_sum = (
            gather_point_is_flat_sum
            + gather_point_flatness.is_flat[active_before].float().sum()
        )

        first_done = done.done & active_before
        done_step = torch.where(first_done, torch.full_like(done_step, step_id + 1), done_step)
        success_seen = success_seen | (done.success & active_before)
        first_collision = done.collision & active_before & ~collision_seen
        first_collision_step = torch.where(
            first_collision,
            torch.full_like(first_collision_step, step_id + 1),
            first_collision_step,
        )
        collision_seen = collision_seen | (done.collision & active_before)
        timeout_seen = timeout_seen | (done.truncated & active_before)
        active_nearest = nearest[active_before]
        if active_nearest.numel() > 0:
            nearest_sum = nearest_sum + active_nearest.sum()
            nearest_count = nearest_count + torch.tensor(float(active_nearest.numel()), device=env.device)
            near_violation_count = near_violation_count + (active_nearest < env.cfg.safety.near_distance).float().sum()
            global_min_nearest = torch.minimum(global_min_nearest, active_nearest.amin())
        terrain_features = step_output.info.get("terrain_features")
        if terrain_features is not None:
            active_terrain = terrain_features[active_before].reshape(-1, terrain_features.shape[-1])
            if active_terrain.numel() > 0:
                heights = active_terrain[:, 0]
                roughness = active_terrain[:, 3]
                traversability = active_terrain[:, 4]
                terrain_height_sum = terrain_height_sum + heights.sum()
                terrain_height_count = terrain_height_count + torch.tensor(float(heights.numel()), device=env.device)
                terrain_height_min = torch.minimum(terrain_height_min, heights.amin())
                terrain_height_max = torch.maximum(terrain_height_max, heights.amax())
                terrain_roughness_sum = terrain_roughness_sum + roughness.sum()
                terrain_roughness_max = torch.maximum(terrain_roughness_max, roughness.amax())
                terrain_traversability_min = torch.minimum(terrain_traversability_min, traversability.amin())
        terrain_speed_scale = step_output.info.get("terrain_speed_scale")
        if terrain_speed_scale is not None:
            per_env_terrain_speed_scale = terrain_speed_scale.mean(dim=-1)
            final_terrain_speed_scale = torch.where(
                active_before,
                per_env_terrain_speed_scale,
                final_terrain_speed_scale,
            )
            active_speed_scale = terrain_speed_scale[active_before].reshape(-1)
            if active_speed_scale.numel() > 0:
                terrain_speed_scale_sum = terrain_speed_scale_sum + active_speed_scale.sum()
                terrain_speed_scale_count = terrain_speed_scale_count + torch.tensor(
                    float(active_speed_scale.numel()),
                    device=env.device,
                )
        path_terrain = step_output.info.get("path_terrain")
        if path_terrain is not None:
            active_path_mean = path_terrain["risk_mean"][active_before].reshape(-1)
            active_path_max = path_terrain["risk_max"][active_before].reshape(-1)
            active_path_height = path_terrain["height_change_mean"][active_before].reshape(-1)
            if active_path_mean.numel() > 0:
                path_risk_mean_sum = path_risk_mean_sum + active_path_mean.sum()
                path_risk_count = path_risk_count + torch.tensor(
                    float(active_path_mean.numel()),
                    device=env.device,
                )
                path_risk_max = torch.maximum(path_risk_max, active_path_max.amax())
                path_height_change_sum = path_height_change_sum + active_path_height.sum()
                path_height_change_count = path_height_change_count + torch.tensor(
                    float(active_path_height.numel()),
                    device=env.device,
                )
        action_filter = step_output.info.get("action_filter")
        if action_filter is not None:
            active_filter = action_filter["applied"][active_before].reshape(-1)
            if active_filter.numel() > 0:
                filter_candidate_count = int(action_filter.get("candidate_count", 0))
                raw_risk = action_filter["raw_path_terrain_risk_mean"][active_before].reshape(-1)
                filtered_risk = action_filter["filtered_path_terrain_risk_mean"][active_before].reshape(-1)
                risk_reduction = action_filter["path_terrain_risk_reduction"][active_before].reshape(-1)
                subgoal_deviation = action_filter["subgoal_deviation"][active_before].reshape(-1)
                suggested_deviation = action_filter["suggested_subgoal_deviation"][
                    active_before
                ].reshape(-1)
                near_violation = action_filter["endpoint_near_violation"][active_before].reshape(-1)
                collision_violation = action_filter["endpoint_collision_violation"][active_before].reshape(-1)
                path_near_violation = action_filter["path_near_violation"][
                    active_before
                ].reshape(-1)
                path_collision_violation = action_filter["path_collision_violation"][
                    active_before
                ].reshape(-1)
                mutual_path_near_violation = action_filter["mutual_path_near_violation"][
                    active_before
                ].reshape(-1)
                mutual_path_collision_violation = action_filter[
                    "mutual_path_collision_violation"
                ][active_before].reshape(-1)
                raw_endpoint_near_violation = action_filter[
                    "raw_endpoint_near_violation"
                ][active_before].reshape(-1)
                raw_endpoint_collision_violation = action_filter[
                    "raw_endpoint_collision_violation"
                ][active_before].reshape(-1)
                raw_path_near_violation = action_filter["raw_path_near_violation"][
                    active_before
                ].reshape(-1)
                raw_path_collision_violation = action_filter[
                    "raw_path_collision_violation"
                ][active_before].reshape(-1)
                raw_mutual_path_near_violation = action_filter[
                    "raw_mutual_path_near_violation"
                ][active_before].reshape(-1)
                raw_mutual_path_collision_violation = action_filter[
                    "raw_mutual_path_collision_violation"
                ][active_before].reshape(-1)
                candidate_feasible = action_filter["candidate_feasible"][
                    active_before
                ].reshape(-1)
                safety_override = action_filter["safety_override"][
                    active_before
                ].reshape(-1)
                collision_override = action_filter["collision_override"][
                    active_before
                ].reshape(-1)
                raw_center_cost = action_filter["raw_visible_center_cost"][
                    active_before
                ].reshape(-1)
                filtered_center_cost = action_filter["filtered_visible_center_cost"][
                    active_before
                ].reshape(-1)
                suggested_center_cost = action_filter["suggested_visible_center_cost"][
                    active_before
                ].reshape(-1)
                center_progress_regression = action_filter["center_progress_regression"][
                    active_before
                ].reshape(-1)
                hold_zone_activation = action_filter["hold_zone_activation"][
                    active_before
                ].reshape(-1)
                hold_zone_rho_cost = action_filter["hold_zone_rho_cost"][
                    active_before
                ].reshape(-1)
                hold_zone_spacing_violation = action_filter[
                    "hold_zone_spacing_violation"
                ][active_before].reshape(-1)
                raw_hold_zone_rho_cost = action_filter["raw_hold_zone_rho_cost"][
                    active_before
                ].reshape(-1)
                raw_hold_zone_spacing_violation = action_filter[
                    "raw_hold_zone_spacing_violation"
                ][active_before].reshape(-1)
                candidate_index = action_filter["candidate_index"][active_before].reshape(-1)
                deterministic_applied = action_filter["deterministic_applied"][
                    active_before
                ].reshape(-1)
                raw_score = action_filter["raw_score"][active_before].reshape(-1)
                filtered_score = action_filter["filtered_score"][active_before].reshape(-1)
                score_margin = action_filter["score_margin"][active_before].reshape(-1)
                sample_count = torch.tensor(float(active_filter.numel()), device=env.device)
                filter_sample_count = filter_sample_count + sample_count
                filter_applied_sum = filter_applied_sum + active_filter.float().sum()
                filter_raw_risk_sum = filter_raw_risk_sum + raw_risk.sum()
                filter_filtered_risk_sum = filter_filtered_risk_sum + filtered_risk.sum()
                filter_risk_reduction_sum = filter_risk_reduction_sum + risk_reduction.sum()
                filter_subgoal_deviation_sum = filter_subgoal_deviation_sum + subgoal_deviation.sum()
                filter_suggested_subgoal_deviation_sum = (
                    filter_suggested_subgoal_deviation_sum + suggested_deviation.sum()
                )
                filter_endpoint_near_violation_sum = filter_endpoint_near_violation_sum + near_violation.sum()
                filter_endpoint_collision_violation_sum = (
                    filter_endpoint_collision_violation_sum + collision_violation.sum()
                )
                filter_endpoint_collision_violation_count = (
                    filter_endpoint_collision_violation_count
                    + (collision_violation > 0.0).float().sum()
                )
                filter_path_near_violation_sum = (
                    filter_path_near_violation_sum + path_near_violation.sum()
                )
                filter_path_collision_violation_sum = (
                    filter_path_collision_violation_sum + path_collision_violation.sum()
                )
                filter_path_collision_violation_count = (
                    filter_path_collision_violation_count
                    + (path_collision_violation > 0.0).float().sum()
                )
                filter_mutual_path_near_violation_sum = (
                    filter_mutual_path_near_violation_sum
                    + mutual_path_near_violation.sum()
                )
                filter_mutual_path_collision_violation_sum = (
                    filter_mutual_path_collision_violation_sum
                    + mutual_path_collision_violation.sum()
                )
                filter_mutual_path_collision_violation_count = (
                    filter_mutual_path_collision_violation_count
                    + (mutual_path_collision_violation > 0.0).float().sum()
                )
                filter_raw_endpoint_near_violation_sum = (
                    filter_raw_endpoint_near_violation_sum
                    + raw_endpoint_near_violation.sum()
                )
                filter_raw_endpoint_collision_violation_sum = (
                    filter_raw_endpoint_collision_violation_sum
                    + raw_endpoint_collision_violation.sum()
                )
                filter_raw_path_near_violation_sum = (
                    filter_raw_path_near_violation_sum + raw_path_near_violation.sum()
                )
                filter_raw_path_collision_violation_sum = (
                    filter_raw_path_collision_violation_sum
                    + raw_path_collision_violation.sum()
                )
                filter_raw_mutual_path_near_violation_sum = (
                    filter_raw_mutual_path_near_violation_sum
                    + raw_mutual_path_near_violation.sum()
                )
                filter_raw_mutual_path_collision_violation_sum = (
                    filter_raw_mutual_path_collision_violation_sum
                    + raw_mutual_path_collision_violation.sum()
                )
                filter_candidate_feasible_sum = (
                    filter_candidate_feasible_sum + candidate_feasible.float().sum()
                )
                filter_feasible_fraction_sum = filter_feasible_fraction_sum + (
                    sample_count * float(action_filter.get("feasible_fraction", 0.0))
                )
                filter_safety_override_sum = (
                    filter_safety_override_sum + safety_override.float().sum()
                )
                filter_collision_override_sum = (
                    filter_collision_override_sum + collision_override.float().sum()
                )
                filter_raw_center_cost_sum = filter_raw_center_cost_sum + raw_center_cost.sum()
                filter_filtered_center_cost_sum = (
                    filter_filtered_center_cost_sum + filtered_center_cost.sum()
                )
                filter_suggested_center_cost_sum = (
                    filter_suggested_center_cost_sum + suggested_center_cost.sum()
                )
                filter_center_progress_regression_sum = (
                    filter_center_progress_regression_sum
                    + center_progress_regression.sum()
                )
                filter_hold_zone_activation_sum = (
                    filter_hold_zone_activation_sum + hold_zone_activation.sum()
                )
                filter_hold_zone_rho_cost_sum = (
                    filter_hold_zone_rho_cost_sum + hold_zone_rho_cost.sum()
                )
                filter_hold_zone_spacing_violation_sum = (
                    filter_hold_zone_spacing_violation_sum
                    + hold_zone_spacing_violation.sum()
                )
                filter_raw_hold_zone_rho_cost_sum = (
                    filter_raw_hold_zone_rho_cost_sum + raw_hold_zone_rho_cost.sum()
                )
                filter_raw_hold_zone_spacing_violation_sum = (
                    filter_raw_hold_zone_spacing_violation_sum
                    + raw_hold_zone_spacing_violation.sum()
                )
                filter_candidate_index_sum = filter_candidate_index_sum + candidate_index.float().sum()
                filter_deterministic_applied_sum = (
                    filter_deterministic_applied_sum + deterministic_applied.float().sum()
                )
                filter_raw_score_sum = filter_raw_score_sum + raw_score.sum()
                filter_filtered_score_sum = filter_filtered_score_sum + filtered_score.sum()
                filter_score_margin_sum = filter_score_margin_sum + score_margin.sum()
                filter_apply_probability_sum = filter_apply_probability_sum + (
                    sample_count * float(action_filter.get("apply_probability", 0.0))
                )
                filter_score_scale_sum = filter_score_scale_sum + (
                    sample_count * float(action_filter.get("score_scale", 0.0))
                )
                filter_schedule_progress_step = max(
                    filter_schedule_progress_step,
                    int(action_filter.get("schedule_progress_step", 0)),
                )
                step_histogram = torch.bincount(
                    candidate_index,
                    minlength=max(filter_candidate_count, 1),
                ).to(dtype=torch.float32, device=env.device)
                if filter_candidate_histogram is None:
                    filter_candidate_histogram = torch.zeros_like(step_histogram)
                if filter_candidate_histogram.numel() < step_histogram.numel():
                    filter_candidate_histogram = F.pad(
                        filter_candidate_histogram,
                        (0, step_histogram.numel() - filter_candidate_histogram.numel()),
                    )
                filter_candidate_histogram[: step_histogram.numel()] += step_histogram
        control_safety = step_output.info.get("control_safety")
        if control_safety is not None:
            control_safety_enabled = control_safety_enabled or bool(
                control_safety.get("enabled", False)
            )
            active_scale = control_safety["linear_scale"][active_before].reshape(-1)
            if active_scale.numel() > 0:
                active_applied = control_safety["applied"][active_before].reshape(-1)
                active_risk = control_safety["pairwise_risk"][active_before].reshape(-1)
                active_predicted = control_safety["predicted_nearest_distance"][
                    active_before
                ].reshape(-1)
                sample_count = torch.tensor(float(active_scale.numel()), device=env.device)
                control_safety_sample_count = control_safety_sample_count + sample_count
                control_safety_applied_sum = (
                    control_safety_applied_sum + active_applied.float().sum()
                )
                control_safety_linear_scale_sum = (
                    control_safety_linear_scale_sum + active_scale.sum()
                )
                control_safety_linear_scale_min = torch.minimum(
                    control_safety_linear_scale_min,
                    active_scale.amin(),
                )
                control_safety_pairwise_risk_sum = (
                    control_safety_pairwise_risk_sum + active_risk.sum()
                )
                control_safety_predicted_nearest_sum = (
                    control_safety_predicted_nearest_sum + active_predicted.sum()
                )
            active_success_zone = control_safety["success_zone_active"][active_before]
            if active_success_zone.numel() > 0:
                env_count = torch.tensor(float(active_success_zone.numel()), device=env.device)
                control_safety_env_count = control_safety_env_count + env_count
                control_safety_success_zone_sum = (
                    control_safety_success_zone_sum + active_success_zone.float().sum()
                )
        formation_center_correction = step_output.info.get("formation_center_correction")
        if formation_center_correction is not None:
            active_correction = formation_center_correction["active"][active_before]
            active_offset = torch.linalg.norm(
                formation_center_correction["offset_xy"][active_before],
                dim=-1,
            )
            if active_correction.numel() > 0:
                env_count = torch.tensor(float(active_correction.numel()), device=env.device)
                formation_center_correction_env_count = (
                    formation_center_correction_env_count + env_count
                )
                formation_center_correction_active_sum = (
                    formation_center_correction_active_sum + active_correction.float().sum()
                )
                formation_center_correction_offset_sum = (
                    formation_center_correction_offset_sum + active_offset.sum()
                )
                formation_center_correction_offset_max = torch.maximum(
                    formation_center_correction_offset_max,
                    active_offset.amax(),
                )
            local_flatness_search_active = formation_center_correction.get(
                "local_flatness_search_active"
            )
            if local_flatness_search_active is not None:
                formation_center_local_flatness_search_active_sum = (
                    formation_center_local_flatness_search_active_sum
                    + local_flatness_search_active[active_before].float().sum()
                )
        terminal_slot_capture = step_output.info.get("terminal_slot_capture")
        if terminal_slot_capture is not None:
            active_capture = terminal_slot_capture["active"][active_before]
            if active_capture.numel() > 0:
                env_count = torch.tensor(float(active_capture.numel()), device=env.device)
                terminal_slot_capture_env_count = terminal_slot_capture_env_count + env_count
                terminal_slot_capture_active_sum = (
                    terminal_slot_capture_active_sum + active_capture.float().sum()
                )
        active = active & ~done.done
        if not active.any():
            break

    initial_dmax_mean = initial_dmax.mean()
    final_dmax_mean = final_dmax.mean()
    timeout_count = int(timeout_seen.sum().detach().cpu())
    timeout_episode_metrics = {
        "count": timeout_count,
        "final_dmax_mean": float(final_dmax[timeout_seen].mean().detach().cpu()) if timeout_seen.any() else None,
        "final_dispersion_mean": float(final_dispersion[timeout_seen].mean().detach().cpu())
        if timeout_seen.any()
        else None,
        "final_mean_speed_mean": float(final_mean_speed[timeout_seen].mean().detach().cpu())
        if timeout_seen.any()
        else None,
        "final_nearest_neighbor_distance_mean": float(
            final_nearest[timeout_seen].mean().detach().cpu()
        )
        if timeout_seen.any()
        else None,
        "final_nearest_neighbor_distance_min": float(
            final_nearest[timeout_seen].amin().detach().cpu()
        )
        if timeout_seen.any()
        else None,
        "final_min_pairwise_ok_rate": float(
            (
                final_nearest[timeout_seen]
                >= float(env.cfg.success_thresholds.min_pairwise_distance)
            )
            .float()
            .mean()
            .detach()
            .cpu()
        )
        if timeout_seen.any() and env.cfg.success_thresholds.min_pairwise_distance > 0.0
        else None,
        "final_flatness_ok_rate": float(
            final_flatness_ok[timeout_seen].float().mean().detach().cpu()
        )
        if timeout_seen.any()
        else None,
        "final_gather_point_is_flat_rate": float(
            final_gather_point_is_flat[timeout_seen].float().mean().detach().cpu()
        )
        if timeout_seen.any()
        else None,
        "final_gather_point_height_range_mean": float(
            final_gather_point_height_range[timeout_seen].mean().detach().cpu()
        )
        if timeout_seen.any()
        else None,
        "final_gather_point_max_slope_mean": float(
            final_gather_point_max_slope[timeout_seen].mean().detach().cpu()
        )
        if timeout_seen.any()
        else None,
        "final_success_hold_count_mean": float(
            final_success_hold_count[timeout_seen].float().mean().detach().cpu()
        )
        if timeout_seen.any()
        else None,
        "max_success_hold_count_mean": float(
            max_success_hold_count[timeout_seen].float().mean().detach().cpu()
        )
        if timeout_seen.any()
        else None,
        "mean_terrain_speed_scale": float(final_terrain_speed_scale[timeout_seen].mean().detach().cpu())
        if timeout_seen.any()
        else None,
    }
    hold_histogram = torch.bincount(
        max_success_hold_count.clamp(max=env.cfg.success_thresholds.hold_steps).detach().cpu(),
        minlength=env.cfg.success_thresholds.hold_steps + 1,
    )
    final_pairwise_safe = (
        final_nearest >= float(env.cfg.success_thresholds.min_pairwise_distance)
        if env.cfg.success_thresholds.min_pairwise_distance > 0.0
        else torch.ones_like(success_seen, dtype=torch.bool)
    )
    final_safe = final_pairwise_safe & final_flatness_ok
    result = {
        "status": "ok",
        "backend": backend,
        "config": str(config),
        "checkpoint": str(checkpoint),
        "run_dir": str(run_dir) if run_dir is not None else None,
        "device": str(env.device),
        "num_envs": env.num_envs,
        "steps": steps,
        "filter_progress_timestep": int(
            env.cfg.planner.subgoal_filter.progress_timestep_override
        ),
        "filter_progress_override": filter_progress_override,
        "gather_point_flatness": {
            "require_flat_for_success": bool(
                env.cfg.gather_point.require_flat_for_success
            ),
            "radius": float(env.cfg.gather_point.flatness_radius),
            "rings": int(env.cfg.gather_point.flatness_rings),
            "samples_per_ring": int(env.cfg.gather_point.flatness_samples_per_ring),
            "max_height_range": float(env.cfg.gather_point.max_height_range),
            "max_slope": float(env.cfg.gather_point.max_slope),
        },
        "oracle_search": {
            "method": env.cfg.gather_point.search_method,
            "feasible_rate": float(
                oracle_search_feasible.float().mean().detach().cpu()
            ),
            "objective_mean": float(oracle_search_objective.mean().detach().cpu()),
            "mean_distance": float(oracle_search_mean_distance.mean().detach().cpu()),
            "max_distance": float(oracle_search_max_distance.mean().detach().cpu()),
            "path_risk_mean": float(oracle_search_path_risk.mean().detach().cpu()),
            "path_height_change_mean": float(
                oracle_search_path_height_change.mean().detach().cpu()
            ),
            "height_range_mean": float(
                oracle_search_height_range.mean().detach().cpu()
            ),
            "max_slope_mean": float(oracle_search_max_slope.mean().detach().cpu()),
        },
        "initial_dmax": float(initial_dmax_mean.detach().cpu()),
        "final_dmax": float(final_dmax_mean.detach().cpu()),
        "dmax_reduction_ratio": float((final_dmax_mean / initial_dmax_mean.clamp_min(1.0e-6)).detach().cpu()),
        "initial_dispersion": float(initial_dispersion.mean().detach().cpu()),
        "final_dispersion": float(final_dispersion.mean().detach().cpu()),
        "final_mean_speed": float(final_mean_speed.mean().detach().cpu()),
        "final_nearest_neighbor_distance": float(final_nearest.mean().detach().cpu()),
        "final_flatness_ok_rate": float(final_flatness_ok.float().mean().detach().cpu()),
        "final_gather_point_is_flat_rate": float(
            final_gather_point_is_flat.float().mean().detach().cpu()
        ),
        "final_gather_point_height_range_mean": float(
            final_gather_point_height_range.mean().detach().cpu()
        ),
        "final_gather_point_max_slope_mean": float(
            final_gather_point_max_slope.mean().detach().cpu()
        ),
        "final_gather_point_mean_slope_mean": float(
            final_gather_point_mean_slope.mean().detach().cpu()
        ),
        "mean_reward": float((reward_sum / reward_count.clamp_min(1.0)).detach().cpu()),
        "success_rate": float(success_seen.float().mean().detach().cpu()),
        "safe_success_rate": float((success_seen & final_safe).float().mean().detach().cpu()),
        "collision_rate": float(collision_seen.float().mean().detach().cpu()),
        "timeout_rate": float(timeout_seen.float().mean().detach().cpu()),
        "finished_rate": float((~active).float().mean().detach().cpu()),
        "mean_done_step": float(done_step[done_step > 0].float().mean().detach().cpu())
        if (done_step > 0).any()
        else None,
        "min_nearest_distance": float(global_min_nearest.detach().cpu()) if torch.isfinite(global_min_nearest) else None,
        "mean_nearest_distance": float((nearest_sum / nearest_count.clamp_min(1.0)).detach().cpu()),
        "near_violation_rate": float((near_violation_count / nearest_count.clamp_min(1.0)).detach().cpu()),
        "first_collision_step_mean": float(first_collision_step[first_collision_step > 0].float().mean().detach().cpu())
        if (first_collision_step > 0).any()
        else None,
        "collision_episode_ids": torch.nonzero(collision_seen, as_tuple=False).flatten().detach().cpu().tolist(),
        "mean_terrain_height": float((terrain_height_sum / terrain_height_count.clamp_min(1.0)).detach().cpu()),
        "terrain_height_range": float((terrain_height_max - terrain_height_min).detach().cpu())
        if torch.isfinite(terrain_height_min) and torch.isfinite(terrain_height_max)
        else 0.0,
        "mean_roughness": float((terrain_roughness_sum / terrain_height_count.clamp_min(1.0)).detach().cpu()),
        "max_roughness": float(terrain_roughness_max.detach().cpu()),
        "min_traversability": float(terrain_traversability_min.detach().cpu())
        if torch.isfinite(terrain_traversability_min)
        else 1.0,
        "mean_terrain_speed_scale": float(
            (terrain_speed_scale_sum / terrain_speed_scale_count.clamp_min(1.0)).detach().cpu()
        ),
        "dmax_ok_rate": float((dmax_ok_sum / gate_sample_count.clamp_min(1.0)).detach().cpu()),
        "dispersion_ok_rate": float((dispersion_ok_sum / gate_sample_count.clamp_min(1.0)).detach().cpu()),
        "speed_ok_rate": float((speed_ok_sum / gate_sample_count.clamp_min(1.0)).detach().cpu()),
        "min_pairwise_ok_rate": float((min_pairwise_ok_sum / gate_sample_count.clamp_min(1.0)).detach().cpu()),
        "flatness_ok_rate": float(
            (flatness_ok_sum / gate_sample_count.clamp_min(1.0)).detach().cpu()
        ),
        "gather_point_is_flat_rate": float(
            (gather_point_is_flat_sum / gate_sample_count.clamp_min(1.0)).detach().cpu()
        ),
        "instant_success_rate": float((instant_success_sum / gate_sample_count.clamp_min(1.0)).detach().cpu()),
        "path_terrain_risk_mean": float((path_risk_mean_sum / path_risk_count.clamp_min(1.0)).detach().cpu()),
        "path_terrain_risk_max": float(path_risk_max.detach().cpu()),
        "path_height_change_mean": float(
            (path_height_change_sum / path_height_change_count.clamp_min(1.0)).detach().cpu()
        ),
        "filter_candidate_count": filter_candidate_count,
        "filter_applied_fraction": float(
            (filter_applied_sum / filter_sample_count.clamp_min(1.0)).detach().cpu()
        ),
        "filter_raw_path_terrain_risk_mean": float(
            (filter_raw_risk_sum / filter_sample_count.clamp_min(1.0)).detach().cpu()
        ),
        "filter_filtered_path_terrain_risk_mean": float(
            (filter_filtered_risk_sum / filter_sample_count.clamp_min(1.0)).detach().cpu()
        ),
        "filter_path_terrain_risk_reduction_mean": float(
            (filter_risk_reduction_sum / filter_sample_count.clamp_min(1.0)).detach().cpu()
        ),
        "filter_subgoal_deviation_mean": float(
            (filter_subgoal_deviation_sum / filter_sample_count.clamp_min(1.0)).detach().cpu()
        ),
        "filter_suggested_subgoal_deviation_mean": float(
            (filter_suggested_subgoal_deviation_sum / filter_sample_count.clamp_min(1.0)).detach().cpu()
        ),
        "filter_endpoint_near_violation_mean": float(
            (filter_endpoint_near_violation_sum / filter_sample_count.clamp_min(1.0)).detach().cpu()
        ),
        "filter_endpoint_collision_violation_mean": float(
            (filter_endpoint_collision_violation_sum / filter_sample_count.clamp_min(1.0)).detach().cpu()
        ),
        "filter_endpoint_collision_violation_fraction": float(
            (filter_endpoint_collision_violation_count / filter_sample_count.clamp_min(1.0)).detach().cpu()
        ),
        "filter_path_near_violation_mean": float(
            (filter_path_near_violation_sum / filter_sample_count.clamp_min(1.0)).detach().cpu()
        ),
        "filter_path_collision_violation_mean": float(
            (filter_path_collision_violation_sum / filter_sample_count.clamp_min(1.0)).detach().cpu()
        ),
        "filter_path_collision_violation_fraction": float(
            (filter_path_collision_violation_count / filter_sample_count.clamp_min(1.0)).detach().cpu()
        ),
        "filter_mutual_path_near_violation_mean": float(
            (
                filter_mutual_path_near_violation_sum
                / filter_sample_count.clamp_min(1.0)
            )
            .detach()
            .cpu()
        ),
        "filter_mutual_path_collision_violation_mean": float(
            (
                filter_mutual_path_collision_violation_sum
                / filter_sample_count.clamp_min(1.0)
            )
            .detach()
            .cpu()
        ),
        "filter_mutual_path_collision_violation_fraction": float(
            (
                filter_mutual_path_collision_violation_count
                / filter_sample_count.clamp_min(1.0)
            )
            .detach()
            .cpu()
        ),
        "filter_raw_endpoint_near_violation_mean": float(
            (filter_raw_endpoint_near_violation_sum / filter_sample_count.clamp_min(1.0)).detach().cpu()
        ),
        "filter_raw_endpoint_collision_violation_mean": float(
            (
                filter_raw_endpoint_collision_violation_sum
                / filter_sample_count.clamp_min(1.0)
            )
            .detach()
            .cpu()
        ),
        "filter_raw_path_near_violation_mean": float(
            (filter_raw_path_near_violation_sum / filter_sample_count.clamp_min(1.0)).detach().cpu()
        ),
        "filter_raw_path_collision_violation_mean": float(
            (filter_raw_path_collision_violation_sum / filter_sample_count.clamp_min(1.0)).detach().cpu()
        ),
        "filter_raw_mutual_path_near_violation_mean": float(
            (
                filter_raw_mutual_path_near_violation_sum
                / filter_sample_count.clamp_min(1.0)
            )
            .detach()
            .cpu()
        ),
        "filter_raw_mutual_path_collision_violation_mean": float(
            (
                filter_raw_mutual_path_collision_violation_sum
                / filter_sample_count.clamp_min(1.0)
            )
            .detach()
            .cpu()
        ),
        "filter_candidate_feasible_fraction": float(
            (filter_candidate_feasible_sum / filter_sample_count.clamp_min(1.0)).detach().cpu()
        ),
        "filter_feasible_fraction": float(
            (filter_feasible_fraction_sum / filter_sample_count.clamp_min(1.0)).detach().cpu()
        ),
        "filter_safety_override_fraction": float(
            (filter_safety_override_sum / filter_sample_count.clamp_min(1.0)).detach().cpu()
        ),
        "filter_collision_override_fraction": float(
            (filter_collision_override_sum / filter_sample_count.clamp_min(1.0)).detach().cpu()
        ),
        "filter_raw_visible_center_cost_mean": float(
            (filter_raw_center_cost_sum / filter_sample_count.clamp_min(1.0)).detach().cpu()
        ),
        "filter_filtered_visible_center_cost_mean": float(
            (filter_filtered_center_cost_sum / filter_sample_count.clamp_min(1.0)).detach().cpu()
        ),
        "filter_suggested_visible_center_cost_mean": float(
            (filter_suggested_center_cost_sum / filter_sample_count.clamp_min(1.0)).detach().cpu()
        ),
        "filter_center_progress_regression_mean": float(
            (filter_center_progress_regression_sum / filter_sample_count.clamp_min(1.0)).detach().cpu()
        ),
        "filter_hold_zone_activation_mean": float(
            (filter_hold_zone_activation_sum / filter_sample_count.clamp_min(1.0)).detach().cpu()
        ),
        "filter_hold_zone_rho_cost_mean": float(
            (filter_hold_zone_rho_cost_sum / filter_sample_count.clamp_min(1.0)).detach().cpu()
        ),
        "filter_hold_zone_spacing_violation_mean": float(
            (
                filter_hold_zone_spacing_violation_sum
                / filter_sample_count.clamp_min(1.0)
            )
            .detach()
            .cpu()
        ),
        "filter_raw_hold_zone_rho_cost_mean": float(
            (filter_raw_hold_zone_rho_cost_sum / filter_sample_count.clamp_min(1.0)).detach().cpu()
        ),
        "filter_raw_hold_zone_spacing_violation_mean": float(
            (
                filter_raw_hold_zone_spacing_violation_sum
                / filter_sample_count.clamp_min(1.0)
            )
            .detach()
            .cpu()
        ),
        "filter_candidate_index_mean": float(
            (filter_candidate_index_sum / filter_sample_count.clamp_min(1.0)).detach().cpu()
        ),
        "filter_deterministic_applied_fraction": float(
            (filter_deterministic_applied_sum / filter_sample_count.clamp_min(1.0)).detach().cpu()
        ),
        "filter_raw_score_mean": float(
            (filter_raw_score_sum / filter_sample_count.clamp_min(1.0)).detach().cpu()
        ),
        "filter_filtered_score_mean": float(
            (filter_filtered_score_sum / filter_sample_count.clamp_min(1.0)).detach().cpu()
        ),
        "filter_score_margin_mean": float(
            (filter_score_margin_sum / filter_sample_count.clamp_min(1.0)).detach().cpu()
        ),
        "filter_apply_probability_mean": float(
            (filter_apply_probability_sum / filter_sample_count.clamp_min(1.0)).detach().cpu()
        ),
        "filter_score_scale_mean": float(
            (filter_score_scale_sum / filter_sample_count.clamp_min(1.0)).detach().cpu()
        ),
        "filter_schedule_progress_step": filter_schedule_progress_step,
        "filter_candidate_index_histogram": (
            {
                str(index): float((count / filter_candidate_histogram.sum().clamp_min(1.0)).detach().cpu())
                for index, count in enumerate(filter_candidate_histogram)
            }
            if filter_candidate_histogram is not None
            else {}
        ),
        "control_safety_enabled": control_safety_enabled,
        "control_safety_applied_fraction": float(
            (control_safety_applied_sum / control_safety_sample_count.clamp_min(1.0)).detach().cpu()
        ),
        "control_safety_linear_scale_mean": float(
            (
                control_safety_linear_scale_sum
                / control_safety_sample_count.clamp_min(1.0)
            )
            .detach()
            .cpu()
        ),
        "control_safety_linear_scale_min": float(
            (
                control_safety_linear_scale_min
                if torch.isfinite(control_safety_linear_scale_min)
                else torch.tensor(1.0, device=env.device)
            )
            .detach()
            .cpu()
        ),
        "control_safety_pairwise_risk_mean": float(
            (
                control_safety_pairwise_risk_sum
                / control_safety_sample_count.clamp_min(1.0)
            )
            .detach()
            .cpu()
        ),
        "control_safety_predicted_nearest_mean": float(
            (
                control_safety_predicted_nearest_sum
                / control_safety_sample_count.clamp_min(1.0)
            )
            .detach()
            .cpu()
        ),
        "control_safety_success_zone_fraction": float(
            (
                control_safety_success_zone_sum
                / control_safety_env_count.clamp_min(1.0)
            )
            .detach()
            .cpu()
        ),
        "formation_center_correction_active_fraction": float(
            (
                formation_center_correction_active_sum
                / formation_center_correction_env_count.clamp_min(1.0)
            )
            .detach()
            .cpu()
        ),
        "formation_center_correction_offset_mean": float(
            (
                formation_center_correction_offset_sum
                / formation_center_correction_env_count.clamp_min(1.0)
            )
            .detach()
            .cpu()
        ),
        "formation_center_correction_offset_max": float(
            formation_center_correction_offset_max.detach().cpu()
        ),
        "formation_center_local_flatness_search_active_fraction": float(
            (
                formation_center_local_flatness_search_active_sum
                / formation_center_correction_env_count.clamp_min(1.0)
            )
            .detach()
            .cpu()
        ),
        "terminal_slot_capture_active_fraction": float(
            (terminal_slot_capture_active_sum / terminal_slot_capture_env_count.clamp_min(1.0))
            .detach()
            .cpu()
        ),
        "max_success_hold_count_mean": float(max_success_hold_count.float().mean().detach().cpu()),
        "final_success_hold_count_mean": float(final_success_hold_count.float().mean().detach().cpu()),
        "hold_count_histogram": {
            str(index): int(count.item()) for index, count in enumerate(hold_histogram)
        },
        "timeout_episode_metrics": timeout_episode_metrics,
    }
    output_path = _resolve_path(output)
    if output_path is not None:
        with output_path.open("w", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2)
        result["artifact"] = str(output_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment/exp_001_minimal.yaml")
    parser.add_argument("--checkpoint", default="outputs/checkpoints/exp_001_minimal_proxy.pt")
    parser.add_argument("--device", default=None)
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument(
        "--filter-progress-override",
        type=int,
        default=None,
        help="Override the subgoal-filter curriculum step for this evaluation only.",
    )
    args = parser.parse_args()
    result = evaluate_checkpoint(
        args.config,
        args.checkpoint,
        device=args.device,
        num_envs=args.num_envs,
        steps=args.steps,
        seed=args.seed,
        output=args.output,
        run_dir=args.run_dir,
        filter_progress_override=args.filter_progress_override,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
