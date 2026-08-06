#!/usr/bin/env python
"""Decompose collision-avoidance controllability across planning layers."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from _common import ROOT, cfg_from_experiment, load_yaml
from analyze_collision_participant_credit_feasibility import _distribution, _upper_triangle
from analyze_joint_action_critic_feasibility import _parse_int_tuple
from analyze_paired_action_interventions import _policy_digest
from lunar_rover_tasks.tasks.multi_rover_gathering.action_interpreter import decode_action
from lunar_rover_tasks.tasks.multi_rover_gathering.gathering_env import MultiRoverGatheringCore
from lunar_rover_tasks.tasks.multi_rover_gathering.mapf_diagnostics import (
    trajectory_pairwise_min_distance,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.metrics import compute_team_metrics
from lunar_rover_tasks.tasks.multi_rover_gathering.simple_controller import compute_control
from lunar_rover_tasks.tasks.multi_rover_gathering.terrain_features import (
    sample_trajectory_terrain_risk,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.trajectory_generator import (
    generate_trajectory,
)
from lunar_rover_tasks.utils.geometry_utils import pairwise_distances_xy
from play import _load_policy_players


EXPERIMENT_ID = "exp152_action_planning_controllability_decomposition"
DEFAULT_RUN_ID = "frozen_exp150_dual_checkpoint_dualseed"
DEFAULT_EXP151_SUMMARY = (
    ROOT
    / "outputs/runs/exp151_collision_credit_causal_validity/_suite/metrics/suite_summary.json"
)
HORIZONS = (1, 2, 4, 8, 16)
LAYER_NAMES = ("unconstrained", "risk", "dmax", "joint")


def _full_planning_outcomes(
    env: MultiRoverGatheringCore,
    actions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    decoded = decode_action(actions, env.positions, env.yaws, env.cfg.planner)
    trajectory = generate_trajectory(
        env.positions,
        decoded.world_subgoal,
        env.cfg.trajectory_generator,
        env.cfg.simulation.planning_dt,
        current_yaws=env.yaws,
    )
    pair_distance = trajectory_pairwise_min_distance(
        trajectory.points,
        trajectory.timestamps,
    )
    terrain_risk = sample_trajectory_terrain_risk(
        trajectory.points,
        env.cfg.terrain,
        env.terrain_runtime,
    )["risk_mean"]
    endpoint_metrics = compute_team_metrics(
        trajectory.points[..., -1, :],
        torch.zeros_like(trajectory.points[..., -1, :2]),
    )
    control = compute_control(
        env.positions,
        env.yaws,
        trajectory,
        env.cfg.low_level_control,
        env.cfg.simulation.planning_dt,
    ).packed
    return pair_distance, terrain_risk, endpoint_metrics.dmax, control


def layered_counterfactual_outcomes(
    env: MultiRoverGatheringCore,
    actions: torch.Tensor,
    *,
    action_delta: float,
    terrain_risk_tolerance: float,
    endpoint_dmax_tolerance: float,
) -> dict[str, torch.Tensor]:
    """Return best safety gain under nested constraints and its control response."""

    if action_delta <= 0.0:
        raise ValueError("action_delta must be positive")
    if terrain_risk_tolerance < 0.0 or endpoint_dmax_tolerance < 0.0:
        raise ValueError("counterfactual tolerances must be non-negative")
    if actions.shape != (env.num_envs, env.n_agents, 2):
        raise ValueError("actions must have shape [environment, agent, 2]")

    baseline_pair, baseline_risk, baseline_dmax, baseline_control = (
        _full_planning_outcomes(env, actions)
    )
    shape = (env.num_envs, env.n_agents, env.n_agents, env.n_agents)
    best = {
        name: torch.zeros(shape, device=env.device, dtype=actions.dtype)
        for name in LAYER_NAMES
    }
    best_radius = torch.zeros_like(best["unconstrained"])
    best_bearing = torch.zeros_like(best["unconstrained"])
    best_control_response = torch.zeros_like(best["unconstrained"])
    best_control_linear_delta = torch.zeros_like(best["unconstrained"])
    best_control_angular_delta = torch.zeros_like(best["unconstrained"])
    best_candidate_linear_saturated = torch.zeros_like(best["unconstrained"])
    best_candidate_angular_saturated = torch.zeros_like(best["unconstrained"])
    baseline_linear_saturated = torch.zeros_like(best["unconstrained"])
    baseline_angular_saturated = torch.zeros_like(best["unconstrained"])
    max_linear = float(env.cfg.low_level_control.max_linear_speed)
    max_angular = float(env.cfg.low_level_control.max_angular_speed)
    saturation_eps = 1.0e-5

    for modifier in range(env.n_agents):
        base_linear_sat = (
            baseline_control[:, modifier, 0] >= max_linear - saturation_eps
        ).float()[:, None, None]
        base_angular_sat = (
            baseline_control[:, modifier, 1].abs() >= max_angular - saturation_eps
        ).float()[:, None, None]
        baseline_linear_saturated[:, modifier] = base_linear_sat
        baseline_angular_saturated[:, modifier] = base_angular_sat
        for dimension in range(2):
            for sign in (-1.0, 1.0):
                candidate = actions.clone()
                candidate[:, modifier, dimension] = (
                    candidate[:, modifier, dimension] + sign * float(action_delta)
                ).clamp(-1.0, 1.0)
                pair, risk, endpoint_dmax, control = _full_planning_outcomes(
                    env, candidate
                )
                gain = pair - baseline_pair
                risk_ok = (
                    risk[:, modifier] - baseline_risk[:, modifier]
                    <= float(terrain_risk_tolerance)
                )
                dmax_ok = endpoint_dmax - baseline_dmax <= float(
                    endpoint_dmax_tolerance
                )
                linear_delta = (
                    control[:, modifier, 0] - baseline_control[:, modifier, 0]
                ).abs()
                angular_delta = (
                    control[:, modifier, 1] - baseline_control[:, modifier, 1]
                ).abs()
                response = torch.sqrt(
                    (linear_delta / max(max_linear, 1.0e-8)).square()
                    + (angular_delta / max(max_angular, 1.0e-8)).square()
                )
                candidate_linear_sat = (
                    control[:, modifier, 0] >= max_linear - saturation_eps
                ).float()
                candidate_angular_sat = (
                    control[:, modifier, 1].abs() >= max_angular - saturation_eps
                ).float()

                current = best["unconstrained"][:, modifier]
                improve = gain > current
                best["unconstrained"][:, modifier] = torch.where(
                    improve, gain, current
                )
                for target, value in (
                    (best_control_response, response),
                    (best_control_linear_delta, linear_delta),
                    (best_control_angular_delta, angular_delta),
                    (best_candidate_linear_saturated, candidate_linear_sat),
                    (best_candidate_angular_saturated, candidate_angular_sat),
                ):
                    target[:, modifier] = torch.where(
                        improve,
                        value[:, None, None].expand_as(gain),
                        target[:, modifier],
                    )

                dimension_target = best_radius if dimension == 0 else best_bearing
                dimension_target[:, modifier] = torch.maximum(
                    dimension_target[:, modifier], gain
                )
                conditions = {
                    "risk": risk_ok,
                    "dmax": dmax_ok,
                    "joint": risk_ok & dmax_ok,
                }
                for name, condition in conditions.items():
                    best[name][:, modifier] = torch.where(
                        condition[:, None, None]
                        & (gain > best[name][:, modifier]),
                        gain,
                        best[name][:, modifier],
                    )

    return {
        "baseline_pair_distance": baseline_pair,
        **{f"best_{name}_pair_gain": value for name, value in best.items()},
        "best_radius_pair_gain": best_radius,
        "best_bearing_pair_gain": best_bearing,
        "best_unconstrained_control_response": best_control_response,
        "best_unconstrained_control_linear_delta": best_control_linear_delta,
        "best_unconstrained_control_angular_delta": best_control_angular_delta,
        "baseline_linear_saturated": baseline_linear_saturated,
        "baseline_angular_saturated": baseline_angular_saturated,
        "best_candidate_linear_saturated": best_candidate_linear_saturated,
        "best_candidate_angular_saturated": best_candidate_angular_saturated,
    }


def collect_layered_timeline(
    *,
    config: str | Path,
    checkpoint_data: dict[str, Any],
    device: str,
    num_envs: int,
    steps: int,
    seed: int,
    action_delta: float,
    terrain_risk_tolerance: float,
    endpoint_dmax_tolerance: float,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    raw_cfg = load_yaml(config)
    cfg = cfg_from_experiment(config)
    cfg.simulation.device = device
    cfg.simulation.num_envs = num_envs
    cfg.seed = seed
    metadata = checkpoint_data.get("metadata") or {}
    if cfg.initial_state.curriculum_enabled:
        cfg.initial_state.progress_timestep_override = int(metadata.get("timesteps", 0))
    if cfg.planner.subgoal_filter.enabled:
        raise ValueError("exp152 requires the strict unfiltered execution chain")
    forbidden = (
        cfg.low_level_control.safety_projection_enabled,
        cfg.low_level_control.success_zone_damping_enabled,
        cfg.low_level_control.formation_center_correction_enabled,
        cfg.low_level_control.terminal_slot_capture_enabled,
        cfg.low_level_control.flat_geometry_capture_enabled,
    )
    if any(forbidden):
        raise ValueError("exp152 requires every execution override to remain disabled")

    env = MultiRoverGatheringCore(cfg)
    act, _ = _load_policy_players(checkpoint_data, cfg, env.device, raw_cfg=raw_cfg)
    actor_obs, _ = env.get_observations()
    policy_std = (
        checkpoint_data["rover_0"]["policy"]["log_std_parameter"]
        .detach()
        .to(env.device)
        .exp()
        .view(1, 1, 2)
    )
    generator = torch.Generator(device=env.device)
    # Deliberately identical to exp151 so the joint layer must reconstruct it.
    generator.manual_seed(seed + 151_151)
    probe_obs = actor_obs[: min(num_envs, 32)].clone()
    with torch.no_grad():
        probe_before = act(probe_obs).clone()

    sequence_names = (
        "baseline_pair_distance",
        "best_unconstrained_pair_gain",
        "best_risk_pair_gain",
        "best_dmax_pair_gain",
        "best_joint_pair_gain",
        "best_radius_pair_gain",
        "best_bearing_pair_gain",
        "best_unconstrained_control_response",
        "best_unconstrained_control_linear_delta",
        "best_unconstrained_control_angular_delta",
        "baseline_linear_saturated",
        "baseline_angular_saturated",
        "best_candidate_linear_saturated",
        "best_candidate_angular_saturated",
    )
    sequences: dict[str, list[torch.Tensor]] = {name: [] for name in sequence_names}
    sequences.update(
        {
            "positions_after": [],
            "done": [],
            "collision_done": [],
            "sampled_actions": [],
        }
    )
    executed_action_max_error = 0.0
    for _ in range(steps):
        with torch.no_grad():
            policy_mean = act(actor_obs)
            noise = torch.randn(
                policy_mean.shape,
                generator=generator,
                device=env.device,
                dtype=policy_mean.dtype,
            )
            sampled_actions = (policy_mean + policy_std * noise).clamp(-1.0, 1.0)
            counterfactual = layered_counterfactual_outcomes(
                env,
                sampled_actions,
                action_delta=action_delta,
                terrain_risk_tolerance=terrain_risk_tolerance,
                endpoint_dmax_tolerance=endpoint_dmax_tolerance,
            )
            action_sent = sampled_actions.clone()
            output = env.step(action_sent)
            executed_action_max_error = max(
                executed_action_max_error,
                float((action_sent - sampled_actions).abs().amax().cpu()),
            )
            for name in sequence_names:
                sequences[name].append(counterfactual[name].cpu())
            sequences["positions_after"].append(output.info["positions"].cpu())
            sequences["done"].append(output.info["done"].done.cpu())
            sequences["collision_done"].append(output.info["done"].collision.cpu())
            sequences["sampled_actions"].append(sampled_actions.cpu())
            actor_obs = output.actor_obs

    with torch.no_grad():
        probe_after = act(probe_obs)
    return (
        {name: torch.stack(values) for name, values in sequences.items()},
        {
            "actor_probe_action_max_abs_change": float(
                (probe_after - probe_before).abs().amax().cpu()
            ),
            "executed_action_max_abs_error": executed_action_max_error,
            "policy_std": policy_std.flatten().cpu().tolist(),
        },
    )


def summarize_layered_timeline(
    timeline: dict[str, torch.Tensor],
    *,
    collision_distance: float,
    actionable_gain: float = 0.02,
    control_response_threshold: float = 0.05,
    horizons: tuple[int, ...] = HORIZONS,
) -> dict[str, Any]:
    required = {
        "positions_after",
        "done",
        "collision_done",
        "sampled_actions",
        "best_radius_pair_gain",
        "best_bearing_pair_gain",
        "best_unconstrained_control_response",
        "best_unconstrained_control_linear_delta",
        "best_unconstrained_control_angular_delta",
        "baseline_linear_saturated",
        "baseline_angular_saturated",
        "best_candidate_linear_saturated",
        "best_candidate_angular_saturated",
        *(f"best_{name}_pair_gain" for name in LAYER_NAMES),
    }
    missing = required - timeline.keys()
    if missing:
        raise ValueError(f"timeline is missing keys: {sorted(missing)}")
    done = timeline["done"].bool()
    collision_done = timeline["collision_done"].bool()
    if done.ndim != 2 or collision_done.shape != done.shape:
        raise ValueError("done and collision_done must have shape [time, environment]")
    time_steps, num_envs = done.shape
    n_agents = timeline["sampled_actions"].shape[2]
    sample_names = (
        *(f"{name}_actionable" for name in LAYER_NAMES),
        "terrain_blocked",
        "dmax_blocked",
        "cross_constraint_incompatibility",
        "combined_feasibility_loss",
        "control_transmitted",
        "control_response",
        "control_linear_delta",
        "control_angular_delta",
        "radius_best",
        "bearing_best",
        "baseline_linear_saturated",
        "baseline_angular_saturated",
        "candidate_linear_saturated",
        "candidate_angular_saturated",
        "nonparticipant_pair_gain_abs",
    )
    samples = {
        horizon: {name: [] for name in sample_names} for horizon in horizons
    }
    collision_episodes = 0
    collision_pairs = 0
    last_done = [-1 for _ in range(num_envs)]
    for t in range(time_steps):
        for env_id in range(num_envs):
            if bool(collision_done[t, env_id]):
                positions = timeline["positions_after"][t, env_id, :, :2]
                distances = pairwise_distances_xy(positions.unsqueeze(0))[0]
                pairs = torch.nonzero(
                    _upper_triangle(distances < float(collision_distance)),
                    as_tuple=False,
                ).tolist()
                if pairs:
                    collision_episodes += 1
                collision_pairs += len(pairs)
                episode_start = last_done[env_id] + 1
                for first, second in pairs:
                    participants = [first, second]
                    for horizon in horizons:
                        step = t - horizon
                        if step < episode_start or step < 0:
                            continue
                        row = samples[horizon]
                        gains = {
                            name: timeline[f"best_{name}_pair_gain"][
                                step, env_id, participants, first, second
                            ].clamp_min(0.0)
                            for name in LAYER_NAMES
                        }
                        flags = {
                            name: bool((value >= float(actionable_gain)).any())
                            for name, value in gains.items()
                        }
                        for name, flag in flags.items():
                            row[f"{name}_actionable"].append(float(flag))
                        if flags["unconstrained"]:
                            row["terrain_blocked"].append(float(not flags["risk"]))
                            row["dmax_blocked"].append(float(not flags["dmax"]))
                            row["cross_constraint_incompatibility"].append(
                                float(flags["risk"] and flags["dmax"] and not flags["joint"])
                            )
                            row["combined_feasibility_loss"].append(
                                float(not flags["joint"])
                            )
                            responsible_local = int(
                                torch.argmax(gains["unconstrained"]).item()
                            )
                            responsible = participants[responsible_local]
                            response = float(
                                timeline["best_unconstrained_control_response"][
                                    step, env_id, responsible, first, second
                                ]
                            )
                            row["control_response"].append(response)
                            row["control_transmitted"].append(
                                float(response >= float(control_response_threshold))
                            )
                            row["control_linear_delta"].append(
                                float(
                                    timeline[
                                        "best_unconstrained_control_linear_delta"
                                    ][step, env_id, responsible, first, second]
                                )
                            )
                            row["control_angular_delta"].append(
                                float(
                                    timeline[
                                        "best_unconstrained_control_angular_delta"
                                    ][step, env_id, responsible, first, second]
                                )
                            )
                            for output_name, tensor_name in (
                                ("baseline_linear_saturated", "baseline_linear_saturated"),
                                ("baseline_angular_saturated", "baseline_angular_saturated"),
                                ("candidate_linear_saturated", "best_candidate_linear_saturated"),
                                ("candidate_angular_saturated", "best_candidate_angular_saturated"),
                            ):
                                row[output_name].append(
                                    float(
                                        timeline[tensor_name][
                                            step, env_id, responsible, first, second
                                        ]
                                    )
                                )
                            radius = float(
                                timeline["best_radius_pair_gain"][
                                    step, env_id, participants, first, second
                                ].amax()
                            )
                            bearing = float(
                                timeline["best_bearing_pair_gain"][
                                    step, env_id, participants, first, second
                                ].amax()
                            )
                            row["bearing_best"].append(float(bearing > radius))
                            row["radius_best"].append(float(radius >= bearing))
                        for modifier in range(n_agents):
                            if modifier not in participants:
                                row["nonparticipant_pair_gain_abs"].append(
                                    abs(
                                        float(
                                            timeline[
                                                "best_unconstrained_pair_gain"
                                            ][step, env_id, modifier, first, second]
                                        )
                                    )
                                )
            if bool(done[t, env_id]):
                last_done[env_id] = t

    horizon_summary: dict[str, Any] = {}
    nonparticipant_max = 0.0
    for horizon, horizon_samples in samples.items():
        row = {"pair_samples": len(horizon_samples["joint_actionable"])}
        for name, values in horizon_samples.items():
            row[name] = _distribution(values)
        local_max = max(horizon_samples["nonparticipant_pair_gain_abs"], default=0.0)
        row["nonparticipant_pair_gain_abs_max"] = local_max
        nonparticipant_max = max(nonparticipant_max, local_max)
        horizon_summary[str(horizon)] = row
    return {
        "collision_episodes": collision_episodes,
        "collision_pairs": collision_pairs,
        "horizons": horizon_summary,
        "nonparticipant_pair_gain_abs_max": nonparticipant_max,
    }


def controllability_decision(combinations: dict[str, Any]) -> dict[str, Any]:
    engineering_passed = all(item["passed"] for item in combinations.values())

    def values(horizon: str, metric: str) -> list[float]:
        return [
            item["summary"]["horizons"][horizon][metric]["mean"]
            for item in combinations.values()
        ]

    u8 = values("8", "unconstrained_actionable")
    u16 = values("16", "unconstrained_actionable")
    control8 = values("8", "control_transmitted")
    joint8 = values("8", "joint_actionable")
    terrain8 = values("8", "terrain_blocked")
    dmax8 = values("8", "dmax_blocked")
    action_quintic_passed = (
        engineering_passed
        and min(u8, default=0.0) >= 0.70
        and min(u16, default=0.0) >= 0.60
    )
    control_passed = action_quintic_passed and min(control8, default=0.0) >= 0.70
    joint_passed = control_passed and min(joint8, default=0.0) >= 0.70
    terrain_dominant = (
        control_passed
        and not joint_passed
        and min(terrain8, default=0.0) >= 0.25
        and all(t >= d + 0.10 for t, d in zip(terrain8, dmax8, strict=True))
    )
    dmax_dominant = (
        control_passed
        and not joint_passed
        and min(dmax8, default=0.0) >= 0.25
        and all(d >= t + 0.10 for t, d in zip(terrain8, dmax8, strict=True))
    )
    if not engineering_passed:
        status = "invalid_diagnostic_stop"
        next_stage = "fix_diagnostic_only"
    elif not action_quintic_passed:
        status = "action_quintic_bottleneck"
        next_stage = "audit_action_range_quintic_geometry_only"
    elif not control_passed:
        status = "low_level_control_bottleneck"
        next_stage = "audit_low_level_control_contract_only"
    elif not joint_passed:
        if terrain_dominant:
            attribution = "terrain_constraint_dominant"
        elif dmax_dominant:
            attribution = "gather_constraint_dominant"
        else:
            attribution = "coupled_constraints"
        status = "objective_tradeoff_bottleneck"
        next_stage = "frozen_multiobjective_geometry_audit_only"
    else:
        attribution = "none"
        status = "inconsistent_with_exp151_stop"
        next_stage = "fix_diagnostic_only"
    return {
        "status": status,
        "next_stage": next_stage,
        "training_authorized": False,
        "engineering_passed": engineering_passed,
        "action_quintic_passed": action_quintic_passed,
        "control_passed": control_passed,
        "joint_passed": joint_passed,
        "attribution": locals().get("attribution", "not_reached"),
        "cross_combination": {
            "h8_unconstrained_actionable_min": min(u8, default=0.0),
            "h16_unconstrained_actionable_min": min(u16, default=0.0),
            "h8_control_transmitted_min": min(control8, default=0.0),
            "h8_joint_actionable_min": min(joint8, default=0.0),
            "h8_terrain_blocked_min": min(terrain8, default=0.0),
            "h8_dmax_blocked_min": min(dmax8, default=0.0),
        },
        "thresholds": {
            "h8_unconstrained_actionable": 0.70,
            "h16_unconstrained_actionable": 0.60,
            "h8_control_transmitted": 0.70,
            "h8_joint_actionable": 0.70,
            "dominant_block_rate": 0.25,
            "dominant_margin": 0.10,
        },
    }


def _combination_checks(
    summary: dict[str, Any],
    *,
    expected_exp151: dict[str, Any],
    actor_digest_unchanged: bool,
    actor_probe_action_max_abs_change: float,
    executed_action_max_abs_error: float,
) -> tuple[dict[str, bool], float]:
    reconstruction_errors = []
    for horizon in HORIZONS:
        expected = expected_exp151["summary"]["horizons"][str(horizon)][
            "any_actionable"
        ]["mean"]
        actual = summary["horizons"][str(horizon)]["joint_actionable"]["mean"]
        reconstruction_errors.append(abs(float(actual) - float(expected)))
    reconstruction_error = max(reconstruction_errors, default=float("inf"))
    finite_values = []
    for row in summary["horizons"].values():
        for value in row.values():
            if isinstance(value, (int, float)):
                finite_values.append(float(value))
            elif isinstance(value, dict):
                finite_values.extend(float(v) for v in value.values())
    finite = bool(torch.isfinite(torch.tensor(finite_values)).all())
    checks = {
        "collision_episodes_ge_100": summary["collision_episodes"] >= 100,
        "actor_checkpoint_unchanged": actor_digest_unchanged,
        "actor_probe_actions_unchanged": actor_probe_action_max_abs_change == 0.0,
        "executed_action_equals_sampled_action": executed_action_max_abs_error == 0.0,
        "nonparticipant_pair_gain_abs_max_le_1e_6": (
            summary["nonparticipant_pair_gain_abs_max"] <= 1.0e-6
        ),
        "exp151_joint_reconstruction_error_le_1e_6": reconstruction_error <= 1.0e-6,
        "all_metrics_finite": finite,
    }
    return checks, reconstruction_error


def analyze_action_planning_controllability_decomposition(
    *,
    config: str | Path,
    checkpoints: tuple[str | Path, ...],
    exp151_summary: str | Path = DEFAULT_EXP151_SUMMARY,
    device: str = "cuda",
    num_envs: int = 128,
    steps: int = 512,
    data_seeds: tuple[int, ...] = (46023, 47023),
    action_delta: float = 0.15,
    terrain_risk_tolerance: float = 0.01,
    endpoint_dmax_tolerance: float = 0.02,
    actionable_gain: float = 0.02,
    control_response_threshold: float = 0.05,
    run_dir: str | Path | None = None,
) -> dict[str, Any]:
    if len(checkpoints) < 2:
        raise ValueError("at least two checkpoints are required")
    expected = json.loads(Path(exp151_summary).read_text(encoding="utf-8"))
    cfg = cfg_from_experiment(config)
    combinations: dict[str, Any] = {}
    for checkpoint in checkpoints:
        checkpoint_path = Path(checkpoint)
        checkpoint_data = torch.load(checkpoint_path, map_location=torch.device(device))
        digest_before = _policy_digest(checkpoint_data)
        checkpoint_label = checkpoint_path.stem
        for seed in data_seeds:
            key = f"{checkpoint_label}_seed{seed}"
            if key not in expected["combinations"]:
                raise ValueError(f"exp151 summary is missing combination {key}")
            timeline, invariance = collect_layered_timeline(
                config=config,
                checkpoint_data=checkpoint_data,
                device=device,
                num_envs=num_envs,
                steps=steps,
                seed=seed,
                action_delta=action_delta,
                terrain_risk_tolerance=terrain_risk_tolerance,
                endpoint_dmax_tolerance=endpoint_dmax_tolerance,
            )
            summary = summarize_layered_timeline(
                timeline,
                collision_distance=float(cfg.safety.collision_distance),
                actionable_gain=actionable_gain,
                control_response_threshold=control_response_threshold,
            )
            digest_after = _policy_digest(checkpoint_data)
            checks, reconstruction_error = _combination_checks(
                summary,
                expected_exp151=expected["combinations"][key],
                actor_digest_unchanged=digest_before == digest_after,
                actor_probe_action_max_abs_change=float(
                    invariance["actor_probe_action_max_abs_change"]
                ),
                executed_action_max_abs_error=float(
                    invariance["executed_action_max_abs_error"]
                ),
            )
            combinations[key] = {
                "checkpoint": str(checkpoint_path),
                "checkpoint_label": checkpoint_label,
                "seed": seed,
                "summary": summary,
                "exp151_joint_reconstruction_max_abs_error": reconstruction_error,
                "invariance": {
                    **invariance,
                    "actor_digest_before": digest_before,
                    "actor_digest_after": digest_after,
                },
                "checks": checks,
                "passed": all(checks.values()),
            }
    decision = controllability_decision(combinations)
    result: dict[str, Any] = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment": EXPERIMENT_ID,
        "status": decision["status"],
        "config": str(config),
        "exp151_summary": str(exp151_summary),
        "collection": {
            "device": device,
            "num_envs": num_envs,
            "steps": steps,
            "data_seeds": list(data_seeds),
            "checkpoints": [str(path) for path in checkpoints],
            "stochastic_policy_actions": True,
            "candidate_actions_executed": False,
        },
        "counterfactual": {
            "action_delta": action_delta,
            "terrain_risk_tolerance": terrain_risk_tolerance,
            "endpoint_dmax_tolerance": endpoint_dmax_tolerance,
            "actionable_gain": actionable_gain,
            "control_response_threshold": control_response_threshold,
            "horizons": list(HORIZONS),
        },
        "combinations": combinations,
        "decision": decision,
    }

    run_dir_path = (
        Path(run_dir)
        if run_dir is not None
        else ROOT / "outputs" / "runs" / EXPERIMENT_ID / DEFAULT_RUN_ID
    )
    if not run_dir_path.is_absolute():
        run_dir_path = ROOT / run_dir_path
    metrics_dir = run_dir_path / "metrics"
    config_dir = run_dir_path / "config"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)
    config_source = Path(config)
    if not config_source.is_absolute():
        config_source = ROOT / config_source
    config_snapshot = config_dir / "experiment.yaml"
    config_snapshot.write_text(config_source.read_text(encoding="utf-8"), encoding="utf-8")
    metrics_path = metrics_dir / "controllability_decomposition.json"
    metrics_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    def artifact_path(path: Path) -> str:
        try:
            return str(path.relative_to(ROOT))
        except ValueError:
            return str(path)

    manifest = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment": EXPERIMENT_ID,
        "producer": "scripts/analyze_action_planning_controllability_decomposition.py",
        "status": result["status"],
        "device": device,
        "collection": result["collection"],
        "artifacts": {
            "config": artifact_path(config_snapshot),
            "metrics": artifact_path(metrics_path),
        },
    }
    (run_dir_path / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    suite_dir = run_dir_path.parent / "_suite"
    suite_metrics_dir = suite_dir / "metrics"
    suite_metrics_dir.mkdir(parents=True, exist_ok=True)
    suite_summary_path = suite_metrics_dir / "suite_summary.json"
    suite_summary_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    suite_manifest = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment": EXPERIMENT_ID,
        "producer": "scripts/analyze_action_planning_controllability_decomposition.py",
        "status": result["status"],
        "artifacts": {
            "suite_summary": artifact_path(suite_summary_path),
            "run_manifest": artifact_path(run_dir_path / "run_manifest.json"),
        },
    }
    (suite_dir / "run_manifest.json").write_text(
        json.dumps(suite_manifest, indent=2), encoding="utf-8"
    )
    result["artifact"] = str(metrics_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--exp151-summary", default=str(DEFAULT_EXP151_SUMMARY))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--steps", type=int, default=512)
    parser.add_argument("--data-seeds", type=_parse_int_tuple, default=(46023, 47023))
    parser.add_argument("--action-delta", type=float, default=0.15)
    parser.add_argument("--terrain-risk-tolerance", type=float, default=0.01)
    parser.add_argument("--endpoint-dmax-tolerance", type=float, default=0.02)
    parser.add_argument("--actionable-gain", type=float, default=0.02)
    parser.add_argument("--control-response-threshold", type=float, default=0.05)
    parser.add_argument("--run-dir", default=None)
    args = parser.parse_args()
    result = analyze_action_planning_controllability_decomposition(
        config=args.config,
        checkpoints=tuple(args.checkpoint),
        exp151_summary=args.exp151_summary,
        device=args.device,
        num_envs=args.num_envs,
        steps=args.steps,
        data_seeds=args.data_seeds,
        action_delta=args.action_delta,
        terrain_risk_tolerance=args.terrain_risk_tolerance,
        endpoint_dmax_tolerance=args.endpoint_dmax_tolerance,
        actionable_gain=args.actionable_gain,
        control_response_threshold=args.control_response_threshold,
        run_dir=args.run_dir,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
