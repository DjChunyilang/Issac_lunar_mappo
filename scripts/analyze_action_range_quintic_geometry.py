#!/usr/bin/env python
"""Frozen separation of local action coverage and quintic geometry losses."""

from __future__ import annotations

import argparse
from dataclasses import replace
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
from lunar_rover_tasks.tasks.multi_rover_gathering.trajectory_generator import (
    generate_trajectory,
)
from lunar_rover_tasks.utils.geometry_utils import pairwise_distances_xy
from play import _load_policy_players


EXPERIMENT_ID = "exp153_action_range_quintic_geometry_audit"
DEFAULT_RUN_ID = "frozen_exp150_dual_checkpoint_dualseed"
DEFAULT_EXP152_SUMMARY = (
    ROOT
    / "outputs/runs/exp152_action_planning_controllability_decomposition"
    / "_suite/metrics/suite_summary.json"
)
HORIZONS = (1, 2, 4, 8, 16)


def _geometry_distances(
    env: MultiRoverGatheringCore,
    actions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    decoded = decode_action(actions, env.positions, env.yaws, env.cfg.planner)
    quintic = generate_trajectory(
        env.positions,
        decoded.world_subgoal,
        env.cfg.trajectory_generator,
        env.cfg.simulation.planning_dt,
        current_yaws=env.yaws,
    )
    line_cfg = replace(env.cfg.trajectory_generator, geometry_method="line")
    line = generate_trajectory(
        env.positions,
        decoded.world_subgoal,
        line_cfg,
        env.cfg.simulation.planning_dt,
        current_yaws=env.yaws,
    )
    return (
        trajectory_pairwise_min_distance(quintic.points, quintic.timestamps),
        trajectory_pairwise_min_distance(line.points, line.timestamps),
        pairwise_distances_xy(decoded.world_subgoal[..., :2]),
    )


def action_range_geometry_outcomes(
    env: MultiRoverGatheringCore,
    actions: torch.Tensor,
    *,
    action_delta: float,
) -> dict[str, torch.Tensor]:
    if action_delta <= 0.0:
        raise ValueError("action_delta must be positive")
    if actions.shape != (env.num_envs, env.n_agents, 2):
        raise ValueError("actions must have shape [environment, agent, 2]")
    baseline_quintic, baseline_line, baseline_endpoint = _geometry_distances(
        env, actions
    )
    shape = (env.num_envs, env.n_agents, env.n_agents, env.n_agents)
    local_quintic = torch.zeros(shape, device=env.device, dtype=actions.dtype)
    axis_quintic = torch.zeros_like(local_quintic)
    grid_quintic = torch.zeros_like(local_quintic)
    grid_line = torch.zeros_like(local_quintic)
    grid_endpoint = torch.zeros_like(local_quintic)
    grid_joint_dimension = torch.zeros_like(local_quintic)

    for modifier in range(env.n_agents):
        for dimension in range(2):
            for sign in (-1.0, 1.0):
                candidate = actions.clone()
                candidate[:, modifier, dimension] = (
                    candidate[:, modifier, dimension] + sign * float(action_delta)
                ).clamp(-1.0, 1.0)
                candidate_quintic, _, _ = _geometry_distances(env, candidate)
                local_quintic[:, modifier] = torch.maximum(
                    local_quintic[:, modifier], candidate_quintic - baseline_quintic
                )

        rho_options = (
            torch.full_like(actions[:, modifier, 0], -1.0),
            actions[:, modifier, 0],
            torch.full_like(actions[:, modifier, 0], 1.0),
        )
        beta_options = (
            torch.full_like(actions[:, modifier, 1], -1.0),
            actions[:, modifier, 1],
            torch.full_like(actions[:, modifier, 1], 1.0),
        )
        for rho_index, rho in enumerate(rho_options):
            for beta_index, beta in enumerate(beta_options):
                if rho_index == 1 and beta_index == 1:
                    continue
                candidate = actions.clone()
                candidate[:, modifier, 0] = rho
                candidate[:, modifier, 1] = beta
                candidate_quintic, candidate_line, candidate_endpoint = (
                    _geometry_distances(env, candidate)
                )
                quintic_gain = candidate_quintic - baseline_quintic
                line_gain = candidate_line - baseline_line
                endpoint_gain = candidate_endpoint - baseline_endpoint
                is_axis = (rho_index == 1) != (beta_index == 1)
                is_joint = rho_index != 1 and beta_index != 1
                if is_axis:
                    axis_quintic[:, modifier] = torch.maximum(
                        axis_quintic[:, modifier], quintic_gain
                    )
                improve = quintic_gain > grid_quintic[:, modifier]
                grid_quintic[:, modifier] = torch.where(
                    improve, quintic_gain, grid_quintic[:, modifier]
                )
                grid_joint_dimension[:, modifier] = torch.where(
                    improve,
                    torch.full_like(quintic_gain, float(is_joint)),
                    grid_joint_dimension[:, modifier],
                )
                grid_line[:, modifier] = torch.maximum(
                    grid_line[:, modifier], line_gain
                )
                grid_endpoint[:, modifier] = torch.maximum(
                    grid_endpoint[:, modifier], endpoint_gain
                )

    return {
        "best_local_quintic_gain": local_quintic,
        "best_axis_quintic_gain": axis_quintic,
        "best_grid_quintic_gain": grid_quintic,
        "best_grid_line_gain": grid_line,
        "best_grid_endpoint_gain": grid_endpoint,
        "best_grid_joint_dimension": grid_joint_dimension,
    }


def collect_geometry_timeline(
    *,
    config: str | Path,
    checkpoint_data: dict[str, Any],
    device: str,
    num_envs: int,
    steps: int,
    seed: int,
    action_delta: float,
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
        raise ValueError("exp153 requires the strict unfiltered execution chain")
    forbidden = (
        cfg.low_level_control.safety_projection_enabled,
        cfg.low_level_control.success_zone_damping_enabled,
        cfg.low_level_control.formation_center_correction_enabled,
        cfg.low_level_control.terminal_slot_capture_enabled,
        cfg.low_level_control.flat_geometry_capture_enabled,
    )
    if any(forbidden):
        raise ValueError("exp153 requires every execution override to remain disabled")
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
    generator.manual_seed(seed + 151_151)
    probe_obs = actor_obs[: min(num_envs, 32)].clone()
    with torch.no_grad():
        probe_before = act(probe_obs).clone()
    outcome_names = (
        "best_local_quintic_gain",
        "best_axis_quintic_gain",
        "best_grid_quintic_gain",
        "best_grid_line_gain",
        "best_grid_endpoint_gain",
        "best_grid_joint_dimension",
    )
    sequences: dict[str, list[torch.Tensor]] = {name: [] for name in outcome_names}
    sequences.update(
        {"positions_after": [], "done": [], "collision_done": [], "sampled_actions": []}
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
            outcomes = action_range_geometry_outcomes(
                env, sampled_actions, action_delta=action_delta
            )
            action_sent = sampled_actions.clone()
            output = env.step(action_sent)
            executed_action_max_error = max(
                executed_action_max_error,
                float((action_sent - sampled_actions).abs().amax().cpu()),
            )
            for name in outcome_names:
                sequences[name].append(outcomes[name].cpu())
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


def summarize_geometry_timeline(
    timeline: dict[str, torch.Tensor],
    *,
    collision_distance: float,
    actionable_gain: float = 0.02,
    action_boundary: float = 0.85,
    horizons: tuple[int, ...] = HORIZONS,
) -> dict[str, Any]:
    gain_names = (
        "local_quintic",
        "axis_quintic",
        "grid_quintic",
        "grid_line",
        "grid_endpoint",
    )
    required = {
        *(f"best_{name}_gain" for name in gain_names),
        "best_grid_joint_dimension",
        "positions_after",
        "done",
        "collision_done",
        "sampled_actions",
    }
    missing = required - timeline.keys()
    if missing:
        raise ValueError(f"timeline is missing keys: {sorted(missing)}")
    done = timeline["done"].bool()
    collision_done = timeline["collision_done"].bool()
    time_steps, num_envs = done.shape
    n_agents = timeline["sampled_actions"].shape[2]
    metric_names = (
        *(f"{name}_actionable" for name in gain_names),
        "range_recovery",
        "joint_dimension_recovery",
        "quintic_geometry_loss",
        "path_crossing_loss",
        "participant_action_boundary",
        "best_local_quintic_gain",
        "best_grid_quintic_gain",
        "best_grid_line_gain",
        "best_grid_endpoint_gain",
        "nonparticipant_pair_gain_abs",
    )
    samples = {h: {name: [] for name in metric_names} for h in horizons}
    collision_episodes = 0
    collision_pairs = 0
    last_done = [-1 for _ in range(num_envs)]
    for t in range(time_steps):
        for env_id in range(num_envs):
            if bool(collision_done[t, env_id]):
                positions = timeline["positions_after"][t, env_id, :, :2]
                pairs = torch.nonzero(
                    _upper_triangle(
                        pairwise_distances_xy(positions.unsqueeze(0))[0]
                        < float(collision_distance)
                    ),
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
                            name: timeline[f"best_{name}_gain"][
                                step, env_id, participants, first, second
                            ].clamp_min(0.0)
                            for name in gain_names
                        }
                        flags = {
                            name: bool((gain >= float(actionable_gain)).any())
                            for name, gain in gains.items()
                        }
                        for name, flag in flags.items():
                            row[f"{name}_actionable"].append(float(flag))
                        row["range_recovery"].append(
                            float(flags["grid_quintic"] and not flags["local_quintic"])
                        )
                        row["quintic_geometry_loss"].append(
                            float(flags["grid_line"] and not flags["grid_quintic"])
                        )
                        row["path_crossing_loss"].append(
                            float(flags["grid_endpoint"] and not flags["grid_line"])
                        )
                        if flags["grid_quintic"]:
                            responsible_local = int(
                                torch.argmax(gains["grid_quintic"]).item()
                            )
                            responsible = participants[responsible_local]
                            row["joint_dimension_recovery"].append(
                                float(
                                    timeline["best_grid_joint_dimension"][
                                        step, env_id, responsible, first, second
                                    ]
                                )
                            )
                        participant_actions = timeline["sampled_actions"][
                            step, env_id, participants
                        ]
                        row["participant_action_boundary"].extend(
                            (participant_actions.abs() >= float(action_boundary))
                            .float()
                            .flatten()
                            .tolist()
                        )
                        for name in (
                            "local_quintic",
                            "grid_quintic",
                            "grid_line",
                            "grid_endpoint",
                        ):
                            row[f"best_{name}_gain"].append(
                                float(gains[name].amax())
                            )
                        for modifier in range(n_agents):
                            if modifier not in participants:
                                row["nonparticipant_pair_gain_abs"].append(
                                    abs(
                                        float(
                                            timeline["best_grid_quintic_gain"][
                                                step, env_id, modifier, first, second
                                            ]
                                        )
                                    )
                                )
            if bool(done[t, env_id]):
                last_done[env_id] = t
    horizons_out: dict[str, Any] = {}
    nonparticipant_max = 0.0
    for horizon, horizon_samples in samples.items():
        row = {"pair_samples": len(horizon_samples["local_quintic_actionable"])}
        for name, values in horizon_samples.items():
            row[name] = _distribution(values)
        local_max = max(horizon_samples["nonparticipant_pair_gain_abs"], default=0.0)
        row["nonparticipant_pair_gain_abs_max"] = local_max
        nonparticipant_max = max(nonparticipant_max, local_max)
        horizons_out[str(horizon)] = row
    return {
        "collision_episodes": collision_episodes,
        "collision_pairs": collision_pairs,
        "horizons": horizons_out,
        "nonparticipant_pair_gain_abs_max": nonparticipant_max,
    }


def geometry_decision(combinations: dict[str, Any]) -> dict[str, Any]:
    engineering_passed = all(item["passed"] for item in combinations.values())

    def values(metric: str) -> list[float]:
        return [
            item["summary"]["horizons"]["8"][metric]["mean"]
            for item in combinations.values()
        ]

    local = values("local_quintic_actionable")
    grid = values("grid_quintic_actionable")
    line = values("grid_line_actionable")
    endpoint = values("grid_endpoint_actionable")
    range_recovery = values("range_recovery")
    geometry_loss = values("quintic_geometry_loss")
    crossing_loss = values("path_crossing_loss")
    if not engineering_passed:
        status = "invalid_diagnostic_stop"
        next_stage = "fix_diagnostic_only"
    elif min(grid, default=0.0) >= 0.80 and min(range_recovery, default=0.0) >= 0.15:
        status = "local_coverage_bottleneck"
        next_stage = "audit_training_action_distribution_coverage_only"
    elif (
        min(grid, default=0.0) < 0.70
        and min(line, default=0.0) >= 0.70
        and min(geometry_loss, default=0.0) >= 0.15
    ):
        status = "quintic_geometry_bottleneck"
        next_stage = "preregister_single_quintic_geometry_fix"
    elif (
        min(line, default=0.0) < 0.70
        and min(endpoint, default=0.0) >= 0.70
        and min(crossing_loss, default=0.0) >= 0.15
    ):
        status = "path_crossing_or_timing_bottleneck"
        next_stage = "audit_path_crossing_time_contract_only"
    elif min(endpoint, default=0.0) < 0.70:
        status = "subgoal_reachability_bottleneck"
        next_stage = "audit_existing_subgoal_range_only"
    else:
        status = "mixed_action_geometry_bottleneck"
        next_stage = "stop_without_single_variable_hypothesis"
    return {
        "status": status,
        "next_stage": next_stage,
        "training_authorized": False,
        "engineering_passed": engineering_passed,
        "cross_combination": {
            "h8_local_quintic_min": min(local, default=0.0),
            "h8_grid_quintic_min": min(grid, default=0.0),
            "h8_grid_line_min": min(line, default=0.0),
            "h8_grid_endpoint_min": min(endpoint, default=0.0),
            "h8_range_recovery_min": min(range_recovery, default=0.0),
            "h8_quintic_geometry_loss_min": min(geometry_loss, default=0.0),
            "h8_path_crossing_loss_min": min(crossing_loss, default=0.0),
        },
        "thresholds": {
            "grid_high": 0.80,
            "actionable": 0.70,
            "recovery_or_loss": 0.15,
        },
    }


def _checks(
    summary: dict[str, Any],
    *,
    expected_exp152: dict[str, Any],
    actor_digest_unchanged: bool,
    actor_probe_action_max_abs_change: float,
    executed_action_max_abs_error: float,
) -> tuple[dict[str, bool], float]:
    errors = []
    for horizon in HORIZONS:
        expected = expected_exp152["summary"]["horizons"][str(horizon)][
            "unconstrained_actionable"
        ]["mean"]
        actual = summary["horizons"][str(horizon)]["local_quintic_actionable"]["mean"]
        errors.append(abs(float(expected) - float(actual)))
    reconstruction_error = max(errors, default=float("inf"))
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
        "exp152_local_reconstruction_error_le_1e_6": reconstruction_error <= 1.0e-6,
        "candidate_counts_fixed_4_4_8": True,
        "all_metrics_finite": finite,
    }
    return checks, reconstruction_error


def analyze_action_range_quintic_geometry(
    *,
    config: str | Path,
    checkpoints: tuple[str | Path, ...],
    exp152_summary: str | Path = DEFAULT_EXP152_SUMMARY,
    device: str = "cuda",
    num_envs: int = 128,
    steps: int = 512,
    data_seeds: tuple[int, ...] = (46023, 47023),
    action_delta: float = 0.15,
    actionable_gain: float = 0.02,
    action_boundary: float = 0.85,
    run_dir: str | Path | None = None,
) -> dict[str, Any]:
    if len(checkpoints) < 2:
        raise ValueError("at least two checkpoints are required")
    expected = json.loads(Path(exp152_summary).read_text(encoding="utf-8"))
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
                raise ValueError(f"exp152 summary is missing combination {key}")
            timeline, invariance = collect_geometry_timeline(
                config=config,
                checkpoint_data=checkpoint_data,
                device=device,
                num_envs=num_envs,
                steps=steps,
                seed=seed,
                action_delta=action_delta,
            )
            summary = summarize_geometry_timeline(
                timeline,
                collision_distance=float(cfg.safety.collision_distance),
                actionable_gain=actionable_gain,
                action_boundary=action_boundary,
            )
            digest_after = _policy_digest(checkpoint_data)
            checks, reconstruction_error = _checks(
                summary,
                expected_exp152=expected["combinations"][key],
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
                "exp152_local_reconstruction_max_abs_error": reconstruction_error,
                "invariance": {
                    **invariance,
                    "actor_digest_before": digest_before,
                    "actor_digest_after": digest_after,
                },
                "checks": checks,
                "passed": all(checks.values()),
            }
    decision = geometry_decision(combinations)
    result: dict[str, Any] = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment": EXPERIMENT_ID,
        "status": decision["status"],
        "config": str(config),
        "exp152_summary": str(exp152_summary),
        "collection": {
            "device": device,
            "num_envs": num_envs,
            "steps": steps,
            "data_seeds": list(data_seeds),
            "checkpoints": [str(path) for path in checkpoints],
            "stochastic_policy_actions": True,
            "candidate_actions_executed": False,
        },
        "candidate_sets": {"local": 4, "axis": 4, "grid": 8},
        "counterfactual": {
            "action_delta": action_delta,
            "actionable_gain": actionable_gain,
            "action_boundary": action_boundary,
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
    metrics_path = metrics_dir / "action_range_quintic_geometry.json"
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
        "producer": "scripts/analyze_action_range_quintic_geometry.py",
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
        "producer": "scripts/analyze_action_range_quintic_geometry.py",
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
    parser.add_argument("--exp152-summary", default=str(DEFAULT_EXP152_SUMMARY))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--steps", type=int, default=512)
    parser.add_argument("--data-seeds", type=_parse_int_tuple, default=(46023, 47023))
    parser.add_argument("--action-delta", type=float, default=0.15)
    parser.add_argument("--actionable-gain", type=float, default=0.02)
    parser.add_argument("--action-boundary", type=float, default=0.85)
    parser.add_argument("--run-dir", default=None)
    args = parser.parse_args()
    result = analyze_action_range_quintic_geometry(
        config=args.config,
        checkpoints=tuple(args.checkpoint),
        exp152_summary=args.exp152_summary,
        device=args.device,
        num_envs=args.num_envs,
        steps=args.steps,
        data_seeds=args.data_seeds,
        action_delta=args.action_delta,
        actionable_gain=args.actionable_gain,
        action_boundary=args.action_boundary,
        run_dir=args.run_dir,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
