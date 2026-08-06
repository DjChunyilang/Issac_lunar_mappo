#!/usr/bin/env python
"""Frozen counterfactual audit of terminal collision-participant credit."""

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
from lunar_rover_tasks.tasks.multi_rover_gathering.terrain_features import (
    sample_trajectory_terrain_risk,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.trajectory_generator import (
    generate_trajectory,
)
from lunar_rover_tasks.utils.geometry_utils import pairwise_distances_xy
from play import _load_policy_players


EXPERIMENT_ID = "exp151_collision_credit_causal_validity"
DEFAULT_RUN_ID = "frozen_exp150_dual_checkpoint_dualseed"
HORIZONS = (1, 2, 4, 8, 16)


def _planning_outcomes(
    env: MultiRoverGatheringCore,
    actions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return time-aligned pair margin, per-agent terrain risk and endpoint dmax."""

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
    return pair_distance, terrain_risk, endpoint_metrics.dmax


def local_counterfactual_outcomes(
    env: MultiRoverGatheringCore,
    actions: torch.Tensor,
    *,
    action_delta: float,
    terrain_risk_tolerance: float,
    endpoint_dmax_tolerance: float,
) -> dict[str, torch.Tensor]:
    """Evaluate four local action perturbations without stepping the environment."""

    if action_delta <= 0.0:
        raise ValueError("action_delta must be positive")
    if terrain_risk_tolerance < 0.0 or endpoint_dmax_tolerance < 0.0:
        raise ValueError("counterfactual tolerances must be non-negative")
    if actions.shape != (env.num_envs, env.n_agents, 2):
        raise ValueError("actions must have shape [environment, agent, 2]")

    baseline_pair, baseline_risk, baseline_dmax = _planning_outcomes(env, actions)
    shape = (env.num_envs, env.n_agents, env.n_agents, env.n_agents)
    best_gain = torch.zeros(shape, device=env.device, dtype=actions.dtype)
    best_gain_risk_delta = torch.zeros_like(best_gain)
    best_gain_dmax_delta = torch.zeros_like(best_gain)
    best_feasible_gain = torch.zeros_like(best_gain)
    best_feasible_risk_delta = torch.zeros_like(best_gain)
    best_feasible_dmax_delta = torch.zeros_like(best_gain)

    for modifier in range(env.n_agents):
        for dimension in range(2):
            for sign in (-1.0, 1.0):
                candidate = actions.clone()
                candidate[:, modifier, dimension] = (
                    candidate[:, modifier, dimension] + sign * float(action_delta)
                ).clamp(-1.0, 1.0)
                pair, risk, endpoint_dmax = _planning_outcomes(env, candidate)
                gain = pair - baseline_pair
                risk_delta = risk[:, modifier] - baseline_risk[:, modifier]
                dmax_delta = endpoint_dmax - baseline_dmax
                risk_matrix = risk_delta[:, None, None].expand_as(gain)
                dmax_matrix = dmax_delta[:, None, None].expand_as(gain)

                current = best_gain[:, modifier]
                improve = gain > current
                best_gain[:, modifier] = torch.where(improve, gain, current)
                best_gain_risk_delta[:, modifier] = torch.where(
                    improve,
                    risk_matrix,
                    best_gain_risk_delta[:, modifier],
                )
                best_gain_dmax_delta[:, modifier] = torch.where(
                    improve,
                    dmax_matrix,
                    best_gain_dmax_delta[:, modifier],
                )

                feasible = (risk_delta <= float(terrain_risk_tolerance)) & (
                    dmax_delta <= float(endpoint_dmax_tolerance)
                )
                feasible_improve = feasible[:, None, None] & (
                    gain > best_feasible_gain[:, modifier]
                )
                best_feasible_gain[:, modifier] = torch.where(
                    feasible_improve,
                    gain,
                    best_feasible_gain[:, modifier],
                )
                best_feasible_risk_delta[:, modifier] = torch.where(
                    feasible_improve,
                    risk_matrix,
                    best_feasible_risk_delta[:, modifier],
                )
                best_feasible_dmax_delta[:, modifier] = torch.where(
                    feasible_improve,
                    dmax_matrix,
                    best_feasible_dmax_delta[:, modifier],
                )

    return {
        "baseline_pair_distance": baseline_pair,
        "best_pair_gain": best_gain,
        "best_pair_gain_risk_delta": best_gain_risk_delta,
        "best_pair_gain_endpoint_dmax_delta": best_gain_dmax_delta,
        "best_feasible_pair_gain": best_feasible_gain,
        "best_feasible_risk_delta": best_feasible_risk_delta,
        "best_feasible_endpoint_dmax_delta": best_feasible_dmax_delta,
    }


def collect_causal_timeline(
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
    """Collect stochastic policy rollouts and non-intervening local alternatives."""

    raw_cfg = load_yaml(config)
    cfg = cfg_from_experiment(config)
    cfg.simulation.device = device
    cfg.simulation.num_envs = num_envs
    cfg.seed = seed
    metadata = checkpoint_data.get("metadata") or {}
    if cfg.initial_state.curriculum_enabled:
        cfg.initial_state.progress_timestep_override = int(metadata.get("timesteps", 0))
    if cfg.planner.subgoal_filter.enabled:
        raise ValueError("exp151 requires the strict unfiltered execution chain")
    forbidden = (
        cfg.low_level_control.safety_projection_enabled,
        cfg.low_level_control.success_zone_damping_enabled,
        cfg.low_level_control.formation_center_correction_enabled,
        cfg.low_level_control.terminal_slot_capture_enabled,
        cfg.low_level_control.flat_geometry_capture_enabled,
    )
    if any(forbidden):
        raise ValueError("exp151 requires every execution override to remain disabled")

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

    sequence_names = (
        "baseline_pair_distance",
        "best_pair_gain",
        "best_pair_gain_risk_delta",
        "best_pair_gain_endpoint_dmax_delta",
        "best_feasible_pair_gain",
        "best_feasible_risk_delta",
        "best_feasible_endpoint_dmax_delta",
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
            counterfactual = local_counterfactual_outcomes(
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
    timeline = {name: torch.stack(values) for name, values in sequences.items()}
    invariance = {
        "actor_probe_action_max_abs_change": float(
            (probe_after - probe_before).abs().amax().cpu()
        ),
        "executed_action_max_abs_error": executed_action_max_error,
        "policy_std": policy_std.flatten().cpu().tolist(),
    }
    return timeline, invariance


def summarize_causal_timeline(
    timeline: dict[str, torch.Tensor],
    *,
    collision_distance: float,
    gamma: float,
    trace_lambda: float,
    horizons: tuple[int, ...] = HORIZONS,
    actionable_gain: float = 0.02,
    insensitive_gain: float = 0.005,
) -> dict[str, Any]:
    """Summarize local counterfactual responsibility for actual collision pairs."""

    required = {
        "baseline_pair_distance",
        "best_pair_gain",
        "best_pair_gain_risk_delta",
        "best_pair_gain_endpoint_dmax_delta",
        "best_feasible_pair_gain",
        "best_feasible_risk_delta",
        "best_feasible_endpoint_dmax_delta",
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
    if done.ndim != 2 or collision_done.shape != done.shape:
        raise ValueError("done and collision_done must have shape [time, environment]")
    time_steps, num_envs = done.shape
    n_agents = timeline["sampled_actions"].shape[2]

    per_horizon: dict[int, dict[str, list[float]]] = {
        horizon: {
            "any_actionable": [],
            "both_actionable": [],
            "participant_actionable": [],
            "equal_credit_supported": [],
            "responsibility_asymmetry": [],
            "locally_optimal_or_insensitive": [],
            "participant_best_gain": [],
            "participant_best_feasible_gain": [],
            "best_feasible_risk_delta": [],
            "best_feasible_endpoint_dmax_delta": [],
            "baseline_pair_distance": [],
            "nonparticipant_pair_gain_abs": [],
        }
        for horizon in horizons
    }
    collision_episode_count = 0
    collision_pair_count = 0
    last_done = [-1 for _ in range(num_envs)]
    for t in range(time_steps):
        for env_id in range(num_envs):
            if bool(collision_done[t, env_id]):
                positions = timeline["positions_after"][t, env_id, :, :2]
                distances = pairwise_distances_xy(positions.unsqueeze(0))[0]
                collision_pairs = _upper_triangle(distances < float(collision_distance))
                pairs = torch.nonzero(collision_pairs, as_tuple=False).tolist()
                if pairs:
                    collision_episode_count += 1
                collision_pair_count += len(pairs)
                episode_start = last_done[env_id] + 1
                for first, second in pairs:
                    for horizon in horizons:
                        step = t - horizon
                        if step < episode_start or step < 0:
                            continue
                        row = per_horizon[horizon]
                        feasible = timeline["best_feasible_pair_gain"][
                            step, env_id, :, first, second
                        ]
                        unconstrained = timeline["best_pair_gain"][
                            step, env_id, :, first, second
                        ]
                        participant_feasible = feasible[[first, second]].clamp_min(0.0)
                        participant_unconstrained = unconstrained[[first, second]].clamp_min(0.0)
                        actionable = participant_feasible >= float(actionable_gain)
                        insensitive = participant_feasible <= float(insensitive_gain)
                        gain_sum = float(participant_feasible.sum())
                        if gain_sum > 1.0e-12:
                            first_share = float(participant_feasible[0]) / gain_sum
                            asymmetry = 2.0 * abs(first_share - 0.5)
                        else:
                            first_share = 0.5
                            asymmetry = 0.0
                        equal_supported = bool(actionable.all()) and (
                            0.25 <= first_share <= 0.75
                        )
                        row["any_actionable"].append(float(actionable.any()))
                        row["both_actionable"].append(float(actionable.all()))
                        row["participant_actionable"].extend(
                            actionable.float().tolist()
                        )
                        row["equal_credit_supported"].append(float(equal_supported))
                        row["responsibility_asymmetry"].append(asymmetry)
                        row["locally_optimal_or_insensitive"].extend(
                            insensitive.float().tolist()
                        )
                        row["participant_best_gain"].extend(
                            participant_unconstrained.tolist()
                        )
                        row["participant_best_feasible_gain"].extend(
                            participant_feasible.tolist()
                        )
                        row["baseline_pair_distance"].append(
                            float(
                                timeline["baseline_pair_distance"][
                                    step, env_id, first, second
                                ]
                            )
                        )
                        for participant in (first, second):
                            row["best_feasible_risk_delta"].append(
                                float(
                                    timeline["best_feasible_risk_delta"][
                                        step, env_id, participant, first, second
                                    ]
                                )
                            )
                            row["best_feasible_endpoint_dmax_delta"].append(
                                float(
                                    timeline["best_feasible_endpoint_dmax_delta"][
                                        step, env_id, participant, first, second
                                    ]
                                )
                            )
                        for modifier in range(n_agents):
                            if modifier not in (first, second):
                                row["nonparticipant_pair_gain_abs"].append(
                                    abs(
                                        float(
                                            timeline["best_pair_gain"][
                                                step,
                                                env_id,
                                                modifier,
                                                first,
                                                second,
                                            ]
                                        )
                                    )
                                )
            if bool(done[t, env_id]):
                last_done[env_id] = t

    horizon_summary: dict[str, Any] = {}
    nonparticipant_max = 0.0
    for horizon, samples in per_horizon.items():
        row: dict[str, Any] = {
            "pair_samples": len(samples["any_actionable"]),
            "trace_weight": float((gamma * trace_lambda) ** horizon),
        }
        for name, values in samples.items():
            row[name] = _distribution(values)
        if samples["nonparticipant_pair_gain_abs"]:
            local_max = max(samples["nonparticipant_pair_gain_abs"])
            nonparticipant_max = max(nonparticipant_max, local_max)
            row["nonparticipant_pair_gain_abs_max"] = local_max
        else:
            row["nonparticipant_pair_gain_abs_max"] = 0.0
        horizon_summary[str(horizon)] = row

    return {
        "collision_episodes": collision_episode_count,
        "collision_pairs": collision_pair_count,
        "horizons": horizon_summary,
        "nonparticipant_pair_gain_abs_max": nonparticipant_max,
    }


def _combination_checks(
    summary: dict[str, Any],
    *,
    actor_digest_unchanged: bool,
    actor_probe_action_max_abs_change: float,
    executed_action_max_abs_error: float,
) -> dict[str, bool]:
    finite_values = []
    for row in summary["horizons"].values():
        for value in row.values():
            if isinstance(value, (int, float)):
                finite_values.append(float(value))
            elif isinstance(value, dict):
                finite_values.extend(float(v) for v in value.values())
    finite = bool(torch.isfinite(torch.tensor(finite_values)).all())
    return {
        "collision_episodes_ge_100": summary["collision_episodes"] >= 100,
        "actor_checkpoint_unchanged": actor_digest_unchanged,
        "actor_probe_actions_unchanged": actor_probe_action_max_abs_change == 0.0,
        "executed_action_equals_sampled_action": executed_action_max_abs_error == 0.0,
        "nonparticipant_pair_gain_abs_max_le_1e_6": (
            summary["nonparticipant_pair_gain_abs_max"] <= 1.0e-6
        ),
        "all_metrics_finite": finite,
    }


def causal_validity_decision(combinations: dict[str, Any]) -> dict[str, Any]:
    """Apply the preregistered cross-combination decision tree."""

    engineering_passed = all(item["passed"] for item in combinations.values())
    h8_actionable = [
        item["summary"]["horizons"]["8"]["any_actionable"]["mean"]
        for item in combinations.values()
    ]
    h16_actionable = [
        item["summary"]["horizons"]["16"]["any_actionable"]["mean"]
        for item in combinations.values()
    ]
    h8_equal = [
        item["summary"]["horizons"]["8"]["equal_credit_supported"]["mean"]
        for item in combinations.values()
    ]
    h8_asymmetry = [
        item["summary"]["horizons"]["8"]["responsibility_asymmetry"]["median"]
        for item in combinations.values()
    ]
    actionability_passed = (
        engineering_passed
        and min(h8_actionable, default=0.0) >= 0.70
        and min(h16_actionable, default=0.0) >= 0.60
    )
    equal_credit_invalid = actionability_passed and (
        max(h8_equal, default=1.0) < 0.50
        or min(h8_asymmetry, default=0.0) >= 0.50
    )
    if not engineering_passed:
        status = "invalid_diagnostic_stop"
        next_stage = "fix_diagnostic_only"
    elif not actionability_passed:
        status = "local_avoidance_not_actionable_stop_credit"
        next_stage = "audit_action_planning_controllability"
    elif equal_credit_invalid:
        status = "equal_participant_credit_causally_invalid"
        next_stage = "frozen_counterfactual_difference_advantage_audit_only"
    else:
        status = "equal_credit_locally_supported_but_learning_failed"
        next_stage = "frozen_score_credit_covariance_audit_only"
    return {
        "status": status,
        "next_stage": next_stage,
        "training_authorized": False,
        "engineering_passed": engineering_passed,
        "actionability_passed": actionability_passed,
        "equal_credit_invalid": equal_credit_invalid,
        "cross_combination": {
            "h8_any_actionable_min": min(h8_actionable, default=0.0),
            "h16_any_actionable_min": min(h16_actionable, default=0.0),
            "h8_equal_credit_supported_max": max(h8_equal, default=0.0),
            "h8_responsibility_asymmetry_min_median": min(
                h8_asymmetry, default=0.0
            ),
        },
        "thresholds": {
            "h8_any_actionable": 0.70,
            "h16_any_actionable": 0.60,
            "h8_equal_credit_supported_exclusive_upper": 0.50,
            "h8_responsibility_asymmetry_median": 0.50,
        },
    }


def analyze_collision_credit_causal_validity(
    *,
    config: str | Path,
    checkpoints: tuple[str | Path, ...],
    device: str = "cuda",
    num_envs: int = 128,
    steps: int = 512,
    data_seeds: tuple[int, ...] = (46023, 47023),
    action_delta: float = 0.15,
    terrain_risk_tolerance: float = 0.01,
    endpoint_dmax_tolerance: float = 0.02,
    actionable_gain: float = 0.02,
    insensitive_gain: float = 0.005,
    run_dir: str | Path | None = None,
) -> dict[str, Any]:
    if len(checkpoints) < 2:
        raise ValueError("at least two checkpoints are required")
    raw_cfg = load_yaml(config)
    cfg = cfg_from_experiment(config)
    algorithm = raw_cfg.get("algorithm", {})
    if not isinstance(algorithm, dict):
        raise ValueError("algorithm config must be a mapping")
    gamma = float(algorithm.get("gamma", 0.99))
    trace_lambda = float(algorithm.get("actor_credit_trace_lambda", 0.95))
    combinations: dict[str, Any] = {}
    for checkpoint in checkpoints:
        checkpoint_path = Path(checkpoint)
        checkpoint_data = torch.load(checkpoint_path, map_location=torch.device(device))
        digest_before = _policy_digest(checkpoint_data)
        checkpoint_label = checkpoint_path.stem
        for seed in data_seeds:
            timeline, invariance = collect_causal_timeline(
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
            summary = summarize_causal_timeline(
                timeline,
                collision_distance=float(cfg.safety.collision_distance),
                gamma=gamma,
                trace_lambda=trace_lambda,
                actionable_gain=actionable_gain,
                insensitive_gain=insensitive_gain,
            )
            digest_after = _policy_digest(checkpoint_data)
            checks = _combination_checks(
                summary,
                actor_digest_unchanged=digest_before == digest_after,
                actor_probe_action_max_abs_change=float(
                    invariance["actor_probe_action_max_abs_change"]
                ),
                executed_action_max_abs_error=float(
                    invariance["executed_action_max_abs_error"]
                ),
            )
            combinations[f"{checkpoint_label}_seed{seed}"] = {
                "checkpoint": str(checkpoint_path),
                "checkpoint_label": checkpoint_label,
                "seed": seed,
                "summary": summary,
                "invariance": {
                    **invariance,
                    "actor_digest_before": digest_before,
                    "actor_digest_after": digest_after,
                },
                "checks": checks,
                "passed": all(checks.values()),
            }
    decision = causal_validity_decision(combinations)
    result: dict[str, Any] = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment": EXPERIMENT_ID,
        "status": decision["status"],
        "config": str(config),
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
            "insensitive_gain": insensitive_gain,
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
    metrics_path = metrics_dir / "collision_credit_causal_validity.json"
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
        "producer": "scripts/analyze_collision_credit_causal_validity.py",
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
        "producer": "scripts/analyze_collision_credit_causal_validity.py",
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
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--steps", type=int, default=512)
    parser.add_argument("--data-seeds", type=_parse_int_tuple, default=(46023, 47023))
    parser.add_argument("--action-delta", type=float, default=0.15)
    parser.add_argument("--terrain-risk-tolerance", type=float, default=0.01)
    parser.add_argument("--endpoint-dmax-tolerance", type=float, default=0.02)
    parser.add_argument("--actionable-gain", type=float, default=0.02)
    parser.add_argument("--insensitive-gain", type=float, default=0.005)
    parser.add_argument("--run-dir", default=None)
    args = parser.parse_args()
    result = analyze_collision_credit_causal_validity(
        config=args.config,
        checkpoints=tuple(args.checkpoint),
        device=args.device,
        num_envs=args.num_envs,
        steps=args.steps,
        data_seeds=args.data_seeds,
        action_delta=args.action_delta,
        terrain_risk_tolerance=args.terrain_risk_tolerance,
        endpoint_dmax_tolerance=args.endpoint_dmax_tolerance,
        actionable_gain=args.actionable_gain,
        insensitive_gain=args.insensitive_gain,
        run_dir=args.run_dir,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
