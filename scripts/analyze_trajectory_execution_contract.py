#!/usr/bin/env python
"""Audit whether planned quintic paths match one-step proxy execution semantics."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from _common import ROOT, cfg_from_experiment
from analyze_actor_gradient_conflicts import _collect_seed_dataset
from analyze_joint_action_critic_feasibility import _parse_int_tuple, tensor_dict_digest


EXPERIMENT_ID = "exp147_trajectory_execution_contract_audit"
DEFAULT_SOURCE_RUN = (
    ROOT
    / "outputs/runs/exp125_decentralized_tiered_b0_pure_rl"
    / "b0_screen_seed23_4m_relative_quintic"
)


def _distribution(values: torch.Tensor) -> dict[str, float]:
    values = values.detach().float().flatten().cpu()
    return {
        "mean": float(values.mean()),
        "median": float(values.median()),
        "p10": float(torch.quantile(values, 0.10)),
        "p90": float(torch.quantile(values, 0.90)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def trajectory_contract_gate(
    *,
    validation: dict[str, dict[str, Any]],
    actor_parameters_unchanged: bool,
    actor_output_change: float,
) -> dict[str, Any]:
    checks = {
        "every_seed_timestamp_speed_violation_rate_ge_0_50": min(
            row["timestamp_speed_violation_rate"] for row in validation.values()
        )
        >= 0.50,
        "every_seed_required_to_declared_horizon_median_ge_2": min(
            row["required_to_declared_horizon_ratio"]["median"]
            for row in validation.values()
        )
        >= 2.0,
        "every_seed_actual_path_utilization_median_le_0_25": max(
            row["actual_path_utilization"]["median"] for row in validation.values()
        )
        <= 0.25,
        "every_seed_first_tracking_segment_fraction_median_le_0_20": max(
            row["first_tracking_segment_fraction"]["median"]
            for row in validation.values()
        )
        <= 0.20,
        "actor_parameters_unchanged": actor_parameters_unchanged,
        "actor_outputs_unchanged": actor_output_change == 0.0,
    }
    return {
        "mismatch_confirmed": all(checks.values()),
        "checks": checks,
        "thresholds": {
            "timestamp_speed_violation_rate": 0.50,
            "required_to_declared_horizon_ratio_median": 2.0,
            "actual_path_utilization_median_max": 0.25,
            "first_tracking_segment_fraction_median_max": 0.20,
        },
    }


def analyze_trajectory_execution_contract(
    *,
    config: str | Path,
    checkpoint: str | Path,
    device: str = "cuda",
    num_envs: int = 64,
    steps: int = 480,
    rollout_length: int = 64,
    seeds: tuple[int, ...] = (40023, 41023),
    run_name: str = "frozen_exp125_seed23",
    write_suite: bool = True,
    output_experiment: str = EXPERIMENT_ID,
) -> dict[str, Any]:
    torch_device = torch.device(device)
    checkpoint_data = torch.load(checkpoint, map_location=torch_device)
    cfg = cfg_from_experiment(config)
    max_linear_speed = float(cfg.low_level_control.max_linear_speed)
    actor_digest_before = tensor_dict_digest(checkpoint_data["rover_0"]["policy"])
    validation: dict[str, Any] = {}
    policy = None
    probe_observations = None
    probe_before = None
    for seed in seeds:
        dataset = _collect_seed_dataset(
            config=config,
            checkpoint_data=checkpoint_data,
            device=device,
            num_envs=num_envs,
            steps=steps,
            rollout_length=rollout_length,
            seed=seed,
            safety_credit_distance=0.72,
        )
        current_policy = dataset.pop("policy")
        if policy is None:
            policy = current_policy
            probe_observations = dataset["observations"][:32].detach().clone()
            with torch.no_grad():
                probe_before = policy.compute(
                    {"observations": probe_observations}, role="policy"
                )[0].detach().clone()
        arc = dataset["planned_path_arc_length"].flatten()
        valid = arc > 1.0e-6
        arc = arc[valid]
        horizon = dataset["planned_path_horizon"].flatten()[valid].clamp_min(1.0e-8)
        reference_speed = dataset["planned_path_reference_speed"].flatten()[valid].clamp_min(
            1.0e-8
        )
        tracking_arc = dataset["planned_tracking_point_arc_length"].flatten()[valid]
        actual = dataset["actual_step_displacement"].flatten()[valid]
        timestamp_speed = arc / horizon
        required_horizon = arc / reference_speed
        validation[str(seed)] = {
            "samples": int(valid.numel()),
            "nonzero_path_samples": int(valid.sum()),
            "zero_path_rate": float((~valid).float().mean().cpu()),
            "planned_arc_length_m": _distribution(arc),
            "declared_horizon_s": _distribution(horizon),
            "reference_speed_mps": _distribution(reference_speed),
            "timestamp_implied_speed_mps": _distribution(timestamp_speed),
            "timestamp_speed_violation_rate": float(
                (timestamp_speed > max_linear_speed).float().mean().cpu()
            ),
            "required_reference_horizon_s": _distribution(required_horizon),
            "required_to_declared_horizon_ratio": _distribution(
                required_horizon / horizon
            ),
            "actual_step_displacement_m": _distribution(actual),
            "actual_path_utilization": _distribution(actual / arc),
            "first_tracking_segment_fraction": _distribution(tracking_arc / arc),
        }

    assert policy is not None and probe_observations is not None and probe_before is not None
    with torch.no_grad():
        probe_after = policy.compute(
            {"observations": probe_observations}, role="policy"
        )[0]
    actor_digest_after = tensor_dict_digest(checkpoint_data["rover_0"]["policy"])
    actor_output_change = float((probe_after - probe_before).abs().amax().cpu())
    gate = trajectory_contract_gate(
        validation=validation,
        actor_parameters_unchanged=actor_digest_before == actor_digest_after,
        actor_output_change=actor_output_change,
    )
    status = (
        "allow_trajectory_contract_fix_plan_only"
        if gate["mismatch_confirmed"]
        else "stop_trajectory_contract_hypothesis"
    )
    if not output_experiment or "/" in output_experiment or "\\" in output_experiment:
        raise ValueError("output_experiment must be one path-safe experiment id.")
    run_dir = ROOT / "outputs/runs" / output_experiment / run_name
    metrics_path = run_dir / "metrics/trajectory_execution_contract.json"
    config_snapshot = run_dir / "config/experiment.yaml"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    config_snapshot.parent.mkdir(parents=True, exist_ok=True)
    source_config = Path(config)
    if not source_config.is_absolute():
        source_config = ROOT / source_config
    config_snapshot.write_text(source_config.read_text(encoding="utf-8"), encoding="utf-8")
    generated_at = datetime.now(timezone.utc).isoformat()
    result = {
        "schema_version": 1,
        "generated_at": generated_at,
        "experiment": output_experiment,
        "run": run_name,
        "status": status,
        "config": str(source_config),
        "checkpoint": str(checkpoint),
        "collection": {
            "device": device,
            "num_envs": num_envs,
            "steps": steps,
            "episode_coverage_seconds": steps * float(cfg.simulation.planning_dt),
            "rollout_length": rollout_length,
            "seeds": list(seeds),
        },
        "fixed_contract": {
            "planning_dt_s": float(cfg.simulation.planning_dt),
            "trajectory_points": int(cfg.trajectory_generator.n_trajectory_points),
            "reference_speed_mps": float(cfg.trajectory_generator.reference_speed),
            "max_linear_speed_mps": max_linear_speed,
            "controller_tracking_point_index": (
                1 if cfg.low_level_control.tracking_point_mode == "fixed_index" else None
            ),
            "controller_uses_reference_speed": False,
            "time_parameterization": cfg.trajectory_generator.time_parameterization,
            "tracking_point_mode": cfg.low_level_control.tracking_point_mode,
            "trajectory_timing_uses_reference_speed": (
                cfg.trajectory_generator.time_parameterization
                == "arc_length_reference_speed"
            ),
            "reward_samples_full_planned_path": True,
            "trajectory_replanned_every_environment_step": True,
        },
        "validation": validation,
        "gate": gate,
        "invariance": {
            "actor_digest_before": actor_digest_before,
            "actor_digest_after": actor_digest_after,
            "actor_probe_output_max_abs_change": actor_output_change,
        },
        "training_authorized": False,
        "four_million_screen_authorized": False,
        "decision": (
            "write_one_trajectory_semantics_fix_plan_but_do_not_train"
            if gate["mismatch_confirmed"]
            else "do_not_modify_trajectory_contract"
        ),
        "artifacts": {
            "metrics": str(metrics_path.relative_to(ROOT)),
            "config": str(config_snapshot.relative_to(ROOT)),
        },
    }
    metrics_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "generated_at": generated_at,
        "experiment": output_experiment,
        "run": run_name,
        "producer": "scripts/analyze_trajectory_execution_contract.py",
        "command": " ".join(sys.argv),
        "status": status,
        "source_checkpoint": str(checkpoint),
        "artifacts": result["artifacts"],
    }
    (run_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    if write_suite:
        suite_dir = ROOT / "outputs/runs" / output_experiment / "_suite"
        suite_metrics = suite_dir / "metrics/audit_summary.json"
        suite_metrics.parent.mkdir(parents=True, exist_ok=True)
        suite_metrics.write_text(json.dumps(result, indent=2), encoding="utf-8")
        (suite_dir / "run_manifest.json").write_text(
            json.dumps(
                {
                    **manifest,
                    "run": "_suite",
                    "artifacts": {
                        "audit_summary": str(suite_metrics.relative_to(ROOT)),
                        "source_run_manifest": str(
                            (run_dir / "run_manifest.json").relative_to(ROOT)
                        ),
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default=str(DEFAULT_SOURCE_RUN / "config/experiment.yaml")
    )
    parser.add_argument(
        "--checkpoint",
        default=str(DEFAULT_SOURCE_RUN / "checkpoints/ppo_timestep_002048.pt"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--steps", type=int, default=480)
    parser.add_argument("--rollout-length", type=int, default=64)
    parser.add_argument("--seeds", type=_parse_int_tuple, default=(40023, 41023))
    parser.add_argument("--run-name", default="frozen_exp125_seed23")
    parser.add_argument("--no-suite", action="store_true")
    parser.add_argument("--output-experiment", default=EXPERIMENT_ID)
    args = parser.parse_args()
    result = analyze_trajectory_execution_contract(
        config=args.config,
        checkpoint=args.checkpoint,
        device=args.device,
        num_envs=args.num_envs,
        steps=args.steps,
        rollout_length=args.rollout_length,
        seeds=args.seeds,
        run_name=args.run_name,
        write_suite=not args.no_suite,
        output_experiment=args.output_experiment,
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
