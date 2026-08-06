#!/usr/bin/env python
"""Run and decide the pre-registered exp140 seed23 4M component screen."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from _common import ROOT, load_yaml
from analyze_failed_episode_repeated_conflicts import (
    analyze_failed_episode_repeated_conflicts,
)
from evaluate_terrain_contrast import evaluate_terrain_contrast
from run_exp125_b0_screen import _quarter_dmax, cuda_free_memory_mb


EXPERIMENT_ID = "exp140_agent_local_near_credit"
RUN_NAME = "local_near_seed23_4m"
TIMESTEPS = 2048
NUM_ENVS = 2048
ROLLOUT_STEPS = 64
CHECKPOINT_INTERVAL = 1024
BASELINE_METRICS = (
    ROOT
    / "outputs"
    / "runs"
    / "exp125_decentralized_tiered_b0_pure_rl"
    / "b0_screen_seed23_4m_relative_quintic"
    / "metrics"
    / "summary.json"
)
BASELINE_CONFLICTS = (
    ROOT
    / "outputs"
    / "runs"
    / "exp136_failed_episode_repeated_conflicts"
    / "frozen_exp125_seed23"
    / "metrics"
    / "failed_episode_repeated_conflicts.json"
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def component_gate(
    summary: dict,
    telemetry: list[dict],
    repeated_conflicts: dict,
    baseline_summary: dict,
    baseline_conflicts: dict,
) -> dict:
    diagnostics = summary.get("training_diagnostics") or {}
    final_eval = summary.get("final_eval") or {}
    baseline_eval = baseline_summary.get("final_eval") or {}
    first_dmax, last_dmax = _quarter_dmax(telemetry)
    dmax_reduction = (
        (first_dmax - last_dmax) / first_dmax
        if first_dmax is not None and last_dmax is not None and first_dmax > 0.0
        else float("-inf")
    )
    success_count = max(
        (
            int(record.get("success_done", record.get("success_count", 0)))
            for record in telemetry
        ),
        default=0,
    )
    train_rows = [row for row in telemetry if row.get("phase") == "train"]
    active_rates = [
        float(row["actor_credit_active_rate"])
        for row in train_rows
        if isinstance(row.get("actor_credit_active_rate"), (int, float))
        and math.isfinite(float(row["actor_credit_active_rate"]))
    ]
    mean_active_rate = (
        sum(active_rates) / len(active_rates) if active_rates else float("-inf")
    )
    reward_errors = [
        float(row.get("actor_credit_team_reward_preservation_error", float("inf")))
        for row in train_rows
    ]
    source_errors = [
        float(row.get("actor_credit_source_reconstruction_error", float("inf")))
        for row in train_rows
    ]
    per_seed = repeated_conflicts.get("per_seed") or {}
    baseline_per_seed = baseline_conflicts.get("per_seed") or {}
    conflict_reductions: dict[str, float] = {}
    for seed in ("28023", "29023"):
        current = float(
            per_seed.get(seed, {})
            .get("failed_repeated_event_count", {})
            .get("median", float("inf"))
        )
        baseline = float(
            baseline_per_seed.get(seed, {})
            .get("failed_repeated_event_count", {})
            .get("median", 0.0)
        )
        conflict_reductions[seed] = (
            (baseline - current) / baseline if baseline > 0.0 else float("-inf")
        )

    baseline_collision = float(baseline_eval.get("collision_rate", 0.0966796875))
    baseline_success = float(baseline_eval.get("success_rate", 0.0517578125))
    baseline_dmax = float(baseline_eval.get("dmax_reduction_ratio", 0.2047232687))
    thresholds = {
        "training_dmax_reduction": 0.20,
        "action_std": 1.0e-4,
        "collision_rate": baseline_collision * 0.70,
        "success_rate": max(0.0, baseline_success - 0.02),
        "dmax_reduction_ratio": baseline_dmax + 0.05,
        "repeated_conflict_reduction": 0.20,
        "credit_active_rate": 0.08,
        "credit_trace_std": 1.0e-4,
        "invariance_error": 1.0e-6,
    }
    checks = {
        "no_nan_or_inf": summary.get("status") == "ok"
        and diagnostics.get("policy_parameters_finite") is True
        and bool(telemetry)
        and all(record.get("nan_flag") is not True for record in telemetry),
        "actor_updated": float(diagnostics.get("policy_parameter_delta_l2", 0.0)) > 0.0,
        "neighbor_encoder_updated": float(
            diagnostics.get("neighbor_encoder_parameter_delta_l2", 0.0)
        )
        > 0.0,
        "terrain_encoder_updated": float(
            diagnostics.get("terrain_encoder_parameter_delta_l2", 0.0)
        )
        > 0.0,
        "action_non_degenerate": float(
            diagnostics.get("post_training_action_std", 0.0)
        )
        > thresholds["action_std"],
        "nonzero_success_episode": success_count > 0
        or float(final_eval.get("success_rate", 0.0)) > 0.0,
        "training_dmax_reduced_20pct": dmax_reduction
        >= thresholds["training_dmax_reduction"],
        "collision_reduced_30pct": float(
            final_eval.get("collision_rate", float("inf"))
        )
        <= thresholds["collision_rate"],
        "success_drop_at_most_2pp": float(final_eval.get("success_rate", 0.0))
        >= thresholds["success_rate"],
        "dmax_degradation_at_most_0_05": float(
            final_eval.get("dmax_reduction_ratio", float("inf"))
        )
        <= thresholds["dmax_reduction_ratio"],
        "every_seed_repeated_conflicts_reduced_20pct": bool(conflict_reductions)
        and min(conflict_reductions.values())
        >= thresholds["repeated_conflict_reduction"],
        "credit_active_rate_ge_0_08": mean_active_rate
        >= thresholds["credit_active_rate"],
        "credit_trace_nonzero": float(
            diagnostics.get("last_actor_credit_std", 0.0)
        )
        > thresholds["credit_trace_std"],
        "team_reward_preserved": max(reward_errors, default=float("inf"))
        <= thresholds["invariance_error"],
        "source_reconstruction_exact": max(source_errors, default=float("inf"))
        <= thresholds["invariance_error"],
        "assignment_is_local_near": diagnostics.get("actor_credit_assignment")
        == "near_potential_local",
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "thresholds": thresholds,
        "evidence": {
            "first_quarter_dmax_mean": first_dmax,
            "last_quarter_dmax_mean": last_dmax,
            "training_dmax_reduction": dmax_reduction,
            "training_success_count": success_count,
            "mean_actor_credit_active_rate": mean_active_rate,
            "maximum_team_reward_preservation_error": max(
                reward_errors, default=float("inf")
            ),
            "maximum_source_reconstruction_error": max(
                source_errors, default=float("inf")
            ),
            "repeated_conflict_reduction_by_seed": conflict_reductions,
            "final_eval": {
                key: final_eval.get(key)
                for key in (
                    "dmax_reduction_ratio",
                    "success_rate",
                    "collision_rate",
                    "timeout_rate",
                )
            },
            "baseline_final_eval": {
                key: baseline_eval.get(key)
                for key in (
                    "dmax_reduction_ratio",
                    "success_rate",
                    "collision_rate",
                    "timeout_rate",
                )
            },
            "training_diagnostics": diagnostics,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/experiment/exp140_agent_local_near_credit.yaml",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--run-name", default=RUN_NAME)
    parser.add_argument("--minimum-free-gpu-mb", type=int, default=8192)
    parser.add_argument("--contrast-num-envs", type=int, default=512)
    parser.add_argument("--contrast-steps", type=int, default=120)
    parser.add_argument("--conflict-num-envs", type=int, default=128)
    parser.add_argument("--conflict-steps", type=int, default=512)
    args = parser.parse_args()

    config = Path(args.config)
    if not config.is_absolute():
        config = ROOT / config
    raw = load_yaml(config)
    experiment = raw.get("experiment") or {}
    algorithm = raw.get("algorithm") or {}
    if experiment.get("name") != EXPERIMENT_ID:
        raise SystemExit(f"Expected experiment.name={EXPERIMENT_ID!r}.")
    if algorithm.get("actor_credit_assignment") != "near_potential_local":
        raise SystemExit("exp140 requires actor_credit_assignment=near_potential_local.")
    if float(algorithm.get("actor_credit_scale", -1.0)) != 0.25:
        raise SystemExit("exp140 requires the pre-registered actor_credit_scale=0.25.")
    if algorithm.get("bc_updates") != 0 or algorithm.get("init_checkpoint") is not None:
        raise SystemExit("exp140 requires random-initialized Pure RL.")

    suite_dir = ROOT / "outputs" / "runs" / EXPERIMENT_ID / "_suite"
    engineering_path = suite_dir / "metrics" / "engineering_gate.json"
    if not engineering_path.is_file():
        raise SystemExit("Missing exp140 engineering gate.")
    engineering = json.loads(engineering_path.read_text(encoding="utf-8"))
    if engineering.get("passed") is not True:
        raise SystemExit("exp140 engineering gate did not pass; refusing to start 4M.")
    if not BASELINE_METRICS.is_file() or not BASELINE_CONFLICTS.is_file():
        raise SystemExit("Missing frozen exp125/exp136 baseline evidence.")

    free_mb = cuda_free_memory_mb() if args.device.startswith("cuda") else None
    if free_mb is not None and free_mb < args.minimum_free_gpu_mb:
        raise SystemExit(
            f"exp140 requires {args.minimum_free_gpu_mb} MB free GPU memory; found {free_mb} MB."
        )

    run_name = str(args.run_name)
    run_dir = ROOT / "outputs" / "runs" / EXPERIMENT_ID / run_name
    if run_dir.exists() and any(run_dir.iterdir()):
        raise SystemExit(f"Refusing to overwrite non-empty run directory: {run_dir}")
    command = [
        str(ROOT / ".venv_isaaclab" / "bin" / "python"),
        str(ROOT / "scripts" / "train_skrl_mappo.py"),
        "--config", str(config),
        "--device", args.device,
        "--timesteps", str(TIMESTEPS),
        "--seed", "23",
        "--num-envs", str(NUM_ENVS),
        "--output-layout", "run",
        "--run-name", run_name,
        "--rollout-steps", str(ROLLOUT_STEPS),
        "--checkpoint-interval", str(CHECKPOINT_INTERVAL),
        "--eval-num-envs", "1024",
        "--eval-steps", "480",
        "--eval-seed-offset", "1000",
        "--bc-updates", "0",
        "--selection-gate", "screen",
    ]
    print(f"running: {' '.join(command)}", flush=True)
    completed = subprocess.run(command, cwd=ROOT, check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)

    metrics_dir = run_dir / "metrics"
    summary = json.loads((metrics_dir / "summary.json").read_text(encoding="utf-8"))
    telemetry = [
        json.loads(line)
        for line in (metrics_dir / "train_metrics.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    terrain_contrast = evaluate_terrain_contrast(
        config=run_dir / "config" / "experiment.yaml",
        checkpoint=run_dir / "checkpoints" / "best.pt",
        device=args.device,
        num_envs=args.contrast_num_envs,
        steps=args.contrast_steps,
        seed=12023,
        initial_state_progress=TIMESTEPS,
        run_dir=run_dir,
    )
    conflict_dir = run_dir / "diagnostics" / "failed_episode_repeated_conflicts"
    repeated_conflicts = analyze_failed_episode_repeated_conflicts(
        config=run_dir / "config" / "experiment.yaml",
        checkpoint=run_dir / "checkpoints" / "best.pt",
        device=args.device,
        num_envs=args.conflict_num_envs,
        steps=args.conflict_steps,
        data_seeds=(28023, 29023),
        run_dir=conflict_dir,
    )
    baseline_summary = json.loads(BASELINE_METRICS.read_text(encoding="utf-8"))
    baseline_conflicts = json.loads(BASELINE_CONFLICTS.read_text(encoding="utf-8"))
    gate = component_gate(
        summary,
        telemetry,
        repeated_conflicts,
        baseline_summary,
        baseline_conflicts,
    )
    result = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment": EXPERIMENT_ID,
        "run": run_name,
        "seed": 23,
        "timesteps": TIMESTEPS,
        "env_steps": TIMESTEPS * NUM_ENVS,
        "free_gpu_memory_mb_at_start": free_mb,
        "engineering_gate": engineering,
        "component_gate": gate,
        "terrain_contrast_descriptive": terrain_contrast,
        "status": "component_gate_passed_plan_only" if gate["passed"] else "stopped_at_component_gate",
        "next_stage": "plan_unified_credit_only" if gate["passed"] else "stop_local_near_credit",
        "forty_m_authorized": False,
    }
    _write_json(metrics_dir / "component_gate.json", result)
    _write_json(suite_dir / "metrics" / "component_screen_summary.json", result)
    _write_json(
        suite_dir / "run_manifest.json",
        {
            "schema_version": 1,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "experiment": EXPERIMENT_ID,
            "producer": "scripts/run_exp140_local_near_credit_screen.py",
            "command": " ".join(sys.argv),
            "status": result["status"],
            "artifacts": {
                "engineering_gate": str(engineering_path.relative_to(ROOT)),
                "component_summary": str(
                    (suite_dir / "metrics" / "component_screen_summary.json").relative_to(ROOT)
                ),
                "run_component_gate": str(
                    (metrics_dir / "component_gate.json").relative_to(ROOT)
                ),
                "terrain_contrast": str(
                    (metrics_dir / "terrain_contrast.json").relative_to(ROOT)
                ),
                "repeated_conflicts": str(
                    (
                        conflict_dir
                        / "metrics"
                        / "failed_episode_repeated_conflicts.json"
                    ).relative_to(ROOT)
                ),
            },
        },
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
