#!/usr/bin/env python
"""Run and decide the pre-registered exp142 seed23 4M component screen."""

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
from run_exp125_b0_screen import cuda_free_memory_mb


EXPERIMENT_ID = "exp142_collision_lagrangian_component"
RUN_NAME = "collision_lagrangian_seed23_4m"
TIMESTEPS = 2048
NUM_ENVS = 2048
ROLLOUT_STEPS = 64
CHECKPOINT_INTERVAL = 1024
BASELINE_METRICS = ROOT / "outputs/runs/exp125_decentralized_tiered_b0_pure_rl/b0_screen_seed23_4m_relative_quintic/metrics/summary.json"
BASELINE_CONFLICTS = ROOT / "outputs/runs/exp136_failed_episode_repeated_conflicts/frozen_exp125_seed23/metrics/failed_episode_repeated_conflicts.json"


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _quarter_means(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    width = max(1, math.ceil(len(values) / 4))
    return sum(values[:width]) / width, sum(values[-width:]) / width


def component_gate(
    summary: dict,
    repeated_conflicts: dict,
    baseline_conflicts: dict,
) -> dict:
    diagnostics = summary.get("training_diagnostics") or {}
    final_eval = summary.get("final_eval") or {}
    history = diagnostics.get("collision_constraint_history") or []
    rates = [
        float(row["episode_equivalent_collision_rate"])
        for row in history
        if math.isfinite(float(row.get("episode_equivalent_collision_rate", float("nan"))))
    ]
    first_rate, last_rate = _quarter_means(rates)
    rate_reduction = (
        (first_rate - last_rate) / first_rate
        if first_rate is not None and last_rate is not None and first_rate > 0.0
        else float("-inf")
    )
    last_width = max(1, math.ceil(len(history) / 4)) if history else 0
    last_rows = history[-last_width:] if last_width else []
    upper_bound_fraction = (
        sum(float(row.get("lagrangian_multiplier", 0.0)) >= 2.0 - 1.0e-8 for row in last_rows)
        / len(last_rows)
        if last_rows
        else 1.0
    )
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
    final_lambda = float(diagnostics.get("lagrangian_multiplier", float("nan")))
    cost_losses = [float(row.get("cost_value_loss", float("nan"))) for row in history]
    checks = {
        "run_finite": summary.get("status") == "ok"
        and diagnostics.get("policy_parameters_finite") is True
        and diagnostics.get("collision_cost_value_parameters_finite") is True
        and bool(history)
        and all(math.isfinite(value) for value in rates + cost_losses),
        "actor_updated": float(diagnostics.get("policy_parameter_delta_l2", 0.0)) > 0.0,
        "neighbor_encoder_updated": float(
            diagnostics.get("neighbor_encoder_parameter_delta_l2", 0.0)
        )
        > 0.0,
        "terrain_encoder_updated": float(
            diagnostics.get("terrain_encoder_parameter_delta_l2", 0.0)
        )
        > 0.0,
        "reward_critic_updated": float(
            diagnostics.get("reward_critic_parameter_delta_l2", 0.0)
        )
        > 0.0,
        "cost_critic_updated": float(
            diagnostics.get("collision_cost_value_parameter_delta_l2", 0.0)
        )
        > 0.0
        and int(diagnostics.get("collision_cost_critic_update_count", 0)) > 0,
        "action_non_degenerate": float(
            diagnostics.get("post_training_action_std", 0.0)
        )
        > 1.0e-4,
        "training_collision_rate_reduced_30pct": rate_reduction >= 0.30,
        "final_collision_le_0_0677": float(
            final_eval.get("collision_rate", float("inf"))
        )
        <= 0.0677,
        "final_success_ge_0_0318": float(final_eval.get("success_rate", 0.0))
        >= 0.0318,
        "final_dmax_ratio_le_0_2547": float(
            final_eval.get("dmax_reduction_ratio", float("inf"))
        )
        <= 0.2547,
        "every_seed_repeated_conflicts_reduced_20pct": bool(conflict_reductions)
        and min(conflict_reductions.values()) >= 0.20,
        "final_lambda_in_open_interval": 0.05 < final_lambda < 2.0,
        "lambda_not_long_pinned_at_upper_bound": upper_bound_fraction < 0.50,
        "pure_rl_without_actor_credit": int(diagnostics.get("bc_updates", -1)) == 0
        and float(diagnostics.get("bc_parameter_delta_l2", -1.0)) == 0.0
        and diagnostics.get("actor_credit_assignment") == "none",
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "thresholds": {
            "training_collision_rate_reduction": 0.30,
            "collision_rate": 0.0677,
            "success_rate": 0.0318,
            "dmax_reduction_ratio": 0.2547,
            "repeated_conflict_reduction": 0.20,
            "lagrangian_multiplier_open_interval": [0.05, 2.0],
            "last_quarter_upper_bound_fraction": 0.50,
        },
        "evidence": {
            "first_quarter_episode_equivalent_collision_rate": first_rate,
            "last_quarter_episode_equivalent_collision_rate": last_rate,
            "training_collision_rate_reduction": rate_reduction,
            "last_quarter_upper_bound_fraction": upper_bound_fraction,
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
            "training_diagnostics": diagnostics,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/experiment/exp142_collision_lagrangian_component.yaml",
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
    algorithm = raw.get("algorithm") or {}
    if (raw.get("experiment") or {}).get("name") != EXPERIMENT_ID:
        raise SystemExit(f"Expected experiment.name={EXPERIMENT_ID!r}.")
    if algorithm.get("collision_constraint_enabled") is not True:
        raise SystemExit("exp142 requires collision_constraint_enabled=true.")
    if algorithm.get("bc_updates") != 0 or algorithm.get("init_checkpoint") is not None:
        raise SystemExit("exp142 requires random-initialized Pure RL.")
    if algorithm.get("actor_credit_assignment") != "none":
        raise SystemExit("exp142 forbids Actor credit stacking.")

    suite_dir = ROOT / "outputs/runs" / EXPERIMENT_ID / "_suite"
    engineering_path = suite_dir / "metrics/engineering_gate.json"
    engineering = json.loads(engineering_path.read_text(encoding="utf-8"))
    if engineering.get("passed") is not True:
        raise SystemExit("exp142 engineering gate did not pass; refusing 4M.")
    if not BASELINE_METRICS.is_file() or not BASELINE_CONFLICTS.is_file():
        raise SystemExit("Missing frozen exp125/exp136 baseline evidence.")
    free_mb = cuda_free_memory_mb() if args.device.startswith("cuda") else None
    if free_mb is not None and free_mb < args.minimum_free_gpu_mb:
        raise SystemExit(
            f"exp142 requires {args.minimum_free_gpu_mb} MB free GPU memory; found {free_mb} MB."
        )

    run_name = str(args.run_name)
    run_dir = ROOT / "outputs/runs" / EXPERIMENT_ID / run_name
    if run_dir.exists() and any(run_dir.iterdir()):
        raise SystemExit(f"Refusing to overwrite non-empty run directory: {run_dir}")
    command = [
        str(ROOT / ".venv_isaaclab/bin/python"),
        str(ROOT / "scripts/train_skrl_mappo.py"),
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
    terrain_contrast = evaluate_terrain_contrast(
        config=run_dir / "config/experiment.yaml",
        checkpoint=run_dir / "checkpoints/best.pt",
        device=args.device,
        num_envs=args.contrast_num_envs,
        steps=args.contrast_steps,
        seed=12023,
        initial_state_progress=TIMESTEPS,
        run_dir=run_dir,
    )
    conflict_dir = run_dir / "diagnostics/failed_episode_repeated_conflicts"
    repeated_conflicts = analyze_failed_episode_repeated_conflicts(
        config=run_dir / "config/experiment.yaml",
        checkpoint=run_dir / "checkpoints/best.pt",
        device=args.device,
        num_envs=args.conflict_num_envs,
        steps=args.conflict_steps,
        data_seeds=(28023, 29023),
        run_dir=conflict_dir,
    )
    baseline_conflicts = json.loads(BASELINE_CONFLICTS.read_text(encoding="utf-8"))
    gate = component_gate(summary, repeated_conflicts, baseline_conflicts)
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
        "next_stage": "plan_unified_terrain_safety_only" if gate["passed"] else "stop_collision_constraint_direction",
        "forty_m_authorized": False,
    }
    _write_json(metrics_dir / "component_gate.json", result)
    _write_json(suite_dir / "metrics/component_screen_summary.json", result)
    _write_json(
        suite_dir / "run_manifest.json",
        {
            "schema_version": 1,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "experiment": EXPERIMENT_ID,
            "producer": "scripts/run_exp142_collision_lagrangian_screen.py",
            "command": " ".join(sys.argv),
            "status": result["status"],
            "artifacts": {
                "engineering_gate": str(engineering_path.relative_to(ROOT)),
                "component_summary": str((suite_dir / "metrics/component_screen_summary.json").relative_to(ROOT)),
                "run_component_gate": str((metrics_dir / "component_gate.json").relative_to(ROOT)),
                "terrain_contrast": str((metrics_dir / "terrain_contrast.json").relative_to(ROOT)),
                "repeated_conflicts": str((conflict_dir / "metrics/failed_episode_repeated_conflicts.json").relative_to(ROOT)),
            },
        },
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
