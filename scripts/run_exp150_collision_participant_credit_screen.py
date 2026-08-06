#!/usr/bin/env python
"""Run and decide the pre-registered exp150 seed23 4M screen."""

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
from run_exp125_b0_screen import cuda_free_memory_mb, screen_acceptance


EXPERIMENT_ID = "exp150_collision_participant_actor_credit"
RUN_NAME = "participant_credit_seed23_4m"
TIMESTEPS = 2048
NUM_ENVS = 2048
ROLLOUT_STEPS = 64
CHECKPOINT_INTERVAL = 1024
EXP148_TIMEOUT = 0.0009765625
EXP148_REPEATED_MEDIAN = 5.0


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def component_gate(
    summary: dict,
    telemetry: list[dict],
    terrain_contrast: dict,
    repeated_conflicts: dict,
) -> dict:
    b0_gate = screen_acceptance(summary, telemetry, terrain_contrast)
    diagnostics = summary.get("training_diagnostics") or {}
    final_eval = summary.get("final_eval") or {}
    train_rows = [row for row in telemetry if row.get("phase") == "train"]

    def max_metric(key: str, default: float) -> float:
        values = [
            float(row[key])
            for row in train_rows
            if isinstance(row.get(key), (int, float))
            and math.isfinite(float(row[key]))
        ]
        return max(values, default=default)

    active_rate = max_metric("actor_credit_active_rate", 0.0)
    zero_sum_error = max_metric("actor_credit_policy_step_sum_abs_max", float("inf"))
    reward_error = max_metric(
        "actor_credit_team_reward_preservation_error", float("inf")
    )
    source_error = max_metric(
        "actor_credit_source_reconstruction_error", float("inf")
    )
    allocation_error = max_metric(
        "actor_credit_allocation_mean_error", float("inf")
    )
    per_seed = repeated_conflicts.get("per_seed") or {}
    repeated_medians = {
        seed: float(
            per_seed.get(seed, {})
            .get("failed_repeated_event_count", {})
            .get("median", float("inf"))
        )
        for seed in ("30023", "31023")
    }
    extra_checks = {
        "assignment_is_collision_participant": diagnostics.get(
            "actor_credit_assignment"
        )
        == "collision_participant_centered",
        "credit_activated": active_rate > 0.0,
        "credit_trace_nonzero": float(
            diagnostics.get("last_actor_credit_std", 0.0)
        )
        > 1.0e-4,
        "credit_step_zero_sum": zero_sum_error <= 1.0e-5,
        "team_reward_preserved": reward_error <= 1.0e-6,
        "source_reconstruction_exact": source_error <= 1.0e-5,
        "allocation_mean_preserved": allocation_error <= 1.0e-5,
        "every_seed_repeated_median_below_exp148": bool(repeated_medians)
        and max(repeated_medians.values()) < EXP148_REPEATED_MEDIAN,
        "timeout_not_shifted_from_collision": float(
            final_eval.get("timeout_rate", float("inf"))
        )
        <= EXP148_TIMEOUT + 0.10,
        "collision_constraint_disabled": diagnostics.get(
            "collision_constraint_enabled"
        )
        is False,
    }
    checks = {**b0_gate["checks"], **extra_checks}
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "thresholds": {
            **b0_gate["thresholds"],
            "credit_trace_std": 1.0e-4,
            "credit_zero_sum_and_source_error": 1.0e-5,
            "team_reward_error": 1.0e-6,
            "repeated_conflict_median_exclusive_upper": EXP148_REPEATED_MEDIAN,
            "timeout_rate": EXP148_TIMEOUT + 0.10,
        },
        "evidence": {
            **b0_gate["evidence"],
            "maximum_actor_credit_active_rate": active_rate,
            "maximum_credit_zero_sum_error": zero_sum_error,
            "maximum_team_reward_preservation_error": reward_error,
            "maximum_source_reconstruction_error": source_error,
            "maximum_allocation_mean_error": allocation_error,
            "repeated_conflict_median_by_seed": repeated_medians,
        },
        "b0_gate_passed": b0_gate["passed"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/experiment/exp150_collision_participant_actor_credit.yaml",
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
    if algorithm.get("actor_credit_assignment") != "collision_participant_centered":
        raise SystemExit(
            "exp150 requires actor_credit_assignment=collision_participant_centered."
        )
    if float(algorithm.get("actor_credit_scale", -1.0)) != 0.25:
        raise SystemExit("exp150 requires actor_credit_scale=0.25.")
    if algorithm.get("bc_updates") != 0 or algorithm.get("init_checkpoint") is not None:
        raise SystemExit("exp150 requires random-initialized Pure RL.")
    if algorithm.get("collision_constraint_enabled") is not False:
        raise SystemExit("exp150 cannot enable the collision constraint component.")

    suite_dir = ROOT / "outputs" / "runs" / EXPERIMENT_ID / "_suite"
    engineering_path = suite_dir / "metrics" / "engineering_gate.json"
    if not engineering_path.is_file():
        raise SystemExit("Missing exp150 engineering gate.")
    engineering = json.loads(engineering_path.read_text(encoding="utf-8"))
    if engineering.get("passed") is not True:
        raise SystemExit("exp150 engineering gate did not pass; refusing to start 4M.")
    free_mb = cuda_free_memory_mb() if args.device.startswith("cuda") else None
    if free_mb is not None and free_mb < args.minimum_free_gpu_mb:
        raise SystemExit(
            f"exp150 requires {args.minimum_free_gpu_mb} MB free GPU memory; found {free_mb} MB."
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
        data_seeds=(30023, 31023),
        run_dir=conflict_dir,
        output_experiment=EXPERIMENT_ID,
    )
    gate = component_gate(summary, telemetry, terrain_contrast, repeated_conflicts)
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
        "status": "screen_passed_plan_40m_only" if gate["passed"] else "stopped_at_4m_gate",
        "next_stage": "preregister_40m" if gate["passed"] else "stop_participant_credit",
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
            "producer": "scripts/run_exp150_collision_participant_credit_screen.py",
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
