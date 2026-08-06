#!/usr/bin/env python
"""Run the pre-registered frozen B0 checkpoint trend audit."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from _common import ROOT
from evaluate_proxy_checkpoint_seed_sweep import run_seed_sweep
from evaluate_terrain_contrast import evaluate_terrain_contrast


EXPERIMENT_ID = "exp144_b0_checkpoint_trend_multiseed_audit"
RUN_NAME = "frozen_exp125_seed23_t1024_t2048"
B0_RUN = ROOT / "outputs/runs/exp125_decentralized_tiered_b0_pure_rl/b0_screen_seed23_4m_relative_quintic"
CHECKPOINTS = ("ppo_timestep_001024.pt", "ppo_timestep_002048.pt")
EVAL_SEEDS = (1023, 2023, 3023, 4023, 5023)
CONTRAST_SEEDS = (12023, 13023, 14023)


def _mean(values: list[float]) -> float:
    if not values:
        raise ValueError("Cannot average an empty metric list.")
    return statistics.fmean(values)


def paired_gate(
    sweep: dict[str, Any],
    contrast_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    by_checkpoint_seed = {
        (str(row["checkpoint"]), int(row["seed"])): row
        for row in sweep.get("rows", [])
    }
    early_label, late_label = CHECKPOINTS
    paired_rows: list[dict[str, float | int]] = []
    for seed in EVAL_SEEDS:
        early = by_checkpoint_seed[(early_label, seed)]
        late = by_checkpoint_seed[(late_label, seed)]
        early_collision = float(early["collision_rate"])
        paired_rows.append(
            {
                "seed": seed,
                "dmax_relative_improvement": (
                    float(early["dmax_reduction_ratio"])
                    - float(late["dmax_reduction_ratio"])
                )
                / float(early["dmax_reduction_ratio"]),
                "collision_relative_improvement": (
                    early_collision - float(late["collision_rate"])
                )
                / max(early_collision, 1.0e-8),
                "success_absolute_gain": float(late["success_rate"])
                - float(early["success_rate"]),
                "early_dmax": float(early["dmax_reduction_ratio"]),
                "late_dmax": float(late["dmax_reduction_ratio"]),
                "early_success": float(early["success_rate"]),
                "late_success": float(late["success_rate"]),
                "early_collision": early_collision,
                "late_collision": float(late["collision_rate"]),
                "early_timeout": float(early["timeout_rate"]),
                "late_timeout": float(late["timeout_rate"]),
            }
        )

    contrast_lookup = {
        (str(row["checkpoint_label"]), int(row["seed"])): row
        for row in contrast_rows
    }
    terrain_pairs: list[dict[str, float | int]] = []
    for seed in CONTRAST_SEEDS:
        early = contrast_lookup[(early_label, seed)]
        late = contrast_lookup[(late_label, seed)]
        early_mse = float(early["action_mse_normal_vs_zero_terrain"])
        late_mse = float(late["action_mse_normal_vs_zero_terrain"])
        early_risk = float(early["path_risk_reduction_fraction"])
        late_risk = float(late["path_risk_reduction_fraction"])
        terrain_pairs.append(
            {
                "seed": seed,
                "early_action_mse": early_mse,
                "late_action_mse": late_mse,
                "action_mse_gain": late_mse - early_mse,
                "early_path_risk_reduction": early_risk,
                "late_path_risk_reduction": late_risk,
                "path_risk_reduction_gain": late_risk - early_risk,
            }
        )

    mean_dmax = _mean(
        [float(row["dmax_relative_improvement"]) for row in paired_rows]
    )
    mean_collision = _mean(
        [float(row["collision_relative_improvement"]) for row in paired_rows]
    )
    mean_success = _mean(
        [float(row["success_absolute_gain"]) for row in paired_rows]
    )
    late_action_mse_mean = _mean(
        [float(row["late_action_mse"]) for row in terrain_pairs]
    )
    path_risk_gain_mean = _mean(
        [float(row["path_risk_reduction_gain"]) for row in terrain_pairs]
    )
    checks = {
        "dmax_improved_in_four_of_five_seeds": sum(
            float(row["dmax_relative_improvement"]) > 0.0 for row in paired_rows
        )
        >= 4,
        "collision_improved_in_four_of_five_seeds": sum(
            float(row["collision_relative_improvement"]) > 0.0
            for row in paired_rows
        )
        >= 4,
        "success_improved_in_four_of_five_seeds": sum(
            float(row["success_absolute_gain"]) > 0.0 for row in paired_rows
        )
        >= 4,
        "mean_dmax_relative_improvement_ge_15pct": mean_dmax >= 0.15,
        "mean_collision_relative_improvement_ge_50pct": mean_collision >= 0.50,
        "mean_success_absolute_gain_ge_2pp": mean_success >= 0.02,
        "terrain_action_mse_improved_in_two_of_three_seeds": sum(
            float(row["action_mse_gain"]) > 0.0 for row in terrain_pairs
        )
        >= 2,
        "terrain_path_risk_improved_in_two_of_three_seeds": sum(
            float(row["path_risk_reduction_gain"]) > 0.0 for row in terrain_pairs
        )
        >= 2,
        "late_terrain_action_mse_mean_ge_0_01": late_action_mse_mean >= 0.01,
        "terrain_path_risk_gain_mean_ge_0_005": path_risk_gain_mean >= 0.005,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "thresholds": {
            "paired_seed_count": 5,
            "minimum_improved_seed_count": 4,
            "mean_dmax_relative_improvement": 0.15,
            "mean_collision_relative_improvement": 0.50,
            "mean_success_absolute_gain": 0.02,
            "contrast_seed_count": 3,
            "minimum_terrain_improved_seed_count": 2,
            "late_terrain_action_mse_mean": 0.01,
            "terrain_path_risk_gain_mean": 0.005,
        },
        "evidence": {
            "paired_eval_rows": paired_rows,
            "terrain_contrast_pairs": terrain_pairs,
            "mean_dmax_relative_improvement": mean_dmax,
            "mean_collision_relative_improvement": mean_collision,
            "mean_success_absolute_gain": mean_success,
            "late_terrain_action_mse_mean": late_action_mse_mean,
            "terrain_path_risk_gain_mean": path_risk_gain_mean,
        },
        "decision": (
            "allow_b0_depth_plan_only"
            if all(checks.values())
            else "stop_b0_depth_training_hypothesis"
        ),
        "forty_m_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--run-name", default=RUN_NAME)
    parser.add_argument("--num-envs", type=int, default=512)
    parser.add_argument("--contrast-num-envs", type=int, default=256)
    args = parser.parse_args()
    run_dir = ROOT / "outputs/runs" / EXPERIMENT_ID / args.run_name
    if run_dir.exists() and any(run_dir.iterdir()):
        raise SystemExit(f"Refusing to overwrite non-empty run directory: {run_dir}")
    config = B0_RUN / "config/experiment.yaml"
    sweep_dir = run_dir / "metrics/checkpoint_seed_sweep"
    sweep = run_seed_sweep(
        config=config,
        run_dir=B0_RUN,
        checkpoints=list(CHECKPOINTS),
        seeds=list(EVAL_SEEDS),
        device=args.device,
        num_envs=args.num_envs,
        steps=480,
        output_dir=sweep_dir,
    )
    contrast_rows: list[dict[str, Any]] = []
    for checkpoint_label in CHECKPOINTS:
        checkpoint = B0_RUN / "checkpoints" / checkpoint_label
        for seed in CONTRAST_SEEDS:
            output = (
                run_dir
                / "metrics/terrain_contrast"
                / f"{checkpoint.stem}_seed{seed}.json"
            )
            print(f"terrain contrast {checkpoint_label} seed={seed}", flush=True)
            row = evaluate_terrain_contrast(
                config=config,
                checkpoint=checkpoint,
                device=args.device,
                num_envs=args.contrast_num_envs,
                steps=120,
                seed=seed,
                output=output,
            )
            row["checkpoint_label"] = checkpoint_label
            contrast_rows.append(row)
    gate = paired_gate(sweep, contrast_rows)
    generated_at = datetime.now(timezone.utc).isoformat()
    result = {
        "schema_version": 1,
        "generated_at": generated_at,
        "experiment": EXPERIMENT_ID,
        "run": args.run_name,
        "frozen_source_run": str(B0_RUN.relative_to(ROOT)),
        "device": args.device,
        "num_envs": args.num_envs,
        "contrast_num_envs": args.contrast_num_envs,
        "eval_seeds": list(EVAL_SEEDS),
        "contrast_seeds": list(CONTRAST_SEEDS),
        "gate": gate,
    }
    result_path = run_dir / "metrics/paired_trend_gate.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "generated_at": generated_at,
        "experiment": EXPERIMENT_ID,
        "run": args.run_name,
        "producer": "scripts/run_exp144_b0_checkpoint_trend_audit.py",
        "command": " ".join(sys.argv),
        "status": gate["decision"],
        "artifacts": {
            "paired_gate": str(result_path.relative_to(ROOT)),
            "checkpoint_seed_sweep": str(
                (sweep_dir / "summary.json").relative_to(ROOT)
            ),
            "terrain_contrast_dir": str(
                (run_dir / "metrics/terrain_contrast").relative_to(ROOT)
            ),
        },
    }
    (run_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    suite_dir = ROOT / "outputs/runs" / EXPERIMENT_ID / "_suite"
    suite_result = suite_dir / "metrics/audit_summary.json"
    suite_result.parent.mkdir(parents=True, exist_ok=True)
    suite_result.write_text(json.dumps(result, indent=2), encoding="utf-8")
    (suite_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                **manifest,
                "run": "_suite",
                "artifacts": {
                    "audit_summary": str(suite_result.relative_to(ROOT)),
                    "source_run_manifest": str(
                        (run_dir / "run_manifest.json").relative_to(ROOT)
                    ),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
