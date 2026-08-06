#!/usr/bin/env python
"""Audit B0 horizon sufficiency and exp142 reward/cost scale competition."""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from _common import ROOT, load_yaml


EXPERIMENT_ID = "exp143_b0_horizon_and_constraint_competition"
DEFAULT_RUN = "frozen_exp125_exp142"
DEFAULT_B0_RUN = ROOT / "outputs/runs/exp125_decentralized_tiered_b0_pure_rl/b0_screen_seed23_4m_relative_quintic"
DEFAULT_CONSTRAINT_RUN = ROOT / "outputs/runs/exp142_collision_lagrangian_component/collision_lagrangian_seed23_4m"
BASE_CONFIG = ROOT / "configs/experiment/exp125_decentralized_tiered_b0_pure_rl_relative_quintic.yaml"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _train_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [row for row in records if row.get("phase") == "train"]
    if not rows:
        raise ValueError("Training telemetry contains no phase=train rows.")
    return rows


def quarter_means(values: list[float]) -> list[float]:
    if len(values) < 4 or len(values) % 4:
        raise ValueError("Audit requires a non-empty telemetry length divisible by four.")
    width = len(values) // 4
    return [
        sum(values[index * width : (index + 1) * width]) / width
        for index in range(4)
    ]


def candidate_metrics(eval_payload: dict[str, Any], timestep: int) -> dict[str, float]:
    matches = [
        row
        for row in eval_payload.get("evaluations", [])
        if int(row.get("candidate_timestep", -1)) == timestep
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one candidate evaluation at timestep {timestep}.")
    return {
        key: float(matches[0][key])
        for key in (
            "dmax_reduction_ratio",
            "success_rate",
            "collision_rate",
            "timeout_rate",
        )
    }


def cumulative_at_or_before(
    rows: list[dict[str, Any]],
    *,
    timestep: int,
    field: str,
) -> int:
    eligible = [row for row in rows if int(row.get("timesteps", -1)) <= timestep]
    if not eligible:
        return 0
    return int(eligible[-1].get(field, 0))


def build_audit(
    *,
    b0_summary: dict[str, Any],
    b0_eval: dict[str, Any],
    b0_telemetry: list[dict[str, Any]],
    constraint_summary: dict[str, Any],
    constraint_telemetry: list[dict[str, Any]],
    curriculum_warmup_timesteps: int,
) -> dict[str, Any]:
    b0_rows = _train_rows(b0_telemetry)
    constraint_rows = _train_rows(constraint_telemetry)
    early = candidate_metrics(b0_eval, 1024)
    late = candidate_metrics(b0_eval, 2048)
    dmax_improvement = (
        early["dmax_reduction_ratio"] - late["dmax_reduction_ratio"]
    ) / early["dmax_reduction_ratio"]
    collision_improvement = (
        early["collision_rate"] - late["collision_rate"]
    ) / early["collision_rate"]
    success_gain = late["success_rate"] - early["success_rate"]
    dmax_quarters = quarter_means([float(row["dmax_mean"]) for row in b0_rows])
    last_quarter_improvement = (
        dmax_quarters[2] - dmax_quarters[3]
    ) / dmax_quarters[2]

    constraint_diagnostics = constraint_summary.get("training_diagnostics") or {}
    history = constraint_diagnostics.get("collision_constraint_history") or []
    if not history:
        raise ValueError("exp142 summary has no collision constraint history.")
    first_half = history[: len(history) // 2]
    second_half = history[len(history) // 2 :]
    unbounded_rate_fraction = sum(
        float(row["episode_equivalent_collision_rate"]) > 1.0 for row in history
    ) / len(history)
    first_competing = next(
        (
            row
            for row in history
            if float(row["lagrangian_multiplier_applied"]) >= 0.5
        ),
        None,
    )
    if first_competing is None:
        constraint_onset_update = None
        constraint_onset_timestep = None
        constraint_success_after_onset = None
        b0_success_after_onset = None
    else:
        constraint_onset_update = int(first_competing["update"])
        constraint_onset_timestep = constraint_onset_update * 64
        constraint_success_after_onset = int(
            constraint_rows[-1].get("success_done", 0)
        ) - cumulative_at_or_before(
            constraint_rows,
            timestep=constraint_onset_timestep,
            field="success_done",
        )
        b0_success_after_onset = int(b0_rows[-1].get("success_done", 0)) - cumulative_at_or_before(
            b0_rows,
            timestep=constraint_onset_timestep,
            field="success_done",
        )
    second_half_cost_ratio_ge_one = sum(
        float(row["lagrangian_multiplier_applied"]) >= 1.0
        for row in second_half
    ) / len(second_half)
    first_half_mean_constraint_rate = sum(
        float(row["episode_equivalent_collision_rate"]) for row in first_half
    ) / len(first_half)
    second_half_mean_constraint_rate = sum(
        float(row["episode_equivalent_collision_rate"]) for row in second_half
    ) / len(second_half)

    diagnostics = b0_summary.get("training_diagnostics") or {}
    trained_timesteps = int(b0_summary.get("timesteps", 0))
    checks = {
        "candidate_dmax_improved_20pct": dmax_improvement >= 0.20,
        "candidate_collision_improved_50pct": collision_improvement >= 0.50,
        "candidate_success_gained_3pp": success_gain >= 0.03,
        "last_quarter_dmax_improved_10pct": last_quarter_improvement >= 0.10,
        "actor_and_encoders_updated": float(
            diagnostics.get("policy_parameter_delta_l2", 0.0)
        )
        > 0.0
        and float(diagnostics.get("neighbor_encoder_parameter_delta_l2", 0.0)) > 0.0
        and float(diagnostics.get("terrain_encoder_parameter_delta_l2", 0.0)) > 0.0,
        "action_non_degenerate": float(
            diagnostics.get("post_training_action_std", 0.0)
        )
        > 1.0e-4,
        "screen_ended_inside_curriculum_warmup": 0 < trained_timesteps
        < curriculum_warmup_timesteps,
        "constraint_success_plateau_after_scale_competition": constraint_success_after_onset
        == 0,
        "constraint_second_half_cost_ratio_ge_one_half": second_half_cost_ratio_ge_one
        >= 0.50,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "thresholds": {
            "candidate_dmax_relative_improvement": 0.20,
            "candidate_collision_relative_improvement": 0.50,
            "candidate_success_absolute_gain": 0.03,
            "last_quarter_dmax_relative_improvement": 0.10,
            "action_std": 1.0e-4,
            "constraint_cost_reward_std_ratio": 1.0,
            "constraint_second_half_fraction": 0.50,
        },
        "evidence": {
            "b0_candidate_1024": early,
            "b0_candidate_2048": late,
            "b0_candidate_dmax_relative_improvement": dmax_improvement,
            "b0_candidate_collision_relative_improvement": collision_improvement,
            "b0_candidate_success_absolute_gain": success_gain,
            "b0_dmax_quarter_means": dmax_quarters,
            "b0_last_quarter_relative_improvement": last_quarter_improvement,
            "b0_training_timesteps": trained_timesteps,
            "curriculum_warmup_timesteps": curriculum_warmup_timesteps,
            "constraint_rate_gt_one_fraction": unbounded_rate_fraction,
            "constraint_first_half_rate_mean": first_half_mean_constraint_rate,
            "constraint_second_half_rate_mean": second_half_mean_constraint_rate,
            "constraint_scale_competition_onset_update": constraint_onset_update,
            "constraint_scale_competition_onset_timestep": constraint_onset_timestep,
            "constraint_success_after_onset": constraint_success_after_onset,
            "b0_success_after_same_timestep": b0_success_after_onset,
            "constraint_second_half_cost_reward_std_ratio_ge_one_fraction": (
                second_half_cost_ratio_ge_one
            ),
            "constraint_final_lagrangian_multiplier": constraint_diagnostics.get(
                "lagrangian_multiplier"
            ),
        },
        "decision": (
            "allow_exp144_b0_12m_plan_only"
            if all(checks.values())
            else "stop_b0_depth_extension"
        ),
        "forty_m_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--b0-run", type=Path, default=DEFAULT_B0_RUN)
    parser.add_argument("--constraint-run", type=Path, default=DEFAULT_CONSTRAINT_RUN)
    parser.add_argument("--run-name", default=DEFAULT_RUN)
    args = parser.parse_args()
    b0_run = args.b0_run if args.b0_run.is_absolute() else ROOT / args.b0_run
    constraint_run = (
        args.constraint_run
        if args.constraint_run.is_absolute()
        else ROOT / args.constraint_run
    )
    raw = load_yaml(BASE_CONFIG)
    warmup = int(raw["initial_state"]["curriculum_warmup_timesteps"])
    audit = build_audit(
        b0_summary=_read_json(b0_run / "metrics/summary.json"),
        b0_eval=_read_json(b0_run / "metrics/eval_metrics.json"),
        b0_telemetry=_read_jsonl(b0_run / "metrics/train_metrics.jsonl"),
        constraint_summary=_read_json(constraint_run / "metrics/summary.json"),
        constraint_telemetry=_read_jsonl(
            constraint_run / "metrics/train_metrics.jsonl"
        ),
        curriculum_warmup_timesteps=warmup,
    )
    generated_at = datetime.now(timezone.utc).isoformat()
    result = {
        "schema_version": 1,
        "generated_at": generated_at,
        "experiment": EXPERIMENT_ID,
        "run": args.run_name,
        "frozen_inputs": {
            "b0_run": str(b0_run.relative_to(ROOT)),
            "constraint_run": str(constraint_run.relative_to(ROOT)),
        },
        **audit,
    }
    run_dir = ROOT / "outputs/runs" / EXPERIMENT_ID / args.run_name
    metrics_path = run_dir / "metrics/horizon_constraint_audit.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "generated_at": generated_at,
        "experiment": EXPERIMENT_ID,
        "run": args.run_name,
        "producer": "scripts/analyze_b0_horizon_and_constraint_competition.py",
        "command": " ".join(sys.argv),
        "status": result["decision"],
        "artifacts": {
            "audit": str(metrics_path.relative_to(ROOT)),
        },
    }
    (run_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    suite_dir = ROOT / "outputs/runs" / EXPERIMENT_ID / "_suite"
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
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
