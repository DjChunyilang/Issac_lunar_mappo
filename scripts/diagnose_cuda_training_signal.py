#!/usr/bin/env python
"""Diagnose CUDA SKRL MAPPO training signal from telemetry JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any


REWARD_COMPONENTS = (
    "gather",
    "oracle",
    "energy",
    "safety",
    "terrain",
    "flatness",
    "motion",
    "consistency",
    "success_hold",
    "terminal",
)

ACTION_SCALE_KEYS = (
    "action_saturation_fraction",
    "action_near_zero_fraction",
    "action_forward_mean",
    "action_forward_std",
    "action_forward_low_saturation_fraction",
    "action_forward_high_saturation_fraction",
    "action_turn_mean",
    "action_turn_std",
    "action_turn_abs_saturation_fraction",
    "physical_rho_mean",
    "physical_rho_std",
    "physical_rho_low_fraction",
    "physical_rho_high_fraction",
    "physical_beta_mean",
    "physical_beta_std",
    "physical_beta_abs_high_fraction",
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _latest_run(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    run_ids = [row.get("run_id") for row in rows if row.get("run_id")]
    if not run_ids:
        return rows
    latest = run_ids[-1]
    return [row for row in rows if row.get("run_id") == latest]


def _series(rows: list[dict[str, Any]], key: str) -> list[float]:
    return [float(row[key]) for row in rows if row.get(key) is not None]


def _trend(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"first": None, "last": None, "min": None, "max": None, "slope_approx": None}
    slope = (values[-1] - values[0]) / max(len(values) - 1, 1)
    return {
        "first": values[0],
        "last": values[-1],
        "min": min(values),
        "max": max(values),
        "slope_approx": slope,
    }


def _first_last(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"first": None, "last": None}
    return {"first": values[0], "last": values[-1]}


def _latest_numeric(rows: list[dict[str, Any]], key: str) -> float | None:
    for row in reversed(rows):
        value = row.get(key)
        if value is not None and isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return None


def _done_summary(rows: list[dict[str, Any]]) -> dict[str, int]:
    done_keys = ("success_done", "timeout_done", "collision_done", "safety_done", "other_done")
    if not rows:
        return {key: 0 for key in done_keys}
    latest = rows[-1]
    return {key: int(latest.get(key, 0) or 0) for key in done_keys}


def _reward_component_trends(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        component: {
            "raw": _trend(_series(rows, f"reward_raw_{component}")),
            "contribution": _trend(_series(rows, f"reward_contribution_{component}")),
            "abs_share": _trend(_series(rows, f"reward_abs_share_{component}")),
        }
        for component in REWARD_COMPONENTS
    }


def _reward_component_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    latest = rows[-1]
    return {
        "weighted_total": latest.get("reward_weighted_total"),
        "contribution_sum": latest.get("reward_contribution_sum"),
        "positive_contribution_sum": latest.get("reward_positive_contribution_sum"),
        "negative_contribution_sum": latest.get("reward_negative_contribution_sum"),
        "abs_contribution_sum": latest.get("reward_abs_contribution_sum"),
        "dominant_positive_component": latest.get("reward_dominant_positive_component"),
        "dominant_negative_component": latest.get("reward_dominant_negative_component"),
        "abs_share": {
            component: latest.get(f"reward_abs_share_{component}")
            for component in REWARD_COMPONENTS
        },
        "contribution": {
            component: latest.get(f"reward_contribution_{component}")
            for component in REWARD_COMPONENTS
        },
    }


def _action_scale_trends(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {key: _trend(_series(rows, key)) for key in ACTION_SCALE_KEYS}


def _action_scale_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    saturation = _latest_numeric(rows, "action_saturation_fraction")
    near_zero = _latest_numeric(rows, "action_near_zero_fraction")
    forward_low = _latest_numeric(rows, "action_forward_low_saturation_fraction")
    forward_high = _latest_numeric(rows, "action_forward_high_saturation_fraction")
    turn_sat = _latest_numeric(rows, "action_turn_abs_saturation_fraction")
    rho_high = _latest_numeric(rows, "physical_rho_high_fraction")
    rho_low = _latest_numeric(rows, "physical_rho_low_fraction")
    beta_high = _latest_numeric(rows, "physical_beta_abs_high_fraction")
    flags = []
    if saturation is not None and saturation > 0.20:
        flags.append("normalized_action_saturation")
    if near_zero is not None and near_zero > 0.80:
        flags.append("near_zero_action_collapse")
    if forward_high is not None and forward_high > 0.20:
        flags.append("forward_high_saturation")
    if forward_low is not None and forward_low > 0.20:
        flags.append("forward_low_saturation")
    if turn_sat is not None and turn_sat > 0.20:
        flags.append("turn_saturation")
    if rho_high is not None and rho_high > 0.20:
        flags.append("physical_rho_high_saturation")
    if rho_low is not None and rho_low > 0.20:
        flags.append("physical_rho_low_saturation")
    if beta_high is not None and beta_high > 0.20:
        flags.append("physical_beta_saturation")
    if not flags:
        flags.append("no_obvious_action_scale_issue")
    return {
        "action_saturation_fraction": saturation,
        "action_near_zero_fraction": near_zero,
        "forward_low_saturation_fraction": forward_low,
        "forward_high_saturation_fraction": forward_high,
        "turn_abs_saturation_fraction": turn_sat,
        "physical_rho_low_fraction": rho_low,
        "physical_rho_high_fraction": rho_high,
        "physical_beta_abs_high_fraction": beta_high,
        "flags": flags,
    }


def _next_experiment_focus(rows: list[dict[str, Any]]) -> list[str]:
    focus: list[str] = []
    action_summary = _action_scale_summary(rows)
    action_flags = set(action_summary["flags"])
    if action_flags - {"no_obvious_action_scale_issue"}:
        focus.append("action_scale_ablation")

    reward_summary = _reward_component_summary(rows)
    abs_share = reward_summary.get("abs_share", {}) if reward_summary else {}
    task_signal_share = sum(
        float(abs_share.get(component) or 0.0)
        for component in ("gather", "oracle", "flatness")
    )
    penalty_share = sum(
        float(abs_share.get(component) or 0.0)
        for component in ("energy", "safety", "terrain", "motion", "consistency", "terminal")
    )
    if task_signal_share < 0.35 and penalty_share > task_signal_share:
        focus.append("reward_signal_balance")

    success_values = _series(rows, "success_rate")
    if success_values and max(success_values) == 0.0:
        focus.append("success_gate_reachability_diagnostic")
    return focus or ["repeat_short_seed_validation"]


def _judge(pairwise: dict, oracle: dict, success_values: list[float], reward_values: list[float]) -> str:
    pair_delta = None if pairwise["first"] is None else pairwise["last"] - pairwise["first"]
    oracle_delta = None if oracle["first"] is None else oracle["last"] - oracle["first"]
    success_gain = (success_values[-1] - success_values[0]) if len(success_values) >= 2 else 0.0
    reward_gain = (reward_values[-1] - reward_values[0]) if len(reward_values) >= 2 else 0.0
    distance_improved = any(delta is not None and delta < -0.05 for delta in (pair_delta, oracle_delta))
    weak_distance = any(delta is not None and delta < 0.0 for delta in (pair_delta, oracle_delta))
    if distance_improved and (success_gain > 0.0 or reward_gain > 0.0):
        return "clear_improvement"
    if weak_distance or success_gain > 0.0 or reward_gain > 0.0:
        return "weak_improvement"
    return "no_clear_improvement"


def diagnose(metrics_path: Path) -> dict[str, Any]:
    all_rows = _read_jsonl(metrics_path)
    rows = _latest_run(all_rows)
    pairwise = _trend(_series(rows, "mean_pairwise_distance"))
    oracle = _trend(_series(rows, "mean_oracle_distance"))
    success_values = _series(rows, "success_rate")
    reward_values = _series(rows, "mean_reward")
    action_keys = ("action_mean", "action_std", "action_min", "action_max")
    action_trend = {key: _first_last(_series(rows, key)) for key in action_keys}
    return {
        "metrics_path": str(metrics_path),
        "run_id": rows[-1].get("run_id") if rows else None,
        "row_count": len(rows),
        "mean_pairwise_distance": pairwise,
        "mean_oracle_distance": oracle,
        "success_rate": {
            "first": success_values[0] if success_values else None,
            "last": success_values[-1] if success_values else None,
            "max": max(success_values) if success_values else None,
        },
        "reward_trend": {
            **_first_last(reward_values),
            "min": min(reward_values) if reward_values else None,
            "max": max(reward_values) if reward_values else None,
            "mean": mean(reward_values) if reward_values else None,
        },
        "action_trend": action_trend,
        "action_scale_trends": _action_scale_trends(rows),
        "action_scale_summary": _action_scale_summary(rows),
        "reward_component_trends": _reward_component_trends(rows),
        "reward_component_summary": _reward_component_summary(rows),
        "done_reason_summary": _done_summary(rows),
        "random_baseline": rows[-1].get("random_baseline") if rows else None,
        "post_training_eval": rows[-1].get("post_training_eval") if rows else None,
        "next_experiment_focus": _next_experiment_focus(rows),
        "judgement": _judge(pairwise, oracle, success_values, reward_values),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", required=True)
    args = parser.parse_args()
    result = diagnose(Path(args.metrics))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
