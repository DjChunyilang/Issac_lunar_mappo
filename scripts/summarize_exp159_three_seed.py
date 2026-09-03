#!/usr/bin/env python3
"""Hierarchical seed/scenario bootstrap for completed exp159 pairs."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from _common import ROOT
from summarize_exp158_three_seed import _seed_bootstrap


def _load(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_phase(phase: str, *, samples: int = 10_000, seed: int = 159) -> dict:
    suite = ROOT / "outputs/runs/exp159_analytical_prd/_suite"
    reports = [
        _load(suite / "metrics" / f"{phase}_seed{run_seed}_pair.json")
        for run_seed in (23, 31, 47)
    ]
    generator = np.random.default_rng(seed)
    success_rows = []
    timeout_rows = []
    for report in reports:
        gae = _load(ROOT / report["gae_run"] / "metrics/paired_strict_acceptance.json")
        prd = _load(ROOT / report["prd_run"] / "metrics/paired_strict_acceptance.json")
        success, timeout = _seed_bootstrap(
            gae, prd, samples=samples, generator=generator
        )
        success_rows.append(success)
        timeout_rows.append(timeout)
    success_stack = np.stack(success_rows)
    timeout_stack = np.stack(timeout_rows)
    selected_seed = generator.integers(0, 3, size=(samples, 3))
    selected_draw = generator.integers(0, samples, size=(samples, 3))
    success_distribution = success_stack[selected_seed, selected_draw].mean(axis=1)
    timeout_distribution = timeout_stack[selected_seed, selected_draw].mean(axis=1)
    success_effect = {
        "point": float(
            np.mean(
                [
                    report["paired_effects"]["success_prd_minus_gae"]["point"]
                    for report in reports
                ]
            )
        ),
        "lower_95": float(np.quantile(success_distribution, 0.05, method="lower")),
        "upper_95": float(np.quantile(success_distribution, 0.95, method="higher")),
    }
    timeout_effect = {
        "point": float(
            np.mean(
                [
                    report["paired_effects"]["timeout_relative_reduction"]["point"]
                    for report in reports
                ]
            )
        ),
        "lower_95": float(np.quantile(timeout_distribution, 0.05, method="lower")),
        "upper_95": float(np.quantile(timeout_distribution, 0.95, method="higher")),
    }
    primary_confident = (
        success_effect["point"] >= 0.10 and success_effect["lower_95"] > 0.0
    ) or (
        timeout_effect["point"] >= 0.20 and timeout_effect["lower_95"] >= 0.20
    )
    safety_keys = (
        "collision_worsening_upper_le_0_02",
        "dmax_worsening_upper_le_0_02",
        "path_risk_not_worse",
        "finite_training",
        "initial_actor_critic_hashes_match",
    )
    relative_wins = sum(
        bool(report["checks"]["relative_primary_improvement"])
        for report in reports
    )
    aggregate_success = float(
        np.mean([report["aggregate"]["prd"]["success"] for report in reports])
    )
    checks = {
        "relative_wins_ge_2_of_3": relative_wins >= 2,
        "aggregate_success_ge_0_50": aggregate_success >= 0.50,
        "hierarchical_primary_effect_confident": primary_confident,
        "all_seed_safety_nonworsening": all(
            all(bool(report["checks"][key]) for key in safety_keys)
            for report in reports
        ),
    }
    if phase == "strict":
        checks["all_seed_all_cells_strict"] = all(
            bool(report.get("strict_all_cells_passed", False))
            for report in reports
        )
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "phase": phase,
        "seeds": [23, 31, 47],
        "bootstrap_samples": samples,
        "relative_wins": relative_wins,
        "aggregate_prd_success": aggregate_success,
        "hierarchical_effects": {
            "success_prd_minus_gae": success_effect,
            "timeout_relative_reduction": timeout_effect,
        },
        "checks": checks,
        "passed": all(checks.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("h1", "strict"), required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--output")
    args = parser.parse_args()
    report = summarize_phase(args.phase, samples=args.bootstrap_samples)
    output = (
        Path(args.output)
        if args.output
        else ROOT
        / "outputs/runs/exp159_analytical_prd/_suite/metrics"
        / f"{args.phase}_three_seed_summary.json"
    )
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "passed": report["passed"]}, indent=2))


if __name__ == "__main__":
    main()
