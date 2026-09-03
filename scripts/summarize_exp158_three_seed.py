#!/usr/bin/env python3
"""Hierarchical seed/scenario bootstrap for completed exp158 pairs."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from _common import ROOT


def _load(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _seed_bootstrap(
    gae_report: dict,
    dae_report: dict,
    *,
    samples: int,
    generator: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    success_cells = []
    timeout_cells = []
    gae_cells = {cell["cell"]: cell for cell in gae_report["cells"]}
    dae_cells = {cell["cell"]: cell for cell in dae_report["cells"]}
    for cell in sorted(gae_cells):
        gae_episodes = gae_cells[cell]["metrics"]["episode_metrics"]
        dae_episodes = dae_cells[cell]["metrics"]["episode_metrics"]
        gae_success = np.asarray([bool(item["success"]) for item in gae_episodes])
        dae_success = np.asarray([bool(item["success"]) for item in dae_episodes])
        gae_timeout = np.asarray([bool(item["timeout"]) for item in gae_episodes])
        dae_timeout = np.asarray([bool(item["timeout"]) for item in dae_episodes])
        index = generator.integers(0, gae_success.size, size=(samples, gae_success.size))
        success_cells.append(
            dae_success[index].mean(axis=1) - gae_success[index].mean(axis=1)
        )
        gae_rate = gae_timeout[index].mean(axis=1)
        dae_rate = dae_timeout[index].mean(axis=1)
        timeout_cells.append((gae_rate - dae_rate) / np.maximum(gae_rate, 1.0e-8))
    return np.stack(success_cells).mean(axis=0), np.stack(timeout_cells).mean(axis=0)


def summarize_phase(
    phase: str,
    *,
    samples: int = 10_000,
    seed: int = 158,
) -> dict:
    suite = ROOT / "outputs/runs/exp158_dae_validation/_suite"
    pair_reports = [
        _load(suite / "metrics" / f"{phase}_seed{run_seed}_pair.json")
        for run_seed in (23, 31, 47)
    ]
    generator = np.random.default_rng(seed)
    seed_success = []
    seed_timeout = []
    for pair in pair_reports:
        gae = _load(ROOT / pair["gae_run"] / "metrics/paired_strict_acceptance.json")
        dae = _load(ROOT / pair["dae_run"] / "metrics/paired_strict_acceptance.json")
        success, timeout = _seed_bootstrap(
            gae, dae, samples=samples, generator=generator
        )
        seed_success.append(success)
        seed_timeout.append(timeout)
    success_stack = np.stack(seed_success)
    timeout_stack = np.stack(seed_timeout)
    selected_seeds = generator.integers(0, 3, size=(samples, 3))
    selected_draws = generator.integers(0, samples, size=(samples, 3))
    success_hierarchical = success_stack[selected_seeds, selected_draws].mean(axis=1)
    timeout_hierarchical = timeout_stack[selected_seeds, selected_draws].mean(axis=1)

    safety_keys = (
        "collision_worsening_upper_le_0_02",
        "dmax_worsening_upper_le_0_02",
        "path_risk_not_worse",
        "finite_training",
        "initial_actor_critic_hashes_match",
    )
    relative_wins = sum(
        bool(report["checks"]["relative_primary_improvement"])
        for report in pair_reports
    )
    aggregate_success = float(
        np.mean([report["aggregate"]["dae"]["success"] for report in pair_reports])
    )
    success_effect = {
        "point": float(
            np.mean(
                [
                    report["paired_effects"]["success_dae_minus_gae"]["point"]
                    for report in pair_reports
                ]
            )
        ),
        "lower_95": float(np.quantile(success_hierarchical, 0.05, method="lower")),
        "upper_95": float(np.quantile(success_hierarchical, 0.95, method="higher")),
    }
    timeout_effect = {
        "point": float(
            np.mean(
                [
                    report["paired_effects"]["timeout_relative_reduction"]["point"]
                    for report in pair_reports
                ]
            )
        ),
        "lower_95": float(np.quantile(timeout_hierarchical, 0.05, method="lower")),
        "upper_95": float(np.quantile(timeout_hierarchical, 0.95, method="higher")),
    }
    primary_confident = (
        success_effect["point"] >= 0.10 and success_effect["lower_95"] > 0.0
    ) or (
        timeout_effect["point"] >= 0.20 and timeout_effect["lower_95"] >= 0.20
    )
    checks = {
        "relative_wins_ge_2_of_3": relative_wins >= 2,
        "aggregate_success_ge_0_50": aggregate_success >= 0.50,
        "hierarchical_primary_effect_confident": primary_confident,
        "all_seed_safety_nonworsening": all(
            all(bool(report["checks"][key]) for key in safety_keys)
            for report in pair_reports
        ),
    }
    if phase == "strict":
        checks["all_seed_all_cells_strict"] = all(
            bool(report.get("strict_all_cells_passed", False))
            for report in pair_reports
        )
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "phase": phase,
        "seeds": [23, 31, 47],
        "bootstrap_samples": samples,
        "relative_wins": relative_wins,
        "aggregate_dae_success": aggregate_success,
        "hierarchical_effects": {
            "success_dae_minus_gae": success_effect,
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
        / "outputs/runs/exp158_dae_validation/_suite/metrics"
        / f"{args.phase}_three_seed_summary.json"
    )
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "passed": report["passed"]}, indent=2))


if __name__ == "__main__":
    main()
