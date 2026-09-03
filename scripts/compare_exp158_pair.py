#!/usr/bin/env python3
"""Compare one frozen-scenario GAE/DAE pair and apply exp158 gates."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np

from _common import ROOT


def _load_run(run_dir: Path) -> tuple[dict, dict, dict]:
    paired_path = run_dir / "metrics/paired_strict_acceptance.json"
    summary_path = run_dir / "metrics/summary.json"
    if not paired_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError(f"Run is missing paired evaluation or summary: {run_dir}")
    paired = json.loads(paired_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    counterfactual_path = run_dir / "metrics/dae_counterfactual_validation.json"
    counterfactual = (
        json.loads(counterfactual_path.read_text(encoding="utf-8"))
        if counterfactual_path.is_file()
        else {}
    )
    return paired, summary, counterfactual


def _episode_value(episode: dict, metric: str) -> float:
    if metric in {"success", "collision", "timeout"}:
        return float(bool(episode[metric]))
    if metric == "dmax_ratio":
        return float(episode[metric])
    raise KeyError(metric)


def stratified_paired_bootstrap(
    gae: dict,
    dae: dict,
    *,
    metric: str,
    transform: Callable[[np.ndarray, np.ndarray], float],
    samples: int = 10_000,
    seed: int = 158,
) -> dict[str, float]:
    gae_cells = {cell["cell"]: cell for cell in gae["cells"]}
    dae_cells = {cell["cell"]: cell for cell in dae["cells"]}
    if gae_cells.keys() != dae_cells.keys():
        raise ValueError("GAE and DAE paired reports have different cells")
    generator = np.random.default_rng(seed)
    estimates = np.empty(samples, dtype=np.float64)
    point_cells = []
    arrays = []
    for name in sorted(gae_cells):
        left = np.asarray(
            [
                _episode_value(item, metric)
                for item in gae_cells[name]["metrics"]["episode_metrics"]
            ],
            dtype=np.float64,
        )
        right = np.asarray(
            [
                _episode_value(item, metric)
                for item in dae_cells[name]["metrics"]["episode_metrics"]
            ],
            dtype=np.float64,
        )
        if left.shape != right.shape or left.size == 0:
            raise ValueError(f"Unpaired episode vectors for {name}")
        arrays.append((left, right))
        point_cells.append(transform(left, right))
    for sample in range(samples):
        cell_values = []
        for left, right in arrays:
            index = generator.integers(0, left.size, size=left.size)
            cell_values.append(transform(left[index], right[index]))
        estimates[sample] = float(np.mean(cell_values))
    return {
        "point": float(np.mean(point_cells)),
        "lower_95": float(np.quantile(estimates, 0.05, method="lower")),
        "upper_95": float(np.quantile(estimates, 0.95, method="higher")),
    }


def _aggregate(report: dict) -> dict[str, float]:
    cells = report["cells"]
    result = {
        key: float(
            np.mean([cell["acceptance"]["point_estimates"][key] for cell in cells])
        )
        for key in ("success", "collision", "timeout", "dmax_ratio")
    }
    result["path_risk"] = float(
        np.mean([cell["metrics"]["path_terrain_risk_mean"] for cell in cells])
    )
    return result


def compare_pair(
    *,
    gae_run: Path,
    dae_run: Path,
    phase: str,
    seed: int,
    bootstrap_samples: int,
) -> dict:
    gae_report, gae_summary, _ = _load_run(gae_run)
    dae_report, dae_summary, dae_counterfactual = _load_run(dae_run)
    if gae_report["manifest"] != dae_report["manifest"]:
        raise ValueError("GAE and DAE did not use the same scenario manifest")
    gae_diag = gae_summary["training_diagnostics"]
    dae_diag = dae_summary["training_diagnostics"]
    initialization_match = (
        gae_diag["initial_policy_sha256"] == dae_diag["initial_policy_sha256"]
        and gae_diag["initial_critic_sha256"] == dae_diag["initial_critic_sha256"]
    )
    success = stratified_paired_bootstrap(
        gae_report,
        dae_report,
        metric="success",
        transform=lambda gae, dae: float(dae.mean() - gae.mean()),
        samples=bootstrap_samples,
        seed=seed + 1,
    )
    timeout_reduction = stratified_paired_bootstrap(
        gae_report,
        dae_report,
        metric="timeout",
        transform=lambda gae, dae: float(
            (gae.mean() - dae.mean()) / max(gae.mean(), 1.0e-8)
        ),
        samples=bootstrap_samples,
        seed=seed + 2,
    )
    collision_worsening = stratified_paired_bootstrap(
        gae_report,
        dae_report,
        metric="collision",
        transform=lambda gae, dae: float(dae.mean() - gae.mean()),
        samples=bootstrap_samples,
        seed=seed + 3,
    )
    dmax_worsening = stratified_paired_bootstrap(
        gae_report,
        dae_report,
        metric="dmax_ratio",
        transform=lambda gae, dae: float(dae.mean() - gae.mean()),
        samples=bootstrap_samples,
        seed=seed + 4,
    )
    gae_metrics = _aggregate(gae_report)
    dae_metrics = _aggregate(dae_report)
    relative_primary = (
        success["point"] >= 0.10 and success["lower_95"] > 0.0
    ) or (
        timeout_reduction["point"] >= 0.20
        and timeout_reduction["lower_95"] >= 0.20
    )
    checks = {
        "initial_actor_critic_hashes_match": initialization_match,
        "dae_success_ge_0_50": dae_metrics["success"] >= 0.50,
        "dae_collision_le_0_20": dae_metrics["collision"] <= 0.20,
        "dae_dmax_ratio_le_0_45": dae_metrics["dmax_ratio"] <= 0.45,
        "relative_primary_improvement": relative_primary,
        "collision_worsening_upper_le_0_02": collision_worsening["upper_95"] <= 0.02,
        "dmax_worsening_upper_le_0_02": dmax_worsening["upper_95"] <= 0.02,
        "path_risk_not_worse": dae_metrics["path_risk"] <= gae_metrics["path_risk"],
        "reward_model_counterfactual_gate": bool(dae_counterfactual.get("passed", False)),
        "dae_advantage_agent_std_nonzero": float(
            dae_diag.get("dae_advantage_agent_std", 0.0)
        )
        > 1.0e-8,
        "finite_training": bool(dae_diag.get("policy_parameters_finite", False))
        and bool(dae_diag.get("reward_model_parameters_finite", False)),
    }
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment": "exp158_dae_validation",
        "phase": phase,
        "seed": seed,
        "gae_run": str(gae_run.relative_to(ROOT)),
        "dae_run": str(dae_run.relative_to(ROOT)),
        "manifest": gae_report["manifest"],
        "episodes": int(gae_report["total_episodes"]),
        "aggregate": {"gae": gae_metrics, "dae": dae_metrics},
        "paired_effects": {
            "success_dae_minus_gae": success,
            "timeout_relative_reduction": timeout_reduction,
            "collision_dae_minus_gae": collision_worsening,
            "dmax_dae_minus_gae": dmax_worsening,
        },
        "checks": checks,
        "passed": all(checks.values()),
        "strict_all_cells_passed": bool(dae_report["passed"]),
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gae-run", required=True)
    parser.add_argument("--dae-run", required=True)
    parser.add_argument("--phase", choices=("h1", "strict"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--output")
    args = parser.parse_args()

    def resolve(value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else ROOT / path

    gae_run = resolve(args.gae_run)
    dae_run = resolve(args.dae_run)
    report = compare_pair(
        gae_run=gae_run,
        dae_run=dae_run,
        phase=args.phase,
        seed=args.seed,
        bootstrap_samples=args.bootstrap_samples,
    )
    output = (
        resolve(args.output)
        if args.output
        else ROOT
        / "outputs/runs/exp158_dae_validation/_suite/metrics"
        / f"{args.phase}_seed{args.seed}_pair.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "passed": report["passed"]}, indent=2))


if __name__ == "__main__":
    main()
