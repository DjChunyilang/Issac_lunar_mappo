#!/usr/bin/env python3
"""Compare one frozen-scenario GAE/ALO-PRD pair and apply exp159 gates."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from _common import ROOT
from compare_exp158_pair import _aggregate, stratified_paired_bootstrap


def _load(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def compare_pair(
    *, gae_run: Path, prd_run: Path, phase: str, seed: int, samples: int
) -> dict:
    gae_report = _load(gae_run / "metrics/paired_strict_acceptance.json")
    prd_report = _load(prd_run / "metrics/paired_strict_acceptance.json")
    gae_summary = _load(gae_run / "metrics/summary.json")
    prd_summary = _load(prd_run / "metrics/summary.json")
    prd_validation = _load(
        prd_run / "prd_checkpoint_validation/metrics/offline_gate.json"
    )
    if gae_report["manifest"] != prd_report["manifest"]:
        raise ValueError("GAE and PRD reports used different manifests")
    gae_diag = gae_summary["training_diagnostics"]
    prd_diag = prd_summary["training_diagnostics"]
    initialization_match = (
        gae_diag["initial_policy_sha256"] == prd_diag["initial_policy_sha256"]
        and gae_diag["initial_critic_sha256"] == prd_diag["initial_critic_sha256"]
    )
    success = stratified_paired_bootstrap(
        gae_report,
        prd_report,
        metric="success",
        transform=lambda gae, prd: float(prd.mean() - gae.mean()),
        samples=samples,
        seed=seed + 1,
    )
    timeout = stratified_paired_bootstrap(
        gae_report,
        prd_report,
        metric="timeout",
        transform=lambda gae, prd: float(
            (gae.mean() - prd.mean()) / max(gae.mean(), 1.0e-8)
        ),
        samples=samples,
        seed=seed + 2,
    )
    collision = stratified_paired_bootstrap(
        gae_report,
        prd_report,
        metric="collision",
        transform=lambda gae, prd: float(prd.mean() - gae.mean()),
        samples=samples,
        seed=seed + 3,
    )
    dmax = stratified_paired_bootstrap(
        gae_report,
        prd_report,
        metric="dmax_ratio",
        transform=lambda gae, prd: float(prd.mean() - gae.mean()),
        samples=samples,
        seed=seed + 4,
    )
    gae_metrics = _aggregate(gae_report)
    prd_metrics = _aggregate(prd_report)
    relative_primary = (
        success["point"] >= 0.10 and success["lower_95"] > 0.0
    ) or (timeout["point"] >= 0.20 and timeout["lower_95"] >= 0.20)
    checks = {
        "initial_actor_critic_hashes_match": initialization_match,
        "prd_success_ge_0_50": prd_metrics["success"] >= 0.50,
        "prd_collision_le_0_20": prd_metrics["collision"] <= 0.20,
        "prd_dmax_ratio_le_0_45": prd_metrics["dmax_ratio"] <= 0.45,
        "relative_primary_improvement": relative_primary,
        "collision_worsening_upper_le_0_02": collision["upper_95"] <= 0.02,
        "dmax_worsening_upper_le_0_02": dmax["upper_95"] <= 0.02,
        "path_risk_not_worse": prd_metrics["path_risk"] <= gae_metrics["path_risk"],
        "prd_checkpoint_gate_passed": bool(prd_validation["passed"]),
        "prd_advantage_agent_std_nonzero": float(
            prd_diag.get("prd_advantage_agent_std", 0.0)
        )
        > 1.0e-4,
        "finite_training": bool(prd_diag.get("policy_parameters_finite", False)),
    }
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment": "exp159_analytical_prd",
        "phase": phase,
        "seed": seed,
        "gae_run": str(gae_run.relative_to(ROOT)),
        "prd_run": str(prd_run.relative_to(ROOT)),
        "manifest": gae_report["manifest"],
        "episodes": int(gae_report["total_episodes"]),
        "aggregate": {"gae": gae_metrics, "prd": prd_metrics},
        "paired_effects": {
            "success_prd_minus_gae": success,
            "timeout_relative_reduction": timeout,
            "collision_prd_minus_gae": collision,
            "dmax_prd_minus_gae": dmax,
        },
        "checks": checks,
        "passed": all(checks.values()),
        "strict_all_cells_passed": bool(prd_report["passed"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gae-run", required=True)
    parser.add_argument("--prd-run", required=True)
    parser.add_argument("--phase", choices=("h1", "strict"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--output")
    args = parser.parse_args()

    def resolve(value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else ROOT / path

    report = compare_pair(
        gae_run=resolve(args.gae_run),
        prd_run=resolve(args.prd_run),
        phase=args.phase,
        seed=args.seed,
        samples=args.bootstrap_samples,
    )
    output = (
        resolve(args.output)
        if args.output
        else ROOT
        / "outputs/runs/exp159_analytical_prd/_suite/metrics"
        / f"{args.phase}_seed{args.seed}_pair.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "passed": report["passed"]}, indent=2))


if __name__ == "__main__":
    main()
