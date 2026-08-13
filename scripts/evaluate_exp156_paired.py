#!/usr/bin/env python3
"""Evaluate one exp156 checkpoint on the frozen 1152-scenario manifest."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

from _common import ROOT, cfg_from_experiment, load_yaml
from evaluate_proxy_policy import evaluate_checkpoint
from exp156_statistics import strict_cell_acceptance
from generate_exp156_scenario_manifest import scenario_snapshot
from lunar_rover_tasks.tasks.multi_rover_gathering.gathering_env import (
    MultiRoverGatheringCore,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--manifest",
        default=(
            "outputs/runs/exp156_differential_multiscale_ablation/_suite/"
            "scenario_manifest.json"
        ),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=480)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    args = parser.parse_args()

    run_dir = ROOT / args.run_dir
    config_path = run_dir / "config/experiment.yaml"
    checkpoint_path = run_dir / "checkpoints/best.pt"
    if not config_path.is_file() or not checkpoint_path.is_file():
        raise SystemExit("Run must contain config/experiment.yaml and checkpoints/best.pt.")
    manifest_path = ROOT / args.manifest
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(manifest["episodes_per_cell"]) != 192:
        raise SystemExit("Formal exp156 evaluation requires 192 scenarios per cell.")

    base = load_yaml(config_path)
    config_dir = run_dir / "metrics/paired_configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    cells = []
    for cell_index, cell in enumerate(manifest["cells"]):
        raw = copy.deepcopy(base)
        raw.setdefault("experiment", {})["seed"] = int(cell["seed"])
        raw["experiment"]["num_envs"] = 192
        raw.setdefault("initial_state", {}).update(cell["initial_state_overrides"])
        raw["initial_state"]["curriculum_enabled"] = False
        raw.setdefault("terrain", {}).update(cell["terrain_overrides"])
        raw.setdefault("safety", {})["collision_termination_enabled"] = True
        cell_config = config_dir / f"{cell['cell']}.json"
        cell_config.write_text(json.dumps(raw, indent=2), encoding="utf-8")
        verification_cfg = cfg_from_experiment(cell_config)
        verification_cfg.simulation.device = "cpu"
        regenerated = {
            key: cell[key]
            for key in (
                "cell",
                "distance",
                "topology",
                "seed",
                "initial_state_overrides",
                "terrain_overrides",
            )
        }
        regenerated["scenarios"] = scenario_snapshot(
            MultiRoverGatheringCore(verification_cfg)
        )
        canonical = json.dumps(regenerated, sort_keys=True, separators=(",", ":"))
        regenerated_hash = hashlib.sha256(canonical.encode()).hexdigest()
        if regenerated_hash != cell["sha256"]:
            raise SystemExit(
                f"Scenario regeneration mismatch for {cell['cell']}; refusing an "
                "unpaired evaluation."
            )
        output = run_dir / "metrics" / f"paired_{cell['cell']}.json"
        metrics = evaluate_checkpoint(
            config=cell_config,
            checkpoint=checkpoint_path,
            device=args.device,
            num_envs=192,
            steps=args.steps,
            seed=int(cell["seed"]),
            output=output,
            run_dir=run_dir,
        )
        episodes = metrics["episode_metrics"]
        acceptance = strict_cell_acceptance(
            success_count=sum(bool(item["success"]) for item in episodes),
            collision_count=sum(bool(item["collision"]) for item in episodes),
            timeout_count=sum(bool(item["timeout"]) for item in episodes),
            dmax_ratios=[item["dmax_ratio"] for item in episodes],
            bootstrap_samples=args.bootstrap_samples,
            seed=156 + cell_index,
        )
        cells.append(
            {
                "cell": cell["cell"],
                "distance": cell["distance"],
                "topology": cell["topology"],
                "scenario_sha256": cell["sha256"],
                "metrics": metrics,
                "acceptance": acceptance,
                "passed": acceptance["passed"],
            }
        )

    report = {
        "manifest": str(manifest_path.relative_to(ROOT)),
        "episodes_per_cell": 192,
        "total_episodes": 1152,
        "cells": cells,
        "passed": all(cell["passed"] for cell in cells),
    }
    report_path = run_dir / "metrics/paired_strict_acceptance.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(report_path), "passed": report["passed"]}, indent=2))


if __name__ == "__main__":
    main()
