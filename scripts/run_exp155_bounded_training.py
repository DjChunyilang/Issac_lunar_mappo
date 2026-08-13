#!/usr/bin/env python3
"""Validate the full-RL ablation winner on seeds 31 and 47.

Seed 23 is already the equal-budget architecture comparison run. This script
starts seed31/47 only when that winner passed all six strata and terrain-use
checks, and it blocks each later seed on the same acceptance contract.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from _common import ROOT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--selection",
        default=(
            "outputs/runs/exp155_full_rl_ablation/_suite/metrics/"
            "full_rl_architecture_ablation.json"
        ),
    )
    parser.add_argument(
        "--config",
        default="configs/experiment/exp155_full_rl_ablation.yaml",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    selection_path = ROOT / args.selection
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    architecture = selection.get("winner")
    if not selection.get("completed") or not isinstance(architecture, str):
        raise SystemExit("A completed three-way full-RL architecture ablation is required.")
    if not selection.get("winner_formal_passed"):
        raise SystemExit(
            "The seed23 winner must pass all six strata and terrain-use checks before "
            "seed31 or seed47 is started."
        )

    records = []
    for seed in (31, 47):
        run_name = f"winner_seed{seed}_full_2400iter"
        command = [
            str(ROOT / ".venv_isaaclab/bin/python3.12"),
            str(ROOT / "scripts/train_skrl_mappo.py"),
            "--config", str(ROOT / args.config),
            "--device", args.device,
            "--num-envs", "256",
            "--rollout-steps", "64",
            "--timesteps", "153600",
            "--seed", str(seed),
            "--actor-architecture", architecture,
            "--output-layout", "run",
            "--run-name", run_name,
            "--selection-gate", "strict",
        ]
        record = {"seed": seed, "run_name": run_name, "command": command}
        records.append(record)
        if not args.execute:
            continue
        completed = subprocess.run(command, cwd=ROOT, check=False)
        record["returncode"] = completed.returncode
        summary_path = ROOT / "outputs/runs/exp155_full_rl_ablation" / run_name / "metrics/summary.json"
        if not summary_path.is_file():
            raise SystemExit(f"Missing training summary for seed {seed}: {summary_path}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        record["bounded_curriculum_status"] = summary["bounded_curriculum"]["status"]
        record["strict_passed"] = bool(summary["strict_acceptance"]["passed"])
        if completed.returncode != 0 or record["bounded_curriculum_status"] != "completed":
            raise SystemExit(f"Seed {seed} failed the fixed curriculum; later seeds are blocked.")
        if not record["strict_passed"]:
            raise SystemExit(f"Seed {seed} failed strict proxy acceptance; later seeds are blocked.")
        stratified_command = [
            str(ROOT / ".venv_isaaclab/bin/python3.12"),
            str(ROOT / "scripts/evaluate_exp155_stratified.py"),
            "--run-dir", str(summary_path.parents[1].relative_to(ROOT)),
            "--device", args.device,
            "--episodes-per-cell", "64",
            "--steps", "480",
            "--seed", str(seed + 20000),
        ]
        stratified = subprocess.run(stratified_command, cwd=ROOT, check=False)
        record["stratified_returncode"] = stratified.returncode
        if stratified.returncode != 0:
            raise SystemExit(f"Seed {seed} failed stratified strict acceptance; later seeds are blocked.")
        terrain_command = [
            str(ROOT / ".venv_isaaclab/bin/python3.12"),
            str(ROOT / "scripts/evaluate_terrain_contrast.py"),
            "--config", str(summary_path.parents[1] / "config/experiment.yaml"),
            "--checkpoint", str(summary_path.parents[1] / "checkpoints/best.pt"),
            "--device", args.device,
            "--num-envs", "512",
            "--steps", "120",
            "--seed", str(seed + 23000),
            "--run-dir", str(summary_path.parents[1].relative_to(ROOT)),
        ]
        terrain = subprocess.run(terrain_command, cwd=ROOT, check=False)
        record["terrain_returncode"] = terrain.returncode
        terrain_path = summary_path.parent / "terrain_contrast.json"
        terrain_result = json.loads(terrain_path.read_text(encoding="utf-8"))
        record["terrain_checks"] = terrain_result["checks"]
        if terrain.returncode != 0 or not all(terrain_result["checks"].values()):
            raise SystemExit(
                f"Seed {seed} failed terrain-use acceptance; later seeds are blocked."
            )

    print(json.dumps({"architecture": architecture, "runs": records}, indent=2))


if __name__ == "__main__":
    main()
