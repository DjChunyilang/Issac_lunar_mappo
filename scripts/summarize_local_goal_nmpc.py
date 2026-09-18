#!/usr/bin/env python3
"""Read-only analysis of policies; writes derived reports within their run directory."""

import argparse
import gzip
import json
from pathlib import Path

import numpy as np


def summarize(run):
    metrics = run / "metrics"
    comparison = {"formal_convergence_claim": False, "cells": {}}
    for phase, filename in (("before", "baseline_eval.json"), ("after", "final_eval_proxy.json")):
        path = metrics / filename
        if not path.exists():
            continue
        report = json.loads(path.read_text())
        for cell, result in report["cells"].items():
            rows = result["episodes"]
            frozen = json.loads((run / "config" / f"{cell}_frozen_scenarios.json").read_text())[
                "scenarios"
            ]
            initial = []
            for row in rows:
                points = np.asarray(frozen[row["scenario_id"]]["positions"])[..., :2]
                initial.append(np.linalg.norm(points[:, None] - points[None, :], axis=-1).max())
            summary = {
                "episodes": len(rows),
                "success_rate": result["success_rate"],
                "collision_rate": result["collision_rate"],
                "timeout_rate": result["timeout_rate"],
                "mean_final_dmax": float(np.mean([r["dmax"] for r in rows])),
                "mean_dmax_reduction_ratio": float(
                    np.mean([r["dmax"] / max(d, 1e-8) for r, d in zip(rows, initial)])
                ),
                "mean_path_length_per_rover_m": float(
                    np.mean([r["motion_progress_m"] for r in rows])
                ),
                "terminal_gate_pass_rates": {
                    key: float(np.mean([r[key] for r in rows]))
                    for key in (
                        "dmax_ok",
                        "dispersion_ok",
                        "speed_ok",
                        "min_pairwise_ok",
                        "flatness_ok",
                        "instant_success",
                    )
                },
                "planner": result["planner"],
                "seconds": result["seconds"],
                "scene_sha256": result["scene_sha256"],
            }
            trajectory = metrics / f"{phase}_{cell}_trajectories.json.gz"
            if trajectory.exists():
                with gzip.open(trajectory, "rt") as stream:
                    records = json.load(stream)
                requests = np.array([r["request"] for r in records])
                commands = np.array([r["commands"] for r in records])
                active = np.array([r["active"] for r in records], dtype=bool)
                velocities = np.array([r["actual_velocity"] for r in records])
                speed = np.linalg.norm(velocities, axis=-1)
                summary["representative_execution"] = {
                    "request_std": requests[active].reshape(-1, 4).std(axis=0).tolist(),
                    "control_std": commands[active].reshape(-1, 2).std(axis=0).tolist(),
                    "stationary_fraction": float(np.mean(speed[active] < 0.01)),
                    "mean_actual_speed_mps": float(np.mean(speed[active])),
                }
            comparison["cells"].setdefault(cell, {})[phase] = summary
    for cell, phases in comparison["cells"].items():
        if (
            "before" in phases
            and "after" in phases
            and phases["before"]["scene_sha256"] != phases["after"]["scene_sha256"]
        ):
            raise ValueError(f"Unpaired scenes: {cell}")
    (metrics / "comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    return comparison


def figures(run):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    destination = run / "figures"
    destination.mkdir(exist_ok=True)
    for path in sorted((run / "metrics").glob("*_trajectories.json.gz")):
        with gzip.open(path, "rt") as stream:
            records = json.load(stream)
        points = np.asarray([r["positions"][0] for r in records])
        active = np.asarray([r["active"][0] for r in records], dtype=bool)
        points = points[active]
        if not len(points):
            continue
        fig, ax = plt.subplots(figsize=(6, 6))
        for rover in range(4):
            ax.plot(points[:, rover, 0], points[:, rover, 1], label=f"rover {rover}")
            ax.scatter(*points[0, rover, :2], marker="o", s=25)
            ax.scatter(*points[-1, rover, :2], marker="x", s=45)
        ax.set(
            xlabel="world x [m] (offline only)",
            ylabel="world y [m]",
            title=path.name.removesuffix("_trajectories.json.gz"),
        )
        ax.set_aspect("equal", adjustable="datalim")
        ax.legend()
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(destination / path.name.replace(".json.gz", ".png"), dpi=150)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--figures", action="store_true")
    args = parser.parse_args()
    report = summarize(args.run_dir)
    if args.figures:
        figures(args.run_dir)
    print(
        json.dumps(
            {
                c: {
                    p: {
                        k: v
                        for k, v in x.items()
                        if k
                        in (
                            "success_rate",
                            "collision_rate",
                            "timeout_rate",
                            "mean_dmax_reduction_ratio",
                        )
                    }
                    for p, x in phases.items()
                }
                for c, phases in report["cells"].items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
