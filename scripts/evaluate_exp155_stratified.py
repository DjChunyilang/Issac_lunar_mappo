#!/usr/bin/env python3
"""Run the fixed 2 x 3 exp155 strict evaluation matrix (64 episodes/cell)."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch
import yaml

from _common import ROOT, cfg_from_experiment, load_yaml
from evaluate_proxy_policy import evaluate_checkpoint
from lunar_rover_tasks.tasks.multi_rover_gathering.mapf_diagnostics import (
    analyze_traversability_topology,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.terrain_features import (
    query_terrain_features,
)


DISTANCES = {
    "near": {
        "spawn_radius_min": 2.4,
        "spawn_radius_max": 3.4,
        "center_xy_range": 1.0,
        "jitter_std": 0.25,
    },
    "far": {
        "spawn_radius_min": 4.5,
        "spawn_radius_max": 6.5,
        "center_xy_range": 2.0,
        "jitter_std": 0.40,
    },
}

TOPOLOGIES = {
    "Open": {
        "topology_profile": "open",
        "crater_count": 0,
        "randomize_per_reset": False,
    },
    "Mixed": {
        "topology_profile": "mixed",
        "crater_count": 30,
        "crater_seed": 11,
        "crater_depth_to_diameter": 0.12,
        "randomize_per_reset": False,
    },
    "Bottleneck": {
        "topology_profile": "bottleneck",
        "crater_count": 100,
        "crater_seed": 4,
        "crater_depth_to_diameter": 0.15,
        "bottleneck_wall_half_width": 0.50,
        "bottleneck_gap_half_width": 0.50,
        "randomize_per_reset": False,
    },
}


def _topology_analysis(config_path: Path) -> dict:
    cfg = cfg_from_experiment(config_path)
    axis = torch.linspace(-12.5, 12.5, 81)
    grid_y, grid_x = torch.meshgrid(axis, axis, indexing="ij")
    xy = torch.stack((grid_x, grid_y), dim=-1)
    traversable = query_terrain_features(xy, cfg.terrain)[..., 4] >= 0.5
    analysis = analyze_traversability_topology(
        traversable,
        max_sources=256,
        seed=0,
    )
    return {
        "topology_class": analysis.topology_class,
        "blocked_ratio": analysis.blocked_ratio,
        "bc_mean": analysis.bc_mean,
        "bc_variance": analysis.bc_variance,
        "bc_high_region_ratio": analysis.bc_high_region_ratio,
    }


def _strict_checks(metrics: dict) -> dict[str, bool]:
    return {
        "dmax_reduction_ratio": float(metrics["dmax_reduction_ratio"]) <= 0.20,
        "success_rate": float(metrics["success_rate"]) >= 0.90,
        "collision_rate": float(metrics["collision_rate"]) <= 0.02,
        "timeout_rate": float(metrics["timeout_rate"]) < 0.10,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--episodes-per-cell", type=int, default=64)
    parser.add_argument("--steps", type=int, default=480)
    parser.add_argument("--seed", type=int, default=23)
    args = parser.parse_args()

    if args.episodes_per_cell != 64:
        raise SystemExit("Formal exp155 evaluation requires exactly 64 episodes per cell.")
    run_dir = ROOT / args.run_dir
    base_config_path = run_dir / "config/experiment.yaml"
    checkpoint_path = run_dir / "checkpoints/best.pt"
    if not base_config_path.is_file() or not checkpoint_path.is_file():
        raise SystemExit("The run directory must contain config/experiment.yaml and checkpoints/best.pt.")

    base = load_yaml(base_config_path)
    config_dir = run_dir / "metrics/strata_configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for distance_index, (distance_name, initial_state) in enumerate(DISTANCES.items()):
        for topology_index, (expected_topology, terrain) in enumerate(TOPOLOGIES.items()):
            raw = copy.deepcopy(base)
            raw.setdefault("initial_state", {}).update(initial_state)
            raw["initial_state"]["curriculum_enabled"] = False
            raw.setdefault("terrain", {}).update(terrain)
            raw.setdefault("safety", {})["collision_termination_enabled"] = True
            config_path = config_dir / f"{distance_name}_{expected_topology.lower()}.yaml"
            config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
            topology = _topology_analysis(config_path)
            if topology["topology_class"] != expected_topology:
                raise SystemExit(
                    f"Topology preset {expected_topology} was classified as "
                    f"{topology['topology_class']}; refusing a mislabeled evaluation."
                )
            output = run_dir / "metrics" / f"eval_{distance_name}_{expected_topology.lower()}.json"
            metrics = evaluate_checkpoint(
                config=config_path,
                checkpoint=checkpoint_path,
                device=args.device,
                num_envs=64,
                steps=args.steps,
                seed=args.seed + distance_index * 1000 + topology_index * 100,
                output=output,
                run_dir=run_dir,
            )
            checks = _strict_checks(metrics)
            records.append(
                {
                    "distance": distance_name,
                    "topology": expected_topology,
                    "topology_analysis": topology,
                    "metrics": metrics,
                    "checks": checks,
                    "passed": all(checks.values()),
                    "timeout_episode_count": int(
                        metrics.get("timeout_episode_metrics", {}).get("count", 0)
                    ),
                }
            )

    result = {
        "episodes_per_cell": 64,
        "total_episodes": 384,
        "thresholds": {
            "dmax_reduction_ratio": "<= 0.20",
            "success_rate": ">= 0.90",
            "collision_rate": "<= 0.02",
            "timeout_rate": "< 0.10 (at most 6/64)",
        },
        "cells": records,
        "passed": all(record["passed"] for record in records),
    }
    output_path = run_dir / "metrics/stratified_strict_acceptance.json"
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
