#!/usr/bin/env python3
"""Freeze the paired 2 x 3 x 192 exp156 evaluation scenarios."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from _common import ROOT, cfg_from_experiment, load_yaml
from lunar_rover_tasks.tasks.multi_rover_gathering.gathering_env import (
    MultiRoverGatheringCore,
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
        "topology_curriculum_stage": "open",
        "crater_count": 0,
        "randomize_per_reset": False,
    },
    "Mixed": {
        "topology_profile": "mixed",
        "topology_curriculum_stage": "mixed_bottleneck",
        "crater_count": 30,
        "crater_seed": 11,
        "crater_depth_to_diameter": 0.12,
        "randomize_per_reset": False,
    },
    "Bottleneck": {
        "topology_profile": "bottleneck",
        "topology_curriculum_stage": "mixed_bottleneck",
        "crater_count": 100,
        "crater_seed": 4,
        "crater_depth_to_diameter": 0.15,
        "bottleneck_wall_half_width": 0.50,
        "bottleneck_gap_half_width": 0.50,
        "randomize_per_reset": False,
    },
}


def scenario_snapshot(core: MultiRoverGatheringCore) -> list[dict]:
    runtime = core.terrain_runtime
    result = []
    for index in range(core.num_envs):
        result.append(
            {
                "scenario_id": index,
                "positions": core.positions[index].detach().cpu().tolist(),
                "yaws": core.yaws[index].detach().cpu().tolist(),
                "terrain": {
                    "translation_xy": runtime.translation_xy[index].detach().cpu().tolist(),
                    "yaw": float(runtime.yaw[index].detach().cpu()),
                    "phase": float(runtime.phase[index].detach().cpu()),
                    "amplitude_scale": float(runtime.amplitude_scale[index].detach().cpu()),
                    "crater_radius_scale": float(runtime.crater_radius_scale[index].detach().cpu()),
                    "crater_depth_scale": float(runtime.crater_depth_scale[index].detach().cpu()),
                    "topology_bucket": int(runtime.topology_bucket[index].detach().cpu()),
                },
            }
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/experiment/exp156_differential_multiscale_ablation.yaml",
    )
    parser.add_argument(
        "--output",
        default=(
            "outputs/runs/exp156_differential_multiscale_ablation/_suite/"
            "scenario_manifest.json"
        ),
    )
    parser.add_argument("--episodes-per-cell", type=int, default=192)
    parser.add_argument("--seed", type=int, default=156_000)
    args = parser.parse_args()
    if args.episodes_per_cell != 192:
        raise SystemExit("The formal exp156 manifest requires 192 episodes per cell.")

    base = load_yaml(args.config)
    cells = []
    cell_index = 0
    for distance, initial_state in DISTANCES.items():
        for topology, terrain in TOPOLOGIES.items():
            seed = args.seed + cell_index * 1000
            raw = copy.deepcopy(base)
            raw.setdefault("experiment", {})["seed"] = seed
            raw["experiment"]["num_envs"] = args.episodes_per_cell
            raw.setdefault("initial_state", {}).update(initial_state)
            raw["initial_state"]["curriculum_enabled"] = False
            raw.setdefault("terrain", {}).update(terrain)
            raw.setdefault("safety", {})["collision_termination_enabled"] = True

            # Use the project loader against a temporary suite config so the
            # manifest records exactly the same resolved interface as evaluation.
            output = ROOT / args.output
            config_dir = output.parent / "scenario_configs"
            config_dir.mkdir(parents=True, exist_ok=True)
            config_path = config_dir / f"{distance}_{topology.lower()}.json"
            config_path.write_text(json.dumps(raw, indent=2), encoding="utf-8")
            cfg = cfg_from_experiment(config_path)
            cfg.simulation.device = "cpu"
            core = MultiRoverGatheringCore(cfg)
            scenarios = scenario_snapshot(core)
            cell_payload = {
                "cell": f"{distance}_{topology.lower()}",
                "distance": distance,
                "topology": topology,
                "seed": seed,
                "initial_state_overrides": initial_state,
                "terrain_overrides": terrain,
                "scenarios": scenarios,
            }
            canonical = json.dumps(cell_payload, sort_keys=True, separators=(",", ":"))
            cell_payload["sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
            cells.append(cell_payload)
            cell_index += 1

    payload = {
        "schema_version": 1,
        "experiment": "exp156_differential_multiscale_ablation",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "base_config": args.config,
        "episodes_per_cell": args.episodes_per_cell,
        "total_episodes": 6 * args.episodes_per_cell,
        "cells": cells,
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "cells": 6, "episodes": 1152}, indent=2))


if __name__ == "__main__":
    main()
