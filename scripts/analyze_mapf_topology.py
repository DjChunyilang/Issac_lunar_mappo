#!/usr/bin/env python
"""Classify a fixed proxy terrain for offline MAPF-stratified evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from _common import cfg_from_experiment
from lunar_rover_tasks.tasks.multi_rover_gathering.mapf_diagnostics import (
    analyze_traversability_topology,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.terrain_features import (
    query_terrain_features,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--grid-size", type=int, default=51)
    parser.add_argument("--max-sources", type=int, default=256)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.grid_size < 3 or args.grid_size % 2 == 0:
        raise SystemExit("--grid-size must be an odd integer >= 3.")

    cfg = cfg_from_experiment(args.config)
    extent = float(cfg.safety.world_xy_limit)
    axis = torch.linspace(-extent, extent, args.grid_size)
    grid_x, grid_y = torch.meshgrid(axis, axis, indexing="ij")
    xy = torch.stack((grid_x, grid_y), dim=-1)
    features = query_terrain_features(xy, cfg.terrain)
    slope = torch.linalg.norm(features[..., 1:3], dim=-1)
    traversable = (
        (features[..., 4] >= float(cfg.terrain.min_speed_scale))
        & (slope <= float(cfg.gather_point.max_slope))
    )
    analysis = analyze_traversability_topology(
        traversable,
        max_sources=args.max_sources,
        seed=int(cfg.seed),
    )
    payload = {
        "schema_version": 1,
        "config": str(args.config),
        "grid_size": args.grid_size,
        "topology_class": analysis.topology_class,
        "blocked_ratio": analysis.blocked_ratio,
        "bc_mean": analysis.bc_mean,
        "bc_variance": analysis.bc_variance,
        "bc_high_region_ratio": analysis.bc_high_region_ratio,
        "traversability_threshold": float(cfg.terrain.min_speed_scale),
        "slope_threshold": float(cfg.gather_point.max_slope),
    }
    serialized = json.dumps(payload, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()
