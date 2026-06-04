#!/usr/bin/env python
"""Sample a configured proxy terrain and write sanity metrics/height map."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from _common import ROOT, cfg_from_experiment
from lunar_rover_tasks.tasks.multi_rover_gathering.terrain_features import query_terrain_features
from terrain_viz import save_height_map


def _resolve(path: str | Path) -> Path:
    output = Path(path)
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def sample_terrain_profile(config: str | Path, resolution: int = 240) -> dict:
    cfg = cfg_from_experiment(config)
    half_size = 0.5 * float(cfg.terrain.crater_field_size)
    xs = torch.linspace(-half_size, half_size, resolution)
    ys = torch.linspace(-half_size, half_size, resolution)
    grid_x, grid_y = torch.meshgrid(xs, ys, indexing="xy")
    xy = torch.stack((grid_x, grid_y), dim=-1)
    with torch.no_grad():
        features = query_terrain_features(xy, cfg.terrain)
    height = features[..., 0]
    roughness = features[..., 3]
    traversability = features[..., 4]
    speed_scale = traversability * torch.exp(-roughness * float(cfg.terrain.slope_speed_scale))
    speed_scale = speed_scale.clamp(min=float(cfg.terrain.min_speed_scale), max=1.0)
    return {
        "config": str(config),
        "terrain_type": cfg.terrain.type,
        "dynamics_enabled": bool(cfg.terrain.dynamics_enabled),
        "field_size_m": float(cfg.terrain.crater_field_size),
        "resolution": resolution,
        "height_min": float(height.min()),
        "height_max": float(height.max()),
        "height_range": float(height.max() - height.min()),
        "roughness_mean": float(roughness.mean()),
        "roughness_max": float(roughness.max()),
        "traversability_min": float(traversability.min()),
        "traversability_mean": float(traversability.mean()),
        "mean_terrain_speed_scale": float(speed_scale.mean()),
        "min_terrain_speed_scale": float(speed_scale.min()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--height-map", required=True)
    parser.add_argument("--resolution", type=int, default=240)
    parser.add_argument("--min-height-range", type=float, default=None)
    parser.add_argument("--max-height-range", type=float, default=None)
    args = parser.parse_args()

    metrics = sample_terrain_profile(args.config, resolution=args.resolution)
    output_path = _resolve(args.output)
    height_map_path = _resolve(args.height_map)
    half_size = 0.5 * metrics["field_size_m"]
    save_height_map(
        cfg_from_experiment(args.config).terrain,
        np.array([-half_size, -half_size]),
        np.array([half_size, half_size]),
        height_map_path,
        title=f"{metrics['terrain_type']} height heatmap",
        resolution=args.resolution,
    )
    metrics["height_map"] = str(height_map_path)
    checks = {}
    if args.min_height_range is not None:
        checks["min_height_range"] = metrics["height_range"] >= args.min_height_range
    if args.max_height_range is not None:
        checks["max_height_range"] = metrics["height_range"] <= args.max_height_range
    metrics["checks"] = checks
    metrics["passed"] = all(checks.values()) if checks else True
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    if not metrics["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
