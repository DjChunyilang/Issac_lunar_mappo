#!/usr/bin/env python3
"""Explicit offline construction of traceable real-base/hypothesis maps."""

import argparse
import json
from pathlib import Path

import yaml
from lunar_rover_tasks.terrain_data.scene_pack import build_scene


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dtm", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--source-id", required=True)
    p.add_argument("--source-url")
    p.add_argument("--group", required=True)
    p.add_argument("--role", choices=["development", "validation", "test"], default="development")
    p.add_argument(
        "--window", type=int, nargs=4, required=True, metavar=("COL", "ROW", "WIDTH", "HEIGHT")
    )
    p.add_argument("--spacing", type=float, required=True)
    p.add_argument("--metric-baseline", type=float, required=True)
    p.add_argument("--source-half-width", type=int, required=True)
    p.add_argument("--height-error", type=Path)
    p.add_argument("--sample-count", type=Path)
    p.add_argument("--error-semantics", default="unavailable")
    p.add_argument("--enhancement-config", type=Path)
    p.add_argument("--stripe-rows", type=int, help="Bound raster working memory; keep one global NASA catalogue")
    p.add_argument("--obstacle-experiment-config", type=Path, help="Also derive labels/coarse ground using this explicit experiment")
    a = p.parse_args()
    quality = {
        k: v for k, v in [("height_error", a.height_error), ("sample_count", a.sample_count)] if v
    }
    enhancement = yaml.safe_load(a.enhancement_config.read_text()) if a.enhancement_config else None
    builder = build_scene
    extra = {}
    if a.stripe_rows is not None:
        from lunar_rover_tasks.terrain_data.scene_build_bounded import build_scene_bounded
        builder = build_scene_bounded
        extra['stripe_rows'] = a.stripe_rows
    out = builder(
        a.dtm,
        a.output,
        source_id=a.source_id,
        source_url=a.source_url,
        geographic_group=a.group,
        role=a.role,
        pixel_window=a.window,
        processing_spacing_m=a.spacing,
        metric_baseline_m=a.metric_baseline,
        source_half_width=a.source_half_width,
        quality_files=quality,
        error_semantics=a.error_semantics,
        enhancement=enhancement,
        **extra,
    )
    print(json.dumps({"scene": str(out)}))
    if a.obstacle_experiment_config is not None:
        from _common import cfg_from_experiment
        from lunar_rover_tasks.terrain_data.obstacle_atlas import build_obstacle_atlas
        from lunar_rover_tasks.terrain_data.scene_pack import file_hash
        cfg=cfg_from_experiment(a.obstacle_experiment_config)
        if cfg.obstacle_input is None:raise ValueError('Expected an obstacle input experiment')
        atlas=build_obstacle_atlas(out,cfg.obstacle_input.atlas_root/file_hash(out),cfg.obstacle_input,cfg.spec)
        print(json.dumps({'obstacle_atlas':str(atlas)}))


if __name__ == "__main__":
    main()
