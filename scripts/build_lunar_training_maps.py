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
    a = p.parse_args()
    quality = {
        k: v for k, v in [("height_error", a.height_error), ("sample_count", a.sample_count)] if v
    }
    enhancement = yaml.safe_load(a.enhancement_config.read_text()) if a.enhancement_config else None
    out = build_scene(
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
    )
    print(json.dumps({"scene": str(out)}))


if __name__ == "__main__":
    main()
