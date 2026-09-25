#!/usr/bin/env python3
"""Build the approved seven-map NASA SFD engineering suite; no learning."""

import argparse
from copy import deepcopy
import gc
from importlib.metadata import version
import json
import os
from pathlib import Path
import resource
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "2")
import numpy as np
import yaml
from lunar_rover_tasks.terrain_data.nasa_sfd import diameter_cdf, sample_diameters
from lunar_rover_tasks.terrain_data.scene_pack import ScenePack, build_scene, file_hash


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False))


def distribution_checks(spec):
    results = {}
    for kind in ("craters", "rocks"):
        cfg = spec[kind]
        low, high = cfg["diameter_min_m"], cfg["diameter_max_m"]
        d = np.sort(sample_diameters(kind, 100000, low, high, np.random.default_rng(23)))
        f = diameter_cdf(kind, d, low, high)
        ks = float(max(np.max(np.arange(1, len(d)+1)/len(d)-f), np.max(f-np.arange(len(d))/len(d))))
        results[kind] = {"samples": len(d), "ks": ks, "limit": .01, "passed": ks < .01}
    if not all(r["passed"] for r in results.values()):
        raise AssertionError("Distribution validation failed")
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference-scene", type=Path, default=Path("outputs/runs/lunar_terrain_validation/nasa_implementation_20260923/scenes/npb_200m_real_base/scene.json"))
    p.add_argument("--spec", type=Path, default=Path("configs/terrain/nasa_sfd_v2.yaml"))
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--resume", action="store_true", help="Verify/reuse completed immutable packs in the same configuration")
    a = p.parse_args()
    reference = ScenePack(a.reference_scene)
    meta = reference.meta
    spec = yaml.safe_load(a.spec.read_text())
    config = {"reference_scene": str(a.reference_scene.resolve()), "reference_sha256": file_hash(a.reference_scene),
              "enhancement": spec, "processing_spacing_m": .1, "seeds": [23, 24, 25]}
    if a.run_dir.exists():
        if not a.resume or json.loads((a.run_dir / "config/experiment.json").read_text()) != config:
            raise FileExistsError("Use a new run directory or --resume with identical configuration")
    else:
        a.run_dir.mkdir(parents=True)
        write_json(a.run_dir / "config/experiment.json", config)
        (a.run_dir / "config/nasa_sfd_v2.yaml").write_text(a.spec.read_text())
    for src in meta["sources"].values():
        if file_hash(src["path"]) != src["sha256"]:
            raise ValueError("Input product changed")
    checks = distribution_checks(spec)
    write_json(a.run_dir / "metrics/distribution_checks.json", checks)
    records = []
    designs = [("real_base", None)]
    for seed in (23, 24, 25):
        for name, rocks in (("craters", False), ("craters_rocks", True)):
            cfg = deepcopy(spec)
            cfg["seed"], cfg["rocks"]["enabled"] = seed, rocks
            designs.append((f"seed{seed}_{name}", cfg))
    for name, enhancement in designs:
        start = time.perf_counter()
        scene_dir = a.run_dir / "scenes" / name
        manifest = scene_dir / "scene.json"
        if not manifest.exists():
            build_scene(
                meta["sources"]["dtm"]["path"], scene_dir,
                source_id=meta["source_id"], geographic_group=meta["geographic_group"],
                pixel_window=meta["pixel_window"], processing_spacing_m=.1,
                metric_baseline_m=.8, source_half_width=2,
                quality_files={k: v["path"] for k, v in meta["sources"].items() if k != "dtm"},
                error_semantics=meta["error_semantics"], enhancement=enhancement,
                source_url=meta["sources"]["dtm"].get("url"), role=meta["role"],
            )
        scene = ScenePack(manifest)
        if scene.meta["enhancement"] != enhancement:
            raise ValueError("Existing scene configuration mismatch")
        base_pack = ScenePack(a.run_dir / "scenes/real_base/scene.json")
        np.testing.assert_array_equal(scene.arrays["base_height"], base_pack.arrays["base_height"])
        reconstructed = scene.arrays["base_height"].copy()
        for kind in ("craters", "rocks"):
            if f"layer_{kind}" in scene.arrays:
                reconstructed += scene.arrays[f"layer_{kind}"]
        np.testing.assert_array_equal(reconstructed, scene.arrays["features"][0])
        good = scene.arrays["metrics_valid"] & base_pack.arrays["metrics_valid"]
        slope = np.hypot(scene.arrays["features"][1], scene.arrays["features"][2])
        base_slope = np.hypot(base_pack.arrays["features"][1], base_pack.arrays["features"][2])
        residual = scene.arrays["features"][0] - scene.arrays["base_height"]
        valid = scene.arrays["height_valid"]
        record = {
            "name": name, "manifest": str(manifest.resolve()), "manifest_sha256": file_hash(manifest),
            "build_or_verify_seconds": time.perf_counter()-start,
            "rss_peak_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
            "base_identical": True, "layer_sum_exact": True,
            "residual_min_m": float(residual[valid].min()), "residual_max_m": float(residual[valid].max()),
            "residual_mean_m": float(residual[valid].mean()), "residual_rms_m": float(np.sqrt(np.mean(residual[valid]**2))),
            "slope_delta_rms": float(np.sqrt(np.mean((slope[good]-base_slope[good])**2))),
            "slope_max": float(slope[good].max()),
            "native_scale_diagnostics": scene.meta.get("enhancement_diagnostics"),
        }
        if enhancement:
            catalog = scene.meta["feature_catalog"]
            record["populations"] = {
                k: {"generated": len(catalog[k]), "target_centres": sum(i["centre_in_target"] for i in catalog[k]),
                    "target_intersecting": sum(i["affects_target"] for i in catalog[k])}
                for k in ("craters", "rocks")
            }
            record["geometry"] = catalog["generation"]["geometry_diagnostics"]
            if enhancement["rocks"]["enabled"]:
                paired = ScenePack(a.run_dir / f"scenes/seed{enhancement['seed']}_craters/scene.json")
                assert catalog["craters"] == paired.meta["feature_catalog"]["craters"]
                np.testing.assert_array_equal(scene.arrays["layer_craters"], paired.arrays["layer_craters"])
                record["paired_craters_identical"] = True
                del paired
        records.append(record)
        write_json(a.run_dir / "metrics/map_checks.json", records)
        print(json.dumps(record), flush=True)
        del scene, base_pack, reconstructed, slope, base_slope, residual, valid, good
        gc.collect()
    for src in meta["sources"].values():
        assert file_hash(src["path"]) == src["sha256"]
    write_json(a.run_dir / "run_manifest.json", {
        "experiment_id": "lunar_terrain_validation", "run_id": a.run_dir.name,
        "command": [sys.executable, *sys.argv], "learning_updates": 0,
        "geographic_group": meta["geographic_group"], "role": meta["role"],
        "source_products_unchanged": True, "map_build_passed": True,
        "environment_acceptance": "pending_separate_benchmark", "maps": records,
        "versions": {k: version(k) for k in ("numpy", "scipy", "opensimplex", "numba", "rasterio", "torch")},
        "limitations": ["Uncalibrated NPB microterrain SFD", "Simplified existing shapes", "Heightfield is not contact/mesh validation"],
    })


if __name__ == "__main__":
    main()
