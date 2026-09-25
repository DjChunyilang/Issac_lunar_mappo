#!/usr/bin/env python3
"""LUPEX data-first audit, geographic benchmark and no-training integration checks."""

from __future__ import annotations

import argparse
import atexit
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source/lunar_rover_tasks"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pyproj import CRS, Transformer
import rasterio
from scipy.ndimage import binary_dilation, label
from shapely.geometry import shape, mapping
from shapely.ops import transform as geometry_transform

from lunar_rover_tasks.terrain_data import ERROR_SEMANTICS, FEATURE_SCHEMA
from lunar_rover_tasks.terrain_data.benchmark import (
    build_benchmark, verify_benchmark, window_catalogue,
)
from lunar_rover_tasks.terrain_data.metrics import (
    audit_half_widths, local_metrics, spatial_structure, summary,
)
from lunar_rover_tasks.terrain_data.products import (
    REGIONS, checksum, extract_verified, geographic_origin, product_paths, raster_metadata,
    read_region, roi_catalogue, write_json,
)
from lunar_rover_tasks.terrain_data.runtime import TerrainProbeRuntime, load_scene


def _atlas(name, data, first_metrics, flags, target):
    z, valid = data["height"], data["raw_valid"]
    slope = np.degrees(np.arctan(np.hypot(first_metrics["slope_x"], first_metrics["slope_y"])))
    panels = [(np.where(valid, z, np.nan), "SfS height (m)", "terrain"),
              (data["inside_roi"].astype(float) + valid.astype(float)*2,
               "Coverage: ROI=1, valid=2, both=3", "viridis"),
              (slope, "Plane slope, native 2 m baseline (deg)", "magma"),
              (first_metrics["roughness"], "Detrended RMS, 2 m baseline (m)", "magma"),
              (np.where(valid, data["error"], np.nan), "Height sensitivity, NOT sigma (m)", "magma"),
              (flags.astype(float), "Flags: missing=1, edge=2, tail=4, jump=8", "viridis")]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    t = data["transform"]
    extent = [t.c, t.c+z.shape[1]*t.a, t.f+z.shape[0]*t.e, t.f]
    for ax, (values, title, cmap) in zip(axes.flat, panels, strict=True):
        finite = values[np.isfinite(values)]
        limits = np.percentile(finite, [1, 99]) if finite.size and "Coverage" not in title and "Flags" not in title else (None, None)
        im = ax.imshow(values, extent=extent, origin="upper", cmap=cmap,
                       vmin=limits[0], vmax=limits[1], interpolation="nearest")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Projected easting (m)")
        ax.set_ylabel("Projected northing (m)")
        fig.colorbar(im, ax=ax, shrink=.8)
    fig.suptitle(f"{name.upper()} | full product | display limits p01-p99; metrics use all valid cells")
    fig.savefig(target, dpi=140)
    plt.close(fig)


def audit_region(root, name, roi, roi_crs, run):
    paths = product_paths(root, name)
    products = {kind: raster_metadata(path) for kind, path in paths.items()}
    dtm = products["dtm"]
    target_crs = CRS(dtm["crs_wkt"])
    roi_transformer = Transformer.from_crs(roi_crs, target_crs, always_xy=True)
    original_roi = roi
    roi = dict(roi, geometry=mapping(geometry_transform(roi_transformer.transform,
                                                       shape(roi["geometry"]))))
    if dtm["horizontal_unit"] != "metre" or dtm["scales"] != [1.0] or dtm["offsets"] != [0.0]:
        raise ValueError("Unimplemented source unit/scale conversion")
    data = read_region(root, name, roi)
    z, valid, error = data["height"], data["raw_valid"], data["error"]
    dx, dy = data["spacing"]
    area = valid.sum()*dx*dy
    regions, count = label(valid)
    sizes = np.bincount(regions.ravel())[1:]
    stats = dict(products=products, roi=roi,
                 roi_registration=dict(source_crs_wkt=roi_crs.to_wkt(),
                                       equivalent_crs_definition=target_crs.equals(roi_crs, ignore_axis_order=True),
                                       transformed_boundary_distance_m=shape(original_roi["geometry"]).hausdorff_distance(shape(roi["geometry"])),
                                       method="pyproj source ROI CRS to DTM CRS; always_xy"),
                 origin=geographic_origin(dtm["crs_wkt"], dtm["bounds"]),
                 grid_registration={}, full_valid_area_m2=float(area),
                 roi_valid_area_m2=float(data["valid"].sum()*dx*dy),
                 coverage_fraction=float(valid.mean()), connected_valid_components=int(count),
                 largest_valid_component_cells=int(sizes.max()) if sizes.size else 0,
                 height_m=summary(z[valid]), roi_height_m=summary(z[data["valid"]]),
                 sensitivity_m=summary(error[valid]), error_semantics=ERROR_SEMANTICS,
                 missing_error_on_valid_dtm=int((valid & ~np.isfinite(error)).sum()),
                 effective_resolution_m=None, scale_metrics=[])
    for kind in ("error", "ortho"):
        meta = products[kind]
        same_crs = CRS(meta["crs_wkt"]).equals(CRS(dtm["crs_wkt"]), ignore_axis_order=True)
        offset = [(meta["transform"][2]-dtm["transform"][2])/dx,
                  (dtm["transform"][5]-meta["transform"][5])/dy]
        stats["grid_registration"][kind] = dict(same_crs=same_crs, origin_offset_pixels=offset,
                                                identical_grid=meta["transform"] == dtm["transform"]
                                                and meta["shape"] == dtm["shape"],
                                                analysis_alignment="coordinate nearest for sensitivity; ortho only displayed in its own coordinates")
    first = None
    for half in audit_half_widths(z.shape):
        print(f"{name}: baseline {2*half*dx:g} m", flush=True)
        metric = local_metrics(z, valid, dx, dy, half)
        if first is None:
            first = metric
        slopes = np.hypot(metric["slope_x"], metric["slope_y"])
        stats["scale_metrics"].append(dict(half_width_cells=half, baseline_xy_m=[2*half*dx, 2*half*dy],
                                            sample_window_cells=2*half+1,
                                            valid_centres=int(metric["valid"].sum()),
                                            coverage_fraction=float(metric["valid"].sum()/valid.sum()),
                                            slope_tangent=summary(slopes),
                                            slope_degrees=summary(np.degrees(np.arctan(slopes))),
                                            detrended_rms_m=summary(metric["roughness"]),
                                            interpretation="diagnostic only; short-baseline smoothing" if 2*half*dx < 10
                                            else "regional morphology candidate; not certified effective resolution"))
    stats["spatial_structure"] = spatial_structure(z, valid, dx, dy)
    edge = valid & binary_dilation(~valid, border_value=1)
    finite_error = error[valid & np.isfinite(error)]
    tail_limit = float(np.percentile(finite_error, 95))
    tail = valid & (error > tail_limit)
    jumps = np.zeros(z.shape, bool)
    differences = []
    for axis in (0, 1):
        delta = np.abs(np.diff(z, axis=axis))
        differences.append(delta[np.isfinite(delta)])
    jump_limit = float(np.percentile(np.concatenate(differences), 99.9))
    # Descriptive candidates only: a real steep crater must not be removed as an artifact.
    jumps[:-1, :] |= np.abs(np.diff(z, axis=0)) > jump_limit
    jumps[:, :-1] |= np.abs(np.diff(z, axis=1)) > jump_limit
    flags = ((~valid).astype(np.uint8) + 2*edge.astype(np.uint8)
             + 4*tail.astype(np.uint8) + 8*jumps.astype(np.uint8))
    stats["quality_flags"] = dict(edge_cells=int(edge.sum()), sensitivity_tail_cells=int(tail.sum()),
                                  within_product_p95_sensitivity_m=tail_limit,
                                  jump_candidate_cells=int(jumps.sum()),
                                  within_product_p999_adjacent_difference_m=jump_limit,
                                  excluded_for_these_flags=0, verified_artifact_mask_available=False,
                                  caveat="Relative audit flags only, not safety/accuracy thresholds; "
                                         "automatically neither remove nor label steep terrain as artifact")
    with rasterio.open(paths["dtm"]) as ds:
        profile = ds.profile.copy()
        profile.update(dtype="uint8", nodata=None, compress="deflate")
        with rasterio.open(run / "data" / f"{name}_quality_flags.tif", "w", **profile) as dst:
            dst.write(flags, 1)
            dst.update_tags(source_sha256=products["dtm"]["sha256"], semantic="diagnostic_flags_v1")
    _atlas(name, data, first, flags, run / "figures" / f"{name}_atlas.png")
    # Ortho uses its actual extent, never assumes pixel-for-pixel registration.
    with rasterio.open(paths["ortho"]) as ds:
        image = ds.read(1, masked=True).astype(float).filled(np.nan)
        fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
        lo, hi = np.nanpercentile(image, [1, 99])
        ax.imshow(image, extent=[ds.bounds.left, ds.bounds.right, ds.bounds.bottom, ds.bounds.top],
                  origin="upper", cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")
        ax.set_title(f"{name.upper()} orthomosaic | native georeferencing")
        ax.set_xlabel("Projected easting (m)"); ax.set_ylabel("Projected northing (m)")
        fig.savefig(run / "figures" / f"{name}_ortho.png", dpi=140)
        plt.close(fig)
    return stats


def create_scenes(regions, products, run):
    results = {}
    for name, region in regions.items():
        target = run / "data" / "scenes" / f"{name}.json"
        paths = product_paths(products, name)
        scene = dict(schema="lunar_scene_v1", terrain_feature_schema=FEATURE_SCHEMA,
                     layer="task_representation", source_id=f"zenodo:17153447/{name}",
                     synthetic_detail=False, representation="native full-product reference; no resampling",
                     origin_xy_m=region["origin"]["projected_xy_m"],
                     native_spacing_m=region["products"]["dtm"]["spacing_m"],
                     processing_spacing_m=region["products"]["dtm"]["spacing_m"],
                     effective_resolution_m=None, orbital_prior_for_actor=False)
        for kind in ("dtm", "error"):
            scene[kind] = dict(path=os.path.relpath(paths[kind], target.parent),
                               sha256=region["products"][kind]["sha256"])
        write_json(target, scene)
        with load_scene(target) as terrain:
            # Choose a supported point only for interface testing, not benchmark inclusion.
            with rasterio.open(paths["dtm"]) as ds:
                z = ds.read(1, masked=True)
                valid = ~np.ma.getmaskarray(z) & np.isfinite(z.data)
                from scipy.ndimage import distance_transform_edt
                # Pad: outside product extent is also missing support.
                distance = distance_transform_edt(np.pad(valid, 1))[1:-1, 1:-1]
                row, col = np.unravel_index(np.argmax(distance), distance.shape)
                xy = np.array(ds.transform * (col+.5, row+.5)) - terrain.origin
            runtime = TerrainProbeRuntime(terrain, half_width=1,
                                          sensor_radius_m=4*max(terrain.spacing))
            runtime.reset(xy)
            observed = runtime.observe_local([[0, 0]])[0]
            steps = [runtime.step([terrain.spacing[0], 0], .25) for _ in range(4)]
            geo = terrain.local_to_geographic(xy)
            roundtrip = float(np.linalg.norm(terrain.geographic_to_local(geo)-xy))
            denied = False
            try:
                runtime.query_orbital_prior(xy)
            except PermissionError:
                denied = True
            results[name] = dict(local_observation_valid=observed["metrics_valid"],
                                 kinematic_steps_moved=sum(s["moved"] for s in steps),
                                 roundtrip_error_m=roundtrip, implicit_orbital_prior_denied=denied,
                                 source=terrain.provenance(), scope="interface only; not vehicle safety or task success")
            if not (observed["metrics_valid"] and all(s["moved"] for s in steps)
                    and roundtrip < 1e-5 and denied):
                raise RuntimeError(f"Failed scene smoke: {name}")
    return results


def plots(regions, benchmark, run):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for name, data in regions.items():
        for ax, direction in zip(axes, ("east", "north"), strict=True):
            curve = data["spatial_structure"]["directions"][direction]["curve"]
            ax.loglog([v["lag_m"] for v in curve],
                      [v["rms_bidirectional_slope_deg"] for v in curve], "o-", label=name.upper())
            ax.set_xlabel("Physical baseline (m)"); ax.set_ylabel("RMS slope angle (deg)")
            ax.set_title(direction); ax.legend()
    fig.savefig(run / "figures/slope_baselines.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 7), constrained_layout=True)
    for group_id, group in enumerate(benchmark["groups"]):
        for name in group:
            xy = np.array(regions[name]["origin"]["projected_xy_m"])/1000
            ax.scatter(*xy, color=f"C{group_id}")
            ax.annotate(name.upper(), xy, xytext=(5, 5), textcoords="offset points")
    for edge in benchmark["relations"]:
        if edge["same_group"]:
            points = np.array([regions[n]["origin"]["projected_xy_m"] for n in (edge["a"], edge["b"])])/1000
            ax.plot(points[:, 0], points[:, 1], "k--", linewidth=.8)
    ax.set(xlabel="Projected easting (km)", ylabel="Projected northing (km)",
           title="Geographic groups; dashed = shared listed control sources", aspect="equal")
    fig.savefig(run / "figures/region_relationships.png", dpi=150); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ROOT / "outputs/runs/lunar_terrain_validation/_suite/data/lupex_17153447_v1/DataS1.zip")
    parser.add_argument("--provenance", type=Path, default=ROOT / "configs/terrain/lupex_sources_v1.json")
    parser.add_argument("--run-name", default="audit_lupex_v1")
    parser.add_argument("--output-layout", choices=["run"], default="run")
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args()
    if Path(args.run_name).name != args.run_name or args.run_name in (".", "..", "_suite"):
        parser.error("Use a single non-reserved run name")
    run = args.run_dir or ROOT / "outputs/runs/lunar_terrain_validation" / args.run_name
    if run.exists():
        parser.error("Run directory already exists; choose a new run to preserve frozen artifacts")
    for subdir in ("config", "data", "figures", "metrics"):
        (run / subdir).mkdir(parents=True)
    archive = args.archive.resolve()
    products = archive.parent / "products"
    manifest = dict(experiment_id="lunar_terrain_validation", run_id=run.name,
                    status="running", command=sys.argv, started_utc=datetime.now(timezone.utc).isoformat(),
                    device="cpu", learning_training_started=False, python=platform.python_version(),
                    rasterio=rasterio.__version__, archive_sha256=checksum(archive),
                    code_sha256={str(p.relative_to(ROOT)): checksum(p) for p in
                                 [Path(__file__), *sorted((ROOT / "source/lunar_rover_tasks/lunar_rover_tasks/terrain_data").glob("*.py"))]})
    write_json(run / "run_manifest.json", manifest)
    def mark_incomplete():
        if manifest["status"] == "running":
            manifest.update(status="failed_or_interrupted", ended_utc=datetime.now(timezone.utc).isoformat())
            write_json(run / "run_manifest.json", manifest)
    atexit.register(mark_incomplete)
    source_manifest = extract_verified(archive, products)
    write_json(run / "data/product_manifest.json", source_manifest)
    provenance = json.loads(args.provenance.read_text())
    write_json(run / "config/lupex_sources_v1.json", provenance)
    evidence = archive.parent / "evidence"
    write_json(run / "data/evidence_manifest.json", dict(
        method="Paper read and Tables 1/2 manually transcribed into versioned provenance config",
        files=[dict(path=str(p), sha256=checksum(p)) for p in sorted(evidence.glob("*")) if p.is_file()],
        paper_url=provenance["author_pdf"]))
    rois, roi_crs = roi_catalogue(products)
    regions = {}
    for name in REGIONS:
        regions[name] = audit_region(products, name, rois[name], roi_crs, run)
        write_json(run / "metrics" / f"{name}_audit.json", regions[name])
    benchmark = build_benchmark(regions, provenance)
    verify_benchmark(benchmark)
    write_json(run / "data/benchmark.json", benchmark)
    windows = window_catalogue(benchmark, regions)
    write_json(run / "data/windows.json", windows)
    write_json(run / "data/layers.json", dict(
        real_products=dict(manifest="product_manifest.json", immutable=True),
        task_representation=dict(scenes="scenes/", windows="windows.json", resampling=False),
        hypothesis_enhancement=dict(status="absent", reason="No justified unresolved-scale model yet")))
    smoke = create_scenes(regions, products, run)
    write_json(run / "metrics/integration_smoke.json", smoke)
    plots(regions, benchmark, run)
    report = dict(schema="lupex_audit_v1", regions=regions,
                  benchmark_hash=benchmark["manifest_sha256"], group_count=len(benchmark["groups"]),
                  nested_fold_count=len(benchmark["folds"]), window_views=len(windows),
                  safety_certification=False, learning_training_started=False,
                  no_data_filling=True, uncertainty_is_sigma=False)
    write_json(run / "metrics/summary.json", report)
    manifest.update(status="completed", completed_utc=datetime.now(timezone.utc).isoformat(),
                    metrics="metrics/summary.json", benchmark="data/benchmark.json",
                    integration="metrics/integration_smoke.json", figures="figures/",
                    benchmark_sha256=benchmark["manifest_sha256"])
    write_json(run / "run_manifest.json", manifest)
    print(json.dumps(dict(run=str(run), groups=benchmark["groups"], windows=len(windows))))


if __name__ == "__main__":
    main()
