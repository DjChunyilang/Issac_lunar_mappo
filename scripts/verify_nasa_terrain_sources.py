#!/usr/bin/env python3
"""Verify the downloaded NPB product and freeze cross-source geographic groups."""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import rasterio
from lunar_rover_tasks.terrain_data.scene_pack import file_hash
from rasterio.warp import transform_bounds


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nasa-root", type=Path, required=True)
    p.add_argument("--lupex-root", type=Path, required=True)
    p.add_argument("--jpl-crop", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    products = {}
    relations = []
    with rasterio.open(a.nasa_root / "npb/dtm.tif") as original:
        common = original.crs
        with rasterio.open(a.jpl_crop) as crop:
            if crop.crs != common:
                raise ValueError("JPL crop CRS differs")
            window = original.window(*crop.bounds)
            z = original.read(1, window=window)
            exact = np.array_equal(z, crop.read(1), equal_nan=True)
            verification = {
                "exact_pixel_match": exact,
                "max_abs_error_m": float(np.nanmax(abs(z - crop.read(1)))),
                "native_pixel_window": list(window.flatten()),
                "original_sha256": file_hash(original.name),
                "jpl_sha256": file_hash(a.jpl_crop),
                "crs_equal": True,
            }
            if not exact:
                raise ValueError("Unexpected JPL crop values")
        for role in ("sample_count", "height_error"):
            with rasterio.open(a.nasa_root / f"npb/{role}.tif") as quality:
                if (quality.crs, quality.transform, quality.shape) != (
                    common,
                    original.transform,
                    original.shape,
                ):
                    raise ValueError("NPB quality raster misregistration")
        # Blockwise audit avoids holding the complete 20 km raster in memory.
        count = sample_zero = finite_error = 0
        lo = float("inf")
        hi = -lo
        with (
            rasterio.open(a.nasa_root / "npb/sample_count.tif") as sc,
            rasterio.open(a.nasa_root / "npb/height_error.tif") as er,
        ):
            for _, w in original.block_windows():
                values = original.read(1, window=w, masked=True).filled(np.nan)
                good = np.isfinite(values)
                count += int(good.sum())
                if good.any():
                    lo = min(lo, float(values[good].min()))
                    hi = max(hi, float(values[good].max()))
                sample_zero += int(((sc.read(1, window=w) == 0) & good).sum())
                finite_error += int((np.isfinite(er.read(1, window=w)) & good).sum())
        verification["full_dtm"] = {
            "valid_pixels": count,
            "min_m": lo,
            "max_m": hi,
            "zero_lola_returns_pixels": sample_zero,
            "finite_error_pixels": finite_error,
            "caution": "Valid DTM includes upstream interpolation; sampling count zero is not a new laser observation",
        }
    paths = {"nasa_npb": a.nasa_root / "npb/dtm.tif"}
    paths.update({f"lupex_{p.name[:3]}": p for p in sorted((a.lupex_root / "DTMs").glob("*.tif"))})
    for name, path in paths.items():
        with rasterio.open(path) as ds:
            products[name] = {
                "path": str(path),
                "sha256": file_hash(path),
                "shape": list(ds.shape),
                "crs_wkt": ds.crs.to_wkt(),
                "bounds_common_m": list(
                    transform_bounds(ds.crs, common, *ds.bounds, densify_pts=41)
                ),
                "native_spacing_xy_m": list(ds.res),
                "source_class": "upstream_interpolated_laser_surface"
                if name == "nasa_npb"
                else "photometric_reconstruction",
                "effective_resolution_m": None,
            }
    parent = {k: k for k in products}

    def root(k):
        while parent[k] != k:
            k = parent[k]
        return k

    def join(a, b):
        parent[root(b)] = root(a)

    for x, y in itertools.combinations(products, 2):
        a0, b0, a1, b1 = products[x]["bounds_common_m"]
        c0, d0, c1, d1 = products[y]["bounds_common_m"]
        overlap = max(0, min(a1, c1) - max(a0, c0)) * max(0, min(b1, d1) - max(b0, d0))
        if overlap > 0:
            join(x, y)
            relations.append(
                {"a": x, "b": y, "type": "conservative_bounds_overlap", "overlap_m2": overlap}
            )
    for x, y in [("lupex_gr1", "lupex_gr2"), ("lupex_mp1", "lupex_mp2")]:
        join(x, y)
        relations.append(
            {
                "a": x,
                "b": y,
                "type": "conservative_shared_processing_group",
                "basis": "Retain audited LUPEX v1 provenance grouping",
            }
        )
    groups = {}
    for k in products:
        groups.setdefault(root(k), []).append(k)
    groups = sorted(groups.values())
    folds = []
    candidates = [g for g in groups if "nasa_npb" not in g]
    for ti, test in enumerate(candidates):
        validation = candidates[(ti + 1) % len(candidates)]
        development = sorted(k for k in products if k not in test + validation)
        folds.append(
            {
                "id": f"heldout_{test[0]}",
                "development": development,
                "validation": validation,
                "test": test,
            }
        )
    manifest = {
        "schema": "lunar_cross_source_groups_v2",
        "products": products,
        "groups": groups,
        "relations": relations,
        "folds": folds,
        "common_crs_wkt": common.to_wkt(),
        "npb_role": "development_only",
        "exclusions": [],
        "split_basis": "geographic/processing groups frozen before new window selection",
        "limitations": [
            "Lunar frame absolute registration uncertainty is not independently measured",
            "Shared reference products imply correlated processing errors even across disjoint geographic groups",
            "LUPEX products have undergone descriptive auditing; these are geographic holdouts, not untouched data",
        ],
    }
    for fold in folds:
        parts = [set(fold[k]) for k in ("development", "validation", "test")]
        assert not any(x & y for x, y in itertools.combinations(parts, 2))
        assert set.union(*parts) == set(products)
        assert all(sum(bool(set(g) & s) for s in parts) == 1 for g in groups)
    (a.output / "source_verification.json").write_text(
        json.dumps(verification, indent=2, allow_nan=False)
    )
    (a.output / "geographic_split_v2.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False)
    )
    print(json.dumps({"verification": verification, "groups": groups}))


if __name__ == "__main__":
    main()
