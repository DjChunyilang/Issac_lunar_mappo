"""Immutable, georeferenced raster scene packs for real-data training maps."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio import Affine
from rasterio.windows import Window
from scipy.ndimage import map_coordinates

from .enhancement import enhance
from .metrics import local_metrics

SCHEMA = "lunar_training_terrain_v2"


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(4 * 1024**2), b""):
            h.update(part)
    return h.hexdigest()


def strict_sample(values, rows, cols, *, order=1):
    """Interpolate only when all nonzero-weight contributors are valid."""
    finite = np.isfinite(values)
    coords = np.stack((rows, cols))
    support = map_coordinates(
        finite.astype(np.float32), coords, order=order, mode="constant", cval=0, prefilter=False
    )
    out = map_coordinates(
        np.where(finite, values, 0), coords, order=order, mode="constant", cval=0, prefilter=False
    )
    inside = (
        (rows >= 0) & (cols >= 0) & (rows <= values.shape[0] - 1) & (cols <= values.shape[1] - 1)
    )
    good = inside & (support >= 1 - 1e-6)
    return np.where(good, out, np.nan).astype(np.float32), good


def build_scene(
    dtm,
    output,
    *,
    source_id,
    geographic_group,
    pixel_window,
    processing_spacing_m,
    metric_baseline_m,
    source_half_width,
    quality_files=None,
    error_semantics="unavailable",
    enhancement=None,
    source_url=None,
    role="development",
):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Scene packs are immutable: {output}")
    if role not in ("development", "validation", "test"):
        raise ValueError("Unknown split role")
    col, row, width, height = map(int, pixel_window)
    if (
        list(pixel_window) != [col, row, width, height]
        or min(width, height) <= 0
        or min(row, col) < 0
    ):
        raise ValueError("Expected positive integer native window")
    spacing = float(processing_spacing_m)
    if spacing <= 0 or metric_baseline_m < 2 * spacing or source_half_width < 1:
        raise ValueError("Invalid processing or analysis scale")
    quality_files = quality_files or {}
    with rasterio.open(dtm) as ds:
        if not ds.crs or not ds.crs.is_projected or ds.crs.linear_units != "metre":
            raise ValueError("Expected projected metre coordinates")
        if ds.transform.b or ds.transform.d or ds.transform.a <= 0 or ds.transform.e >= 0:
            raise ValueError("Expected north-up axis-aligned raster")
        if col + width > ds.width or row + height > ds.height:
            raise ValueError("Window outside product")
        dx, dy = ds.res
        target_w, target_h = round(width * dx / spacing), round(height * dy / spacing)
        if target_w < 3 or target_h < 3:
            raise ValueError("Scene needs at least three samples per axis")
        actual_dx, actual_dy = width * dx / target_w, height * dy / target_h
        transform = rasterio.windows.transform(
            Window(col, row, width, height), ds.transform
        ) * Affine.scale(width / target_w, height / target_h)
        # Native halo supports strict interpolation and original-baseline metrics.
        halo = source_half_width + 2
        c0, r0 = max(0, col - halo), max(0, row - halo)
        c1, r1 = min(ds.width, col + width + halo), min(ds.height, row + height + halo)
        native = (
            ds.read(1, window=Window(c0, r0, c1 - c0, r1 - r0), masked=True)
            .astype(np.float64)
            .filled(np.nan)
        )
        native = native * ds.scales[0] + ds.offsets[0]
        if not np.any(np.isfinite(native)):
            raise ValueError("Window has no valid heights")
        z_origin = float(np.nanmedian(native))
        native -= z_origin
        rr, cc = np.meshgrid(
            (np.arange(target_h) + 0.5) * height / target_h + row - r0 - 0.5,
            (np.arange(target_w) + 0.5) * width / target_w + col - c0 - 0.5,
            indexing="ij",
        )
        base, valid = strict_sample(native, rr, cc)
        native_metrics = local_metrics(native, np.isfinite(native), dx, dy, source_half_width)
        source_features = np.stack(
            [
                strict_sample(native_metrics[k], rr, cc)[0]
                for k in ("slope_x", "slope_y", "roughness")
            ]
        )
        crs_wkt = ds.crs.to_wkt()
        origin_xy = list(transform * (target_w / 2, target_h / 2))
        native_transform = list(ds.transform)[:6]
    qualities = {}
    refs = {"dtm": {"path": str(Path(dtm).resolve()), "sha256": file_hash(dtm), "url": source_url}}
    yy, xx = np.meshgrid(np.arange(target_h) + 0.5, np.arange(target_w) + 0.5, indexing="ij")
    projected_x, projected_y = transform * (xx, yy)
    for name, path in quality_files.items():
        with rasterio.open(path) as qs:
            if qs.crs.to_wkt() != crs_wkt:
                from pyproj import Transformer

                xq, yq = Transformer.from_crs(crs_wkt, qs.crs, always_xy=True).transform(
                    projected_x, projected_y
                )
            else:
                xq, yq = projected_x, projected_y
            qc, qr = ~qs.transform * (xq, yq)
            # Read only intersecting region. Quality values are attributes, not
            # independent draws or an uncertainty propagation operation.
            qc, qr = qc - 0.5, qr - 0.5
            q0 = max(0, int(np.floor(qc.min())) - 1)
            p0 = max(0, int(np.floor(qr.min())) - 1)
            q1 = min(qs.width, int(np.ceil(qc.max())) + 2)
            p1 = min(qs.height, int(np.ceil(qr.max())) + 2)
            if q0 >= q1 or p0 >= p1:
                qualities[name] = np.full(base.shape, np.nan, np.float32)
            else:
                v = (
                    qs.read(1, window=Window(q0, p0, q1 - q0, p1 - p0), masked=True)
                    .astype(float)
                    .filled(np.nan)
                )
                v = v * qs.scales[0] + qs.offsets[0]
                v[v < 0] = np.nan
                qualities[name] = strict_sample(v, qr - p0, qc - q0, order=0)[0]
        refs[name] = {"path": str(Path(path).resolve()), "sha256": file_hash(path)}
    final = base.copy()
    layers = {}
    catalog = {}
    if enhancement is not None:
        final, layers, catalog = enhance(base, valid, (actual_dx, actual_dy), enhancement)
    half = max(1, int(np.ceil(metric_baseline_m / (2 * min(actual_dx, actual_dy)))))
    metrics = local_metrics(final, valid, actual_dx, actual_dy, half)
    features = np.stack(
        [final, metrics["slope_x"], metrics["slope_y"], metrics["roughness"]]
    ).astype(np.float32)
    arrays = {
        "features": features,
        "base_height": base,
        "height_valid": valid,
        "metrics_valid": metrics["valid"],
        "source_features": source_features,
    }
    arrays.update({f"quality_{k}": v for k, v in qualities.items()})
    arrays.update({f"layer_{k}": v for k, v in layers.items()})
    output.mkdir(parents=True)
    files = {}
    for name, array in arrays.items():
        path = output / f"{name}.npy"
        np.save(path, array, allow_pickle=False)
        files[name] = {
            "file": path.name,
            "sha256": file_hash(path),
            "shape": list(array.shape),
            "dtype": str(array.dtype),
        }
    with rasterio.open(
        output / "height.tif",
        "w",
        driver="GTiff",
        height=target_h,
        width=target_w,
        count=1,
        dtype="float32",
        crs=crs_wkt,
        transform=transform,
        nodata=np.nan,
        compress="deflate",
    ) as dst:
        dst.write(final, 1)
        dst.update_tags(
            height_origin_m=z_origin,
            representation="local_height; add height_origin_m for source datum",
        )
    files["height_geotiff"] = {"file": "height.tif", "sha256": file_hash(output / "height.tif")}
    metadata = {
        "schema": SCHEMA,
        "source_id": source_id,
        "geographic_group": geographic_group,
        "role": role,
        "sources": refs,
        "pixel_window": [col, row, width, height],
        "crs_wkt": crs_wkt,
        "native_transform": native_transform,
        "transform": list(transform)[:6],
        "origin_projected_xy_m": origin_xy,
        "height_origin_m": z_origin,
        "native_spacing_xy_m": [dx, dy],
        "processing_spacing_xy_m": [actual_dx, actual_dy],
        "metric_baseline_xy_m": [2 * half * actual_dx, 2 * half * actual_dy],
        "source_metric_baseline_xy_m": [2 * source_half_width * dx, 2 * source_half_width * dy],
        "effective_support_m": None,
        "source_support_caveat": "native grid spacing is not effective measurement resolution",
        "shape": [target_h, target_w],
        "extent_xy_m": [width * dx, height * dy],
        "height_interpolation": "bilinear_strict_support",
        "quality_sampling": "nearest_source_attribute_no_uncertainty_propagation",
        "error_semantics": error_semantics,
        "classification": "real_plus_hypothesis" if enhancement else "real_base",
        "enhancement": enhancement,
        "feature_catalog": catalog,
        "files": files,
        "feature_names": [
            "relative_height_m",
            "simulation_slope_x",
            "simulation_slope_y",
            "simulation_detrended_rms_m",
        ],
    }
    if enhancement:
        residual = final - base
        from rasterio.warp import Resampling, reproject

        reduced = np.full((height, width), np.nan, np.float32)
        reproject(
            residual,
            reduced,
            src_transform=transform,
            src_crs=crs_wkt,
            dst_transform=Affine(*native_transform) * Affine.translation(col, row),
            dst_crs=crs_wkt,
            src_nodata=np.nan,
            dst_nodata=np.nan,
            resampling=Resampling.average,
        )
        native_good = np.isfinite(reduced)
        metadata["enhancement_diagnostics"] = {
            "native_area_mean_residual_rms_m": float(np.sqrt(np.mean(reduced[native_good] ** 2))),
            "native_area_mean_residual_max_abs_m": float(np.max(np.abs(reduced[native_good]))),
            "residual_rms_m": float(np.sqrt(np.mean(residual[valid] ** 2))),
            "residual_max_abs_m": float(np.max(np.abs(residual[valid]))),
            "low_frequency_change": "not asserted zero; retain base and inspect source-scale downsample differences",
        }
    (output / "scene.json").write_text(json.dumps(metadata, indent=2, allow_nan=False))
    return output / "scene.json"


class ScenePack:
    def __init__(self, manifest, *, verify=True):
        self.path = Path(manifest)
        self.meta = json.loads(self.path.read_text())
        if self.meta["schema"] != SCHEMA:
            raise ValueError("Unsupported scene schema")
        self.arrays = {}
        for name, item in self.meta["files"].items():
            path = self.path.parent / item["file"]
            if path.parent.resolve() != self.path.parent.resolve():
                raise ValueError("Scene file must be local to pack")
            if verify and file_hash(path) != item["sha256"]:
                raise ValueError(f"Scene checksum mismatch: {name}")
            if path.suffix == ".npy":
                self.arrays[name] = np.load(path, mmap_mode="r", allow_pickle=False)
        self.transform = Affine(*self.meta["transform"])

    def local_to_projected(self, xy):
        return np.asarray(xy) + self.meta["origin_projected_xy_m"]

    def projected_to_local(self, xy):
        return np.asarray(xy) - self.meta["origin_projected_xy_m"]

    def checkpoint_contract(self):
        return {
            "terrain_feature_schema": SCHEMA,
            "scene_manifest_sha256": file_hash(self.path),
            "geographic_group": self.meta["geographic_group"],
        }
