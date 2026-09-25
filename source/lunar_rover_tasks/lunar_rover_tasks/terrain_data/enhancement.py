"""Deterministic, metre-based hypothesis layers inspired by Allan et al. (2019).

These are explicit simulation hypotheses, not NASA's unreleased generator or
recovered centimetre-scale observations. No upstream LuNaSynth code is copied.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.ndimage import gaussian_filter


def diamond_square(shape, *, seed, rms_m, cutoff_m, spacing_xy_m, hurst):
    """Classical diamond-square residual with an explicit physical high pass.

    Classical midpoint averaging is named deliberately: this is not claimed to
    reproduce NASA's modified Catmull-Rom implementation. Borders are generated
    on a larger field, then cropped. Statistical parameters must be declared.
    """
    dx, dy = spacing_xy_m
    if min(dx, dy, cutoff_m) <= 0 or rms_m < 0 or not 0 < hurst < 1:
        raise ValueError("Invalid physical residual parameters")
    side = 2 ** math.ceil(math.log2(max(shape) + 2))
    rng = np.random.default_rng(seed)
    z = np.zeros((side + 1, side + 1), dtype=np.float32)
    step, amplitude = side, 1.0
    while step > 1:
        half = step // 2
        square = (
            z[0:side:step, 0:side:step]
            + z[step : side + 1 : step, 0:side:step]
            + z[0:side:step, step : side + 1 : step]
            + z[step : side + 1 : step, step : side + 1 : step]
        ) * 0.25
        z[half:side:step, half:side:step] = square + rng.normal(0, amplitude, square.shape)
        for row_start, col_start in ((0, half), (half, 0)):
            rows = np.arange(row_start, side + 1, step)
            cols = np.arange(col_start, side + 1, step)
            rr, cc = np.meshgrid(rows, cols, indexing="ij")
            total, count = np.zeros(rr.shape, np.float32), np.zeros(rr.shape, np.float32)
            for dr, dc in ((-half, 0), (half, 0), (0, -half), (0, half)):
                nr, nc = rr + dr, cc + dc
                inside = (nr >= 0) & (nr <= side) & (nc >= 0) & (nc <= side)
                total += np.where(inside, z[nr.clip(0, side), nc.clip(0, side)], 0)
                count += inside
            z[rr, cc] = total / count + rng.normal(0, amplitude, rr.shape)
        step, amplitude = half, amplitude * 2 ** (-hurst)
    r0, c0 = (side - shape[0]) // 2, (side - shape[1]) // 2
    result = z[r0 : r0 + shape[0], c0 : c0 + shape[1]].copy()
    # Remove low spatial frequencies before setting the stated residual RMS.
    result -= gaussian_filter(
        result, sigma=(cutoff_m / (2 * dy), cutoff_m / (2 * dx)), mode="reflect"
    )
    result -= result.mean()
    std = float(result.std())
    if std > 0:
        result *= rms_m / std
    return result


def crater_profile(radius_normalized, *, depth_m, rim_height_m, outer_radius_ratio=2.0):
    """C1 compact polynomial bowl/rim hypothesis, finite at its centre."""
    if depth_m < 0 or rim_height_m < 0 or outer_radius_ratio <= 1:
        raise ValueError("Invalid crater shape")
    r = np.asarray(radius_normalized)
    if np.any(r < 0):
        raise ValueError("Radius must be nonnegative")
    inside = -depth_m * (1 - r * r) ** 2 + rim_height_m * r * r * (2 - r * r)
    t = np.clip((r - 1) / (outer_radius_ratio - 1), 0, 1)
    outside = rim_height_m * (1 - 3 * t * t + 2 * t * t * t)
    return np.where(r <= 1, inside, np.where(r < outer_radius_ratio, outside, 0))


def _patch(shape, spacing, x, y, radius):
    dx, dy = spacing
    h, w = shape
    c0, c1 = max(0, int(np.floor((x - radius) / dx))), min(w, int(np.ceil((x + radius) / dx)))
    r0, r1 = (
        max(0, int(np.floor((h * dy - y - radius) / dy))),
        min(h, int(np.ceil((h * dy - y + radius) / dy))),
    )
    if c0 >= c1 or r0 >= r1:
        return None
    xx = (np.arange(c0, c1) + 0.5) * dx
    yy = (h - np.arange(r0, r1) - 0.5) * dy
    xx, yy = np.meshgrid(xx - x, yy - y)
    return (slice(r0, r1), slice(c0, c1)), xx, yy


def _powerlaw(rng, n, low, high, exponent):
    if low <= 0 or high < low or exponent <= 0:
        raise ValueError("Invalid diameter distribution")
    if low == high:
        return np.full(n, low)
    u = rng.random(n)
    return (low ** (-exponent) + u * (high ** (-exponent) - low ** (-exponent))) ** (-1 / exponent)


def enhance(base, valid, spacing_xy_m, spec):
    """Return separate residual/crater/rock layers plus metre-valued catalogues.

    Source-resolved terrain is never silently replaced. Random generated
    features are constrained below the declared unresolved diameter limit.
    """
    if spec.get("method") == "nasa_sfd_v2":
        from .nasa_sfd import enhance_nasa

        final, layers, catalog, _ = enhance_nasa(base, valid, spacing_xy_m, spec)
        return final, layers, catalog
    if spec.get("method", "legacy_hypothesis_v1") != "legacy_hypothesis_v1":
        raise ValueError("Unknown enhancement method")
    if spec.get("classification") != "hypothesis" or not spec.get("parameter_evidence"):
        raise ValueError("Enhancement requires explicit hypothesis classification and evidence")
    dx, dy = spacing_xy_m
    h, w = base.shape
    layers = {k: np.zeros_like(base, dtype=np.float32) for k in ("fractal", "craters", "rocks")}
    catalog = {"craters": [], "rocks": []}
    if spec.get("fractal", {}).get("enabled", False):
        f = spec["fractal"]
        layers["fractal"] = diamond_square(
            base.shape,
            seed=f["seed"],
            rms_m=f["rms_m"],
            cutoff_m=f["cutoff_m"],
            spacing_xy_m=spacing_xy_m,
            hurst=f["hurst"],
        )
    for kind in ("craters", "rocks"):
        cfg = spec.get(kind, {})
        if not cfg.get("enabled", False):
            continue
        rng = np.random.default_rng(cfg["seed"])
        max_d = float(cfg["diameter_max_m"])
        if max_d > float(spec["unresolved_diameter_limit_m"]):
            raise ValueError("Enhancement would duplicate potentially source-resolved features")
        count = int(rng.poisson(float(cfg["density_per_m2"]) * h * dy * w * dx))
        if count > int(spec.get("max_features", 10000)):
            raise ValueError("Feature count exceeds declared generation resource bound")
        diameters = _powerlaw(
            rng, count, float(cfg["diameter_min_m"]), max_d, float(cfg["size_exponent"])
        )
        for index, diameter in enumerate(diameters):
            x, y = rng.uniform(0, w * dx), rng.uniform(0, h * dy)
            radius = diameter / 2
            support = 2 * radius if kind == "craters" else radius
            patch = _patch(base.shape, spacing_xy_m, x, y, support)
            if patch is None:
                continue
            slices, xx, yy = patch
            distance = np.hypot(xx, yy)
            footprint = distance <= support
            # Quality rejection is recorded; terrain slope is not a rejection rule.
            supported = (
                bool(np.all(valid[slices][footprint]))
                and x >= support
                and y >= support
                and x + support <= w * dx
                and y + support <= h * dy
            )
            item = {
                "id": index,
                "x_m": float(x),
                "y_m": float(y),
                "diameter_m": float(diameter),
                "accepted": bool(supported),
                "rejection_reason": None if supported else "invalid_or_boundary_support",
            }
            catalog[kind].append(item)
            if not supported:
                continue
            if kind == "craters":
                depth = float(cfg["depth_to_diameter"]) * diameter
                rim = float(cfg["rim_to_diameter"]) * diameter
                layers[kind][slices] += crater_profile(
                    distance / radius, depth_m=depth, rim_height_m=rim
                )
                item.update(
                    depth_m=float(depth),
                    rim_height_m=float(rim),
                    profile="c1_polynomial_hypothesis_v1",
                )
            else:
                height = float(cfg["height_to_diameter"]) * diameter
                bump = height * np.sqrt(np.maximum(1 - (distance / radius) ** 2, 0))
                # Heightfield envelope is the single proxy geometry/collision source.
                layers[kind][slices] = np.maximum(layers[kind][slices], bump)
                item.update(
                    height_m=float(height), profile="upper_half_ellipsoid_height_envelope_v1"
                )
    final = base.copy()
    for value in layers.values():
        value[~valid] = 0
        final += value
    final[~valid] = np.nan
    return final, layers, catalog
