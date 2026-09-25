"""NASA VIPER engineering distributions and static ejecta-conditioned placement.

Adapted from NASA Ames synthterrain (Apache-2.0), pinned by UPSTREAM_COMMIT.
Copyright 2024-2025 United States Government, represented by the Administrator
of NASA. See third_party/synthterrain/{LICENSE,NOTICE.md} for attribution and
changes. These are hypothesis populations, not NPB observations.
"""

from importlib.metadata import version

import numpy as np
from scipy.ndimage import distance_transform_edt

from .enhancement import _patch, crater_profile

UPSTREAM_COMMIT = "e10ba36e9aa5b923bc7e6746ef089859af9ccaf8"
UPSTREAM = "https://github.com/NeoGeographyToolkit/synthterrain"
MODELS = {"craters": (0.029174, 1.92), "rocks": (0.0003, 2.482)}


def csfd(kind, diameter):
    """Cumulative count per m² for diameter in m (craters <= 80 m only)."""
    d = np.asarray(diameter, dtype=float)
    if np.any(~np.isfinite(d)) or np.any(d <= 0) or (kind == "craters" and np.any(d > 80)):
        raise ValueError("Diameter outside supported positive finite model range")
    k, exponent = MODELS[kind]
    return k * d ** -exponent


def check_interval(kind, low, high):
    if not 0 < low < high:
        raise ValueError("Require 0 < diameter_min < diameter_max")
    csfd(kind, [low, high])


def population_count(kind, area, low, high):
    check_interval(kind, low, high)
    if not np.isfinite(area) or area <= 0:
        raise ValueError("No positive finite generation area")
    return int(np.floor(area * csfd(kind, low)) - np.floor(area * csfd(kind, high)))


def diameter_cdf(kind, d, low, high):
    check_interval(kind, low, high)
    x = np.clip(d, low, high)
    return (csfd(kind, low) - csfd(kind, x)) / (csfd(kind, low) - csfd(kind, high))


def sample_diameters(kind, count, low, high, rng):
    check_interval(kind, low, high)
    exponent = MODELS[kind][1]
    u = rng.random(count)
    return (low ** -exponent + u * (high ** -exponent - low ** -exponent)) ** (-1 / exponent)


def eligible_centres(valid, spacing, support_m):
    """Conservative whole-cell eligibility, including jitter and invalid-cell area."""
    dx, dy = spacing
    if min(dx, dy) <= 0 or support_m < 0:
        raise ValueError("Invalid support/grid spacing")
    distance = distance_transform_edt(np.pad(valid, 1), sampling=(dy, dx))[1:-1, 1:-1]
    # Two half-diagonals account for centre jitter and the closest invalid cell.
    return valid & (distance > support_m + np.hypot(dx, dy))


def normalize_probability(values, mask, *, fallback=False):
    p = np.where(mask, values, 0).astype(np.float64)
    if not np.isfinite(p).all() or np.any(p < 0):
        raise ValueError("Invalid probability weights")
    total = float(p.sum())
    used_fallback = total == 0
    if total == 0:
        if not fallback or not mask.any():
            raise ValueError("Empty probability support")
        p = mask.astype(float)
        total = float(p.sum())
    return p / total, used_fallback


def sample_positions(probability, count, spacing, rng):
    """Lower-left metre coordinates; north-up rows, unbiased pixel jitter."""
    dx, dy = spacing
    h, w = probability.shape
    p = np.asarray(probability, dtype=float)
    if not np.isfinite(p).all() or np.any(p < 0) or not np.isclose(p.sum(), 1):
        raise ValueError("Position probabilities must sum to one")
    cells = rng.choice(p.size, count, p=(p / p.sum()).ravel())
    row, col = np.divmod(cells, w)
    jitter = rng.random((count, 2))
    return np.column_stack(((col + jitter[:, 0]) * dx, (h - row - jitter[:, 1]) * dy))


def background_probability(shape, mask, seed):
    from opensimplex import OpenSimplex

    if min(shape) < 10:
        weights = np.ones(shape)
    else:
        noise = OpenSimplex(seed).noise2array(np.arange(shape[1]), np.arange(shape[0]))
        weights = (noise + 1) / 2
        weights[weights <= 0.5] = 0
    return normalize_probability(weights, mask, fallback=True)


def ejecta_probability(shape, spacing, craters):
    """NASA radial law, sampled in metres at true centres; no age attenuation.

    Stable catalogue order replaces age ordering; exterior contributions add,
    interiors overwrite, as upstream. Physical evaluation avoids upstream's
    linspace/window discretization and supports rectangular cells.
    """
    result = np.zeros(shape, dtype=np.float64)
    for item in craters:
        d = item["diameter_m"]
        patch = _patch(shape, spacing, item["x_m"], item["y_m"], d)
        if patch is None:
            continue
        slices, xx, yy = patch
        radius = np.hypot(xx, yy) / (d / 2)
        outer = np.where((radius > 1) & (radius <= 2), np.exp(-radius / 0.7), 0)
        view = result[slices]
        view += outer
        view[radius <= 1] = 0.05 * outer.max()
    return result


def generate_catalog(valid, spacing, spec):
    """Generate catalogues/probabilities independently of height rasterization."""
    if spec.get("classification") != "hypothesis" or not spec.get("parameter_evidence"):
        raise ValueError("NASA enhancement requires declared hypothesis evidence")
    if spec.get("fractal", {}).get("enabled", False):
        raise ValueError("nasa_sfd_v2 does not include a fractal residual")
    if not np.allclose(spacing, spec["probability_spacing_m"], rtol=0, atol=1e-9):
        raise ValueError("v2 requires the fixed physical probability grid as processing grid")
    streams = np.random.SeedSequence(spec["seed"]).spawn(5)
    rngs = [np.random.default_rng(s) for s in streams]
    catalog = {"craters": [], "rocks": []}
    probabilities, stats = {}, {}
    for index, kind in enumerate(("craters", "rocks")):
        cfg = spec[kind]
        if not cfg["enabled"]:
            continue
        low, high = cfg["diameter_min_m"], cfg["diameter_max_m"]
        check_interval(kind, low, high)
        if high > spec["unresolved_diameter_limit_m"]:
            raise ValueError("Diameter exceeds declared enhancement limit")
        if low < 5 * max(spacing) - 1e-9:
            raise ValueError("Minimum feature needs at least five processing cells")
        support = high if kind == "craters" else high / 2
        eligible = eligible_centres(valid, spacing, support)
        area = float(eligible.sum() * spacing[0] * spacing[1])
        count = population_count(kind, area, low, high)
        if count > spec.get("max_features", 10000):
            raise ValueError("Feature count exceeds resource bound; do not reduce density silently")
        diameters = sample_diameters(kind, count, low, high, rngs[index * 2])
        if kind == "craters":
            probability, _ = normalize_probability(eligible, eligible)
        else:
            bg_seed = int(rngs[4].integers(0, 2**63 - 1))
            background, bg_fallback = background_probability(valid.shape, eligible, bg_seed)
            ejecta = ejecta_probability(valid.shape, spacing, catalog["craters"])
            ejecta[~eligible] = 0
            has_ejecta = bool(ejecta.sum() > 0)
            if has_ejecta:
                ejecta, _ = normalize_probability(ejecta, eligible)
                probability = 0.5 * background + 0.5 * ejecta
            else:
                probability = background.copy()
            probabilities.update(background=background, ejecta=ejecta, rocks=probability)
            stats["spatial"] = {
                "background_seed": bg_seed, "background_uniform_fallback": bg_fallback,
                "ejecta_available": has_ejecta, "background_weight": 0.5 if has_ejecta else 1.0,
                "age_decay": "disabled_static", "crater_order": "stable_catalog_id",
                "kernel": "NASA_radial_law_physical_cell_centres_v2",
            }
        positions = sample_positions(probability, count, spacing, rngs[index * 2 + 1])
        for i, (d, (x, y)) in enumerate(zip(diameters, positions)):
            catalog[kind].append({
                "id": i, "diameter_m": float(d), "x_m": float(x), "y_m": float(y),
                "accepted": True, "rejection_reason": None,
            })
        probabilities[f"{kind}_eligible"] = eligible
        stats[kind] = {"area_m2": area, "generated_count": count,
                       "diameter_min_m": low, "diameter_max_m": high,
                       "coefficient_per_m2": MODELS[kind][0], "exponent": MODELS[kind][1]}
    catalog["generation"] = {
        "model": "VIPER_Env_Spec", "method": "nasa_sfd_v2", "upstream": UPSTREAM,
        "upstream_commit": UPSTREAM_COMMIT, "count_rule": "floor(A*C(min))-floor(A*C(max))",
        "classification": "hypothesis_not_regional_observations", "seed": spec["seed"],
        "stream_spawn_keys": [list(s.spawn_key) for s in streams],
        "coordinate_convention": "lower_left_metres_north_up_rows",
        "grid_origin_lower_left_m": [0, 0], "spacing_xy_m": list(spacing),
        "shape": list(valid.shape), "stats": stats,
        "versions": {k: version(k) for k in ("numpy", "scipy", "opensimplex", "numba")},
    }
    return catalog, probabilities


def render_catalog(base, valid, spacing, catalog, spec):
    layers = {k: np.zeros_like(base, dtype=np.float32) for k in ("craters", "rocks")}
    diagnostics = {}
    for kind in ("craters", "rocks"):
        coverage = np.zeros(base.shape, np.uint16)
        cfg = spec[kind]
        for item in catalog[kind]:
            d = item["diameter_m"]
            support = d if kind == "craters" else d / 2
            patch = _patch(base.shape, spacing, item["x_m"], item["y_m"], support)
            if patch is None:
                continue
            slices, xx, yy = patch
            r = np.hypot(xx, yy) / (d / 2)
            footprint = r < (2 if kind == "craters" else 1)
            if not valid[slices][footprint].all():
                raise ValueError("Catalogue footprint intersects invalid data")
            coverage[slices] += footprint
            if kind == "craters":
                depth, rim = cfg["depth_to_diameter"] * d, cfg["rim_to_diameter"] * d
                layers[kind][slices] += crater_profile(r, depth_m=depth, rim_height_m=rim)
                item.update(depth_m=depth, rim_height_m=rim, profile="c1_polynomial_hypothesis_v1")
            else:
                height = cfg["height_to_diameter"] * d
                bump = height * np.sqrt(np.maximum(1 - r * r, 0))
                layers[kind][slices] = np.maximum(layers[kind][slices], bump)
                item.update(height_m=height, profile="upper_half_ellipsoid_height_envelope_v1")
        diagnostics[kind] = {
            "generation_coverage_fraction": float(np.mean(coverage[valid] > 0)),
            "generation_overlap_fraction": float(np.mean(coverage[valid] > 1)),
            "max_overlapping_footprints": int(coverage.max()),
            "visible_feature_count": None,
            "visible_count_note": "Not inferred from overlapping heightfields; catalogue count only",
        }
    final = base.copy()
    for layer in layers.values():
        layer[~valid] = 0
        final += layer
    final[~valid] = np.nan
    catalog["generation"]["geometry_diagnostics"] = diagnostics
    return final, layers


def enhance_nasa(base, valid, spacing, spec):
    catalog, probabilities = generate_catalog(valid, spacing, spec)
    final, layers = render_catalog(base, valid, spacing, catalog, spec)
    return final, layers, catalog, probabilities
