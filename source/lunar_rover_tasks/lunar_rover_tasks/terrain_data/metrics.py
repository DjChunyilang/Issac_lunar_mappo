"""Physical-baseline terrain descriptors; missing samples are never filled for analysis."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter


def summary(values: np.ndarray) -> dict:
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if not values.size:
        return {"count": 0}
    q = np.percentile(values, [0, 5, 25, 50, 75, 95, 100])
    return dict(count=int(values.size), mean=float(values.mean()), std=float(values.std()),
                **dict(zip(("min", "p05", "p25", "p50", "p75", "p95", "max"),
                           map(float, q), strict=True)))


def detrend_plane(z: np.ndarray, valid: np.ndarray, dx: float, dy: float):
    """Least-squares plane on ALL valid cells, y north/up for a north-up raster."""
    valid = valid & np.isfinite(z)
    if np.count_nonzero(valid) < 3:
        raise ValueError("A plane requires at least three valid samples")
    yy, xx = np.indices(z.shape, dtype=np.float64)
    xx = (xx - (z.shape[1] - 1) / 2) * dx
    yy = ((z.shape[0] - 1) / 2 - yy) * dy
    x, y, h = xx[valid], yy[valid], z[valid].astype(np.float64)
    offset = float(h.mean())
    h = h - offset
    a = np.array([[x @ x, x @ y, x.sum()], [x @ y, y @ y, y.sum()],
                  [x.sum(), y.sum(), float(h.size)]])
    b = np.array([x @ h, y @ h, h.sum()])
    coeff = np.linalg.solve(a, b)
    residual = np.where(valid, z - offset - coeff[0] * xx - coeff[1] * yy - coeff[2],
                        np.nan)
    return residual, [float(coeff[0]), float(coeff[1]), float(coeff[2] + offset)]


def local_metrics(z: np.ndarray, valid: np.ndarray, dx: float, dy: float,
                  half_width: int) -> dict[str, np.ndarray]:
    """Square-window plane gradient and plane-residual RMS in physical metres.

    A window of 2h+1 native samples spans 2h*spacing between outer sample centres.
    All samples must be valid. Zero padding is only an implementation detail;
    incomplete windows, including array edges, are invalid in every output.
    """
    if half_width < 1 or int(half_width) != half_width or min(dx, dy) <= 0:
        raise ValueError("Positive spacing and integer half_width >= 1 required")
    valid = np.asarray(valid, bool) & np.isfinite(z)
    n = 2 * half_width + 1
    full = uniform_filter(valid.astype(float), n, mode="constant") > 1 - 1e-10
    if not np.any(full):
        empty = np.full(z.shape, np.nan)
        return dict(slope_x=empty.copy(), slope_y=empty.copy(), roughness=empty.copy(),
                    valid=full)
    # Removing a whole-raster plane reduces cancellation in the second moments.
    residual, plane = detrend_plane(z, valid, dx, dy)
    r = np.where(valid, residual, 0.0)
    yy, xx = np.indices(z.shape, dtype=float)
    xx *= dx
    yy *= -dy
    mean = uniform_filter(r, n, mode="constant")
    cx = uniform_filter(xx * r, n, mode="constant") - xx * mean
    cy = uniform_filter(yy * r, n, mode="constant") - yy * mean
    vx = half_width * (half_width + 1) * dx * dx / 3
    vy = half_width * (half_width + 1) * dy * dy / 3
    variance = uniform_filter(r * r, n, mode="constant") - mean * mean
    rms = np.sqrt(np.maximum(variance - cx * cx / vx - cy * cy / vy, 0.0))
    return dict(slope_x=np.where(full, plane[0] + cx / vx, np.nan),
                slope_y=np.where(full, plane[1] + cy / vy, np.nan),
                roughness=np.where(full, rms, np.nan), valid=full)


def audit_half_widths(shape: tuple[int, int]) -> list[int]:
    """Dyadic descriptive scales with at least four complete widths across the extent.

    This is an audit sampling design, NOT an assertion of effective resolution.
    """
    widths = []
    h = 1
    while 4 * (2 * h + 1) <= min(shape):
        widths.append(h)
        h *= 2
    return widths


def spatial_structure(z: np.ndarray, valid: np.ndarray, dx: float, dy: float) -> dict:
    residual, plane = detrend_plane(z, valid, dx, dy)
    curves = {}
    for axis, spacing, name in ((1, dx, "east"), (0, dy, "north")):
        curve, lag = [], 1
        while lag <= z.shape[axis] // 2:
            if axis == 1:
                a, b = residual[:, :-lag], residual[:, lag:]
            else:
                a, b = residual[:-lag], residual[lag:]
            good = np.isfinite(a) & np.isfinite(b)
            av, bv = a[good], b[good]
            if av.size < 3:
                break
            denom = av.std() * bv.std()
            corr = float(np.mean((av - av.mean()) * (bv - bv.mean())) / denom) if denom else None
            # Use original heights for slope-baseline diagnostics, not the detrended plane.
            original_delta = (z[:, :-lag] - z[:, lag:] if axis == 1
                              else z[:-lag] - z[lag:])[good]
            slope_angles = np.degrees(np.arctan(original_delta / (lag * spacing)))
            curve.append(dict(lag_m=lag * spacing, pairs=int(av.size), correlation=corr,
                              rms_bidirectional_slope_deg=float(np.sqrt(np.mean(slope_angles**2))),
                              semivariance_m2=float(np.mean((av - bv) ** 2) / 2)))
            lag *= 2
        crossing = next((p["lag_m"] for p in curve if p["correlation"] is not None
                         and p["correlation"] <= np.exp(-1)), None)
        curves[name] = dict(curve=curve, first_sampled_e_folding_m=crossing,
                            censored=crossing is None)
    return dict(global_plane_gradient_xy=plane[:2], directions=curves,
                caveat="Directional descriptive correlation, not proof of spatial independence")
