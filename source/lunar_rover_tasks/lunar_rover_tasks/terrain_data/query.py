"""On-demand source-grid queries with explicit scale, error and feature contracts."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pyproj import CRS, Transformer
import rasterio
from rasterio.windows import Window

from . import ERROR_SEMANTICS, FEATURE_SCHEMA


class RasterBlocks:
    """Bounded LRU of native raster cells. Cache density cannot create observations."""

    def __init__(self, path: Path, max_bytes: int = 16 * 1024**2, block_size: int = 128):
        self.dataset = rasterio.open(path)
        if block_size < 1 or max_bytes < block_size**2 * 8:
            self.dataset.close()
            raise ValueError("Cache must hold at least one float64 block")
        self.max_bytes, self.block_size = max_bytes, block_size
        self.blocks = OrderedDict()
        self.bytes = 0

    def cells(self, rows, cols):
        rows, cols = np.broadcast_arrays(np.asarray(rows, int), np.asarray(cols, int))
        result = np.full(rows.shape, np.nan)
        inside = ((rows >= 0) & (cols >= 0) & (rows < self.dataset.height)
                  & (cols < self.dataset.width))
        keys = np.stack((rows[inside] // self.block_size, cols[inside] // self.block_size), -1)
        for br, bc in np.unique(keys, axis=0):
            key = (int(br), int(bc))
            if key not in self.blocks:
                window = Window(bc*self.block_size, br*self.block_size,
                                min(self.block_size, self.dataset.width-bc*self.block_size),
                                min(self.block_size, self.dataset.height-br*self.block_size))
                data = self.dataset.read(1, window=window, masked=True).astype(float).filled(np.nan)
                data = data * self.dataset.scales[0] + self.dataset.offsets[0]
                while self.bytes + data.nbytes > self.max_bytes and self.blocks:
                    self.bytes -= self.blocks.popitem(last=False)[1].nbytes
                self.blocks[key] = data
                self.bytes += data.nbytes
            self.blocks.move_to_end(key)
            selected = inside & (rows // self.block_size == br) & (cols // self.block_size == bc)
            result[selected] = self.blocks[key][rows[selected] % self.block_size,
                                                cols[selected] % self.block_size]
        return result

    def close(self):
        self.blocks.clear()
        self.bytes = 0
        self.dataset.close()


@dataclass(frozen=True)
class TerrainSample:
    height_m: float
    slope_xy: tuple[float, float]
    detrended_rms_m: float
    height_valid: bool
    metrics_valid: bool
    error_valid: bool
    height_sensitivity_m: float
    support_max_sensitivity_m: float
    metric_center_local_xy_m: tuple[float, float]
    baseline_xy_m: tuple[float, float]
    native_spacing_xy_m: tuple[float, float]
    processing_spacing_xy_m: tuple[float, float]
    source_id: str
    height_interpolation: str
    schema: str = FEATURE_SCHEMA
    error_semantics: str = ERROR_SEMANTICS
    effective_resolution_m: None = None
    measurement_class: str = "orbital_multiview_sfs_reconstruction"
    vehicle_safety: str = "unknown"


class RasterTerrain:
    """Read raw products; query positions are projected-local metres, never lon/lat."""

    def __init__(self, dtm: Path, error: Path, *, source_id: str, origin_xy=None,
                 max_cache_bytes: int = 32 * 1024**2):
        self.dtm = RasterBlocks(dtm, max_cache_bytes // 2)
        self.error = RasterBlocks(error, max_cache_bytes // 2)
        ds = self.dtm.dataset
        crs = CRS(ds.crs)
        if (not crs.is_projected or any(a.unit_conversion_factor != 1 for a in crs.axis_info)
                or ds.transform.b != 0 or ds.transform.d != 0
                or ds.transform.a <= 0 or ds.transform.e >= 0):
            self.close()
            raise ValueError("Expected a north-up projected metre grid")
        self.origin = np.asarray(origin_xy if origin_xy is not None else
                                 [(ds.bounds.left + ds.bounds.right)/2,
                                  (ds.bounds.bottom + ds.bounds.top)/2], dtype=float)
        self.source_id = source_id
        self.spacing = ds.res
        self.to_geo = Transformer.from_crs(crs, crs.geodetic_crs, always_xy=True)
        self.from_geo = Transformer.from_crs(crs.geodetic_crs, crs, always_xy=True)
        self.to_error = Transformer.from_crs(crs, CRS(self.error.dataset.crs), always_xy=True)

    def local_to_geographic(self, xy):
        p = np.asarray(xy) + self.origin
        return np.stack(self.to_geo.transform(p[..., 0], p[..., 1]), -1)

    def geographic_to_local(self, lonlat):
        p = np.asarray(lonlat)
        return np.stack(self.from_geo.transform(p[..., 0], p[..., 1]), -1) - self.origin

    def _errors(self, rows, cols):
        x, y = self.dtm.dataset.transform * (cols + .5, rows + .5)
        ex, ey = self.to_error.transform(x, y)
        ec, er = ~self.error.dataset.transform * (ex, ey)
        return self.error.cells(np.floor(er).astype(int), np.floor(ec).astype(int))

    def query(self, xy, *, half_width: int, interpolation: str = "nearest") -> TerrainSample:
        if half_width < 1 or int(half_width) != half_width:
            raise ValueError("Metrics need integer half_width >= 1 native cells")
        if interpolation not in ("nearest", "bilinear"):
            raise ValueError("Only explicitly labelled nearest/bilinear height queries supported")
        xy = np.asarray(xy, float)
        if xy.shape != (2,) or not np.all(np.isfinite(xy)):
            raise ValueError("Query must be a finite local (x, y) in metres")
        c, r = ~self.dtm.dataset.transform * tuple(xy + self.origin)
        row, col = int(np.floor(r)), int(np.floor(c))
        height = float(self.dtm.cells(row, col))
        if interpolation == "bilinear":
            # Coordinates of cell centres; all four contributors must be valid.
            rr, cc = int(np.floor(r-.5)), int(np.floor(c-.5))
            fr, fc = r-.5-rr, c-.5-cc
            values = self.dtm.cells(np.array([rr, rr, rr+1, rr+1]),
                                    np.array([cc, cc+1, cc, cc+1]))
            weights = np.array([(1-fr)*(1-fc), (1-fr)*fc, fr*(1-fc), fr*fc])
            height = float(values @ weights) if np.all(np.isfinite(values)) else float("nan")
        offsets = np.arange(-half_width, half_width+1)
        yy, xx = np.meshgrid(offsets, offsets, indexing="ij")
        values = self.dtm.cells(row+yy, col+xx)
        errors = self._errors(row+yy, col+xx)
        good = bool(np.isfinite(values).all() and np.isfinite(height))
        dx, dy = self.spacing
        slope, rms = (float("nan"), float("nan")), float("nan")
        if good:
            design = np.stack((xx.ravel()*dx, -yy.ravel()*dy, np.ones(xx.size)), -1)
            relative = values.ravel() - height
            fitted = np.linalg.lstsq(design, relative, rcond=None)[0]
            slope = (float(fitted[0]), float(fitted[1]))
            rms = float(np.sqrt(np.mean((relative - design @ fitted)**2)))
        centre = np.asarray(self.dtm.dataset.transform * (col+.5, row+.5)) - self.origin
        error_good = bool(np.all(np.isfinite(errors)) and np.all(errors >= 0))
        return TerrainSample(height, slope, rms, bool(np.isfinite(height)), good, error_good,
                             float(errors[half_width, half_width]),
                             float(np.max(errors)) if error_good else float("nan"),
                             tuple(centre), (2*half_width*dx, 2*half_width*dy),
                             self.spacing, self.spacing, self.source_id, interpolation)

    def native_patch(self, pixel_window: tuple[int, int, int, int]) -> dict:
        """Native heightfield + validity for downstream planners/simulator adapters.

        No filling, horizontal scaling, or claim that missing cells are physical obstacles.
        The consumer must preserve the validity mask and source metadata.
        """
        col, row, width, height = pixel_window
        if (any(int(v) != v for v in pixel_window) or min(row, col) < 0
                or min(width, height) <= 0 or row+height > self.dtm.dataset.height
                or col+width > self.dtm.dataset.width):
            raise ValueError("Patch must be an integer window wholly inside the source product")
        rr, cc = np.meshgrid(np.arange(row, row+height), np.arange(col, col+width), indexing="ij")
        z = self.dtm.cells(rr, cc)
        error = self._errors(rr, cc)
        transform = rasterio.windows.transform(Window(col, row, width, height),
                                               self.dtm.dataset.transform)
        return dict(height_m=z, valid=np.isfinite(z), height_sensitivity_m=error,
                    transform=transform, crs_wkt=self.dtm.dataset.crs.to_wkt(),
                    pixel_window=pixel_window, provenance=self.provenance())

    def provenance(self) -> dict:
        return dict(source_id=self.source_id, dtm=str(self.dtm.dataset.name),
                    error=str(self.error.dataset.name), schema=FEATURE_SCHEMA,
                    error_semantics=ERROR_SEMANTICS, native_spacing_m=self.spacing,
                    effective_resolution_m=None, synthetic_detail=False,
                    raw_source_retained=True, origin_xy_m=self.origin.tolist())

    def close(self):
        self.dtm.close()
        self.error.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
