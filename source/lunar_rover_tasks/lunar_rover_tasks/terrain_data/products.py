"""Immutable-product ingestion and coordinate-aware LUPEX catalogue construction."""

from __future__ import annotations

import hashlib
import json
import shutil
import stat
import zipfile
from pathlib import Path

import numpy as np
import rasterio
from pyproj import CRS, Proj, Transformer
from rasterio.features import geometry_mask
from rasterio.warp import Resampling, reproject
import shapefile

from .metrics import summary

REGIONS = ("cr1", "gr1", "gr2", "lp1", "mp1", "mp2")
ARCHIVE_MD5 = "4ad67232e49b502897ecff197336da29"


def checksum(path: Path, algorithm: str = "sha256") -> str:
    h = hashlib.new(algorithm)
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def extract_verified(archive: Path, destination: Path) -> dict:
    """Verify original, reject path traversal/symlinks, never overwrite altered products."""
    if checksum(archive, "md5") != ARCHIVE_MD5:
        raise ValueError("LUPEX original archive MD5 mismatch")
    destination.mkdir(parents=True, exist_ok=True)
    records = []
    with zipfile.ZipFile(archive) as bundle:
        for info in sorted(bundle.infolist(), key=lambda item: item.filename):
            path = destination / info.filename
            if not path.resolve().is_relative_to(destination.resolve()):
                raise ValueError("Unsafe archive path")
            if stat.S_ISLNK(info.external_attr >> 16):
                raise ValueError("Archive symlinks are not supported")
            if info.is_dir():
                path.mkdir(parents=True, exist_ok=True)
                continue
            with bundle.open(info) as source:
                digest = hashlib.sha256(source.read()).hexdigest()
            if path.exists():
                if checksum(path) != digest:
                    raise ValueError(f"Existing product has changed: {path}")
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(info) as source, path.open("xb") as target:
                    shutil.copyfileobj(source, target)
                path.chmod(0o444)
            records.append(dict(path=info.filename, bytes=info.file_size, sha256=digest))
    return dict(source_url="https://zenodo.org/records/17153447", archive_md5=ARCHIVE_MD5,
                archive_sha256=checksum(archive), files=records)


def roi_catalogue(root: Path) -> tuple[dict, CRS]:
    reader = shapefile.Reader(str(root / "ROIs/ROIs.shp"))
    crs = CRS.from_wkt((root / "ROIs/ROIs.prj").read_text())
    result = {}
    for record in reader.iterShapeRecords():
        attrs = record.record.as_dict()
        names = [str(value).lower() for value in attrs.values()
                 if str(value).lower() in REGIONS]
        if len(names) != 1 or names[0] in result:
            raise ValueError(f"Ambiguous ROI: {attrs}")
        result[names[0]] = dict(geometry=record.shape.__geo_interface__, attributes=attrs)
    if set(result) != set(REGIONS):
        raise ValueError("Expected six distinct LUPEX ROIs")
    return result, crs


def product_paths(root: Path, region: str) -> dict[str, Path]:
    if region not in REGIONS:
        raise ValueError(f"Unknown LUPEX region: {region}")
    return dict(dtm=root / f"DTMs/{region}_roi_sfs_1m-DEM.tif",
                ortho=root / f"orthomosaics/{region}_roi_sfs_1m-ORTHO.tif",
                error=root / f"uncertainties/{region}_sfs-height-error-final.tif")


def raster_metadata(path: Path) -> dict:
    with rasterio.open(path) as ds:
        z = ds.read(1, masked=True)
        valid = ~np.ma.getmaskarray(z) & np.isfinite(z.data)
        crs = CRS(ds.crs)
        return dict(shape=list(ds.shape), spacing_m=list(ds.res), bounds=list(ds.bounds),
                    transform=list(ds.transform)[:6], crs_wkt=crs.to_wkt(),
                    horizontal_unit=crs.axis_info[0].unit_name,
                    vertical_unit_tag=ds.units[0], dtype=ds.dtypes[0], nodata=ds.nodata,
                    scales=list(ds.scales), offsets=list(ds.offsets),
                    valid_cells=int(valid.sum()), missing_cells=int((~valid).sum()),
                    native_values=summary(z.data[valid]),
                    explicit_vertical_crs_available=False,
                    vertical_unit_evidence="paper: DTM and height-error in metres; ortho radiometry",
                    sha256=checksum(path))


def read_region(root: Path, region: str, roi: dict) -> dict:
    paths = product_paths(root, region)
    with rasterio.open(paths["dtm"]) as ds:
        z = ds.read(1, masked=True)
        valid = ~np.ma.getmaskarray(z) & np.isfinite(z.data)
        inside = geometry_mask([roi["geometry"]], ds.shape, ds.transform, invert=True)
        # Missing source cells remain NaN. No display fill or extrapolation is used.
        error = np.full(ds.shape, np.nan, dtype=np.float32)
        with rasterio.open(paths["error"]) as source:
            reproject(rasterio.band(source, 1), error, src_transform=source.transform,
                      src_crs=source.crs, src_nodata=source.nodata,
                      dst_transform=ds.transform, dst_crs=ds.crs, dst_nodata=np.nan,
                      resampling=Resampling.nearest)
        return dict(height=np.where(valid, z.data, np.nan), valid=valid & inside,
                    raw_valid=valid, inside_roi=inside, error=error, transform=ds.transform,
                    crs=ds.crs, spacing=ds.res)


def geographic_origin(crs_wkt: str, bounds: list[float]) -> dict:
    crs = CRS.from_wkt(crs_wkt)
    origin = [(bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2]
    to_lonlat = Transformer.from_crs(crs, crs.geodetic_crs, always_xy=True)
    lon, lat = to_lonlat.transform(*origin)
    factors = Proj(crs).get_factors(lon, lat)
    return dict(projected_xy_m=origin, longitude_deg=lon, latitude_deg=lat,
                local_axes="x=projected easting - origin_x; y=projected northing - origin_y",
                inverse="projected_xy = local_xy + origin_xy; inverse projection to lunar lon/lat",
                datum_radius_m=crs.ellipsoid.semi_major_metre,
                projection_scale_at_origin=factors.meridional_scale,
                distance_note="Local coordinates are projected metres, not exact surface geodesics; "
                              "retain projection and scale factor for mission-distance calculations")
