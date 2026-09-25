"""Real-terrain contracts tested on analytic rasters, without downloading LUPEX in CI."""

import copy
import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")
pytest.importorskip("pyproj")
pytest.importorskip("shapely")
pytest.importorskip("shapefile")
pytest.importorskip("scipy")
from rasterio.transform import from_origin

from lunar_rover_tasks.terrain_data import FEATURE_SCHEMA
from lunar_rover_tasks.terrain_data.benchmark import (
    build_benchmark, content_hash, verify_benchmark, window_catalogue,
)
from lunar_rover_tasks.terrain_data.metrics import local_metrics, spatial_structure
from lunar_rover_tasks.terrain_data.query import RasterBlocks, RasterTerrain
from lunar_rover_tasks.terrain_data.runtime import TerrainProbeRuntime, require_feature_schema

MOON_CRS = "+proj=stere +lat_0=-90 +lat_ts=-90 +lon_0=0 +R=1737400 +units=m +no_defs"


def make_rasters(tmp_path, *, hole=False, error_shift=True):
    yy, xx = np.indices((25, 25))
    z = (3000 + .2 * (xx+.5) + .3 * (25-yy-.5)).astype("float32")
    if hole:
        z[:, 12] = -9999
    dtm, error = tmp_path / "dtm.tif", tmp_path / "error.tif"
    with rasterio.open(dtm, "w", driver="GTiff", width=25, height=25, count=1,
                       dtype="float32", crs=MOON_CRS, transform=from_origin(10000, 20025, 1, 1),
                       nodata=-9999) as ds:
        ds.write(z, 1)
    # Mirrors MP2's extra west column and south row; coordinates, not indices, must match.
    errors = np.full((26, 26), .04, dtype="float32")
    errors[10, 11 if error_shift else 10] = .73
    with rasterio.open(error, "w", driver="GTiff", width=26, height=26, count=1,
                       dtype="float32", crs=MOON_CRS,
                       transform=from_origin(9999 if error_shift else 10000, 20025, 1, 1),
                       nodata=-9999) as ds:
        ds.write(errors, 1)
    return dtm, error


@pytest.mark.parametrize("half", [1, 2, 4])
def test_plane_gradient_roughness_and_boundary(half):
    y, x = np.indices((41, 43))
    z = 2000 + .2*x - .3*y
    result = local_metrics(z, np.ones(z.shape, bool), 1, 1, half)
    assert np.allclose(result["slope_x"][result["valid"]], .2, atol=1e-10)
    assert np.allclose(result["slope_y"][result["valid"]], .3, atol=1e-10)
    assert np.nanmax(result["roughness"]) < 1e-8
    assert not result["valid"][:half].any()
    assert not result["valid"][:, -half:].any()


def test_roughness_matches_explicit_plane_fit_and_has_scale_dependence():
    y, x = np.indices((41, 41), dtype=float)
    z = 1700 + .2*x - .1*y + .3*np.sin(x*1.3) + .2*np.cos(y*.4)
    values = []
    for half in (1, 4):
        metrics = local_metrics(z, np.ones(z.shape, bool), 2, 3, half)
        yy, xx = np.indices((2*half+1, 2*half+1))
        design = np.stack([xx.ravel()*2, -yy.ravel()*3, np.ones(xx.size)], axis=-1)
        patch = z[20-half:21+half, 20-half:21+half].ravel()
        residual = patch - design @ np.linalg.lstsq(design, patch, rcond=None)[0]
        rms = np.sqrt(np.mean(residual**2))
        assert metrics["roughness"][20, 20] == pytest.approx(rms, abs=1e-8)
        values.append(rms)
    assert values[1] > values[0]


def test_no_data_cannot_become_flat():
    z = np.zeros((21, 21)); z[10, 10] = np.nan
    result = local_metrics(z, np.isfinite(z), 1, 1, 2)
    assert not result["valid"][8:13, 8:13].any()
    assert np.isnan(result["roughness"][10, 10])
    assert result["valid"][4, 4]


def test_query_coordinates_error_registration_and_bilinear_provenance(tmp_path):
    paths = make_rasters(tmp_path)
    with RasterTerrain(*paths, source_id="analytic", origin_xy=[10000, 20000]) as terrain:
        sample = terrain.query([10.5, 14.5], half_width=2)
        assert sample.height_valid and sample.metrics_valid and sample.error_valid
        assert sample.slope_xy == pytest.approx((.2, .3), abs=1e-4)
        assert sample.height_sensitivity_m == pytest.approx(.73)
        assert sample.vehicle_safety == "unknown"
        xy = np.array([[10.5, 14.5], [1, 3]])
        assert np.allclose(terrain.geographic_to_local(terrain.local_to_geographic(xy)), xy, atol=1e-6)
        interp = terrain.query([10.7, 14.8], half_width=2, interpolation="bilinear")
        assert interp.height_m == pytest.approx(3000+.2*10.7+.3*14.8, abs=1e-3)
        assert interp.native_spacing_xy_m == interp.processing_spacing_xy_m == (1, 1)
        assert interp.effective_resolution_m is None
        assert not terrain.provenance()["synthetic_detail"]
        assert interp.source_id == "analytic" and interp.height_interpolation == "bilinear"
        patch = terrain.native_patch((9, 9, 4, 4))
        assert patch["height_m"].shape == (4, 4)
        assert patch["provenance"]["native_spacing_m"] == (1, 1)
        assert patch["transform"] * (.5, .5) == (10009.5, 20015.5)
        with pytest.raises(ValueError):
            terrain.native_patch((24, 24, 4, 4))


def test_queries_reject_holes_edges_and_unknown_interpolation(tmp_path):
    paths = make_rasters(tmp_path, hole=True)
    with RasterTerrain(*paths, source_id="hole", origin_xy=[10000, 20000]) as terrain:
        sample = terrain.query([12.5, 12.5], half_width=1)
        assert not sample.height_valid and not sample.metrics_valid
        assert not terrain.query([11.9, 12.5], half_width=1, interpolation="bilinear").height_valid
        assert not terrain.query([-1, 1], half_width=1).height_valid
        edge = terrain.query([.5, .5], half_width=1)
        assert edge.height_valid and not edge.metrics_valid
        with pytest.raises(ValueError):
            terrain.query([3, 3], half_width=0)
        with pytest.raises(ValueError):
            terrain.query([3, 3], half_width=1, interpolation="fill")


def test_bounded_cache(tmp_path):
    paths = make_rasters(tmp_path)
    cache = RasterBlocks(paths[0], max_bytes=4*4*8, block_size=4)
    for i in range(25):
        assert np.isfinite(cache.cells(i, i))
        assert cache.bytes <= cache.max_bytes
    assert np.isnan(cache.cells(-1, 100))
    cache.close()


def test_runtime_permissions_checkpoint_guard_and_no_teleport(tmp_path):
    paths = make_rasters(tmp_path, hole=True)
    with RasterTerrain(*paths, source_id="hole", origin_xy=[10000, 20000]) as terrain:
        runtime = TerrainProbeRuntime(terrain, half_width=1, sensor_radius_m=4)
        runtime.reset([9.5, 12.5])
        assert runtime.observe_local([[0, 0]])[0]["metrics_valid"]
        with pytest.raises(PermissionError):
            runtime.observe_local([[3.9, 0]])  # centre fits, metric footprint does not
        with pytest.raises(PermissionError):
            runtime.query_orbital_prior([20, 20])
        step = runtime.step([6, 0], 1)
        assert not step["moved"] and step["task_success"] is None
        prior = TerrainProbeRuntime(terrain, half_width=1, sensor_radius_m=4, orbital_prior=True)
        assert prior.query_orbital_prior([20, 20])["schema"] == FEATURE_SCHEMA
    for metadata in ({}, {"terrain_feature_schema": "legacy"}):
        with pytest.raises(ValueError):
            require_feature_schema(metadata)
    require_feature_schema({"terrain_feature_schema": FEATURE_SCHEMA})


def benchmark_fixture():
    regions, source = {}, {"qualitative_feature_scale_m": 10, "regions": {}}
    for index, name in enumerate("abcde"):
        regions[name] = dict(products={"dtm": dict(crs_wkt="same", bounds=[100*index, 0, 100*index+80, 80],
                                                   shape=[80, 80], spacing_m=[1, 1], sha256=name)},
                             spatial_structure={"directions": {"east": dict(censored=False, first_sampled_e_folding_m=16*(index+1))}})
        source["regions"][name] = dict(nac_pairs=[name], tc_controls=["shared" if name in "ab" else name])
    return regions, source


def test_split_groups_shared_controls_without_overlap_and_detects_leakage():
    regions, source = benchmark_fixture()
    manifest = build_benchmark(regions, source)
    assert manifest["groups"] == [["a", "b"], ["c"], ["d"], ["e"]]
    assert len(manifest["folds"]) == 12
    verify_benchmark(manifest)
    assert manifest == build_benchmark(regions, source)
    windows = window_catalogue(manifest, regions)
    assert any(w.get("partial_edge") for w in windows)
    corrupt = copy.deepcopy(manifest)
    corrupt["folds"][0]["train"].append(corrupt["folds"][0]["test"][0])
    corrupt["manifest_sha256"] = content_hash({k:v for k,v in corrupt.items() if k != "manifest_sha256"})
    with pytest.raises(ValueError, match="split"):
        verify_benchmark(corrupt)


def test_test_statistics_do_not_fit_windows():
    regions, source = benchmark_fixture()
    manifest = build_benchmark(regions, source)
    before = manifest["folds"][0]
    for name in before["test"] + before["validation"]:
        regions[name]["spatial_structure"]["directions"]["east"]["first_sampled_e_folding_m"] = 1e8
    after = build_benchmark(regions, source)["folds"][0]
    assert before["window_design"] == after["window_design"]


def test_constant_plane_slope_baseline_uses_original_not_detrended_height():
    y, x = np.indices((32, 32), dtype=float)
    result = spatial_structure(.2*x-.3*y, np.ones(x.shape, bool), 1, 1)
    for point in result["directions"]["east"]["curve"]:
        assert point["rms_bidirectional_slope_deg"] == pytest.approx(np.degrees(np.arctan(.2)))


def test_legacy_environment_cannot_silently_treat_real_map_as_flat():
    from lunar_rover_tasks.tasks.multi_rover_gathering.gathering_env_cfg import TerrainCfg
    from lunar_rover_tasks.tasks.multi_rover_gathering.terrain_features import is_flat_terrain
    cfg = TerrainCfg()
    cfg.type = "lupex_dtm"
    with pytest.raises(ValueError, match="cannot be reused silently"):
        is_flat_terrain(cfg)
