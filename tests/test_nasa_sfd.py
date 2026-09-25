"""Distribution, placement and integration contracts for NASA engineering maps."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import yaml

from lunar_rover_tasks.terrain_data.nasa_sfd import (
    csfd, diameter_cdf, eligible_centres, ejecta_probability, enhance_nasa,
    generate_catalog, normalize_probability, population_count,
    sample_diameters, sample_positions,
)


@pytest.fixture
def spec():
    return yaml.safe_load(Path("configs/terrain/nasa_sfd_v2.yaml").read_text())


@pytest.mark.parametrize("kind,lo,hi,count", [("craters", .8, 2, 1483), ("rocks", .6, 1.2, 35)])
def test_formula_count_and_distribution(kind, lo, hi, count):
    assert population_count(kind, 40000, lo, hi) == count
    np.testing.assert_allclose(diameter_cdf(kind, [lo, hi], lo, hi), [0, 1])
    d = np.sort(sample_diameters(kind, 100000, lo, hi, np.random.default_rng(23)))
    assert d.min() >= lo and d.max() <= hi
    f = diameter_cdf(kind, d, lo, hi)
    ks = max(np.max(np.arange(1, len(d)+1)/len(d)-f), np.max(f-np.arange(len(d))/len(d)))
    assert ks < .01
    assert csfd(kind, 1) == (0.029174 if kind == "craters" else 0.0003)


def test_invalid_ranges_and_probabilities():
    for args in [("craters", 1, 81), ("rocks", 0, 1), ("rocks", 1, 1)]:
        with pytest.raises(ValueError):
            sample_diameters(*args[:1], 10, *args[1:], np.random.default_rng(0))
    with pytest.raises(ValueError):
        normalize_probability(np.ones((3, 3)), np.zeros((3, 3), bool), fallback=True)
    with pytest.raises(ValueError):
        sample_positions(np.ones((2, 2)), 10, (1, 1), np.random.default_rng(0))


def test_spatial_sampling_and_signed_north_up_jitter():
    p = np.array([[.1, .2], [.3, .4]])
    xy = sample_positions(p, 100000, (.2, .3), np.random.default_rng(23))
    col = (xy[:, 0]/.2).astype(int)
    row = ((.6-xy[:, 1])/.3).astype(int)
    counts = np.bincount(2*row+col, minlength=4)/len(xy)
    np.testing.assert_allclose(counts, p.ravel(), atol=.005)
    np.testing.assert_allclose(np.mean(xy[:, 0]/.2-col), .5, atol=.005)
    np.testing.assert_allclose(np.mean((.6-xy[:, 1])/.3-row), .5, atol=.005)


def test_holes_and_complete_footprint_eligibility():
    valid = np.ones((120, 150), bool)
    valid[55:65, 70:80] = False
    mask = eligible_centres(valid, (.1, .2), 1)
    assert not mask[:6].any() and not mask[:, :11].any()
    assert not mask[55:65, 59:91].any()
    p, fallback = normalize_probability(np.zeros(valid.shape), mask, fallback=True)
    assert fallback and np.isclose(p.sum(), 1) and np.all(p[~mask] == 0)


def test_single_crater_radial_law_and_mixture():
    d = 2
    crater = {"diameter_m": d, "x_m": 5.05, "y_m": 5.05}
    weights = ejecta_probability((101, 101), (.1, .1), [crater])
    assert weights[50, 50] < weights[50, 62]
    np.testing.assert_allclose(weights[50, 62], np.exp(-1.2/.7))
    assert weights[50, 62] > weights[50, 67] > weights[50, 70]
    assert weights[0, 0] == 0
    mask = np.ones_like(weights, bool)
    ej, _ = normalize_probability(weights, mask)
    bg, _ = normalize_probability(mask, mask)
    p = .5*(ej+bg)
    xy = sample_positions(p, 100000, (.1, .1), np.random.default_rng(9))
    inside = np.hypot(xy[:, 0]-5.05, xy[:, 1]-5.05) < 2
    yy, xx = np.indices(p.shape)
    # Statistical test uses cell assignment, avoiding circle-edge jitter effects.
    selected = np.hypot((xx+.5)*.1-5.05, (100.5-yy)*.1-5.05) < 2
    rows = ((10.1-xy[:, 1])/.1).astype(int)
    cols = (xy[:, 0]/.1).astype(int)
    assert abs(selected[rows, cols].mean()-p[selected].sum()) < .01
    assert inside.mean() > .4


def test_reproducible_geometry_craters_independent_of_rocks(spec):
    z = np.zeros((220, 260), np.float32)
    valid = np.ones_like(z, bool)
    valid[100:110, 110:120] = False
    z[~valid] = np.nan
    final, layers, cat, probs = enhance_nasa(z, valid, (.1, .1), spec)
    other, _, cat2, _ = enhance_nasa(z, valid, (.1, .1), spec)
    np.testing.assert_array_equal(final, other)
    assert cat == cat2
    only = deepcopy(spec)
    only["rocks"]["enabled"] = False
    _, cl, cc, _ = enhance_nasa(z, valid, (.1, .1), only)
    assert cat["craters"] == cc["craters"]
    np.testing.assert_array_equal(layers["craters"], cl["craters"])
    assert np.isnan(final[~valid]).all()
    assert np.isfinite(final[valid]).all()
    assert np.isclose(probs["rocks"].sum(), 1)
    assert np.all(probs["rocks"][~valid] == 0)
    np.testing.assert_allclose(final, z+layers["craters"]+layers["rocks"], equal_nan=True)


def test_no_crater_fallback_and_no_data_error(spec):
    spec["craters"]["enabled"] = False
    cat, p = generate_catalog(np.ones((100, 100), bool), (.1, .1), spec)
    assert not cat["generation"]["stats"]["spatial"]["ejecta_available"]
    np.testing.assert_array_equal(p["rocks"], p["background"])
    with pytest.raises(ValueError, match="area"):
        generate_catalog(np.zeros((100, 100), bool), (.1, .1), spec)


def test_buffered_scene_coordinates_and_source_preservation(tmp_path, spec):
    import rasterio
    from rasterio.transform import from_origin
    from lunar_rover_tasks.terrain_data.scene_pack import ScenePack, build_scene, file_hash

    source = tmp_path / "native.tif"
    crs = "+proj=stere +lat_0=-90 +lon_0=0 +R=1737400 +units=m +no_defs"
    with rasterio.open(source, "w", driver="GTiff", width=20, height=20, count=1,
                       dtype="float32", crs=crs, transform=from_origin(1000, 2000, 5, 5)) as ds:
        y, x = np.indices((20, 20))
        ds.write((100+x*.2-y*.3).astype(np.float32), 1)
    before = file_hash(source)
    args = dict(source_id="test", geographic_group="one", pixel_window=[7, 7, 6, 6],
                processing_spacing_m=.1, metric_baseline_m=.8, source_half_width=2)
    base = ScenePack(build_scene(source, tmp_path / "base", **args))
    scene = ScenePack(build_scene(source, tmp_path / "full", enhancement=spec, **args))
    assert file_hash(source) == before
    np.testing.assert_array_equal(base.arrays["base_height"], scene.arrays["base_height"])
    gen = scene.meta["feature_catalog"]["generation"]
    assert gen["target_crop_rc"] == [50, 50, 300, 300]
    assert scene.arrays["probability_rocks"].shape == (400, 400)
    for kind in ("craters", "rocks"):
        for item in scene.meta["feature_catalog"][kind]:
            np.testing.assert_allclose(scene.local_to_projected(item["local_xy_m"]), item["projected_xy_m"])
            assert item["centre_in_target"] == (0 <= item["x_m"] < 30 and 0 <= item["y_m"] < 30)
    reconstructed = scene.arrays["base_height"] + scene.arrays["layer_craters"] + scene.arrays["layer_rocks"]
    np.testing.assert_array_equal(reconstructed, scene.arrays["features"][0])
