"""Required rejection witnesses for the S01 review gaps."""

from pathlib import Path
from dataclasses import replace
from copy import deepcopy
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from _common import cfg_from_experiment
from diagnose_upstream_execution import synthetic_local_frame
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_execution import rollout_numpy
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_footprint import (
    FootprintNeighbor, FootprintReviewer, FootprintTerrain,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.protocol import TerrainSnapshot


@pytest.fixture(scope="module")
def spec():
    return cfg_from_experiment("configs/experiment/upstream_scale_s01.yaml").spec


def test_tiny_neighbor_radius_cannot_certify_overlapping_vehicles(spec):
    terrain = synthetic_local_frame(np.zeros(3), spec, 0.)
    plan = rollout_numpy([0., 0., 0.], [[0., 0.]], spec)
    neighbor = FootprintNeighbor(1, np.array([0., .2]), np.array([[1., 0.], [1., 0.]]), np.array([.1, .1]))
    report = FootprintReviewer(spec).trajectory(plan, [0., 0.], terrain, neighbors=[neighbor])
    assert not report.passed


@pytest.mark.parametrize("conservative", [False, True])
def test_invented_flat_terrain_gain_cannot_shorten_execution_prediction(spec, conservative):
    terrain = synthetic_local_frame(np.zeros(3), spec, 0.)
    plan = rollout_numpy([0., 0., 0.], [[1., 0.]], spec, gain=.25)
    report = FootprintReviewer(spec).trajectory(plan, [1., 0.], terrain, conservative_reserve=conservative)
    assert not report.passed
    assert report.reason == "velocity_gain_inconsistent"


@pytest.mark.parametrize("radius", [.1, .28, 2.974, 5.949, 6.5])
def test_profile_bound_neighbor_factory_rejects_small_positive_radii(spec, radius):
    with pytest.raises(ValueError, match="neighbor_radius_below_geometry"):
        FootprintNeighbor.for_spec(1, [0., .2], [[8., 0.], [8., 0.]], [radius, radius], spec)


@pytest.mark.parametrize("invalid", ["missing", "other_profile", "old_version"])
def test_legal_radius_without_matching_profile_and_version_is_rejected(spec, invalid):
    radius = 2 * spec.vehicle.effective_radius
    peer = FootprintNeighbor.for_spec(1, [0., .2], [[8., 0.], [8., 0.]], [radius, radius], spec)
    if invalid == "missing":peer = replace(peer, profile_fingerprint=None)
    if invalid == "other_profile":peer = replace(peer, profile_fingerprint="different_vehicle")
    if invalid == "old_version":peer = replace(peer, version="identified_neighbor_upstream_v1")
    plan = rollout_numpy([0., 0., 0.], [[0., 0.]], spec)
    terrain = synthetic_local_frame(np.zeros(3), spec, 0., query_version="batch_ties_v2")
    report = FootprintReviewer(spec).trajectory(plan, [0., 0.], terrain, neighbors=[peer])
    assert not report.passed and report.reason == "neighbor_geometry_contract_mismatch"


def test_legal_neighbor_uncertainty_is_not_added_twice(spec):
    radius = 2 * spec.vehicle.effective_radius + 1.
    peer = FootprintNeighbor.for_spec(1, [0., .2], [[radius + .05, 0.]] * 2, [radius] * 2, spec)
    np.testing.assert_array_equal(peer.radii, [radius] * 2)
    plan = rollout_numpy([0., 0., 0.], [[0., 0.]], spec)
    assert FootprintReviewer(spec).trajectory(plan, [0., 0.], synthetic_local_frame(np.zeros(3), spec, 0.), neighbors=[peer]).passed


def uniform_terrain(spec, feature):
    snapshot = synthetic_local_frame(np.zeros(3), spec, 0.).snapshot()
    features = snapshot.features.copy(); features[:] = feature
    return FootprintTerrain(TerrainSnapshot(snapshot.xy, features, snapshot.observed_at, snapshot.support_radius), spec)


@pytest.mark.parametrize("control,yaw", [([.5, .2], 0.), ([-.5, -.2], 1.), ([0., .2], .4)])
@pytest.mark.parametrize("traversability", [.25, .65, 1.])
def test_legal_directional_and_minimum_terrain_gain_remains_feasible(spec, control, yaw, traversability):
    from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_execution import terrain_gain_numpy
    terrain = uniform_terrain(spec, [0., .06, -.02, 0., traversability])
    def gain(pose):return terrain_gain_numpy(pose, terrain.lookup(pose[:2])[0][0], spec)
    plan = rollout_numpy([0., 0., yaw], [control], spec, gain=gain)
    assert FootprintReviewer(spec).trajectory(plan, control, terrain).passed


def test_wrong_slope_direction_cannot_change_predicted_speed(spec):
    from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_execution import terrain_gain_numpy
    feature = np.array([0., .15, .02, 0., .8]); terrain = uniform_terrain(spec, feature)
    wrong_gain = terrain_gain_numpy([0., 0., 0.], feature, spec)
    plan = rollout_numpy([0., 0., np.pi / 2], [[.5, .1]], spec, gain=wrong_gain)
    assert FootprintReviewer(spec).trajectory(plan, [.5, .1], terrain).reason == "velocity_gain_inconsistent"


def test_correct_speed_but_wrong_successor_state_is_rejected(spec):
    plan = rollout_numpy([0., 0., 0.], [[.5, .1]], spec)
    plan.states[-1, 0] += .1
    assert FootprintReviewer(spec).trajectory(plan, [.5, .1], synthetic_local_frame(np.zeros(3), spec, 0.)).reason == "dynamics_inconsistent"


def test_float32_legal_model_has_explicit_tolerance(spec):
    terrain = uniform_terrain(spec, [0., .1, .03, 0., .6])
    from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_execution import terrain_gain_numpy
    gain = terrain_gain_numpy([0., 0., .8], [0., .1, .03, 0., .6], spec)
    plan = rollout_numpy([0., 0., .8], [[.5, .1]], spec, gain=gain)
    for name in ("states", "controls", "velocities"):setattr(plan, name, getattr(plan, name).astype(np.float32))
    assert FootprintReviewer(spec).trajectory(plan, [.5, .1], terrain).passed


def test_terrain_from_other_profile_and_invalid_time_are_rejected(spec):
    terrain = synthetic_local_frame(np.zeros(3), spec, 0.)
    other = replace(spec, vehicle=replace(spec.vehicle, length=6.))
    plan = rollout_numpy([0., 0., 0.], [[0., 0.]], spec)
    with pytest.raises(ValueError, match="profile mismatch"):
        FootprintReviewer(other).trajectory(plan, [0., 0.], terrain)
    plan.timestamp = 1.
    with pytest.raises(ValueError, match="information time"):
        FootprintReviewer(spec).trajectory(plan, [0., 0.], terrain)


@pytest.mark.parametrize("block", [1, 7, 256])
def test_batch_lookup_preserves_all_channels_order_and_unknown_masks(spec, block, monkeypatch):
    import lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_footprint as module
    monkeypatch.setattr(module, "QUERY_BLOCK", block)
    rng = np.random.default_rng(31)
    xy = np.array([[x, y] for x in np.arange(-1., 1.01, .2) for y in np.arange(-1., 1.01, .2)])
    f = rng.uniform(-.5, .5, (len(xy), 5)); f[:12, 1:3] = np.tile([[.3, .4], [-.3, .4]], (6, 1))
    queries = np.vstack([xy, xy[:-12] + [.1, .1], xy[:-12] + [.1, 0.], rng.uniform(-1.5, 1.5, (500, 2))])
    for ids in (np.arange(len(xy)), np.arange(len(xy))[::3], np.array([0]), np.array([], dtype=int)):
        snapshot = TerrainSnapshot(xy[ids], f[ids], np.zeros(len(ids)), spec.perception.support_radius)
        a = FootprintTerrain(snapshot, spec, query_version="scalar_ties_v1")
        b = FootprintTerrain(snapshot, spec, query_version="batch_ties_v2")
        for reference, actual in zip(a.lookup(queries), b.lookup(queries)):
            np.testing.assert_array_equal(reference, actual)
        assert b.lookup(np.empty((0, 2)))[0].shape == (0, 5)


def test_dense_coincident_ties_bound_each_ragged_allocation(spec):
    from profile_upstream_footprint import CountedTree
    import lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_footprint as module
    count = 4096
    xy = np.zeros((count, 2)); features = np.zeros((count, 5)); features[:, 4] = 1.
    features[-1, 0] = .4; features[-1, 1] = .3; features[-1, 4] = 0.
    terrain = FootprintTerrain(TerrainSnapshot(xy, features, np.zeros(count), .155), spec)
    class BoundedTree(CountedTree):
        def query_ball_point(self, points, *args, **kwargs):
            result = super().query_ball_point(points, *args, **kwargs)
            if not kwargs.get("return_length", False):
                assert sum(len(ids) for ids in result) <= module.TIE_ENTRY_BUDGET
            return result
    terrain._tree = BoundedTree(terrain._tree)
    f, known = terrain.lookup(np.zeros((128, 2)))
    assert known.all() and (f[:, 0] == .4).all() and (f[:, 4] == 0.).all()


@pytest.mark.parametrize("defect", ["slope", "step", "clearance", "unknown", "flat"])
def test_scalar_and_batch_full_footprint_reports_match(spec, defect):
    from dataclasses import asdict
    snapshot = synthetic_local_frame(np.zeros(3), spec, 0.).snapshot()
    f = snapshot.features.copy(); xy = snapshot.xy.copy()
    if defect == "slope":f[:, 1] = .2
    if defect == "step":f[np.linalg.norm(xy, axis=-1) < .35, 0] = .15
    if defect == "clearance":f[np.linalg.norm(xy, axis=-1) < .35, 0] = .5
    if defect == "unknown":
        keep = np.linalg.norm(xy - [1.35, .9], axis=-1) > .45; xy, f = xy[keep], f[keep]
    snap = TerrainSnapshot(xy, f, np.zeros(len(xy)), .155)
    scalar = FootprintTerrain(snap, spec, query_version="scalar_ties_v1")
    batch = FootprintTerrain(snap, spec, query_version="batch_ties_v2")
    plan = rollout_numpy([0., 0., 0.], [[0., 0.]], spec)
    reviewer = FootprintReviewer(spec)
    assert asdict(reviewer.trajectory(plan, [0., 0.], scalar)) == asdict(reviewer.trajectory(plan, [0., 0.], batch))


def test_versions_preserve_legacy_metadata_and_refuse_cross_restore():
    from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_proxy import UpstreamActuatorProxy
    old = cfg_from_experiment("configs/experiment/upstream_scale_s01.yaml")
    new = cfg_from_experiment("configs/experiment/upstream_scale_s01_review_v2.yaml")
    assert "execution_review_version" not in old.metadata
    assert new.metadata["geometry_version"] == "footprint_upstream_v2"
    assert new.metadata["solver_cache_key"] != old.metadata["solver_cache_key"]
    state = UpstreamActuatorProxy(old, np.zeros((2, 4, 3))).state_dict()
    with pytest.raises(ValueError, match="mismatch"):
        UpstreamActuatorProxy(new, np.zeros((2, 4, 3))).load_state_dict(state)
    broken = deepcopy(new.metadata); broken["terrain_query_version"] = "scalar_ties_v1"
    with pytest.raises(ValueError, match="mismatch"):new.validate_state_identity({"contracts": broken})


def test_legacy_review_is_explicit_and_retains_original_reproduction_only(spec):
    terrain = synthetic_local_frame(np.zeros(3), spec, 0.)
    plan = rollout_numpy([0., 0., 0.], [[1., 0.]], spec, gain=.25)
    assert FootprintReviewer(spec, version="footprint_review_v1").trajectory(plan, [1., 0.], terrain).passed
    assert not FootprintReviewer(spec).trajectory(plan, [1., 0.], terrain).passed


@pytest.mark.parametrize("previous,traversability", [([25/9, 0.], .25), ([2.4, .3], .65), ([-.5, -.2], 1.)])
def test_complete_joint_stop_uses_the_terrain_model_for_every_step(spec, previous, traversability):
    from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_execution import stop_envelope, terrain_gain_numpy
    terrain = uniform_terrain(spec, [0., .06, -.02, 0., traversability])
    def gain(pose):return terrain_gain_numpy(pose, terrain.lookup(pose[:2])[0][0], spec)
    envelope = stop_envelope([0., 0., 0.], previous, previous, spec, latency=.2, gain=gain)
    assert envelope.fits_horizon
    report = FootprintReviewer(spec).trajectory(envelope.trajectory, previous, terrain,
                                              require_stop=True, conservative_reserve=True)
    assert report.passed
    changed = deepcopy(envelope.trajectory)
    changed.velocities[0] *= .5
    assert not FootprintReviewer(spec).trajectory(changed, previous, terrain, conservative_reserve=True).passed


@pytest.mark.parametrize("review,query", [("footprint_review_v1", "batch_ties_v2"), ("typo", "scalar_ties_v1"), ("footprint_review_v2", "typo")])
def test_unknown_or_inconsistent_review_query_configuration_is_rejected(review, query):
    from _common import ROOT, load_yaml
    from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_spec import parse_upstream_experiment
    data = load_yaml("configs/experiment/upstream_scale_s01.yaml")
    data["upstream_scale"].update(execution_review_version=review, terrain_query_version=query)
    with pytest.raises(ValueError):parse_upstream_experiment(data, ROOT)
