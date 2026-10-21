"""New-scale baselines: native semantics, real actuator prediction and isolation."""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from _common import cfg_from_experiment
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.protocol import TerrainSnapshot
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_baselines import (
    DeterministicAdapter, PrimitiveAdapter, UpstreamPrimitiveInput,
    PRIMITIVE_LIBRARY_VERSION, primitive_library, primitive_fingerprint,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_execution import (
    control_error, midpoint_numpy, terrain_gain_numpy,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_footprint import FootprintTerrain
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_planning import (
    UpstreamIntent, UpstreamPlannerInput,
)


@pytest.fixture(scope="module")
def cfg():
    return replace(cfg_from_experiment("configs/experiment/upstream_scale_s01_review_v2.yaml"),
                   simulated_latency=0., planner_adapter="line_upstream_v1")


@pytest.fixture(scope="module")
def terrain(cfg):
    x, y = np.meshgrid(np.arange(-15., 15.01, .2), np.arange(-8., 8.01, .2), indexing="ij")
    xy = np.c_[x.ravel(), y.ravel()]
    features = np.zeros((len(xy), 5)); features[:, 4] = 1.
    return FootprintTerrain(TerrainSnapshot(xy, features, np.zeros(len(xy)),
                                          cfg.spec.perception.support_radius), cfg.spec)


def request(cfg, terrain, *, goal=(4., 0.), pose=(0., 0., 0.), previous=(0., 0.),
            cap=None, now=0., stop=False):
    return UpstreamPlannerInput(env_id=0, agent_id=0, ego_pose=np.array(pose),
        ego_velocity=np.array(previous), intent=UpstreamIntent(np.array(goal), False, 0., 1.,
                                                               speed_cap=cap, stop_at_goal=stop),
        terrain=terrain, neighbors=(), previous_control=np.array(previous), now=now,
        times=cfg.spec.grid.times(now), workspace=(-100., -100., 100., 100.), spec=cfg.spec)


def test_frozen47_index_and_native_semantics_mapped_without_mutating_legacy(cfg):
    path = Path("configs/experiment/primitive47_library_v1.json")
    original_bytes = path.read_bytes()
    legacy = json.loads(original_bytes)["library"]
    library = primitive_library(cfg.spec)
    assert library["count"] == 47 and library["version"] == PRIMITIVE_LIBRARY_VERSION
    for old, new in zip(legacy["entries"], library["entries"], strict=True):
        assert old["index"] == new["index"]
        assert old["primitive_type"] == new["primitive_type"]
        assert old["motion_direction"] == new["motion_direction"]
        assert old["planned_yaw_delta"] == pytest.approx(new["planned_yaw_delta"], abs=3e-8)
        np.testing.assert_allclose(new["local_xy"], np.array(old["local_xy"]) * 2.5, atol=2e-7)
        expected = old["reference_speed"] * (25 / 9) / 1.15
        if old["motion_direction"] < 0:expected = max(-.5, expected)
        assert new["reference_speed"] == pytest.approx(expected, abs=1e-7)
    assert path.read_bytes() == original_bytes
    assert max(x["reference_speed"] for x in library["entries"]) == pytest.approx(25 / 9)
    assert library["entries"][40]["reference_speed"] == -.5


def test_library_fingerprint_is_repeatable_copied_and_profile_bound(cfg):
    old = primitive_fingerprint(cfg.spec)
    first = primitive_library(cfg.spec); first["entries"][1]["local_xy"][0] = 999.
    assert primitive_fingerprint(cfg.spec) == old
    changed = replace(cfg.spec, vehicle=replace(cfg.spec.vehicle, forward_limit=2.))
    assert primitive_fingerprint(changed) != old


def test_frozen_new_library_file_matches_registered_content(cfg):
    snapshot = json.loads(Path("configs/experiment/upstream_primitive47_library_v1.json").read_text())
    assert snapshot == {"library": primitive_library(cfg.spec), "sha256": primitive_fingerprint(cfg.spec)}


@pytest.mark.parametrize("primitive_id", [True, -1, 47, 1.5, "1"])
def test_native_payload_rejects_invalid_index(cfg, terrain, primitive_id):
    payload = UpstreamPrimitiveInput.for_context(request(cfg, terrain), primitive_id)
    with pytest.raises(ValueError, match="primitive"):
        payload.validate()


@pytest.mark.parametrize("field,value", [("library_version", "exp156_primitive47_v1"),
                                       ("library_fingerprint", "forged")])
def test_native_payload_rejects_old_or_tampered_library(cfg, terrain, field, value):
    payload = UpstreamPrimitiveInput.for_context(request(cfg, terrain), 43)
    with pytest.raises(ValueError, match="primitive"):
        replace(payload, **{field: value}).validate()


@pytest.mark.parametrize("geometry", ["line", "quintic"])
@pytest.mark.parametrize("goal", [(4., 0.), (3., 2.), (-3., 0.)])
def test_controller_prediction_obeys_new_joint_limits_and_direction(cfg, terrain, geometry, goal):
    adapter = DeterministicAdapter(replace(cfg, planner_adapter=f"{geometry}_upstream_v1"), geometry)
    context = request(cfg, terrain, goal=goal)
    controls = adapter.candidate_controls(context)[0].controls
    assert controls.shape == (40, 2) and np.isfinite(controls).all()
    state, previous = context.ego_pose.copy(), context.previous_control.copy()
    for control in controls:
        assert not control_error(control, previous, cfg.spec)
        state, _ = midpoint_numpy(state, control, cfg.spec)
        previous = control
    assert np.sign(state[0]) == np.sign(goal[0])
    # The first physical advance is 0.04 m/s * 0.2 s. Geometry sampling is not execution.
    assert controls[0, 0] == pytest.approx(.04 * np.sign(goal[0]))
    if goal[0] < 0:assert controls[:, 0].min() >= -.5 - 1e-10


@pytest.mark.parametrize("geometry", ["line", "quintic"])
def test_speed_cap_controls_reference_without_instantly_clipping_moving_state(cfg, terrain, geometry):
    adapter = DeterministicAdapter(replace(cfg, planner_adapter=f"{geometry}_upstream_v1"), geometry)
    from_rest = adapter.candidate_controls(request(cfg, terrain, cap=.3))[0].controls
    assert from_rest[:, 0].max() <= .3 + 1e-10
    moving = adapter.candidate_controls(request(cfg, terrain, previous=(1., 0.), cap=.3))[0].controls
    assert moving[0, 0] == pytest.approx(.92)
    assert not control_error(moving[0], [1., 0.], cfg.spec)


def test_rolling_target_is_not_an_implicit_stop(cfg, terrain):
    adapter = DeterministicAdapter(cfg)
    rolling = adapter.candidate_controls(request(cfg, terrain, goal=(1., 0.)))[0].controls
    stop = adapter.candidate_controls(request(cfg, terrain, goal=(1., 0.), stop=True))[0].controls
    assert rolling[-1, 0] > .5
    assert abs(stop[-1, 0]) < rolling[-1, 0]


@pytest.mark.parametrize("geometry", ["line", "quintic"])
def test_zero_distance_move_does_not_invent_a_forward_goal(cfg, terrain, geometry):
    config = replace(cfg, planner_adapter=f"{geometry}_upstream_v1")
    adapter = DeterministicAdapter(config, geometry)
    context = request(cfg, terrain, goal=(0., 0.))
    assert not context.intent.hold
    controls = adapter.candidate_controls(context)[0].controls
    np.testing.assert_array_equal(controls, 0.)
    result = adapter.plan(context)
    assert result.valid
    np.testing.assert_array_equal(result.trajectory.controls, 0.)


@pytest.mark.parametrize("primitive_id", [0, 6, 33, 40, 41, 42, 43, 44, 45, 46])
def test_native_replay_control_semantics(cfg, terrain, primitive_id):
    adapter = PrimitiveAdapter(replace(cfg, planner_adapter="primitive47_upstream_v1"))
    context = request(cfg, terrain)
    entry = primitive_library(cfg.spec)["entries"][primitive_id]
    native = replace(context, intent=replace(context.intent, goal_xy=np.array(entry["local_xy"]),
                       terminal_yaw=entry["planned_yaw_delta"], stop_at_goal=True))
    controls = adapter._controls(native, entry)
    previous = context.previous_control
    for command in controls:
        assert not control_error(command, previous, cfg.spec)
        previous = command
    if primitive_id == 0:np.testing.assert_array_equal(controls, 0.)
    elif primitive_id in (43, 44):
        np.testing.assert_array_equal(controls[:, 0], 0.)
        assert np.sign(controls[0, 1]) == (1 if primitive_id == 43 else -1)
        assert np.max(np.abs(controls[:, 1])) <= .2 + 1e-10
    elif primitive_id in (40, 41, 42):
        assert controls[:, 0].min() < 0 and controls[:, 0].min() >= -.5 - 1e-10
    else:assert controls[:, 0].max() > 0


def test_primitive_requires_native_payload_not_polar_context(cfg, terrain):
    adapter = PrimitiveAdapter(replace(cfg, planner_adapter="primitive47_upstream_v1"))
    with pytest.raises(ValueError, match="native"):
        adapter.plan(request(cfg, terrain))


@pytest.mark.parametrize("geometry", ["line", "quintic"])
def test_real_baseline_plan_independently_repredicts_and_restores(cfg, terrain, geometry):
    config = replace(cfg, planner_adapter=f"{geometry}_upstream_v1")
    adapter = DeterministicAdapter(config, geometry)
    context = request(cfg, terrain, cap=.4)
    output = adapter.plan(context)
    assert output.valid, output.fallback_reason
    assert output.trajectory.states.shape == (41, 3)
    states = [context.ego_pose]
    for command in output.trajectory.controls:
        features, known = context.terrain.lookup(states[-1][:2]); assert known[0]
        state, velocity = midpoint_numpy(states[-1], command, cfg.spec,
                                          gain=terrain_gain_numpy(states[-1], features[0], cfg.spec))
        states.append(state)
    np.testing.assert_allclose(output.trajectory.states, states, atol=1e-10, rtol=1e-8)
    restored = DeterministicAdapter(config, geometry)
    restored.load_state_dict(adapter.state_dict())
    replay = restored.plan(context)
    np.testing.assert_array_equal(output.trajectory.controls, replay.trajectory.controls)


def test_native_spin_freezes_request_heading_and_snapshot(cfg, terrain):
    config = replace(cfg, planner_adapter="primitive47_upstream_v1")
    adapter = PrimitiveAdapter(config)
    first = adapter.plan(UpstreamPrimitiveInput.for_context(request(cfg, terrain), 43))
    assert first.valid, first.fallback_reason
    saved = adapter.state_dict()
    restored = PrimitiveAdapter(config); restored.load_state_dict(saved)
    later = request(cfg, terrain, pose=(0., 0., .1), previous=(0., .1), now=.2)
    payload = UpstreamPrimitiveInput.for_context(later, 43)
    a, b = adapter.plan(payload), restored.plan(payload)
    assert a.valid and b.valid
    assert adapter._resolved.terminal_yaw == pytest.approx(np.pi / 4)
    np.testing.assert_array_equal(a.trajectory.controls, b.trajectory.controls)


def test_native_snapshot_rejects_library_change_without_mutating_state(cfg, terrain):
    config = replace(cfg, planner_adapter="primitive47_upstream_v1")
    adapter = PrimitiveAdapter(config)
    adapter.plan(UpstreamPrimitiveInput.for_context(request(cfg, terrain), 0))
    state = adapter._adapter_state(); bad = deepcopy(state); bad["library_fingerprint"] = "old"
    with pytest.raises(ValueError, match="library"):
        adapter._load_adapter_state(bad)
    assert adapter._adapter_state()["library_fingerprint"] == state["library_fingerprint"]


@pytest.mark.parametrize("adapter_type", ["line", "quintic", "primitive"])
def test_moving_hold_and_expiry_reach_stationary_tail(cfg, terrain, adapter_type):
    is_primitive = adapter_type == "primitive"
    adapter_id = "primitive47_upstream_v1" if is_primitive else f"{adapter_type}_upstream_v1"
    config = replace(cfg, planner_adapter=adapter_id)
    adapter = PrimitiveAdapter(config) if is_primitive else DeterministicAdapter(config, adapter_type)
    context = request(cfg, terrain, previous=(.5, 0.))
    hold = replace(context, intent=replace(context.intent, hold=True, goal_xy=context.ego_pose[:2]))
    payload = UpstreamPrimitiveInput.for_context(hold, 0) if is_primitive else hold
    result = adapter.plan(payload)
    assert result.valid
    if is_primitive:
        assert result.diagnostics["primitive_id"] == 0
        assert result.diagnostics["library_fingerprint"] == primitive_fingerprint(cfg.spec)
    assert result.trajectory.controls[0, 0] == pytest.approx(.42)
    np.testing.assert_array_equal(result.trajectory.controls[-1], [0., 0.])
    expired = request(cfg, terrain, previous=(.5, 0.), now=1.2)
    payload = UpstreamPrimitiveInput.for_context(expired, 33) if is_primitive else expired
    result = adapter.plan(payload)
    assert not result.valid and result.fallback_reason == "expired_intent"
    if is_primitive:assert result.diagnostics["primitive_id"] == 33
    assert result.trajectory.controls[0, 0] == pytest.approx(.42)


def test_native_same_request_tightened_cap_preserves_frozen_target(cfg, terrain):
    config = replace(cfg, planner_adapter="primitive47_upstream_v1")
    adapter = PrimitiveAdapter(config)
    first = request(cfg, terrain, cap=.4)
    adapter.plan(UpstreamPrimitiveInput.for_context(first, 33))
    frozen = adapter._resolved.goal_xy.copy()
    later = request(cfg, terrain, pose=(.1, 0., 0.), cap=.1, now=.2)
    result = adapter.plan(UpstreamPrimitiveInput.for_context(later, 33))
    assert result.valid
    np.testing.assert_array_equal(adapter._resolved.goal_xy, frozen)
    assert result.trajectory.controls[:, 0].max() <= .1 + 1e-10


def test_default_latency_uses_delivered_state_and_actual_activation_pose(cfg, terrain):
    config = replace(cfg, simulated_latency=.5)
    context = request(cfg, terrain, previous=(.2, 0.), cap=.4, pose=(.12, 0., 0.), now=.6)
    context = replace(context, information_pose=np.zeros(3), information_time=0.)
    output = DeterministicAdapter(config).plan(context)
    assert output.valid, output.fallback_reason
    np.testing.assert_array_equal(output.trajectory.states[0], context.ego_pose)
