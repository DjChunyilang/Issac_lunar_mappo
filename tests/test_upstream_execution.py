"""Independent physics, actuator, footprint and stopping reserve witnesses."""

from pathlib import Path
from dataclasses import replace
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from _common import cfg_from_experiment
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_execution import (
    brake_sequence, control_error, limit_control_numpy, limit_control_torch, midpoint_numpy,
    midpoint_torch, rollout_numpy, rollout_torch, stop_envelope, wheel_rates,
    terrain_gain_numpy, terrain_gain_torch,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_footprint import (
    FootprintTerrain, FootprintReviewer, FootprintNeighbor, admit_control,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.protocol import TerrainSnapshot


@pytest.fixture(scope="module")
def spec():
    return cfg_from_experiment("configs/experiment/upstream_scale_s00.yaml").spec


def local_terrain(spec, *, slope=0., anomaly=None, remove=None, bounds=(-16., 16., -9., 9.)):
    x0, x1, y0, y1 = bounds
    xx, yy = np.meshgrid(np.arange(x0, x1 + .05, .1), np.arange(y0, y1 + .05, .1), indexing="ij")
    xy = np.c_[xx.ravel(), yy.ravel()]
    features = np.zeros((len(xy), 5)); features[:, 0] = xy[:, 0] * slope
    features[:, 1] = slope; features[:, 4] = 1.
    if anomaly is not None:anomaly(xy, features)
    if remove is not None:
        keep = ~remove(xy); xy, features = xy[keep], features[keep]
    return FootprintTerrain(TerrainSnapshot(xy, features, np.zeros(len(xy)), spec.perception.support_radius), spec)


@pytest.mark.parametrize("previous,target", [
    ([0., 0.], [5., 1.]), ([.06, .35], [0., 0.]), ([.5, -.2], [-.5, .35]),
    ([-.5, .2], [2., -.35]), ([25/9, 0.], [25/9, .35]), ([2.4, .35], [0., 0.]),
])
def test_projection_reference_and_torch_satisfy_joint_constraints(spec, previous, target):
    reference = limit_control_numpy(target, previous, spec)
    actual = limit_control_torch(torch.tensor([target], dtype=torch.float64),
                                 torch.tensor([previous], dtype=torch.float64), spec).numpy()[0]
    np.testing.assert_allclose(actual, reference, atol=1e-10, rtol=1e-8)
    assert not control_error(actual, previous, spec)


@pytest.mark.parametrize("speed", [1., 1.5, 2., 25/9])
def test_actual_acceleration_reaches_declared_speed_without_hidden_cap(spec, speed):
    command = np.zeros(2); pose = np.zeros(3); maximum = 0.
    for _ in range(80):
        next_command = limit_control_numpy([speed, 0.], command, spec)
        assert not control_error(next_command, command, spec)
        pose, actual = midpoint_numpy(pose, next_command, spec)
        maximum = max(maximum, actual[0]); command = next_command
    assert maximum == pytest.approx(speed, abs=1e-10)
    assert pose[0] > 1 and pose[1] == 0
    assert wheel_rates(command, spec)[0] == pytest.approx(speed / .5)


def test_angular_speed_reduced_before_crossing_low_speed_threshold(spec):
    previous = np.array([.06, .35]); commands = brake_sequence(previous, spec)
    assert commands[0, 0] > spec.vehicle.low_speed_threshold
    assert commands[0, 1] == pytest.approx(.31)
    for control in commands:
        assert not control_error(control, previous, spec)
        if control[0] <= .05:assert abs(control[1]) <= .2 + 1e-10
        previous = control
    np.testing.assert_array_equal(commands[-1], [0., 0.])


def test_reverse_acceleration_uses_point_two_and_direction_passes_zero(spec):
    command = np.array([.06, 0.]); sequence = []
    for _ in range(8):
        next_command = limit_control_numpy([-.5, 0.], command, spec)
        assert not control_error(next_command, command, spec)
        sequence.append(next_command[0]); command = next_command
    assert 0. in sequence
    zero = sequence.index(0.)
    assert sequence[zero + 1] == pytest.approx(-.04)


@pytest.mark.parametrize("gain", [.25, .6, 1.])
def test_midpoint_numpy_torch_and_analytic_reference(spec, gain):
    p = np.array([2., -1., .9]); u = np.array([1.3, -.2])
    mid = .9 + .5 * .2 * u[1] * gain
    expected = p + .2 * np.array([u[0] * gain * np.cos(mid), u[0] * gain * np.sin(mid), u[1] * gain])
    actual, velocity = midpoint_numpy(p, u, spec, gain=gain)
    other, speed = midpoint_torch(torch.tensor(p), torch.tensor(u), spec, gain=torch.tensor(gain, dtype=torch.float64))
    np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-8)
    np.testing.assert_allclose(other.numpy(), expected, atol=1e-10, rtol=1e-8)
    np.testing.assert_allclose(speed.numpy(), velocity, atol=1e-10, rtol=1e-8)


def test_tensor_rollout_batch_permutation_and_empty_batch(spec):
    generator = np.random.default_rng(31)
    raw = generator.normal([1., 0.], [.6, .4], (3, 4, 10, 2))
    poses = np.zeros((3, 4, 3)); previous = np.zeros((3, 4, 2))
    a = rollout_torch(torch.tensor(poses), torch.tensor(raw), torch.tensor(previous), spec)
    perm = np.array([2, 0, 1])
    b = rollout_torch(torch.tensor(poses[perm]), torch.tensor(raw[perm]), torch.tensor(previous[perm]), spec)
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x.numpy()[perm], y.numpy())
    for batch in range(3):
        for sample in range(4):
            command = previous[batch, sample]; commands = []
            for target in raw[batch, sample]:
                command = limit_control_numpy(target, command, spec); commands.append(command)
            plan = rollout_numpy(poses[batch, sample], commands, spec)
            np.testing.assert_allclose(a[0][batch, sample], plan.states, atol=1e-10, rtol=1e-8)
            np.testing.assert_allclose(a[1][batch, sample], plan.controls, atol=1e-10, rtol=1e-8)
    empty = rollout_torch(torch.zeros((0, 3), dtype=torch.float64), torch.zeros((0, 40, 2), dtype=torch.float64),
                          torch.zeros((0, 2), dtype=torch.float64), spec)
    assert empty[0].shape == (0, 41, 3)


def test_float32_projection_matches_reference_with_same_raw_commands(spec):
    raw = np.random.default_rng(47).normal([1.4, 0.], [.6, .3], (2, 3, 12, 2))
    a = rollout_torch(torch.zeros((2, 3, 3)), torch.tensor(raw, dtype=torch.float32), torch.zeros((2, 3, 2)), spec)
    b = rollout_torch(torch.zeros((2, 3, 3), dtype=torch.float64), torch.tensor(raw),
                      torch.zeros((2, 3, 2), dtype=torch.float64), spec)
    for x, y in zip(a, b):
        np.testing.assert_allclose(x.numpy(), y.numpy(), atol=1e-5, rtol=1e-5)


def test_full_speed_discrete_stop_envelope_and_latency_clock(spec):
    cap = spec.vehicle.forward_limit
    direct = stop_envelope([0., 0., 0.], [cap, 0.], [cap, 0.], spec, commit=False)
    after = stop_envelope([0., 0., 0.], [cap, 0.], [cap, 0.], spec, information_time=12., commit=True)
    assert after.delay_steps == 3 and after.braking_steps == 35 and after.stationary_steps == 1
    assert len(after.trajectory.controls) == 40 and after.fits_horizon
    assert after.information_time == 12. and after.activation_time == pytest.approx(12.6)
    assert direct.conservative_center_distance + spec.vehicle.length / 2 + .3 == pytest.approx(14.3986111111)
    assert after.conservative_center_distance + spec.vehicle.length / 2 + .3 == pytest.approx(14.9541666667)
    assert after.conservative_center_distance + spec.vehicle.effective_radius > 15.
    too_long = stop_envelope([0., 0., 0.], [cap, 0.], [cap, 0.], spec, latency=1.)
    assert not too_long.fits_horizon


@pytest.mark.parametrize("previous", [[25/9, 0.], [2.4, .35], [-.5, -.2], [0., .2]])
def test_complete_stop_never_reverses_or_increases_motion(spec, previous):
    controls = brake_sequence(previous, spec)
    prior = np.asarray(previous)
    for control in controls:
        assert not control_error(control, prior, spec)
        assert (abs(control) <= abs(prior) + 1e-8).all()
        prior = control
    np.testing.assert_array_equal(controls[-1], [0., 0.])


@pytest.mark.parametrize("target,previous", [([float('nan'), 0.], [0., 0.]), ([0., 0.], [4., 0.]), ([0., 0.], [0., .35])])
def test_nonfinite_or_illegal_previous_state_is_rejected(spec, target, previous):
    with pytest.raises(ValueError):limit_control_numpy(target, previous, spec)
    with pytest.raises(ValueError):limit_control_torch(torch.tensor([target], dtype=torch.float64),
                                                     torch.tensor([previous], dtype=torch.float64), spec)


def test_flat_footprint_and_detrended_slope_differ_from_raw_height_rms(spec):
    reviewer = FootprintReviewer(spec)
    slope = math_tan(4)
    terrain = local_terrain(spec, slope=slope)
    report = reviewer.pose([0., 0., 0.], terrain, parking=True)
    assert report.passed and report.roughness_rms < .01
    assert report.slope == pytest.approx(slope, abs=.005)
    assert np.std(terrain.snapshot().features[:, 0]) > .05
    assert reviewer.pose([0., 0., 0.], local_terrain(spec, slope=math_tan(7))).passed
    assert reviewer.pose([0., 0., 0.], local_terrain(spec, slope=math_tan(7)), parking=True).reason == "parking_slope"


def math_tan(degrees):
    return np.tan(np.deg2rad(degrees))


@pytest.mark.parametrize("defect,reason", [("wheel_unknown", "unknown_footprint"),
    ("body_unknown", "unknown_footprint"), ("underbody", "body_clearance"),
    ("step", "terrain_step"), ("wheel_rough", "contact_roughness"),
    ("steep", "driving_slope"), ("impassable", "impassable_footprint")])
def test_footprint_rejects_contact_body_and_terrain_failures(spec, defect, reason):
    def missing(xy):
        center = [1.35, .9] if defect == "wheel_unknown" else [0., 0.]
        return np.linalg.norm(xy - center, axis=-1) < .4
    def anomaly(xy, features):
        center = np.linalg.norm(xy, axis=-1) < .35
        if defect == "underbody":features[center, 0] = .5
        if defect == "step":features[center, 0] = .15
        if defect == "wheel_rough":features[np.linalg.norm(xy - [1.35, .9], axis=-1) < .4, 0] = .25
        if defect == "steep":features[center, 1] = math_tan(11)
        if defect == "impassable":features[center, 4] = 0.
    terrain = local_terrain(spec, anomaly=anomaly, remove=missing if defect.endswith("unknown") else None)
    assert FootprintReviewer(spec).pose([0., 0., 0.], terrain).reason == reason


def test_tied_dangerous_height_and_slope_not_hidden_by_nearest_neighbor(spec):
    xy = np.array([[-.1, 0.], [.1, 0.]])
    features = np.array([[0., 0., 0., 0., 1.], [.4, .4, 0., 0., .1]])
    terrain = FootprintTerrain(TerrainSnapshot(xy, features, np.zeros(2), .155), spec)
    f, known = terrain.lookup([[0., 0.]])
    assert known[0] and f[0, 0] == .4 and f[0, 1] == .4 and f[0, 4] == .1
    assert not terrain.lookup([[0., .3]])[1][0]


def test_rotation_sweep_unknown_corner_and_workspace_clearance(spec):
    reviewer = FootprintReviewer(spec)
    previous = [0., .2]; plan = rollout_numpy([0., 0., 0.], [[0., .2]] * 3, spec)
    assert reviewer.trajectory(plan, previous, local_terrain(spec)).passed
    assert reviewer.trajectory(plan, previous, local_terrain(spec), workspace=[[-3., -3.], [3., 3.]]).reason == "workspace_clearance"
    gap = lambda xy: np.linalg.norm(xy - [2.8, 1.95], axis=-1) < .3
    assert reviewer.trajectory(plan, previous, local_terrain(spec, remove=gap)).reason == "unknown_footprint"


def test_sender_prediction_segment_collision_and_forecast_coverage(spec):
    plan = rollout_numpy([0., 0., 0.], [[0., 0.]], spec)
    crossing = FootprintNeighbor.for_spec(3, np.array([0., .2]), np.array([[-8., 0.], [8., 0.]]), np.full(2, 2 * spec.vehicle.effective_radius), spec)
    report = FootprintReviewer(spec).trajectory(plan, [0., 0.], local_terrain(spec), neighbors=[crossing])
    assert report.reason == "neighbor_swept_clearance"
    with pytest.raises(ValueError, match="Duplicate"):
        FootprintReviewer(spec).trajectory(plan, [0., 0.], local_terrain(spec), neighbors=[crossing, crossing])
    with pytest.raises(ValueError, match="cover"):
        crossing.at(1.)


def test_reachable_radius_growth_enters_swept_clearance_bound_once(spec):
    plan = rollout_numpy([0., 0., 0.], [[0., 0.]], spec)
    # All static grid nodes remain nominally outside the radius. The finite
    # review must retain its uncertainty-growth allowance at the boundary.
    growing = FootprintNeighbor.for_spec(3, np.array([0., .2]), np.array([[7.001, 0.], [7.001, 0.]]), np.array([2 * spec.vehicle.effective_radius, 7.]), spec)
    terrain = local_terrain(spec)
    report = FootprintReviewer(spec).trajectory(plan, [0., 0.], terrain, neighbors=[growing])
    assert report.reason == "neighbor_swept_clearance"
    far = FootprintNeighbor.for_spec(3, np.array([0., .2]), np.array([[8., 0.], [8., 0.]]), np.array([2 * spec.vehicle.effective_radius, 7.]), spec)
    assert FootprintReviewer(spec).trajectory(plan, [0., 0.], terrain, neighbors=[far]).passed


def test_admission_verifies_full_speed_rectangle_stop_not_circle_terrain(spec):
    terrain = local_terrain(spec, bounds=(-15., 15., -8., 8.))
    decision = admit_control([25/9, 0.], [25/9, 0.], [0., 0., 0.], terrain, spec)
    assert decision.verified and decision.control[0] == pytest.approx(25/9)
    assert decision.envelope.fits_horizon
    assert decision.envelope.trajectory.states[-1, 0] + spec.vehicle.effective_radius > 15.


def test_reduced_horizon_and_unknown_do_not_silently_certify_braking(spec):
    terrain = local_terrain(spec)
    decision = admit_control([25/9, 0.], [25/9, 0.], [0., 0., 0.], terrain, spec, latency=1., candidates=3)
    assert not decision.verified and decision.status == "unverified_brake"
    assert decision.reason == "stop_horizon_exceeded"
    assert decision.control[0] == pytest.approx(25/9 - .08)
    assert decision.transient_overspeed > 0
    unknown = FootprintTerrain(TerrainSnapshot(np.empty((0, 2)), np.empty((0, 5)), np.empty(0), .155), spec)
    other = admit_control([0., 0.], [0., 0.], [0., 0., 0.], unknown, spec, hold=True)
    assert not other.verified and other.reason == "unknown_footprint"


def test_terrain_kinematic_gain_is_shared_and_actual_speed_is_separate(spec):
    pose = np.array([0., 0., .4]); features = np.array([1., .1, -.05, 0., .8])
    numpy_gain = terrain_gain_numpy(pose, features, spec)
    other = terrain_gain_torch(torch.tensor(pose), torch.tensor(features), spec)
    assert other.item() == pytest.approx(numpy_gain, abs=1e-10)
    _, actual = midpoint_numpy(pose, [1.5, .2], spec, gain=numpy_gain)
    assert actual[0] < 1.5 and actual[1] < .2
    assert terrain_gain_numpy(pose, np.array([0., 3., 3., 0., 0.]), spec) == .25


def test_proxy_batch_execution_async_reset_and_transactional_restore(spec):
    from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_proxy import UpstreamActuatorProxy
    cfg = cfg_from_experiment("configs/experiment/upstream_scale_s00.yaml")
    initial = np.zeros((2, 4, 3)); initial[:, :, 0] = np.arange(4) * 10.
    core = UpstreamActuatorProxy(cfg, initial)
    controls = np.ones((2, 4, 2)); controls[..., 1] = 0.
    first = core.step(controls)
    assert not first["clearance_violation"].any()
    np.testing.assert_allclose(first["applied_control"][..., 0], .04)
    np.testing.assert_allclose(first["actual_velocity"], first["applied_control"])
    state = core.state_dict(); core.step(controls)
    uninterrupted = core.state_dict()
    restored = UpstreamActuatorProxy(cfg, initial); restored.load_state_dict(state); restored.step(controls)
    for key in ("poses", "previous_control", "actual_velocity", "times", "episodes"):
        np.testing.assert_array_equal(restored.state_dict()[key], uninterrupted[key])
    env_one = restored.poses[1].copy(); before = restored.state_dict()
    restored.reset([0], initial[:1])
    np.testing.assert_array_equal(restored.poses[1], env_one)
    assert restored.times[0] == 0 and restored.times[1] == pytest.approx(.4)
    bad = restored.state_dict(); bad["previous_control"][0, 0, 0] = 5.
    unchanged = restored.poses.copy()
    with pytest.raises(ValueError, match="actuator state"):restored.load_state_dict(bad)
    np.testing.assert_array_equal(restored.poses, unchanged)
    bad = before; bad["contracts"]["planner_protocol"] = "planner_protocol_v1"
    with pytest.raises(ValueError, match="mismatch"):restored.load_state_dict(bad)


def test_injected_latency_advances_held_control_only_once(spec):
    from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_proxy import UpstreamActuatorProxy
    cfg = cfg_from_experiment("configs/experiment/upstream_scale_s00.yaml")
    initial = np.zeros((2, 4, 3)); initial[:, :, 1] = np.arange(4) * 20.
    core = UpstreamActuatorProxy(cfg, initial)
    target = np.zeros((2, 4, 2)); target[..., 0] = 25/9
    for _ in range(70):core.step(target)
    state = core.state_dict(); start = core.poses[0, 0].copy(); now = float(core.times[0])
    envelope = stop_envelope(start, core.previous_control[0, 0], core.previous_control[0, 0], spec, information_time=now)
    for _ in range(envelope.delay_steps):core.step(core.previous_control.copy())
    np.testing.assert_allclose(core.poses[0, 0], envelope.trajectory.states[3], atol=1e-10, rtol=1e-8)
    first = core.step(core.previous_control.copy())
    assert first["activation_time"][0] == pytest.approx(envelope.activation_time)
    np.testing.assert_allclose(core.poses[0, 0], envelope.trajectory.states[4], atol=1e-10, rtol=1e-8)
    assert state["times"][0] == pytest.approx(14.)


@pytest.mark.parametrize("latency", [.2, .5, 1.])
@pytest.mark.parametrize("deceleration", [.2, .4, .6])
def test_latency_and_braking_sensitivity_require_complete_joint_stop_horizon(spec, latency, deceleration):
    changed = replace(spec, vehicle=replace(spec.vehicle, linear_deceleration=deceleration))
    cap = spec.vehicle.forward_limit
    envelope = stop_envelope([0., 0., 0.], [cap, 0.], [cap, 0.], changed, latency=latency)
    n = int(np.ceil(cap / (deceleration * .2)))
    delay = int(np.ceil(latency / .2))
    assert envelope.braking_steps == n
    assert envelope.fits_horizon == (delay + 1 + n + 1 <= 40)
    expected = .2 * (cap * (delay + 1) + sum(max(0., cap - i * deceleration * .2) for i in range(n)))
    assert envelope.conservative_center_distance == pytest.approx(expected, abs=1e-10)


def test_contract_evidence_rejects_missing_collected_tests_and_foreign_run(tmp_path):
    from planner_evidence import EvidenceRun, file_sha256, json_write
    from upstream_evidence import require_contract_tests
    config = "configs/experiment/upstream_scale_s01.yaml"
    run = EvidenceRun(tmp_path / "run", config)
    tests = ("tests/test_upstream_spec.py",)
    path = run.path / "metrics/contracts.xml"
    path.write_text('<testsuites><testsuite tests="27" errors="0"><testcase classname="tests.test_upstream_spec" name="fake"/></testsuite></testsuites>')
    json_write(path.with_suffix(".source.json"), {"source_snapshot": run.identity["source_snapshot"],
              "test_sources": {name: file_sha256(name) for name in tests}})
    with pytest.raises(ValueError, match="Incomplete"):
        require_contract_tests(path, run, tests)
    with pytest.raises(ValueError, match="current run"):
        require_contract_tests(tmp_path / "foreign.xml", run, tests)


@pytest.mark.parametrize("change", ["clock", "device"])
def test_offline_probe_rejects_silent_realtime_or_cuda_coverage(change):
    from diagnose_upstream_execution import diagnose
    cfg = cfg_from_experiment("configs/experiment/upstream_scale_s01.yaml")
    cfg = replace(cfg, clock_mode="realtime_measured_latency") if change == "clock" else replace(cfg, planner_device="cuda")
    with pytest.raises(ValueError, match="CPU/offline"):
        diagnose(cfg, None)
