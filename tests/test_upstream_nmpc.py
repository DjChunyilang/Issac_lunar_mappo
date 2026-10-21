"""Real acados and analytic contracts for the separate 40-step NMPC model.

These tests deliberately do not skip a missing required acados installation.
Use the documented acados runtime environment for this test module.
"""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from _common import cfg_from_experiment
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_nmpc import (
    UpstreamNmpcAdapter, UpstreamNmpcSolver, build_upstream_ocp, build_upstream_solver,
    corridor_support, guided_controls, local_rollout, solver_identity,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_execution import (
    brake_sequence, control_error, midpoint_numpy, terrain_gain_numpy,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_footprint import FootprintTerrain, FootprintReviewer
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.protocol import TerrainSnapshot
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_planning import UpstreamIntent, UpstreamPlannerInput


@pytest.fixture(scope="module")
def cfg():
    return replace(cfg_from_experiment("configs/experiment/upstream_scale_s01_review_v2.yaml"),
                   planner_adapter="nmpc_upstream_v1", simulated_latency=0.)


def terrain(spec, slope=0.):
    x, y = np.meshgrid(np.arange(-15., 15.01, .2), np.arange(-8., 8.01, .2), indexing="ij")
    xy = np.c_[x.ravel(), y.ravel()]
    feature = np.zeros((len(xy), 5)); feature[:, 0] = slope * xy[:, 0]
    feature[:, 1] = slope; feature[:, 4] = 1.
    return FootprintTerrain(TerrainSnapshot(xy, feature, np.zeros(len(xy)), spec.perception.support_radius), spec)


def item(cfg, goal=(4., 0.), *, pose=None, previous=None, hold=False, now=0., expires=1., slope=0.):
    return UpstreamPlannerInput(env_id=0, agent_id=0,
        ego_pose=np.zeros(3) if pose is None else np.asarray(pose, dtype=float),
        ego_velocity=np.zeros(2) if previous is None else np.asarray(previous, dtype=float),
        intent=UpstreamIntent(np.asarray(goal), hold, 0., expires), terrain=terrain(cfg.spec, slope),
        neighbors=(), previous_control=np.zeros(2) if previous is None else np.asarray(previous, dtype=float),
        now=now, times=cfg.spec.grid.times(now), workspace=(-100., -100., 100., 100.), spec=cfg.spec)


@pytest.fixture(scope="module")
def solver(cfg, tmp_path_factory):
    return build_upstream_solver(cfg.spec, tmp_path_factory.mktemp("upstream_acados"))


def test_real_model_midpoint_and_terrain_gain_match_reference(cfg):
    import casadi as ca
    ocp = build_upstream_ocp(cfg.spec); model = ocp.model
    evaluate = ca.Function("step_reference", [model.x, model.u, model.p], [model.disc_dyn_expr])
    rng = np.random.default_rng(31)
    for _ in range(30):
        pose = rng.normal(size=3); control = rng.uniform([-.5, -.2], [2.4, .2])
        feature = np.array([0., .08, -.03, 0., .85])
        p = np.zeros(model.p.shape[0]); p[4:9] = feature
        actual = np.asarray(evaluate(np.r_[pose, [0., 0.]], control, p)).ravel()
        expected, _ = midpoint_numpy(pose, control, cfg.spec, gain=terrain_gain_numpy(pose, feature, cfg.spec))
        np.testing.assert_allclose(actual[:3], expected, atol=1e-10, rtol=1e-8)
        np.testing.assert_array_equal(actual[3:], control)
    assert ocp.solver_options.N_horizon == 40 and ocp.solver_options.tf == 8.


@pytest.mark.parametrize("previous,command,legal", [
    ([0., 0.], [.04, .03], True), ([1., 0.], [.92, .03], True),
    ([-.3, 0.], [-.34, 0.], True), ([.03, 0.], [-.01, 0.], False),
    ([1., 0.], [1.06, 0.], False), ([.02, .18], [.02, .21], False),
    ([25/9, 0.], [25/9, .03], False),
])
def test_real_model_directional_wheel_and_yaw_constraints(cfg, previous, command, legal):
    import casadi as ca
    ocp = build_upstream_ocp(cfg.spec); m = ocp.model
    evaluate = ca.Function("actuator_constraints", [m.x, m.u, m.p], [m.con_h_expr])
    p = np.zeros(m.p.shape[0]); p[3] = cfg.spec.vehicle.forward_limit
    p[-3:-1] = np.sign(previous)
    p[-1] = 0. if abs(command[0]) <= cfg.spec.vehicle.low_speed_threshold else np.sign(command[0])
    h = np.asarray(evaluate(np.r_[np.zeros(3), previous], command, p)).ravel()[:12]
    passed = bool(((h >= ocp.constraints.lh[:12] - 1e-10) & (h <= ocp.constraints.uh[:12] + 1e-10)).all())
    assert passed == legal
    assert (not control_error(command, previous, cfg.spec)) == legal


def test_support_selection_covers_far_rolling_and_stop_corridors(cfg):
    inp = item(cfg, previous=[25/9, 0.])
    warm = local_rollout(inp, np.tile([1.8, 0.], (40, 1)))
    stop = local_rollout(inp, brake_sequence(inp.previous_control, cfg.spec))
    support = corridor_support(inp, warm, stop)
    assert len(support.xy) > 128 and support.xy[:, 0].max() > 13.
    assert support.reference_nodes == 41 and support.stop_nodes > 30
    assert support.warm_known and support.stop_known
    for state in [warm.states[-1], stop.states[-1]]:
        disks = support.stage_disks(state[:2])
        assert (np.linalg.norm(disks[:, :2] - state[:2], axis=-1) <= disks[:, 2] + 1e-9).any()
    np.testing.assert_array_equal(support.radii, cfg.spec.perception.support_radius)


@pytest.mark.parametrize("goal,yaw,sign", [((4., 0.), 0., 1), ((-2., 0.), 0., -1),
                                           ((2., 2.), np.pi/4, 1), ((0., 3.), np.pi/2, 2), ((0., 0.), 0., 0)])
def test_real_acados_move_cases_have_legal_controls_and_progress(cfg, solver, goal, yaw, sign):
    inp = item(cfg, goal); engine = UpstreamNmpcSolver(cfg.spec, solver=solver, deadline=30.)
    engine.reset(); controls, detail = engine.solve(inp, yaw, cfg.spec.vehicle.forward_limit)
    assert controls is not None, detail
    assert detail["solver_status"] in (0, 2) and 1 <= detail["solver_passes"] <= 2
    previous = inp.previous_control
    for control in controls:
        assert not control_error(control, previous, cfg.spec)
        previous = control
    trajectory = local_rollout(inp, controls)
    assert trajectory.states.shape == (41, 3) and trajectory.controls.shape == (40, 2)
    if sign:
        assert np.linalg.norm(trajectory.states[-1, :2] - goal) < np.linalg.norm(goal)
        if abs(sign) == 1:assert np.sign(controls[0, 0]) == sign
    else:
        np.testing.assert_allclose(controls, 0., atol=1e-8)
    report = FootprintReviewer(cfg.spec).trajectory(trajectory, inp.previous_control, inp.terrain,
            neighbors=inp.neighbors, workspace=np.asarray(inp.workspace).reshape(2, 2))
    assert report.passed, report


def test_real_acados_nonflat_dynamics_and_low_speed_cap(cfg, solver):
    inp = item(cfg, slope=.05)
    engine = UpstreamNmpcSolver(cfg.spec, solver=solver, deadline=30.); engine.reset()
    controls, detail = engine.solve(inp, 0., .6)
    assert controls is not None, detail
    assert abs(controls[:, 0]).max() <= .6 + 1e-6
    trajectory = local_rollout(inp, controls)
    assert np.max(abs(trajectory.velocities[:, 0] - controls[:, 0])) > 1e-3
    assert trajectory.states[-1, 0] > 1.


def test_cache_identity_is_bound_to_complete_spec_and_implementation(cfg, tmp_path):
    identity = solver_identity(cfg.spec)
    assert identity["steps"] == 40 and identity["dt"] == .2
    assert identity["profile_fingerprint"] == cfg.spec.fingerprint
    changed = replace(cfg.spec, vehicle=replace(cfg.spec.vehicle, reverse_limit=.4))
    assert solver_identity(changed) != identity
    with pytest.raises(ValueError, match="Missing upstream acados cache"):
        build_upstream_solver(cfg.spec, tmp_path, build=False)


class UnusedSolver:
    def reset(self):pass
    def solve(self):raise AssertionError("HOLD/expired must not invoke acados")


def test_implicit_constraints_freeze_and_restore_without_solver_objects(cfg):
    adapter = UpstreamNmpcAdapter(cfg, solver=UnusedSolver())
    inp = item(cfg, (-3., 0.))
    assert adapter._resolve(inp) == (0., cfg.spec.vehicle.forward_limit)
    moved = replace(inp, ego_pose=np.array([0., 1., .8]))
    assert adapter._resolve(moved) == adapter._resolve(inp)
    state = adapter._adapter_state()
    other = UpstreamNmpcAdapter(cfg, solver=UnusedSolver())
    other._validate_adapter_state(state); other._load_adapter_state(state)
    assert other._resolve(moved) == adapter._resolve(moved)
    state["model_identity"]["dt"] = .1
    with pytest.raises(ValueError, match="model/state mismatch"):
        other._validate_adapter_state(state)


@pytest.mark.parametrize("hold,now", [(True, 0.), (False, 1.2)])
def test_public_hold_and_expiry_use_common_limited_stop(cfg, hold, now):
    inp = item(cfg, previous=[.4, .1], hold=hold, now=now)
    adapter = UpstreamNmpcAdapter(cfg, solver=UnusedSolver())
    output = adapter.plan(inp)
    assert np.isfinite(output.trajectory.controls).all()
    np.testing.assert_allclose(output.trajectory.controls[-1], 0., atol=1e-9)
    prior = inp.previous_control
    for command in output.trajectory.controls:
        assert not control_error(command, prior, cfg.spec)
        assert (np.abs(command) <= np.abs(prior) + 1e-8).all()
        prior = command


def test_real_adapter_snapshot_next_step_and_delayed_information_reserve(cfg, solver):
    adapter = UpstreamNmpcAdapter(cfg, solver=solver, deadline=30.)
    first_input = item(cfg)
    first = adapter.plan(first_input)
    assert first.valid, first.diagnostics
    assert first.diagnostics["selected_source"] == "nmpc"
    assert first.feedback.solver_passes >= 1
    state = adapter.state_dict()
    next_input = replace(first_input, ego_pose=first.trajectory.states[1],
                         ego_velocity=first.trajectory.velocities[0],
                         previous_control=first.trajectory.controls[0], now=.2,
                         times=cfg.spec.grid.times(.2))
    expected = adapter.plan(next_input)
    restored = UpstreamNmpcAdapter(cfg, solver=solver, deadline=30.)
    restored.load_state_dict(state)
    actual = restored.plan(next_input)
    assert actual.valid and expected.valid
    np.testing.assert_allclose(actual.trajectory.controls, expected.trajectory.controls, atol=1e-7, rtol=1e-6)
    np.testing.assert_allclose(actual.trajectory.states, expected.trajectory.states, atol=1e-7, rtol=1e-6)
    assert actual.diagnostics["frozen_terminal_yaw"] == first.diagnostics["frozen_terminal_yaw"]
    delayed_cfg = replace(cfg, simulated_latency=.5)
    delayed = item(delayed_cfg, pose=[.48, 0., 0.], previous=[.8, 0.], now=.6)
    delayed = replace(delayed, information_pose=np.zeros(3), information_time=0.)
    output = UpstreamNmpcAdapter(delayed_cfg, solver=solver, deadline=30.).plan(delayed)
    assert output.valid, output.diagnostics
    assert output.diagnostics["delay_steps"] == 3 and output.diagnostics["information_reserve_verified"]


class FailedSolver:
    def __init__(self, status=4, nonfinite=False):
        self.status = status; self.nonfinite = nonfinite; self.values = {}; self.calls = 0
    def reset(self):self.values = {}
    def constraints_set(self, *args):pass
    def set(self, k, key, value):self.values[(k, key)] = np.array(value)
    def get(self, k, key):
        value = self.values[(k, key)].copy()
        if self.nonfinite:value[0] = np.nan
        return value
    def solve(self):
        self.calls += 1
        return self.status


@pytest.mark.parametrize("nonfinite,reason,passes", [(False, "solver_failure", 2), (True, "nonfinite_solver_output", 1)])
def test_solver_failures_have_bounded_recovery_and_preserve_actual_reason(cfg, nonfinite, reason, passes):
    solver = FailedSolver(nonfinite=nonfinite)
    output = UpstreamNmpcAdapter(cfg, solver=solver, deadline=30.).plan(item(cfg))
    assert not output.valid and output.fallback_reason == reason
    assert solver.calls == passes == output.feedback.solver_passes
    assert np.isfinite(output.trajectory.controls).all()
    np.testing.assert_allclose(output.trajectory.controls, 0., atol=1e-9)


def test_solver_timeout_returns_no_stale_candidate(cfg, monkeypatch):
    from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal import upstream_nmpc
    ticks = iter([0., 1., 2.])
    monkeypatch.setattr(upstream_nmpc, "time", SimpleNamespace(monotonic=lambda: next(ticks)))
    solver = FailedSolver(status=0)
    controls, detail = UpstreamNmpcSolver(cfg.spec, solver=solver, deadline=.1).solve(item(cfg), 0., 1.)
    assert controls is None and detail["fallback_reason"] == "solver_timeout" and solver.calls == 1


def test_second_pass_timeout_discards_first_candidate(cfg, monkeypatch):
    from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal import upstream_nmpc
    class InconsistentShooting(FailedSolver):
        def get(self, k, key):
            value = super().get(k, key)
            if key == "x":value[0] += 1.
            return value
    ticks = iter([0., .01, 1., 2.])
    monkeypatch.setattr(upstream_nmpc, "time", SimpleNamespace(monotonic=lambda: next(ticks)))
    solver = InconsistentShooting(status=0)
    controls, detail = UpstreamNmpcSolver(cfg.spec, solver=solver, deadline=.1).solve(item(cfg), 0., 1.)
    assert controls is None and detail["fallback_reason"] == "solver_timeout" and solver.calls == 2
