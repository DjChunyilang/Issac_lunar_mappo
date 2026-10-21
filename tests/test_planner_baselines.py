"""T05 adapters, real low-dimensional NMPC, and frozen categorical semantics."""

from copy import deepcopy
from dataclasses import replace
import os

import numpy as np
import pytest

from test_planner_protocol import adapter as legacy_adapter, request, EchoSolver
from _common import cfg_from_experiment
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.baselines import (
    NMPCPolarAdapter,
    DeterministicAdapter,
    PrimitiveAdapter,
    PrimitiveReplayInput,
    primitive_library,
    primitive_fingerprint,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.protocol import LocalIntent
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.planner import (
    NMPCPlanner,
    build_solver,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.contracts import contract_metadata
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.environment import (
    LocalGoalEnvironment,
)


def low_input(goal=(0.8, 0), **kwargs):
    item = request(**kwargs)
    return replace(item, intent=LocalIntent(np.asarray(goal), False, item.now, item.now + 1))


@pytest.mark.parametrize("goal,yaw", [((1, 0), 0), ((-1, 0), 0), ((0, 1), np.pi / 2), ((0, 0), 0)])
def test_low_dim_defaults_frozen_and_restored(goal, yaw):
    p = NMPCPolarAdapter(NMPCPlanner(EchoSolver(), 0.75, 10))
    item = low_input(goal)
    output = p.plan(item)
    assert output.feedback.reason != "unsupported_intent"
    assert p.resolved_intent.terminal_yaw == pytest.approx(yaw)
    assert p.resolved_intent.speed_cap == 1.15
    state = p.state_dict()
    q = NMPCPolarAdapter(NMPCPlanner(EchoSolver(), 0.75, 10))
    q.load_state_dict(state)
    changed = replace(item, ego_pose=np.array([0.1, 0.2, 0.3]), now=0.2, times=item.times + 0.2)
    q.plan(changed)
    assert q.resolved_intent.terminal_yaw == p.resolved_intent.terminal_yaw
    bad = deepcopy(state)
    bad["state_version"] = 1
    with pytest.raises(ValueError):
        q.load_state_dict(bad)


def test_complete_legacy_request_exact_parity():
    item = request()
    a = legacy_adapter()
    b = NMPCPolarAdapter(NMPCPlanner(EchoSolver(), 0.75, 10))
    x, y = a.plan(item), b.plan(item)
    np.testing.assert_array_equal(x.trajectory.controls, y.trajectory.controls)
    np.testing.assert_array_equal(x.trajectory.states, y.trajectory.states)
    assert x.feedback.status == y.feedback.status


@pytest.mark.parametrize("geometry", ["line", "quintic"])
def test_deterministic_physical_prediction_and_restore(geometry):
    p = DeterministicAdapter(geometry)
    item = low_input()
    output = p.plan(item)
    assert output.valid and output.trajectory.states[1, 0] > 0
    assert output.trajectory.states[1, 0] <= 0.060001  # slew, not path interpolation
    q = DeterministicAdapter(geometry)
    q.load_state_dict(p.state_dict())
    shifted = replace(
        item,
        now=0.2,
        times=item.times + 0.2,
        ego_pose=output.trajectory.states[1],
        previous_control=output.trajectory.controls[0],
    )
    np.testing.assert_array_equal(
        p.plan(shifted).trajectory.controls, q.plan(shifted).trajectory.controls
    )
    q.reset()
    assert q.previous is None


def test_expired_and_hold_do_not_follow_old_target():
    p = DeterministicAdapter()
    item = low_input(control=(0.8, 0.5))
    expired = replace(item, now=2, times=item.times + 2)
    out = p.plan(expired)
    assert not out.valid and out.fallback_reason == "expired_intent"
    np.testing.assert_allclose(out.trajectory.controls[0], [0.5, 0], atol=1e-12)
    held = replace(item, intent=LocalIntent.for_hold(item.ego_pose, 0))
    out = p.plan(held)
    assert not out.valid
    assert np.max(abs(out.trajectory.controls[-1])) == 0


@pytest.mark.parametrize("index", range(47))
def test_frozen_primitive_native_replay(index):
    p = PrimitiveAdapter()
    item = low_input()
    out = p.plan(PrimitiveReplayInput(item, index))
    out.validate(item)
    assert primitive_library()["count"] == 47 and len(primitive_fingerprint()) == 64
    assert np.isfinite(out.trajectory.controls).all()
    if index in (40, 41, 42):
        assert out.trajectory.controls[0, 0] <= 0
    if index in (43, 44):
        assert out.trajectory.controls[0, 0] == 0
        assert out.trajectory.controls[0, 1] * (1 if index == 43 else -1) > 0
    if index == 0:
        assert np.max(abs(out.trajectory.controls)) == 0
    with pytest.raises(ValueError):
        p.plan(item)


@pytest.mark.parametrize("name", ["deterministic_line_v1", "deterministic_quintic_v1"])
def test_non_solver_factory_and_environment(name, tmp_path):
    cfg = cfg_from_experiment("configs/experiment/t05_planner_baselines.yaml")
    cfg.simulation.num_envs = 2
    cfg.local_goal_contract.planner_adapter = name
    cfg.local_goal_contract.planner_revision = "deterministic_v1"

    def forbidden(*a, **kw):
        raise AssertionError("acados construction forbidden")

    env = LocalGoalEnvironment(cfg, tmp_path, solver_factory=forbidden)
    assert len(env.planners) == 8
    before = [p.state_dict() for p in env.planners]
    env.reset([0])
    assert before[4:] == [p.state_dict() for p in env.planners[4:]]
    state = env.state_dict()
    env.load_state_dict(state)
    env.pool.shutdown()
    assert contract_metadata(cfg)["planner_revision"] == "deterministic_v1"


@pytest.mark.parametrize("goal", [(0.8, 0), (0, 0.8), (-0.8, 0), (0, 0)])
def test_real_acados_low_dimensional_pair(tmp_path, goal, record_property):
    assert os.environ.get("ACADOS_SOURCE_DIR"), "Actual acados required; no skip"
    item = low_input(goal)
    left = NMPCPolarAdapter(NMPCPlanner(build_solver(tmp_path / "new", 0.75), 0.75, 10))
    right = legacy_adapter()  # replace its deterministic solver with actual acados
    right._planner = NMPCPlanner(build_solver(tmp_path / "old", 0.75), 0.75, 10)
    a = left.plan(item)
    b = right.plan(replace(item, intent=left.resolved_intent))
    assert a.feedback.reason != "unsupported_intent"
    assert a.valid and b.valid
    for x, y in (
        (a.trajectory.states, b.trajectory.states),
        (a.trajectory.controls, b.trajectory.controls),
    ):
        np.testing.assert_allclose(x, y, atol=1e-7, rtol=1e-6)
    record_property(
        "paired_max_error", float(np.max(abs(a.trajectory.controls - b.trajectory.controls)))
    )
    snapshot = left.state_dict()
    left.load_state_dict(snapshot)
    if goal == (-0.8, 0):
        assert a.trajectory.controls[0, 0] < 0


def test_primitive_library_frozen_content_matches_decoder():
    import json
    from pathlib import Path

    frozen = json.loads(Path("configs/experiment/primitive47_library_v1.json").read_text())
    assert frozen["sha256"] == primitive_fingerprint()
    assert frozen["library"] == json.loads(json.dumps(primitive_library()))
    assert len(frozen["library"]["entries"]) == 47


@pytest.mark.parametrize("expired", [False, True])
def test_real_acados_polar_hold_and_expiry(tmp_path, expired):
    assert os.environ.get("ACADOS_SOURCE_DIR"), "Actual acados required; no skip"
    r = low_input(control=(0.8, 0.5))
    if expired:
        r = replace(r, now=2, times=r.times + 2)
    else:
        r = replace(r, intent=LocalIntent.for_hold(r.ego_pose, 0))
    p = NMPCPolarAdapter(NMPCPlanner(build_solver(tmp_path, 0.75), 0.75, 10))
    out = p.plan(r)
    assert not out.valid
    if expired:
        assert out.fallback_reason == "expired_intent"
    else:
        assert out.feedback.emergency_brake and out.fallback_reason in (
            "speed_limit",
            "solver_failure",
        )
    np.testing.assert_allclose(out.trajectory.controls[0], [0.5, 0], atol=1e-7, rtol=1e-6)
    assert abs(out.trajectory.controls[-1]).max() == 0


def test_native_primitive_rejects_non_protocol_context():
    with pytest.raises(ValueError):
        PrimitiveReplayInput(object(), 0).validate()
