"""T03: replaceable planner contracts, parity and actual acados bridge validation."""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import os
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from _common import cfg_from_experiment, load_yaml
from train_local_goal_nmpc import validate_checkpoint_contract, validate_training_gate
from test_local_goal_nmpc import FailingSolver, flat_memory
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.contracts import (
    DT,
    N,
    LocalGoalRequest,
    NeighborPrediction,
    PlannerFeedback,
    TimedTrajectory,
    contract_metadata,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.environment import (
    LocalGoalEnvironment,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.planner import (
    NMPCPlanner,
    build_solver,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.protocol import (
    LocalIntent,
    TerrainView,
    IdentifiedNeighbor,
    PlannerInput,
    PlannerOutput,
    NMPCAdapter,
    plan_batch,
    execution_passthrough,
)


class EchoSolver(FailingSolver):
    """Deterministically returns the configured warm candidate; not a planning baseline."""

    def __init__(self, status=0):
        self.status = status
        self.calls = 0
        self.values = {}

    def set(self, k, key, value):
        self.values[k, key] = np.array(value).copy()

    def get(self, k, key):
        return self.values[k, key]

    def solve(self):
        self.calls += 1
        return self.status


def request(*, pose=None, action=(0.4, 0, 0, 0), now=0.0, control=None, memory=None):
    pose = np.zeros(3) if pose is None else np.asarray(pose)
    goal = LocalGoalRequest.from_action(action, pose, now)
    return PlannerInput(
        0,
        0,
        pose,
        np.zeros(2),
        LocalIntent.from_request(goal),
        TerrainView(flat_memory() if memory is None else memory),
        (),
        np.zeros(2) if control is None else control,
        now,
        now + np.arange(N + 1) * DT,
    )


def adapter(status=0, deadline=10):
    return NMPCAdapter(NMPCPlanner(EchoSolver(status), 0.75, deadline=deadline))


class StationaryTestAdapter:
    def reset(self):
        pass

    def state_dict(self):
        return {"test_only": True}

    def load_state_dict(self, state):
        assert state == self.state_dict()

    def plan(self, item):
        item.validate()
        trajectory = TimedTrajectory(
            np.tile(item.ego_pose, (N + 1, 1)),
            np.zeros((N, 2)),
            np.zeros((N, 2)),
            item.times.copy(),
            item.now,
        )
        return PlannerOutput(trajectory, PlannerFeedback(status="feasible"), True)


def test_same_input_consumed_by_distinct_adapters_and_sparse_batch():
    item = request(action=(0, 0, 0, -1))
    a, b = adapter(), StationaryTestAdapter()
    for p in (a, b):
        output = plan_batch([p], [item])[0]
        assert output.valid and output.total_cost is None
        assert output.predicted_states is output.trajectory.states
        assert output.nominal_controls is output.trajectory.controls
    inputs = [replace(item, env_id=2, agent_id=3), replace(item, env_id=0, agent_id=1)]
    output = plan_batch([StationaryTestAdapter(), StationaryTestAdapter()], inputs)
    assert len(output) == 2
    assert plan_batch([], []) == []
    with pytest.raises(ValueError):
        plan_batch([a, a], inputs)
    with pytest.raises(ValueError):
        plan_batch([a, b], [item, item])


@pytest.mark.parametrize(
    "change",
    [
        {"version": "bad"},
        {"ego_pose": [0, 0]},
        {"ego_pose": [0, np.nan, 0]},
        {"ego_velocity": [np.inf, 0]},
        {"previous_control": [0]},
        {"env_id": -1},
        {"agent_id": True},
        {"now": -1},
        {"times": np.arange(N + 1) * 0.1},
        {"times": np.zeros(2)},
        {"terrain": flat_memory()},
    ],
)
def test_invalid_input_rejected_before_solve(change):
    bridge = adapter()
    with pytest.raises(ValueError):
        bridge.plan(replace(request(), **change))
    assert bridge._planner.solver.calls == 0


@pytest.mark.parametrize(
    "changes",
    [
        {"goal_xy": [0]},
        {"hold": 1},
        {"timestamp": 0.1},
        {"expires_at": -1},
        {"terminal_yaw": np.nan},
        {"speed_cap": -1},
        {"speed_cap": np.inf},
        {"timestamp": np.nan},
        {"hold": True, "terminal_yaw": None},
    ],
)
def test_invalid_intent_rejected_before_solve(changes):
    item, bridge = request(), adapter()
    with pytest.raises(ValueError):
        bridge.plan(replace(item, intent=replace(item.intent, **changes)))
    assert bridge._planner.solver.calls == 0


def test_frozen_intent_coordinate_conversion_and_terrain_snapshot():
    pose = np.array([1.0, -0.2, np.pi / 2])
    legacy = LocalGoalRequest.from_action([0.5, 0, 0.25, 0], pose, 3)
    intent = LocalIntent.from_request(legacy)
    np.testing.assert_allclose(intent.goal_xy, [1, 0.6])
    assert intent.terminal_yaw == pytest.approx(3 * np.pi / 4)
    legacy.target[:] = 99
    np.testing.assert_allclose(intent.goal_xy, [1, 0.6])
    with pytest.raises(ValueError):
        intent.goal_xy[0] = 8
    memory = flat_memory()
    view = TerrainView(memory)
    before = view.lookup([[0, 0]])[0]
    memory.reset()
    np.testing.assert_array_equal(before, view.lookup([[0, 0]])[0])
    assert view.safe([[0, 0]])[0]
    for name in ("core", "terrain_runtime", "oracle_point", "observe", "reset"):
        assert not hasattr(view, name)


@pytest.mark.parametrize("now,reason", [(1.0, ""), (1.000001, "expired_intent")])
def test_expiry_boundary_and_brake_without_solver(now, reason):
    item = request(control=np.array([0.9, 1.2]))
    item = replace(item, now=now, times=now + np.arange(N + 1) * DT)
    bridge = adapter()
    result = bridge.plan(item)
    if reason:
        assert result.fallback_reason == reason and not result.valid
        assert bridge._planner.solver.calls == 0
        assert bridge.state_dict()["previous"] is None
        np.testing.assert_allclose(result.nominal_controls[0], [0.6, 0.6])
    else:
        assert bridge._planner.solver.calls > 0
        assert result.fallback_reason != "expired_intent"


def test_unsupported_low_dimensional_intent_and_frozen_hold():
    item, bridge = request(), adapter()
    output = bridge.plan(
        replace(item, intent=replace(item.intent, terminal_yaw=None, speed_cap=None))
    )
    assert output.fallback_reason == "unsupported_intent" and not output.valid
    assert bridge._planner.solver.calls == 0
    hold = LocalIntent.for_hold(np.array([0, 0, 0.3]), 0)
    moved = replace(item, ego_pose=[0.02, 0, 0.4], intent=hold, previous_control=[0.8, 0.5])
    result = bridge.plan(moved)
    assert hold.terminal_yaw == 0.3
    np.testing.assert_array_equal(hold.goal_xy, [0, 0])
    assert (
        np.max(
            np.abs(np.diff(np.vstack([moved.previous_control, result.nominal_controls]), axis=0))[
                :, 0
            ]
        )
        <= 0.3001
    )


@pytest.mark.parametrize("status", [0, 2, 4])
def test_deterministic_direct_bridge_parity(status, record_property):
    item = request()
    direct, bridge = NMPCPlanner(EchoSolver(status), 0.75, deadline=10), adapter(status)
    intent = item.intent
    goal = LocalGoalRequest(np.r_[intent.goal_xy, intent.terminal_yaw], intent.speed_cap, 0, 1)
    old, feedback = direct.plan(item.ego_pose, goal, item.terrain, [], item.previous_control, 0)
    new = bridge.plan(item)
    for field in ("states", "controls", "velocities", "times"):
        np.testing.assert_array_equal(getattr(old, field), getattr(new.trajectory, field))
    assert feedback.status == new.feedback.status
    assert feedback.reason == new.fallback_reason
    record_property("max_trajectory_error", 0.0)
    decision = execution_passthrough(item.ego_pose, new)
    np.testing.assert_array_equal(decision.nominal_control, decision.applied_control)
    assert decision.safety_status == "projection_disabled"


def test_failure_previous_revalidation_and_timeout():
    item = request(action=(0, 0, 0, -1))
    bridge = adapter()
    assert bridge.plan(item).valid
    bridge._planner.solver.status = 4
    result = bridge.plan(item)
    assert (
        result.valid
        and result.feedback.used_previous
        and result.fallback_reason == "solver_failure"
    )
    empty = flat_memory()
    empty.reset()
    result = bridge.plan(replace(item, terrain=TerrainView(empty)))
    assert not result.valid and result.feedback.emergency_brake
    timed = adapter(deadline=-1).plan(item)
    assert not timed.valid and timed.fallback_reason == "solver_timeout"


def test_prefix_feedback_mapping(monkeypatch):
    bridge, item = adapter(), request(action=(0, 0, 0, -1))
    original = bridge._planner.plan

    def prefix(*args):
        trajectory, feedback = original(*args)
        feedback.status, feedback.prefix_steps = "feasible_prefix", 3
        return trajectory, feedback

    monkeypatch.setattr(bridge._planner, "plan", prefix)
    output = bridge.plan(item)
    assert output.valid and output.feedback.prefix_steps == 3


@pytest.mark.parametrize(
    "change",
    [
        {"version": "bad"},
        {"valid": False},
        {"total_cost": np.inf},
        {"fallback_reason": "invented"},
    ],
)
def test_output_consistency(change):
    item = request(action=(0, 0, 0, -1))
    with pytest.raises(ValueError):
        replace(adapter().plan(item), **change).validate(item)


def test_neighbor_identity_and_stale_conservative_input():
    item = request(now=2)
    prediction = NeighborPrediction(np.tile([1, 1], (N + 1, 1)), np.ones(N + 1), 0, 1)
    peer = IdentifiedNeighbor(3, prediction)
    item = replace(item, neighbors=(peer,))
    item.validate()  # expired communicated prediction is permitted, never refreshed with truth
    prediction.positions[:] = 999
    np.testing.assert_array_equal(peer.prediction.positions[0], [1, 1])
    for peers in (
        (peer, peer),
        (IdentifiedNeighbor(0, prediction),),
        (IdentifiedNeighbor(2, replace(prediction, version=99)),),
    ):
        with pytest.raises(ValueError):
            replace(item, neighbors=peers).validate()


def test_adapter_state_restore_is_detached_versioned_and_solver_free():
    item, a, b = request(), adapter(), adapter()
    a.plan(item)
    saved = a.state_dict()
    b.load_state_dict(saved)
    for key in ("controls", "states"):
        np.testing.assert_array_equal(
            getattr(a.plan(item).trajectory, key), getattr(b.plan(item).trajectory, key)
        )
    saved["previous"].states[:] = 100
    assert not (b.state_dict()["previous"].states == 100).any()
    assert set(saved) == {
        "protocol",
        "adapter",
        "state_version",
        "planner_revision",
        "previous",
        "blocked",
    }
    for key in ("protocol", "adapter", "state_version", "planner_revision"):
        bad = a.state_dict()
        bad.pop(key)
        with pytest.raises(ValueError):
            b.load_state_dict(bad)
    a.reset()
    assert a.state_dict()["previous"] is None and a.state_dict()["blocked"] == 0


@pytest.fixture
def env():
    torch.set_num_threads(1)
    torch.manual_seed(23)
    cfg = cfg_from_experiment(ROOT / "configs/experiment/t03_planner_protocol.yaml")
    cfg.simulation.num_envs = 2
    instance = LocalGoalEnvironment(cfg, "unused", solver_factory=lambda *a, **kw: EchoSolver(4))
    yield instance
    instance.close()


def test_environment_batch_reset_restore_and_recording(env):
    transition = env.step(np.zeros((2, 4, 4)), max_low_steps=1)
    records = transition.trace[0] if hasattr(transition, "trace") else transition.records[0]
    assert [(x["env_id"], x["agent_id"]) for x in records["planner_execution"]] == [
        (e, a) for e in range(2) for a in range(4)
    ]
    assert all(x["safety_status"] == "projection_disabled" for x in records["planner_execution"])
    saved = env.state_dict()
    assert "warm" not in saved
    before = [p.state_dict()["blocked"] for p in env.planners]
    env.reset([0])
    assert [p.state_dict()["blocked"] for p in env.planners] == [0] * 4 + before[4:]
    env.load_state_dict(saved)
    assert [p.state_dict()["blocked"] for p in env.planners] == before
    env.step(np.zeros((2, 4, 4)), max_low_steps=1, enabled=torch.tensor([False, True]))
    assert [p.state_dict()["blocked"] for p in env.planners[:4]] == before[:4]
    broken = deepcopy(saved)
    broken["planner_states"][0].pop("state_version")
    with pytest.raises(ValueError):
        env.load_state_dict(broken)


def test_protocol_config_metadata_and_checkpoint_rejection(env):
    metadata = env.contracts
    assert metadata["planner_protocol"] == "planner_protocol_v1"
    assert metadata["planner_state_version"] == 1
    old = contract_metadata(cfg_from_experiment(ROOT / "configs/experiment/p0_contract_v2.yaml"))
    assert "planner_protocol" not in old
    with pytest.raises(ValueError):
        validate_checkpoint_contract({"contracts": old}, metadata)
    with pytest.raises(ValueError):
        validate_training_gate(load_yaml(ROOT / "configs/experiment/t03_planner_protocol.yaml"))
    for key, value in (
        ("planner_protocol", "unknown"),
        ("planner_adapter", "unknown"),
        ("training_paused", False),
    ):
        cfg = deepcopy(env.core.cfg)
        setattr(cfg.local_goal_contract, key, value)
        with pytest.raises(ValueError):
            contract_metadata(cfg)


def test_environment_deadline_keeps_other_planner_states(env):
    env.core.step_count[0] = env.core.max_episode_steps - 1
    transition = env.step(np.zeros((2, 4, 4)), max_low_steps=2)
    assert transition.duration.tolist() == [1, 2]
    assert [p.state_dict()["blocked"] for p in env.planners[:4]] == [0] * 4
    assert all(p.state_dict()["blocked"] > 0 for p in env.planners[4:])


def test_neighbor_sender_permissions_and_hidden_truth(env, monkeypatch):
    monkeypatch.setattr(env, "_transport", lambda poses: None)
    before = env.neighbors(0, 0, identified=True)
    env.core.oracle_point.add_(1000)
    env.core.positions[0, 1:, :2].add_(3)
    after = env.neighbors(0, 0, identified=True)
    assert [n.sender_id for n in before] == [n.sender_id for n in after]
    for a, b in zip(before, after):
        np.testing.assert_array_equal(a.prediction.positions, b.prediction.positions)
    sender = before[0].sender_id
    env.core.communication_cache.valid[0, 0, sender] = False
    assert sender not in [n.sender_id for n in env.neighbors(0, 0, identified=True)]
    env.received.pop((0, 0, before[-1].sender_id))
    assert before[-1].sender_id not in [n.sender_id for n in env.neighbors(0, 0, identified=True)]


@pytest.mark.parametrize(
    "action,hold,control",
    [
        ([0.6, 0, 0, 0], False, [0, 0]),
        ([-0.6, 0, 0, 0], False, [0, 0]),
        ([0, 0, 0.5, -1], False, [0, 0]),
        ([0, 0, 0, -1], True, [0.8, 0.5]),
    ],
)
def test_real_acados_bridge_pair(tmp_path, action, hold, control, record_property):
    assert os.environ.get("ACADOS_SOURCE_DIR"), "T03 acceptance requires actual acados; cannot skip"
    direct = NMPCPlanner(build_solver(tmp_path, 0.75), 0.75, deadline=10)
    bridge = NMPCAdapter(NMPCPlanner(build_solver(tmp_path, 0.75, build=False), 0.75, deadline=10))
    item = request(action=action, control=np.array(control))
    if hold:
        item = replace(item, intent=LocalIntent.for_hold(item.ego_pose, 0))
    intent = item.intent
    goal = LocalGoalRequest(np.r_[intent.goal_xy, intent.terminal_yaw], intent.speed_cap, 0, 1)
    old, feedback = direct.plan(item.ego_pose, goal, item.terrain, [], item.previous_control, 0)
    new = bridge.plan(item)
    errors = []
    for field in ("states", "controls", "velocities", "times"):
        left, right = getattr(old, field), getattr(new.trajectory, field)
        errors.append(float(np.max(np.abs(left - right))))
        np.testing.assert_allclose(left, right, rtol=1e-6, atol=1e-7)
    assert feedback.status == new.feedback.status
    assert feedback.reason == new.fallback_reason
    if not hold:
        assert new.valid
    else:
        assert (
            np.max(
                np.abs(np.diff(np.vstack([item.previous_control, new.nominal_controls]), axis=0))[
                    :, 0
                ]
            )
            <= 0.3001
        )
    record_property("max_trajectory_error", max(errors))


def test_environment_can_execute_a_different_protocol_adapter(env):
    env.planners = [StationaryTestAdapter() for _ in env.planners]
    result = env.step(np.zeros((2, 4, 4)), max_low_steps=1)
    assert all(r["valid"] for r in result.records[0]["planner_execution"])
    snapshot = env.state_dict()
    env.reset([1])
    env.load_state_dict(snapshot)
    assert all(p.state_dict() == {"test_only": True} for p in env.planners)


def test_new_and_old_environment_paths_pair_with_restoration(env, record_property):
    cfg = deepcopy(env.core.cfg)
    cfg.local_goal_contract.planner_protocol = "legacy_direct"
    torch.manual_seed(23)
    direct = LocalGoalEnvironment(cfg, "unused", solver_factory=lambda *a, **kw: EchoSolver(4))
    try:
        for p in direct.planners:
            p.deadline = 10
        for p in env.planners:
            p._planner.deadline = 10
        actions = np.tile([0.2, 0.1, 0.1, 0], (2, 4, 1))
        for _ in range(2):
            left = direct.step(actions, max_low_steps=1)
            right = env.step(actions, max_low_steps=1)
            np.testing.assert_array_equal(left.records[0]["commands"], right.records[0]["commands"])
            torch.testing.assert_close(left.reward, right.reward, rtol=0, atol=0)
            torch.testing.assert_close(left.actor, right.actor, rtol=0, atol=0)
            np.testing.assert_array_equal(direct.core.positions, env.core.positions)
        saved = env.state_dict()
        first = env.step(actions, max_low_steps=1)
        env.load_state_dict(saved)
        second = env.step(actions, max_low_steps=1)
        np.testing.assert_array_equal(first.records[0]["commands"], second.records[0]["commands"])
        torch.testing.assert_close(first.reward, second.reward, rtol=0, atol=0)
        record_property("max_environment_command_error", 0.0)
    finally:
        direct.close()


def test_protocol_actor_slots_follow_sender_crossing(env, monkeypatch):
    from test_p0_learning_contracts import (
        test_neighbor_slots_follow_sender_across_distance_crossing,
    )

    test_neighbor_slots_follow_sender_across_distance_crossing(env, monkeypatch)


@pytest.mark.parametrize("mode", ["missing", "expired", "sparse", "unsupported", "invalid_sender"])
def test_protocol_actor_communication_permissions(env, monkeypatch, mode):
    from test_p0_learning_contracts import test_neighbor_permissions

    test_neighbor_permissions(env, monkeypatch, mode)


def test_actual_conservative_stale_predictions_remain_sender_bound(env, monkeypatch):
    monkeypatch.setattr(env, "_transport", lambda poses: None)
    env.core.step_count[0] = 10
    for (e, a, sender), message in env.received.items():
        if e == 0 and a == 0:
            message["timestamp"], message["expires_at"] = 0.0, 1.0
            message["position"] = np.array([10.0 + sender, 20.0 + sender])
    peers = env.neighbors(0, 0, identified=True)
    for peer in peers:
        np.testing.assert_array_equal(
            peer.prediction.positions[0], [10.0 + peer.sender_id, 20.0 + peer.sender_id]
        )
        assert peer.prediction.radii[-1] > peer.prediction.radii[0]
    item = replace(request(now=2), neighbors=tuple(peers))
    item.validate()
    env.core.communication_cache.full[0, 0, peers[0].sender_id] = False
    sparse = env.neighbors(0, 0, identified=True)
    assert [p.sender_id for p in sparse] == [p.sender_id for p in peers]


def test_real_acados_bridge_warm_restore_and_reverse(tmp_path, record_property):
    assert os.environ.get("ACADOS_SOURCE_DIR"), "T03 acceptance requires actual acados; cannot skip"
    bridge = NMPCAdapter(NMPCPlanner(build_solver(tmp_path, 0.75), 0.75, deadline=10))
    initial = bridge.plan(request(action=(0.6, 0, 0, 0)))
    assert initial.valid
    saved = bridge.state_dict()
    direct = NMPCPlanner(build_solver(tmp_path, 0.75, build=False), 0.75, deadline=10)
    direct.previous, direct.blocked = deepcopy(saved["previous"]), saved["blocked"]
    bridge.reset()
    bridge.load_state_dict(saved)
    item = request(
        pose=initial.predicted_states[1],
        action=(-0.6, 0, 0, 0),
        now=0.2,
        control=initial.nominal_controls[0],
    )
    goal = LocalGoalRequest(
        np.r_[item.intent.goal_xy, item.intent.terminal_yaw], item.intent.speed_cap, 0.2, 1.2
    )
    old, feedback = direct.plan(item.ego_pose, goal, item.terrain, [], item.previous_control, 0.2)
    new = bridge.plan(item)
    assert new.valid and new.feedback.status == feedback.status
    errors = []
    for name in ("states", "controls", "velocities", "times"):
        left, right = getattr(old, name), getattr(new.trajectory, name)
        errors.append(float(np.max(np.abs(left - right))))
        np.testing.assert_allclose(left, right, rtol=1e-6, atol=1e-7)
    assert new.nominal_controls[:, 0].min() < -0.05
    record_property("max_trajectory_error", max(errors))


def test_terrain_snapshot_rejects_nonfinite_and_drops_owner_references():
    memory = flat_memory()
    memory.core = object()
    memory.terrain_runtime = object()
    view = TerrainView(memory)
    assert not hasattr(view._TerrainView__memory, "core")
    assert not hasattr(view._TerrainView__memory, "terrain_runtime")
    memory._features[0, 0] = np.nan
    with pytest.raises(ValueError):
        TerrainView(memory)


def test_legacy_terrain_ties_unchanged_and_mppi_opt_in_is_detached():
    from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.memory import LocalTerrainMemory

    memory = LocalTerrainMemory(support_radius=0.3)
    memory.observe(
        np.array([[-0.1, 0], [0.1, 0]]), np.array([[0, 0, 0, 0, 1], [0, 1, 0, 0, 0.1]]), 0
    )
    view = TerrainView(memory)
    for actual, expected in zip(view.lookup([[0, 0]]), memory.lookup([[0, 0]])):
        np.testing.assert_array_equal(actual, expected)
    strict = view.conservative()
    assert not strict.safe([[0, 0]])[0]
    assert strict.lookup([[0, 0]])[0][0, 4] == 0.1
    memory.features[:] = 0
    assert strict.lookup([[0, 0]])[0][0, 1] == 1
    np.testing.assert_array_equal(view.lookup([[0, 0]])[0][0], [0, 0, 0, 0, 1])
