"""40-step protocol, activation-time reserve and public lifecycle contracts."""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from _common import cfg_from_experiment
from diagnose_upstream_execution import synthetic_local_frame
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_execution import control_error, rollout_numpy
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_footprint import FootprintNeighbor, FootprintReport
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_planning import (
    Candidate, PlanningAdapter, UpstreamIntent, UpstreamPlannerInput,
)


@pytest.fixture(scope="module")
def cfg():
    return replace(cfg_from_experiment("configs/experiment/upstream_scale_s01_review_v2.yaml"), simulated_latency=0.)


@pytest.fixture(scope="module")
def terrain(cfg):
    return synthetic_local_frame(np.zeros(3), cfg.spec, 0., query_version="batch_ties_v2")


def context(cfg, terrain, **changes):
    data = dict(env_id=0, agent_id=0, ego_pose=np.zeros(3), ego_velocity=np.zeros(2),
                intent=UpstreamIntent([4., 0.], False, 0., 1.), terrain=terrain, neighbors=(),
                previous_control=np.zeros(2), now=0., times=np.arange(41) * .2,
                workspace=(-100., -100., 100., 100.), spec=cfg.spec)
    data.update(changes)
    return UpstreamPlannerInput(**data)


class ScriptAdapter(PlanningAdapter):
    adapter_id = "upstream_test_script_v1"

    def __init__(self, cfg, candidates=()):
        super().__init__(cfg)
        self.candidates = list(candidates)
        self.calls = 0

    def candidate_controls(self, item):
        self.calls += 1
        return self.candidates


class AcceptingReviewer:
    """Budget/state tests only; real geometry tests retain the production reviewer."""

    def trajectory(self, *args, **kwargs):
        return FootprintReport(True)


def cheap(adapter):
    adapter.reviewer = AcceptingReviewer()
    return adapter


def test_input_owns_readonly_arrays_and_explicit_time_grid(cfg, terrain):
    pose = np.zeros(3)
    item = context(cfg, terrain, ego_pose=pose)
    pose[0] = 100.
    assert item.ego_pose[0] == 0. and not item.ego_pose.flags.writeable
    assert not item.intent.goal_xy.flags.writeable
    assert item.times.shape == (41,) and item.times[-1] == 8.


@pytest.mark.parametrize("changes,match", [
    ({"env_id": True}, "env_id"), ({"agent_id": 4}, "agent_id"),
    ({"ego_pose": [0., np.nan, 0.]}, "finite"), ({"ego_velocity": [np.inf, 0.]}, "finite"),
    ({"ego_velocity": [.2, 0.]}, "inconsistent"),
    ({"previous_control": [.2, .1], "ego_velocity": [.1, .1]}, "terrain-scaled"),
    ({"previous_control": [3., 0.]}, "Illegal"), ({"now": -1.}, "activation"),
    ({"times": np.arange(16) * .2}, "shape"), ({"times": np.arange(41) * .1}, "grid"),
    ({"workspace": (-1., -1., -2., 2.)}, "workspace"),
    ({"protocol_version": "planner_protocol_v1"}, "version"),
    ({"information_pose": [0., 0., 0.]}, "together"),
    ({"intent": UpstreamIntent([4., 0.], False, .1, 1.)}, "future intent"),
    ({"intent": UpstreamIntent([4., 0.], False, 0., 1., speed_cap=3.)}, "physical"),
])
def test_bad_inputs_rejected_before_solver(cfg, terrain, changes, match):
    with pytest.raises(ValueError, match=match):
        context(cfg, terrain, **changes)


@pytest.mark.parametrize("changes", [dict(hold=1), dict(stop_at_goal=1), dict(timestamp=-1.),
                                       dict(expires_at=-1.), dict(terminal_yaw=np.inf), dict(speed_cap=-.1)])
def test_invalid_intent(cfg, changes):
    arguments = dict(goal_xy=[1., 0.], hold=False, timestamp=0., expires_at=1.)
    arguments.update(changes)
    with pytest.raises(ValueError):
        UpstreamIntent(**arguments)


def test_neighbor_identity_profile_and_coverage(cfg, terrain):
    radius = 2 * cfg.spec.vehicle.effective_radius
    peer = FootprintNeighbor.for_spec(1, [0., 8.], [[20., 0.]] * 2, [radius] * 2, cfg.spec)
    assert context(cfg, terrain, neighbors=(peer,)).neighbors[0].sender_id == 1
    for peers in ((peer, peer), (replace(peer, sender_id=0),), (replace(peer, times=np.array([0., 7.])),),
                  (replace(peer, profile_fingerprint="other"),)):
        with pytest.raises(ValueError):
            context(cfg, terrain, neighbors=peers)


def test_expiry_boundary_and_hold_do_not_call_candidate_solver(cfg, terrain):
    adapter = cheap(ScriptAdapter(cfg, [Candidate(np.zeros((40, 2)))]))
    edge = context(cfg, terrain, now=1., times=1. + np.arange(41) * .2)
    assert adapter.plan(edge).valid and adapter.calls == 1
    expired = replace(edge, now=1.2, times=1.2 + np.arange(41) * .2)
    result = adapter.plan(expired)
    assert not result.valid and result.fallback_reason == "expired_intent" and adapter.calls == 1
    adapter.reset()
    hold = context(cfg, terrain, intent=UpstreamIntent([0., 0.], True, 0., 1.))
    assert adapter.plan(hold).valid and adapter.calls == 1


def test_missing_information_state_refuses_valid_before_solver(cfg, terrain):
    delayed = replace(cfg, simulated_latency=.5)
    adapter = cheap(ScriptAdapter(delayed, [Candidate(np.zeros((40, 2)))]))
    result = adapter.plan(context(delayed, terrain))
    assert not result.valid and result.fallback_reason == "missing_information_state"
    assert adapter.calls == 0 and result.trajectory.timestamp == 0.


def test_delay_propagation_is_verified_and_not_executed_twice(cfg, terrain):
    delayed = replace(cfg, simulated_latency=.5)
    previous = np.array([1., 0.])
    item = context(delayed, terrain, previous_control=previous, ego_velocity=previous,
                   information_pose=np.zeros(3), information_time=0., ego_pose=[.6, 0., 0.],
                   now=.6, times=.6 + np.arange(41) * .2,
                   intent=UpstreamIntent([0., 0.], True, 0., 2.))
    adapter = cheap(ScriptAdapter(delayed))
    result = adapter.plan(item)
    assert result.valid and result.diagnostics["delay_steps"] == 3
    assert result.trajectory.timestamp == .6 and result.trajectory.states[0, 0] == .6
    assert result.trajectory.controls[0, 0] == pytest.approx(.92)
    assert result.trajectory.states[1, 0] == pytest.approx(.6 + .92 * .2)
    assert result.diagnostics["information_reserve_verified"]
    with pytest.raises(ValueError, match="actual activation"):
        adapter.validate_input(replace(item, ego_pose=[0., 0., 0.]))
    with pytest.raises(ValueError, match="rounded"):
        adapter.validate_input(replace(item, information_time=.1))


def test_state_restore_is_version_bound_atomic_and_isolated(cfg, terrain):
    adapter = cheap(ScriptAdapter(cfg, [Candidate(np.zeros((40, 2)))]))
    item = context(cfg, terrain)
    first = adapter.plan(item)
    saved = adapter.state_dict()
    restored = cheap(ScriptAdapter(cfg, [Candidate(np.zeros((40, 2)))]))
    restored.load_state_dict(saved)
    np.testing.assert_array_equal(restored.plan(item).trajectory.states, first.trajectory.states)
    for key, bad in (("state_version", 99), ("adapter_id", "other"), ("identity", (0, 7)), ("adapter_state", {"bad": True})):
        invalid = deepcopy(saved)
        invalid[key] = bad
        with pytest.raises(ValueError):
            restored.load_state_dict(invalid)
        np.testing.assert_array_equal(restored.state_dict()["previous_trajectory"].states, saved["previous_trajectory"].states)
    with pytest.raises(ValueError, match="another environment"):
        adapter.plan(replace(item, env_id=1))
    adapter.reset()
    assert adapter.plan(replace(item, env_id=1)).valid
    assert restored.state_dict()["identity"] == (0, 0)


@pytest.mark.parametrize("defect", ["negative_time", "actuator_jump"])
def test_corrupt_restored_trajectory_is_rejected_before_state_mutation(cfg, terrain, defect):
    adapter = cheap(ScriptAdapter(cfg, [Candidate(np.zeros((40, 2)))]))
    adapter.plan(context(cfg, terrain))
    saved = adapter.state_dict()
    if defect == "negative_time":
        saved["previous_trajectory"].timestamp = -1.
        saved["previous_trajectory"].times -= 1.
    else:
        saved["previous_trajectory"].controls[1, 0] = 1.
    with pytest.raises(ValueError):
        adapter.load_state_dict(saved)
    np.testing.assert_array_equal(adapter.state_dict()["previous_trajectory"].controls, np.zeros((40, 2)))


def test_sample_review_budget_excludes_single_updated_mean(cfg, terrain):
    class RejectingReviewer:
        def trajectory(self, *args, **kwargs):
            return FootprintReport(False, "synthetic_rejection")
    controls = np.zeros((40, 2))
    adapter = ScriptAdapter(cfg, [Candidate(controls, "updated_mean")] * 3 + [Candidate(controls, "sample")] * 20)
    adapter.reviewer = RejectingReviewer()
    result = adapter.plan(context(cfg, terrain))
    assert not result.valid
    assert result.diagnostics["candidate_reviews"] == 17
    assert result.diagnostics["sampled_candidate_reviews"] == 16
    assert result.diagnostics["updated_mean_reviews"] == 1
    assert result.diagnostics["fallback_reviews"] == 4
    assert result.diagnostics["unverified_brake"]


def test_raw_candidate_cost_is_not_reported_after_public_reprediction(cfg, terrain):
    adapter = cheap(ScriptAdapter(cfg, [Candidate(np.zeros((40, 2)), total_cost=123.)]))
    assert adapter.plan(context(cfg, terrain)).total_cost is None


def test_previous_plan_is_shifted_by_elapsed_actual_control_steps(cfg, terrain):
    controls = np.zeros((40, 2)); controls[:5, 0] = np.arange(1, 6) * .04
    adapter = cheap(ScriptAdapter(cfg, [Candidate(controls)]))
    item = context(cfg, terrain)
    first = adapter.plan(item)
    adapter.candidates = []
    elapsed = 2
    later = replace(item, ego_pose=first.trajectory.states[elapsed], ego_velocity=first.trajectory.velocities[elapsed - 1],
                    previous_control=first.trajectory.controls[elapsed - 1], now=.4, times=.4 + np.arange(41) * .2)
    recovered = adapter.plan(later)
    assert recovered.valid and recovered.feedback.used_previous
    np.testing.assert_allclose(recovered.trajectory.controls[0], first.trajectory.controls[elapsed])
    assert recovered.trajectory.timestamp == .4
    with pytest.raises(ValueError, match="whole actual control steps"):
        adapter.validate_input(replace(later, now=.5, times=.5 + np.arange(41) * .2))


def test_new_cap_admits_limited_brake_but_not_held_overspeed(cfg, terrain):
    previous = np.array([1., 0.])
    item = context(cfg, terrain, previous_control=previous, ego_velocity=previous,
                   intent=UpstreamIntent([0., 0.], True, 0., 1., speed_cap=.1))
    adapter = cheap(ScriptAdapter(cfg))
    output = adapter.plan(item)
    assert output.valid and output.trajectory.controls[0, 0] == pytest.approx(.92)
    previous_step = previous
    for command in output.trajectory.controls:
        assert not control_error(command, previous_step, cfg.spec)
        previous_step = command
    bad = rollout_numpy(np.zeros(3), np.tile(previous, (40, 1)), cfg.spec)
    assert adapter._review(item, bad) == "intent_speed_cap"


def test_real_full_footprint_moving_hold_stops_without_goal_teleport(cfg, terrain):
    item = context(cfg, terrain, previous_control=[.2, .03], ego_velocity=[.2, .03],
                   intent=UpstreamIntent([100., 100.], True, 0., 1.))
    output = ScriptAdapter(cfg).plan(item)
    assert output.valid and output.feedback.status == "hold"
    assert output.trajectory.states.shape == (41, 3)
    np.testing.assert_array_equal(output.trajectory.controls[-1], np.zeros(2))
    assert 0 < output.trajectory.states[-1, 0] < .1
    assert output.diagnostics["information_reserve_verified"]


def test_real_workspace_failure_is_unverified_brake(cfg, terrain):
    item = context(cfg, terrain, workspace=(-1., -1., 1., 1.), intent=UpstreamIntent([0., 0.], True, 0., 1.))
    output = ScriptAdapter(cfg).plan(item)
    assert not output.valid and output.diagnostics["unverified_brake"]
    assert output.diagnostics["execution_review_reason"] == "workspace_clearance"


def test_real_overlap_cannot_be_certified_by_stopping(cfg, terrain):
    radius = 2 * cfg.spec.vehicle.effective_radius
    peer = FootprintNeighbor.for_spec(1, [0., 8.], [[1., 0.]] * 2, [radius] * 2, cfg.spec)
    item = context(cfg, terrain, neighbors=(peer,), intent=UpstreamIntent([0., 0.], True, 0., 1.))
    output = ScriptAdapter(cfg).plan(item)
    assert not output.valid
    assert output.diagnostics["execution_review_reason"] == "neighbor_swept_clearance"



def test_direct_adapter_rejects_unimplemented_realtime_latency(cfg):
    with pytest.raises(ValueError, match="measured latency chain"):
        ScriptAdapter(replace(cfg, clock_mode="realtime_measured_latency"))


@pytest.mark.parametrize("reason", ["all_samples_numerically_invalid", "planner_timeout", None])
def test_no_candidate_preserves_producer_failure_reason(cfg, terrain, reason):
    adapter = cheap(ScriptAdapter(cfg))
    output = adapter.finish(context(cfg, terrain), [], diagnostics={"failure_reason": reason})
    expected = reason or "no_candidate"
    assert not output.valid
    assert output.fallback_reason == expected == output.feedback.reason
    assert output.feedback.candidate_rejection == expected


def test_actual_candidate_rejection_has_priority_over_producer_failure(cfg, terrain):
    class RejectingReviewer:
        def trajectory(self, *args, **kwargs):
            return FootprintReport(False, "unknown_footprint")
    adapter = ScriptAdapter(cfg)
    adapter.reviewer = RejectingReviewer()
    output = adapter.finish(context(cfg, terrain), [Candidate(np.zeros((40, 2)))],
                            diagnostics={"failure_reason": "planner_timeout"})
    assert not output.valid
    assert output.fallback_reason == "unknown_footprint" == output.feedback.candidate_rejection
