"""Finite actual-state collection, drift, priority and asynchronous lifecycle."""
from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from _common import cfg_from_experiment
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_footprint import FootprintReport
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_task import UpstreamTask


@pytest.fixture(scope="module")
def spec():return cfg_from_experiment("configs/experiment/upstream_scale_s02.yaml").spec


def inputs(spec, envs=2):
    poses = np.zeros((envs, 4, 3));poses[..., :2] = np.asarray(spec.task.slots)
    speed = np.zeros((envs, 4, 2))
    reports = [[FootprintReport(True, minimum_body_clearance=.3, checked_poses=1) for _ in range(4)] for _ in range(envs)]
    return poses, speed, reports


def step(task, values, time, **kw):
    return task.advance(*values, np.broadcast_to(np.asarray(time), (task.num_envs,)), **kw)


def test_full_square_needs_actual_continuous_25_steps(spec):
    task = UpstreamTask(spec, 2);values = inputs(spec)
    for i in range(24):
        result = step(task, values, (i + 1) * .2)
        assert not result["terminated"].any()
        assert result["hold_count"].tolist() == [i + 1] * 2
    result = step(task, values, 5.)
    assert result["terminal_reason"].tolist() == ["success"] * 2
    assert result["ended_this_step"].all() and not result["bootstrap_allowed"].any()
    np.testing.assert_allclose(result["dmax"], 7.2 * np.sqrt(2))
    np.testing.assert_allclose(result["dispersion"], 25.92)


@pytest.mark.parametrize("defect", ["linear", "angular", "terrain", "drift", "heading"])
def test_any_actual_parking_break_resets_hold(spec, defect):
    task = UpstreamTask(spec, 1);values = inputs(spec, 1)
    for i in range(3):step(task, values, (i + 1) * .2)
    changed = deepcopy(values)
    if defect == "linear":changed[1][0, 1, 0] = .011
    elif defect == "angular":changed[1][0, 2, 1] = .004
    elif defect == "terrain":changed[2][0][3] = FootprintReport(False, "parking_slope", checked_poses=1)
    elif defect == "drift":changed[0][0, 0, 0] += .101
    else:changed[0][0, 0, 2] += np.deg2rad(1.01)
    result = step(task, changed, .8)
    assert result["hold_count"][0] == 0 and not result["terminated"][0]
    assert result["drift_broken"][0] == (defect in ("drift", "heading"))
    assert step(task, values, 1.)["hold_count"][0] == 1


def test_actual_zero_motion_does_not_imply_valid_parking_terrain(spec):
    task = UpstreamTask(spec, 1);values = inputs(spec, 1)
    # A driving pass at 7 degrees is not a parking pass at 5 degrees.
    values[2][0][0] = FootprintReport(True, slope=np.tan(np.deg2rad(7)), minimum_body_clearance=.3, checked_poses=1)
    result = step(task, values, .2)
    assert result["actual_stopped"][0] and not result["parking_terrain_satisfied"][0]
    assert result["hold_count"][0] == 0
    values[2][0][0] = FootprintReport(True)  # no actual individual footprint measurements
    assert not step(task, values, .4)["parking_terrain_satisfied"][0]


@pytest.mark.parametrize("reason", ["environment_invalid", "collision", "out_of_bounds"])
def test_unsafe_state_has_priority_over_success_and_deadline(spec, reason):
    task = UpstreamTask(spec, 1, 5.);values = inputs(spec, 1)
    for i in range(24):step(task, values, (i + 1) * .2)
    changed = deepcopy(values);flags = np.array([True])
    if reason == "environment_invalid":flags[0] = False
    elif reason == "collision":changed[0][0, 1] = changed[0][0, 0]
    else:changed[0][0, 0, 0] = 99.
    result = step(task, changed, 5., environment_valid=flags)
    assert result["terminal_reason"][0] == reason
    assert not result["bootstrap_allowed"][0] and result["hold_count"][0] == 0
    assert result["learning_valid"][0] == (reason != "environment_invalid")


def test_success_on_same_deadline_step_wins_in_legal_state(spec):
    task = UpstreamTask(spec, 1, 5.);values = inputs(spec, 1)
    for i in range(25):result = step(task, values, (i + 1) * .2)
    assert result["terminal_reason"][0] == "success"


def test_deadline_is_terminal_without_bootstrap_and_not_a_buffer_cut(spec):
    task = UpstreamTask(spec, 1, .4);values = inputs(spec, 1)
    first = step(task, values, .2)
    assert first["bootstrap_allowed"][0] and first["hold_count"][0] == 1
    saved = task.state_dict();restored = UpstreamTask(spec, 1, .4);restored.load_state_dict(saved)
    final = step(restored, values, .4)
    assert final["terminal_reason"][0] == "task_deadline"
    assert final["terminated"][0] and not final["truncated"][0]
    assert not final["bootstrap_allowed"][0] and final["learning_valid"][0]


def test_nonfinite_actual_state_is_environment_failure_not_timeout(spec):
    task = UpstreamTask(spec, 1, .2);values = inputs(spec, 1)
    values[0][0, 0, 0] = np.nan
    result = step(task, values, .2)
    assert result["terminal_reason"][0] == "environment_invalid"
    assert not result["learning_valid"][0] and not result["bootstrap_allowed"][0]


def test_clearance_and_physical_envelope_overlap_are_separate(spec):
    task = UpstreamTask(spec, 1);values = inputs(spec, 1)
    values[0][0, :, :2] = [[0.,0.],[6.2,0.],[0.,7.2],[7.2,7.2]]
    result = step(task, values, .2)
    assert not result["clearance_satisfied"][0]
    assert not result["terminated"][0] and result["hold_count"][0] == 0


def test_async_termination_reset_and_inactive_statistics_do_not_cross_environments(spec):
    task = UpstreamTask(spec, 2, [.4, 360.]);values = inputs(spec)
    step(task, values, .2);result = step(task, values, .4)
    assert result["terminal_reason"].tolist() == ["task_deadline", "none"]
    result = step(task, values, [.4,.6], active_envs=np.array([False, True]))
    assert not result["ended_this_step"].any() and result["hold_count"].tolist() == [2,3]
    assert result["learning_valid"].tolist() == [False,True]
    task.reset([0])
    result = step(task, values, [.2,.8])
    assert result["hold_count"].tolist() == [1,4]
    assert result["terminal_reason"].tolist() == ["none","none"]


@pytest.mark.parametrize("bad", ["hold", "anchor", "time", "reason", "profile", "deadline"])
def test_corrupt_task_restore_is_rejected_before_mutation(spec, bad):
    task = UpstreamTask(spec, 2);values = inputs(spec);step(task, values, .2);saved = task.state_dict()
    altered = deepcopy(saved)
    if bad == "hold":altered["hold_count"][0] = 26
    elif bad == "anchor":altered["anchors"][0,0] = np.nan
    elif bad == "time":altered["last_time"][0] = .1
    elif bad == "reason":altered["terminal_reason"][0] = "success"
    elif bad == "profile":altered["contracts"]["profile_fingerprint"] = "old"
    else:altered["contracts"]["episode_seconds"] = [180.,180.]
    with pytest.raises(ValueError):task.load_state_dict(altered)
    np.testing.assert_array_equal(task.state_dict()["hold_count"], saved["hold_count"])
    np.testing.assert_array_equal(task.state_dict()["anchors"], saved["anchors"])


def test_restore_preserves_hold_anchor_and_next_step_exactly(spec):
    task = UpstreamTask(spec, 2);values = inputs(spec)
    for i in range(8):step(task, values, (i + 1) * .2)
    restored = UpstreamTask(spec, 2);restored.load_state_dict(task.state_dict())
    expected, actual = step(task, values, 1.8), step(restored, values, 1.8)
    for name in expected:np.testing.assert_array_equal(actual[name], expected[name])
    with pytest.raises(ValueError, match="one actual control step"):step(task, values, 2.2)
