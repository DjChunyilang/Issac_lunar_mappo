"""Independent N40 MPPI numerical, batch, lifecycle and physical contracts."""

from copy import deepcopy
from dataclasses import replace
import sys
from pathlib import Path
import json

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from _common import cfg_from_experiment
from diagnose_upstream_execution import synthetic_local_frame
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_execution import (
    control_error, limit_control_numpy, midpoint_numpy, terrain_gain_numpy,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_footprint import FootprintReport
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_planning import UpstreamIntent, UpstreamPlannerInput
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_mppi import (
    UpstreamMPPIAdapter, UpstreamMPPIKernel, UpstreamMPPISettings,
)


@pytest.fixture(scope="module", autouse=True)
def single_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@pytest.fixture(scope="module")
def cfg():
    return replace(cfg_from_experiment("configs/experiment/upstream_scale_s01_review_v2.yaml"),
                   simulated_latency=0., planner_adapter="mppi_upstream_v1")


@pytest.fixture(scope="module")
def terrain(cfg):
    return synthetic_local_frame(np.zeros(3), cfg.spec, 0., query_version="batch_ties_v2")


def request(cfg, terrain, **changes):
    values = dict(env_id=0, agent_id=0, ego_pose=np.zeros(3), ego_velocity=np.zeros(2),
                  intent=UpstreamIntent([4., 0.], False, 0., 1.), terrain=terrain, neighbors=(),
                  previous_control=np.zeros(2), now=0., times=np.arange(41) * .2,
                  workspace=(-100., -100., 100., 100.), spec=cfg.spec)
    values.update(changes)
    return UpstreamPlannerInput(**values)


class BudgetReviewer:
    """RNG/probability tests isolate geometry; physical tests use real reviewer."""
    def trajectory(self, *args, **kwargs):
        return FootprintReport(True)


def adapter(cfg, *, cheap=True, **settings):
    result = UpstreamMPPIAdapter(cfg, UpstreamMPPIKernel(cfg, UpstreamMPPISettings(samples=8, **settings)))
    if cheap:
        result.reviewer = BudgetReviewer()
    return result


def next_request(item, output, steps=1):
    now = item.now + steps * .2
    return replace(item, ego_pose=output.trajectory.states[steps], previous_control=output.trajectory.controls[steps - 1],
                   ego_velocity=output.trajectory.velocities[steps - 1], now=now, times=now + np.arange(41) * .2)


@pytest.mark.parametrize("settings", [dict(samples=0), dict(iterations=True), dict(candidate_limit=17),
    dict(device="cuda:8"), dict(dtype="float16"), dict(noise_w=0.), dict(temperature=float("nan")), dict(deadline_s=-1.)])
def test_invalid_settings_rejected(settings):
    with pytest.raises(ValueError):
        UpstreamMPPISettings(**settings)


@pytest.mark.parametrize("previous", [[0., 0.], [2., .2], [-.4, -.1], [.06, .35]])
def test_float64_batch_rollout_matches_independent_numpy_time_loop(cfg, terrain, previous):
    item = request(cfg, terrain, previous_control=previous, ego_velocity=previous)
    kernel = UpstreamMPPIKernel(cfg, UpstreamMPPISettings(samples=3, dtype="float64"))
    raw = np.random.default_rng(23).normal([.8, 0.], [1.2, .8], (1, 3, 40, 2))
    packed = kernel.pack([item])
    states, controls, velocities, numeric, _ = kernel.rollout([item], kernel.tensor(raw), packed[0])
    assert numeric.all()
    for k in range(3):
        pose, prior, stopped = np.zeros(3), np.asarray(previous, dtype=float), False
        for t in range(40):
            desired = raw[0, k, t].copy()
            desired[0] = np.clip(desired[0], -.5, 25 / 9)
            stopped |= np.ceil(abs(prior[0]) / .08 - 1e-10) + np.ceil(abs(prior[1]) / .04 - 1e-10) + 1 >= 40 - t
            command = limit_control_numpy(np.zeros(2) if stopped else desired, prior, cfg.spec)
            command[abs(command) <= 1e-10] = 0.
            feature, _ = terrain.lookup(pose[:2][None])
            pose, actual = midpoint_numpy(pose, command, cfg.spec, gain=terrain_gain_numpy(pose, feature[0], cfg.spec))
            np.testing.assert_allclose(controls[0, k, t].numpy(), command, atol=1e-10, rtol=1e-8)
            np.testing.assert_allclose(states[0, k, t + 1].numpy(), pose, atol=1e-10, rtol=1e-8)
            np.testing.assert_allclose(velocities[0, k, t].numpy(), actual, atol=1e-10, rtol=1e-8)
            prior = command
    assert controls[:, :, -1].abs().max() < 1e-10


def test_rolling_target_does_not_force_stop_at_4m(cfg, terrain):
    item = request(cfg, terrain, previous_control=[2., 0.], ego_velocity=[2., 0.])
    kernel = UpstreamMPPIKernel(cfg, UpstreamMPPISettings(samples=1, dtype="float64"))
    raw = kernel.tensor(np.tile([25 / 9, 0.], (1, 1, 40, 1)))
    evaluated = kernel.evaluate([item], raw)
    assert evaluated[0][0, 0, -1, 0] > 4. and evaluated[5].all()
    parked = replace(item, intent=replace(item.intent, stop_at_goal=True))
    stopped_cost = kernel.cost([parked], evaluated[0], evaluated[1], *kernel.pack([parked]))[0]
    assert stopped_cost["goal"] > evaluated[-1]["goal"]


def test_batch_reorder_and_single_use_identical_injected_noise(cfg, terrain):
    a, b = request(cfg, terrain), request(cfg, terrain, env_id=1, intent=UpstreamIntent([0., 4.], False, 0., 1.))
    noise = np.random.default_rng(31).normal(size=(2, 2, 8, 40, 2)) * [.25, .10]
    batch = [adapter(cfg), adapter(cfg)]
    expected = batch[0].batch_kernel.plan_batch(batch, [a, b], noise=noise)
    reversed_handles = [adapter(cfg), adapter(cfg)]
    reversed_out = reversed_handles[0].batch_kernel.plan_batch(reversed_handles, [b, a], noise=noise[:, ::-1].copy())
    for i, item in enumerate([a, b]):
        single = adapter(cfg)
        out = single.batch_kernel.plan_batch([single], [item], noise=noise[:, i:i + 1])[0]
        np.testing.assert_allclose(out.trajectory.controls, expected[i].trajectory.controls, atol=1e-5, rtol=1e-5)
        np.testing.assert_array_equal(reversed_out[1 - i].trajectory.states, expected[i].trajectory.states)
        assert out.valid and expected[i].diagnostics["batch_size"] == 2


def test_rng_restore_sparse_batch_and_other_environment_reset_are_independent(cfg, terrain):
    a, b = request(cfg, terrain), request(cfg, terrain, env_id=1)
    handles = [adapter(cfg), adapter(cfg)]
    first = handles[0].batch_kernel.plan_batch(handles, [a, b])
    saved = handles[0].state_dict()
    restored = adapter(cfg); restored.load_state_dict(saved)
    handles[1].reset()
    later = next_request(a, first[0])
    second = handles[0].plan(later)
    replay = restored.plan(later)
    np.testing.assert_array_equal(second.trajectory.states, replay.trajectory.states)
    torch.testing.assert_close(handles[0].last_samples["raw_noise"], restored.last_samples["raw_noise"], atol=0, rtol=0)
    alone = adapter(cfg); alone.plan(a)
    torch.testing.assert_close(alone.state_dict()["adapter_state"]["generator"], saved["adapter_state"]["generator"], atol=0, rtol=0)


def test_invalid_batch_does_not_bind_or_advance_rng(cfg, terrain):
    a = adapter(cfg); item = request(cfg, terrain)
    before = a.state_dict()
    with pytest.raises(ValueError, match="noise shape"):
        a.batch_kernel.plan_batch([a], [item], noise=np.zeros((2, 1, 7, 40, 2)))
    assert a.state_dict()["identity"] is None
    torch.testing.assert_close(a.state_dict()["adapter_state"]["generator"], before["adapter_state"]["generator"])
    with pytest.raises(ValueError, match="independent"):
        a.batch_kernel.plan_batch([a, a], [item, item])
    assert a.batch_kernel.plan_batch([], []) == []


def test_all_nonfinite_samples_have_zero_weight_and_explicit_brake(cfg, terrain):
    a = adapter(cfg); item = request(cfg, terrain)
    out = a.batch_kernel.plan_batch([a], [item], noise=np.full((2, 1, 8, 40, 2), np.nan))[0]
    assert not out.valid and out.feedback.emergency_brake
    assert out.diagnostics["numerical_valid_fraction"] == 0.
    assert out.diagnostics["effective_samples"] == 0.
    assert out.diagnostics["failure_reason"] == "all_samples_numerically_invalid"
    assert np.isfinite(out.trajectory.states).all()
    assert out.diagnostics["importance_correction"] == [None] * 8
    json.dumps(out.diagnostics, allow_nan=False)


def test_deadline_before_sampling_is_separate_failure(cfg, terrain):
    a = adapter(cfg, deadline_s=.1)
    times = iter([0.])
    out = a.batch_kernel.plan_batch([a], [request(cfg, terrain)], clock=lambda: next(times, 1.))[0]
    assert not out.valid and out.diagnostics["timed_out"]
    assert out.diagnostics["failure_reason"] == "planner_timeout"


@pytest.mark.parametrize("kind", ["hold", "expired", "missing_information"])
def test_parking_paths_do_not_sample_or_keep_stale_samples(cfg, terrain, kind):
    config = replace(cfg, simulated_latency=.5) if kind == "missing_information" else cfg
    a = adapter(config)
    a.last_samples = {"stale": True}
    intent = UpstreamIntent([4., 0.], kind == "hold", 0., 1.)
    now = 1.2 if kind == "expired" else 0.
    out = a.plan(request(config, terrain, intent=intent, now=now, times=now + np.arange(41) * .2,
                         previous_control=[.2, 0.], ego_velocity=[.2, 0.]))
    assert a.last_samples is None and a.batch_kernel.batch_calls == 0
    assert out.trajectory.controls[0, 0] == pytest.approx(.12)
    assert out.valid == (kind == "hold")


def test_restoration_rejects_settings_and_corrupt_rng_atomically(cfg, terrain):
    a = adapter(cfg); a.plan(request(cfg, terrain)); saved = a.state_dict()
    wrong = adapter(cfg, noise_v=.5)
    with pytest.raises(ValueError, match="settings"):
        wrong.load_state_dict(saved)
    bad = deepcopy(saved); bad["adapter_state"]["generator"] = torch.zeros(3)
    with pytest.raises(ValueError, match="generator"):
        a.load_state_dict(bad)
    torch.testing.assert_close(a.state_dict()["adapter_state"]["generator"], saved["adapter_state"]["generator"], atol=0, rtol=0)


def test_real_full_review_forward_move_and_moving_hold(cfg, terrain):
    a = adapter(cfg, cheap=False)
    item = request(cfg, terrain)
    out = a.plan(item)
    assert out.valid and out.trajectory.states[-1, 0] > .5
    assert out.diagnostics["information_reserve_verified"]
    assert out.total_cost == pytest.approx(sum(out.diagnostics["selected_cost_parts"].values()))
    later = next_request(item, out)
    hold = a.plan(replace(later, intent=UpstreamIntent(later.ego_pose[:2], True, later.now, later.now + 1.)))
    assert hold.valid and hold.feedback.status == "hold"
    previous = later.previous_control
    for control in hold.trajectory.controls:
        assert not control_error(control, previous, cfg.spec)
        previous = control
    np.testing.assert_array_equal(hold.trajectory.controls[-1], [0., 0.])


def test_required_cuda_same_injected_noise_and_same_device_restore(cfg, terrain):
    assert torch.cuda.is_available(), "S02 CUDA acceptance requires an actual device"
    item = request(cfg, terrain)
    noise = np.random.default_rng(47).normal(size=(2, 1, 8, 40, 2)) * [.25, .10]
    cpu, gpu = adapter(cfg), adapter(cfg, device="cuda")
    co = cpu.batch_kernel.plan_batch([cpu], [item], noise=noise)[0]
    go = gpu.batch_kernel.plan_batch([gpu], [item], noise=noise)[0]
    np.testing.assert_allclose(go.trajectory.states, co.trajectory.states, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(go.trajectory.controls, co.trajectory.controls, atol=1e-5, rtol=1e-5)
    restored = adapter(cfg, device="cuda"); restored.load_state_dict(gpu.state_dict())
    later = next_request(item, go)
    np.testing.assert_array_equal(gpu.plan(later).trajectory.states, restored.plan(later).trajectory.states)
    with pytest.raises(ValueError, match="device/version"):
        cpu.load_state_dict(gpu.state_dict())


def test_expiry_boundary_is_inclusive_and_mixed_hold_batch_preserves_noise_slots(cfg, terrain):
    hold = request(cfg, terrain, intent=UpstreamIntent([0., 0.], True, 0., 1.))
    move = request(cfg, terrain, env_id=1, now=1., times=1. + np.arange(41) * .2)
    noise = np.random.default_rng(23).normal(size=(2, 2, 8, 40, 2)) * [.25, .10]
    handles = [adapter(cfg), adapter(cfg)]
    outputs = handles[0].batch_kernel.plan_batch(handles, [hold, move], noise=noise)
    single = adapter(cfg)
    expected = single.batch_kernel.plan_batch([single], [move], noise=noise[:, 1:])[0]
    assert outputs[0].valid and outputs[0].diagnostics["samples"] == 0
    np.testing.assert_array_equal(outputs[1].trajectory.states, expected.trajectory.states)
    assert outputs[0].diagnostics["end_to_end_s"] == outputs[1].diagnostics["end_to_end_s"]


def test_last_iteration_deadline_discards_updated_candidate(cfg, terrain, monkeypatch):
    a = adapter(cfg, iterations=1, deadline_s=.1)
    elapsed = [0.]
    original = a.batch_kernel.evaluate
    def expensive(*args, **kwargs):
        out = original(*args, **kwargs)
        elapsed[0] = 99.
        return out
    monkeypatch.setattr(a.batch_kernel, "evaluate", expensive)
    out = a.batch_kernel.plan_batch([a], [request(cfg, terrain)], clock=lambda: elapsed[0])[0]
    assert not out.valid and out.fallback_reason == "planner_timeout"
    assert out.diagnostics["timed_out"] and out.diagnostics["candidate_reviews"] == 0


def test_full_review_deadline_cannot_save_timed_out_candidate_as_old_plan(cfg, terrain, monkeypatch):
    a = adapter(cfg, deadline_s=.1)
    elapsed = [0.]
    original = a.finish
    def expensive(*args, **kwargs):
        result = original(*args, **kwargs)
        elapsed[0] = 99.
        return result
    monkeypatch.setattr(a, "finish", expensive)
    out = a.batch_kernel.plan_batch([a], [request(cfg, terrain)], clock=lambda: elapsed[0])[0]
    assert not out.valid and out.fallback_reason == "planner_timeout"
    assert out.diagnostics["timeout_phase"] == "full_review"
    assert not out.diagnostics["full_review_passed"] and out.diagnostics["selected_cost_parts"] == {}
    assert a.state_dict()["previous_trajectory"] is None and a.warm is None


def test_review_budget_exhaustion_separate_from_no_feasible_samples(cfg, terrain):
    class Reject:
        def trajectory(self, *args, **kwargs):
            return FootprintReport(False, "test_full_footprint_rejection")
    a = adapter(cfg, candidate_limit=2)
    a.reviewer = Reject()
    out = a.plan(request(cfg, terrain))
    assert not out.valid and out.diagnostics["prescreen_count"] == 8
    assert out.diagnostics["sampled_candidate_reviews"] == 2
    assert out.diagnostics["unreviewed_sample_count"] == 6
    assert out.diagnostics["sample_review_budget_exhausted"]


@pytest.mark.parametrize("defect", ["rng_binding", "warm_jump", "warm_missing_stop"])
def test_corrupt_warm_and_rng_identity_rejected_atomically(cfg, terrain, defect):
    a = adapter(cfg); a.plan(request(cfg, terrain)); saved = a.state_dict()
    invalid = deepcopy(saved)
    if defect == "rng_binding":
        invalid["adapter_state"]["rng_bound"] = False
    else:
        index = 1 if defect == "warm_jump" else -1
        invalid["adapter_state"]["warm"][index] = [2., 0.]
        invalid["previous_trajectory"].controls[index] = [2., 0.]
    with pytest.raises(ValueError):
        a.load_state_dict(invalid)
    np.testing.assert_array_equal(a.warm, saved["adapter_state"]["warm"])


def test_score_uses_exact_reviewed_trajectory_without_float32_reprojection(cfg, terrain):
    a = adapter(cfg); item = request(cfg, terrain); out = a.plan(item)
    trajectory = deepcopy(out.trajectory)
    trajectory.states[-1, 0] += 2e-7
    cost = a.score_trajectory(item, trajectory)
    reference = UpstreamMPPIKernel(cfg, UpstreamMPPISettings(samples=8, dtype="float64"))
    parts, _ = reference.cost([item], reference.tensor(trajectory.states)[None, None],
                              reference.tensor(trajectory.controls)[None, None], *reference.pack([item]))
    assert cost == pytest.approx(float(sum(parts.values())[0, 0]), abs=1e-10, rel=1e-8)


def test_memory_chunking_changes_neither_sampling_nor_selected_trajectory(cfg, terrain):
    item = request(cfg, terrain)
    noise = np.random.default_rng(47).normal(size=(2, 1, 8, 40, 2)) * [.25, .10]
    wide = adapter(cfg)
    small = adapter(cfg, rollout_chunk=3, query_chunk=7, query_memory_bytes=4096)
    a = wide.batch_kernel.plan_batch([wide], [item], noise=noise)[0]
    b = small.batch_kernel.plan_batch([small], [item], noise=noise)[0]
    np.testing.assert_array_equal(a.trajectory.states, b.trajectory.states)
    assert a.total_cost == b.total_cost


def test_lower_speed_cap_requires_transition_and_keeps_asymmetric_reverse_bound(cfg, terrain):
    item = request(cfg, terrain, previous_control=[1., 0.], ego_velocity=[1., 0.],
                   intent=UpstreamIntent([-4., 0.], False, 0., 1., speed_cap=.1))
    a = adapter(cfg, cheap=False)
    output = a.plan(item)
    assert output.valid
    assert output.trajectory.controls[0, 0] == pytest.approx(.92, abs=1e-6)
    assert output.trajectory.controls[:, 0].min() >= -.1 - 1e-8
    assert np.max(abs(output.trajectory.controls[-1])) == 0.
