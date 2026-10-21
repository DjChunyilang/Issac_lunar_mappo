"""CPU MPPI numerical, batch, state, fallback and actual local execution contracts."""

from dataclasses import replace
from copy import deepcopy

import numpy as np
import pytest
import torch

from test_planner_baselines import low_input
from _common import cfg_from_experiment
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.mppi import (
    MPPIConfig,
    MPPIKernel,
    MPPIAdapter,
    rollout_batch,
    importance_update,
    base_cost,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.execution import (
    limit_control,
    dynamics,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.protocol import (
    plan_batch,
    LocalIntent,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.environment import (
    LocalGoalEnvironment,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.actions import PolicyAction
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.memory import LocalTerrainMemory
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.contracts import contract_metadata


def kernel(**kwargs):
    torch.set_num_threads(1)
    return MPPIKernel(MPPIConfig(samples=64, **kwargs))


def test_numpy_rollout_reference():
    rng = np.random.default_rng(23)
    raw = rng.normal(size=(2, 5, 15, 2))
    pose = np.array([[0.2, -0.1, 0.3], [-0.1, 0.2, -0.4]])
    previous = np.array([[0.4, 0.2], [-0.2, -0.1]])
    cap = np.array([1.15, 0.6])
    states, controls, velocities = rollout_batch(
        torch.tensor(pose), torch.tensor(raw), torch.tensor(previous), torch.tensor(cap)
    )
    for b in range(2):
        for k in range(5):
            s, last = pose[b].copy(), previous[b].copy()
            reference = [s.copy()]
            for t in range(15):
                last = limit_control(raw[b, k, t], last, cap[b])
                s, v = dynamics(s, last, np.array([0, 0, 0, 0, 1]), 0.75)
                reference.append(s.copy())
                np.testing.assert_allclose(controls[b, k, t], last, atol=1e-10, rtol=1e-8)
                np.testing.assert_allclose(velocities[b, k, t], v, atol=1e-10, rtol=1e-8)
            np.testing.assert_allclose(states[b, k], reference, atol=1e-10, rtol=1e-8)


def test_importance_analytic_raw_noise_and_nan_mask():
    mean = torch.full((1, 15, 2), 0.2, dtype=torch.float64)
    noise = torch.tensor(np.random.default_rng(23).normal(size=(1, 4, 15, 2)), dtype=torch.float64)
    cost = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float64)
    sigma = torch.tensor([0.35, 0.7], dtype=torch.float64)
    out, w, score, valid = importance_update(
        mean, noise, cost, torch.ones(1, 4, dtype=torch.bool), 1.0, sigma
    )
    expected = cost.numpy() + (mean[:, None] * noise / sigma.square()).sum((-1, -2)).numpy()
    weights = np.exp(-(expected - expected.min(-1, keepdims=True)))
    weights /= weights.sum(-1, keepdims=True)
    np.testing.assert_allclose(w, weights, atol=1e-10, rtol=1e-8)
    np.testing.assert_allclose(
        out, mean.numpy() + (weights[..., None, None] * noise.numpy()).sum(1), atol=1e-10, rtol=1e-8
    )
    noise[:] = torch.nan
    out, w, _, valid = importance_update(
        mean, noise, cost, torch.ones(1, 4, dtype=torch.bool), 1.0, sigma
    )
    assert not valid.any() and torch.equal(out, mean) and torch.count_nonzero(w) == 0
    assert torch.isfinite(out).all()


def test_cost_reference_components():
    states = torch.zeros(1, 2, 16, 3, dtype=torch.float64)
    controls = torch.full((1, 2, 15, 2), -0.1, dtype=torch.float64)
    previous = torch.zeros(1, 2, dtype=torch.float64)
    parts = base_cost(
        states,
        controls,
        previous,
        torch.ones(1, 2, dtype=torch.float64),
        torch.zeros(1),
        torch.zeros(1),
    )
    assert parts["goal"][0, 0] == 84
    assert parts["control"][0, 0] == pytest.approx(0.012)
    assert parts["smooth"][0, 0] == pytest.approx(0.007)
    assert parts["reverse"][0, 0] == pytest.approx(0.03)


def test_same_injected_noise_single_batch_and_reorder():
    k = kernel()
    req = [low_input(), replace(low_input((-0.8, 0.2)), env_id=2, agent_id=3)]
    noise = np.random.default_rng(31).normal(size=(2, 2, 64, 15, 2)) * [0.35, 0.7]
    p = [MPPIAdapter(k) for _ in req]
    batch = k.plan_batch(p, req, noise=noise)
    for i, r in enumerate(req):
        single = k.plan_batch([MPPIAdapter(k)], [r], noise=noise[:, i : i + 1])[0]
        np.testing.assert_allclose(
            batch[i].trajectory.controls, single.trajectory.controls, atol=1e-5, rtol=1e-5
        )
    reversed_out = k.plan_batch(
        [MPPIAdapter(k), MPPIAdapter(k)], req[::-1], noise=noise[:, ::-1].copy()
    )
    for a, b in zip(batch, reversed_out[::-1]):
        np.testing.assert_allclose(a.trajectory.states, b.trajectory.states, atol=1e-5, rtol=1e-5)


def test_rng_per_identity_reset_and_snapshot():
    k = kernel()
    requests = [replace(low_input(), agent_id=i) for i in range(4)]
    a, b = [MPPIAdapter(k) for _ in range(4)], [MPPIAdapter(k) for _ in range(4)]
    x = plan_batch(a, requests)
    y = plan_batch(b[::-1], requests[::-1])
    for p, q in zip(x, y[::-1]):
        np.testing.assert_allclose(
            p.trajectory.controls, q.trajectory.controls, atol=1e-5, rtol=1e-5
        )
    saved = a[2].state_dict()
    restored = MPPIAdapter(k)
    restored.load_state_dict(saved)
    r = replace(
        requests[2],
        now=0.2,
        times=requests[2].times + 0.2,
        ego_pose=x[2].trajectory.states[1],
        previous_control=x[2].trajectory.controls[0],
    )
    a[0].reset()
    p, q = a[2].plan(r), restored.plan(r)
    np.testing.assert_array_equal(p.trajectory.controls, q.trajectory.controls)
    bad = deepcopy(saved)
    bad["rng_device"] = "cuda"
    before = restored.state_dict()
    with pytest.raises(ValueError):
        restored.load_state_dict(bad)
    assert torch.equal(before["rng_state"], restored.state_dict()["rng_state"])
    restored.reset()
    assert restored.warm is None and restored.previous is None


def test_hold_expiry_empty_and_no_feasible_samples():
    k = kernel()
    p = MPPIAdapter(k)
    r = low_input(control=(0.8, 0.5))
    hold = replace(r, intent=LocalIntent.for_hold(r.ego_pose, 0))
    out = p.plan(hold)
    assert not out.valid
    np.testing.assert_allclose(out.trajectory.controls[0], [0.5, 0], atol=1e-12)
    assert p.last_diagnostics["samples"] == 0
    expired = replace(r, now=2, times=r.times + 2)
    out = p.plan(expired)
    assert out.fallback_reason == "expired_intent"
    assert plan_batch([], []) == []
    from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.protocol import TerrainView

    unknown = replace(low_input(), terrain=TerrainView(LocalTerrainMemory()))
    out = MPPIAdapter(k).plan(unknown)
    assert not out.valid and out.fallback_reason == "no_feasible_samples"
    assert np.isfinite(out.trajectory.controls).all()


def test_invalid_noise_fallback_and_timeout():
    k = kernel()
    p = MPPIAdapter(k)
    r = low_input()
    noise = torch.full((2, 1, 64, 15, 2), torch.nan)
    out = k.plan_batch([p], [r], noise=noise)[0]
    assert not out.valid and out.fallback_reason == "all_samples_invalid"
    assert np.isfinite(out.trajectory.controls).all()
    assert p.last_diagnostics["effective_samples"] >= 0
    k = kernel(deadline_s=0.1)
    ticks = iter([0, 1, 2, 3, 4])
    out = k.plan_batch([MPPIAdapter(k)], [r], clock=lambda: next(ticks))[0]
    assert not out.valid and out.fallback_reason == "planner_timeout"


@pytest.mark.parametrize("samples", [512, 1024, 2048])
def test_sample_profiles_actual_cpu_plan(samples, record_property):
    torch.set_num_threads(1)
    k = MPPIKernel(MPPIConfig(samples=samples))
    out = MPPIAdapter(k).plan(low_input())
    assert out.valid
    record_property("samples", samples)
    record_property("seconds", out.feedback.solve_seconds)


def test_batch_environment_no_solver_async_restore(tmp_path):
    cfg = cfg_from_experiment("configs/experiment/t06_mppi_base.yaml")
    cfg.simulation.num_envs = 2
    cfg.local_goal_contract.mppi_samples = 64

    def forbidden(*a, **kw):
        raise AssertionError("acados forbidden")

    env = LocalGoalEnvironment(cfg, tmp_path, solver_factory=forbidden)
    actions = PolicyAction(torch.full((2, 4, 2), 0.0), torch.zeros(2, 4, dtype=torch.bool))
    actions = PolicyAction(
        torch.stack((torch.full((2, 4), 0.4), torch.zeros(2, 4)), dim=-1), actions.hold
    )
    out = env.step(actions, max_low_steps=1)
    assert env.planners[0].batch_kernel.batch_calls == 1
    assert len(out.records[0]["planner_execution"]) == 8
    state = env.state_dict()
    env.load_state_dict(state)
    saved = env.planners[4].state_dict()
    env.reset([0])
    assert torch.equal(saved["rng_state"], env.planners[4].state_dict()["rng_state"])
    np.testing.assert_array_equal(saved["warm"], env.planners[4].warm)
    env.pool.shutdown()


@pytest.mark.parametrize(
    "changes", [{"temperature": 0}, {"samples": 0}, {"noise_v": float("nan")}, {"iterations": True}]
)
def test_invalid_config(changes):
    with pytest.raises(ValueError):
        MPPIConfig(**changes)


def test_cuda_explicit_unavailable_and_version_rejection():
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError):
            MPPIKernel(MPPIConfig(device="cuda"))
    cfg = cfg_from_experiment("configs/experiment/t06_mppi_base.yaml")
    assert contract_metadata(cfg)["mppi"]["samples"] == 512
    cfg.local_goal_contract.planner_revision = "prefix_consistency_v2"
    with pytest.raises(ValueError):
        contract_metadata(cfg)


def test_stable_numerics_contract_is_explicit_and_legacy_metadata_unchanged():
    legacy = MPPIConfig().metadata()
    assert set(legacy) == {
        "adapter", "samples", "iterations", "temperature", "noise_v", "noise_w",
        "device", "dtype", "query_chunk", "terrain_cost", "neighbor_cost", "unknown_cost",
        "terrain_dynamics", "slope_scale", "deadline_s", "time_grid", "cost_version",
        "actuator_mapping", "sampling", "dynamics_version", "terrain_snapshot",
    }
    cfg = cfg_from_experiment("configs/experiment/t07_mppi_terrain_neighbor.yaml")
    before = contract_metadata(cfg)
    cfg.local_goal_contract.planner_adapter = "mppi_terrain_neighbor_stable_v2"
    cfg.local_goal_contract.planner_revision = "mppi_terrain_neighbor_stable_v2"
    metadata = contract_metadata(cfg)
    assert before["planner_state_version"] == 1 and metadata["planner_state_version"] == 2
    assert metadata["mppi"]["dtype"] == "float32"
    assert metadata["mppi"]["numerics_version"] == "fp32_fp64_accum_v2"
    assert metadata["mppi"]["accumulation_dtype"] == "float64"
    assert metadata["mppi"]["terrain_query_memory_budget_bytes"] == 256 * 1024**2
    from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.registry import create_adapters

    planner = create_adapters(1, None, cfg.terrain.slope_speed_scale, cfg.local_goal_contract)[0]
    assert planner.state_dict()["state_version"] == 2


def test_legacy_snapshot_schema_and_stable_precision_reject_without_mutation():
    legacy = MPPIAdapter(kernel())
    saved_legacy = legacy.state_dict()
    assert saved_legacy["state_version"] == 1
    assert "numerics_version" not in saved_legacy["configuration"]["mppi"]
    restored_legacy = MPPIAdapter(kernel())
    restored_legacy.load_state_dict(saved_legacy)
    assert torch.equal(restored_legacy.rng.get_state(), legacy.rng.get_state())
    stable_kernel = MPPIKernel(
        MPPIConfig(adapter="mppi_terrain_neighbor_stable_v2", samples=64)
    )
    stable = MPPIAdapter(stable_kernel)
    saved = stable.state_dict()
    for incompatible in (saved_legacy, {**saved, "state_version": 1}):
        with pytest.raises(ValueError):
            stable.load_state_dict(incompatible)
        assert torch.equal(stable.rng.get_state(), saved["rng_state"])
    forged = deepcopy(saved)
    forged["configuration"]["mppi"].pop("numerics_version")
    with pytest.raises(ValueError, match="configuration"):
        stable.load_state_dict(forged)
    different_dtype = MPPIAdapter(MPPIKernel(replace(stable_kernel.config, dtype="float64")))
    before = different_dtype.state_dict()
    with pytest.raises(ValueError, match="configuration"):
        different_dtype.load_state_dict(saved)
    assert torch.equal(different_dtype.rng.get_state(), before["rng_state"])


def test_extreme_costs_have_finite_normalized_weights():
    mean = torch.zeros(1, 15, 2)
    noise = torch.ones(1, 3, 15, 2)
    cost = torch.tensor([[1e30, -1e30, float("inf")]])
    out, w, _, usable = importance_update(
        mean, noise, cost, torch.ones(1, 3, dtype=torch.bool), 1.0, torch.tensor([0.35, 0.7])
    )
    assert torch.isfinite(out).all() and torch.isfinite(w).all()
    assert w.sum() == 1 and not usable[0, 2] and w[0, 1] == 1


@pytest.mark.parametrize("seed", [23, 31, 47])
def test_reproducible_seed_and_nonfinite_batch_atomic_rejection(seed):
    k = kernel()
    r = low_input()
    a, b = MPPIAdapter(k, seed), MPPIAdapter(k, seed)
    np.testing.assert_array_equal(a.plan(r).trajectory.controls, b.plan(r).trajectory.controls)
    p, q = MPPIAdapter(k), MPPIAdapter(k)
    invalid = replace(r, agent_id=1, ego_pose=np.array([np.nan, 0, 0]))
    before = p.state_dict()
    with pytest.raises(ValueError):
        k.plan_batch([p, q], [r, invalid])
    assert (
        p.identity is None
        and p.warm is None
        and torch.equal(before["rng_state"], p.state_dict()["rng_state"])
    )


def test_base_can_plan_with_optional_solver_imports_forbidden(monkeypatch):
    import builtins

    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in {"casadi", "acados_template"}:
            raise ImportError("Optional solver deliberately unavailable")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    assert MPPIAdapter(kernel()).plan(low_input()).valid


def test_snapshot_contains_request_state_and_rejects_missing():
    p = MPPIAdapter(kernel())
    r = low_input()
    p.plan(r)
    saved = p.state_dict()
    np.testing.assert_array_equal(saved["intent"].goal_xy, r.intent.goal_xy)
    wrong = deepcopy(saved)
    del wrong["intent"]
    with pytest.raises(ValueError):
        p.load_state_dict(wrong)
    q = MPPIAdapter(p.batch_kernel)
    q.load_state_dict(saved)
    np.testing.assert_array_equal(q.intent.goal_xy, r.intent.goal_xy)
    q.reset()
    assert q.intent is None


def test_reported_cost_uses_actual_selected_trajectory():
    r = low_input()
    out = MPPIAdapter(kernel()).plan(r)
    t = out.trajectory
    d2 = ((t.states[1:, :2] - r.intent.goal_xy) ** 2).sum(-1)
    delta = np.diff(np.vstack((r.previous_control, t.controls)), axis=0)
    expected = (
        2 * d2.sum()
        + 12 * d2[-1]
        + (t.controls**2 * np.array([0.05, 0.03])).sum()
        + (delta**2 * np.array([0.5, 0.2])).sum()
        + 0.2 * np.minimum(t.controls[:, 0], 0).dot(np.minimum(t.controls[:, 0], 0))
    )
    assert out.total_cost == pytest.approx(expected, abs=1e-5, rel=1e-5)


def test_invalid_sampling_reuses_only_revalidated_previous():
    from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.protocol import IdentifiedNeighbor
    from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.contracts import (
        NeighborPrediction,
    )

    k = kernel()
    p = MPPIAdapter(k)
    r = low_input()
    out = p.plan(r)
    r = replace(
        r,
        now=0.2,
        times=r.times + 0.2,
        ego_pose=out.trajectory.states[1],
        previous_control=out.trajectory.controls[0],
    )
    noise = torch.full((2, 1, 64, 15, 2), torch.nan)
    reused = k.plan_batch([p], [r], noise=noise)[0]
    assert (
        reused.valid
        and reused.feedback.used_previous
        and reused.fallback_reason == "all_samples_invalid"
    )
    neighbor = IdentifiedNeighbor(
        1, NeighborPrediction(np.tile(r.ego_pose[:2], (16, 1)), np.full(16, 0.46), 0, 1)
    )
    rejected = k.plan_batch([p], [replace(r, neighbors=(neighbor,))], noise=noise)[0]
    assert not rejected.valid and rejected.feedback.emergency_brake
    assert rejected.feedback.candidate_rejection == "neighbor_swept_collision"


def test_python_numpy_integer_identity_same_rng_stream():
    k = kernel()
    a, b = MPPIAdapter(k), MPPIAdapter(k)
    r = low_input()
    other = replace(r, env_id=np.int64(r.env_id), agent_id=np.int64(r.agent_id))
    np.testing.assert_array_equal(a.plan(r).trajectory.controls, b.plan(other).trajectory.controls)
    a.reset()
    b.reset()
    assert torch.equal(a.rng.get_state(), b.rng.get_state())
