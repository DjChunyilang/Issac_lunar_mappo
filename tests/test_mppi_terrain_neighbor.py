"""T07 local information, functional risk costs and real CPU closure evidence."""

from dataclasses import replace
from copy import deepcopy
import numpy as np
import pytest
import torch

from test_mppi import low_input
from _common import cfg_from_experiment
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.protocol import (
    TerrainView,
    LocalIntent,
    IdentifiedNeighbor,
    plan_batch,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.contracts import NeighborPrediction
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.memory import LocalTerrainMemory
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.tensor_views import (
    TerrainBatch,
    NeighborBatch,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.mppi import (
    MPPIConfig,
    MPPIKernel,
    MPPIAdapter,
    feasible_samples_cpu,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.execution import (
    validate_trajectory,
    dynamics,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.environment import (
    LocalGoalEnvironment,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.actions import PolicyAction


def view(*, risk=False, blocked=False, boundary=False, slope=0):
    memory = LocalTerrainMemory()
    axis = np.arange(-1.6, 1.61, 0.1)
    xy = np.stack(np.meshgrid(axis, axis), -1).reshape(-1, 2)
    if boundary:
        xy = xy[xy[:, 0] <= 0.4]
    features = np.zeros((len(xy), 5))
    features[:, 4] = 1
    features[:, 1] = slope
    if risk:
        features[xy[:, 1] < -0.02, 4] = 0.35
    if blocked:
        features[(xy[:, 0] > 0.4) & (xy[:, 0] < 0.7), 4] = 0.1
    memory.observe(xy, features, 0.5)
    return TerrainView(memory).conservative()


def kernel(**kwargs):
    torch.set_num_threads(1)
    values = dict(
        adapter="mppi_terrain_neighbor_v1",
        samples=64,
        terrain_cost=True,
        neighbor_cost=True,
        unknown_cost=True,
        terrain_dynamics=True,
    )
    values.update(kwargs)
    return MPPIKernel(MPPIConfig(**values))


def peer(sender, positions, radius=0.46, timestamp=0, expiry=1):
    return IdentifiedNeighbor(
        sender, NeighborPrediction(np.asarray(positions), np.full(16, radius), timestamp, expiry)
    )


@pytest.mark.parametrize("chunk", [1, 7, 256])
def test_snapshot_tensor_nearest_and_unknown(chunk):
    terrain = view(slope=0.2)
    snapshot = terrain.snapshot()
    assert snapshot.version == "local_terrain_samples_v1"
    assert not snapshot.xy.flags.writeable and not snapshot.features.flags.writeable
    with pytest.raises(ValueError):
        snapshot.features[0, 4] = 0
    assert not hasattr(snapshot, "core") and not hasattr(terrain, "terrain_runtime")
    k = kernel(dtype="float64", query_chunk=chunk)
    batch = TerrainBatch([terrain, TerrainView(LocalTerrainMemory())], k.tensor, chunk)
    points = np.random.default_rng(23).uniform(-2, 2, size=(2, 9, 3, 2))
    points[0, 0, 0] = snapshot.xy[0]
    f, known, direct = batch.lookup(k.tensor(points))
    for b, v in enumerate([terrain, TerrainView(LocalTerrainMemory())]):
        expected = v.lookup(points[b].reshape(-1, 2))
        np.testing.assert_allclose(f[b].reshape(-1, 5), expected[0], atol=1e-10, rtol=1e-8)
        np.testing.assert_array_equal(known[b].reshape(-1), expected[1])
        np.testing.assert_array_equal(direct[b].reshape(-1), expected[2])
    assert batch.max_query_block <= chunk


def test_equidistant_conflicting_terrain_is_conservative():
    memory = LocalTerrainMemory(support_radius=0.3)
    memory.observe(
        np.array([[-0.1, 0], [0.1, 0]]), np.array([[0, 0, 0, 0, 1], [0, 1, 0, 0, 0.1]]), 0
    )
    terrain = TerrainView(memory).conservative()
    k = kernel(dtype="float64")
    f, known, _ = TerrainBatch([terrain], k.tensor).lookup(k.tensor([[[0, 0]]]))
    assert known.all() and f[0, 0, 4] == 0.1 and f[0, 0, 1] == 1
    assert not terrain.safe([[0, 0]])[0]
    np.testing.assert_allclose(f[0, 0], terrain.lookup([[0, 0]])[0][0])


def test_stable_neighbor_cost_matches_float64_formula_from_float32_inputs():
    r = replace(low_input(), neighbors=(peer(2, np.tile([0.6, 0.2], (16, 1))),))
    k = kernel()
    packed = NeighborBatch([r], k.tensor)
    states = k.tensor(np.random.default_rng(23).normal(size=(1, 7, 16, 3)) * 0.2)
    delta = states.numpy()[:, :, 1:, None, :2].astype(np.float64) - packed.positions.numpy()[
        :, None, :, 1:
    ].transpose(0, 1, 3, 2, 4).astype(np.float64)
    radius = packed.radii.numpy()[:, None, :, 1:].transpose(0, 1, 3, 2).astype(np.float64)
    expected = 8 * np.square(
        np.maximum((radius + 0.2 - np.linalg.norm(delta, axis=-1)) / 0.2, 0)
    ).sum(axis=(-1, -2))
    actual = packed.cost(states, accumulation_dtype=torch.float64)
    assert actual.dtype == torch.float64 and packed.cost(states).dtype == torch.float32
    np.testing.assert_allclose(actual.numpy(), expected, atol=1e-12, rtol=1e-12)


def test_query_memory_budget_is_opt_in_and_deterministic():
    k = kernel()
    terrain = view()
    count = len(terrain.snapshot().xy)
    budget = 2 * count * (8 * 4 + 4) * 3
    bounded = TerrainBatch([terrain, terrain], k.tensor, 256, memory_budget_bytes=budget)
    assert bounded.chunk == 3
    legacy = TerrainBatch([terrain, terrain], k.tensor, 256)
    assert legacy.chunk == 256
    points = k.tensor(np.random.default_rng(31).uniform(-2, 2, size=(2, 11, 2)))
    for a, b in zip(bounded.lookup(points), legacy.lookup(points)):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    assert bounded.max_query_block == 3
    with pytest.raises(ValueError, match="memory budget"):
        TerrainBatch([terrain], k.tensor, memory_budget_bytes=0)


def test_terrain_scaled_motion_matches_numpy():
    k = kernel(dtype="float64")
    r = replace(low_input(), terrain=view(slope=0.3))
    raw = k.tensor(np.random.default_rng(23).normal(size=(1, 3, 15, 2)) * 0.15)
    states, controls, velocities, *_ = k.evaluate([r], raw)
    p = MPPIAdapter(k)
    for i in range(3):
        host = p.predict(r, controls[0, i].numpy())
        np.testing.assert_allclose(states[0, i], host.states, atol=1e-10, rtol=1e-8)
        np.testing.assert_allclose(velocities[0, i], host.velocities, atol=1e-10, rtol=1e-8)


def test_risk_changes_ranking_and_selected_control_same_noise():
    r = replace(low_input(), terrain=view(risk=True))
    on = kernel(terrain_dynamics=False)
    off = kernel(terrain_dynamics=False, terrain_cost=False)
    raw = np.zeros((1, 2, 15, 2))
    raw[..., 0] = 0.3
    raw[0, 0, :, 1] = 0.5
    raw[0, 1, :, 1] = -0.5
    *_, cost_on, valid_on, parts = on.evaluate([r], on.tensor(raw))
    *_, cost_off, _, _ = off.evaluate([r], off.tensor(raw))
    assert valid_on.all() and parts["terrain"][0, 1] > parts["terrain"][0, 0]
    assert cost_on[0, 1] > cost_on[0, 0]
    assert abs(float(cost_off[0, 1] - cost_off[0, 0])) < 1e-4
    noise = np.random.default_rng(31).normal(size=(2, 1, 64, 15, 2)) * [0.35, 0.7]
    a = on.plan_batch([MPPIAdapter(on)], [r], noise=noise)[0]
    b = off.plan_batch([MPPIAdapter(off)], [r], noise=noise)[0]
    assert a.valid and b.valid
    assert np.max(abs(a.trajectory.controls - b.trajectory.controls)) > 1e-4


@pytest.mark.parametrize("blocked,boundary", [(True, False), (False, True)])
def test_impassable_unknown_and_soft_switches_keep_hard_checks(blocked, boundary):
    r = replace(low_input((1.2, 0)), terrain=view(blocked=blocked, boundary=boundary))
    k = kernel(terrain_dynamics=False, terrain_cost=False, unknown_cost=False)
    raw = np.zeros((1, 1, 15, 2))
    raw[..., 0] = 1.15
    states, controls, _, _, valid, _ = k.evaluate([r], k.tensor(raw))
    assert not valid.any()
    np.testing.assert_array_equal(
        valid[0], feasible_samples_cpu(r, states[0].numpy(), controls[0].numpy())
    )
    out = MPPIAdapter(k).plan(r)
    assert out.valid and out.feedback.status == "feasible_prefix"
    assert out.feedback.prefix_steps > 0 and abs(out.trajectory.controls[-1]).max() == 0
    assert not validate_trajectory(out.trajectory, r.terrain, [], 1.15, r.previous_control)


@pytest.mark.parametrize("mode", ["cross", "head_on", "stopped", "expired"])
def test_neighbor_hard_checks_sender_and_soft_cost(mode):
    t = np.arange(16) * 0.2
    xy = (
        np.stack((np.full(16, 0.45), 0.7 - 0.5 * t), -1)
        if mode == "cross"
        else np.stack((1 - 0.5 * t, np.zeros(16)), -1)
    )
    if mode in ("stopped", "expired"):
        xy = np.tile([0.6, 0.25], (16, 1))
    n = peer(
        3, xy, expiry=-0.1 if mode == "expired" else 1, timestamp=-1 if mode == "expired" else 0
    )
    r = replace(low_input(), terrain=view(), neighbors=(n,))
    k = kernel(terrain_dynamics=False)
    raw = np.zeros((1, 3, 15, 2))
    raw[..., 0] = 0.3
    states, controls, _, _, valid, parts = k.evaluate([r], k.tensor(raw))
    assert parts["neighbor"].max() > 0
    np.testing.assert_array_equal(
        valid[0], feasible_samples_cpu(r, states[0].numpy(), controls[0].numpy())
    )
    off = kernel(terrain_dynamics=False, neighbor_cost=False)
    *_, valid_off, parts_off = off.evaluate([r], off.tensor(raw))
    np.testing.assert_array_equal(valid, valid_off)
    assert not parts_off["neighbor"].any()
    n2 = peer(1, np.tile([1.4, 1.4], (16, 1)))
    a = replace(r, neighbors=(n, n2))
    b = replace(r, neighbors=(n2, n))
    packed = NeighborBatch([a, b], k.tensor)
    assert packed.senders.tolist() == [[3, 1], [1, 3]]
    aa = k.evaluate([a], k.tensor(raw))
    bb = k.evaluate([b], k.tensor(raw))
    torch.testing.assert_close(aa[3], bb[3])
    torch.testing.assert_close(aa[4], bb[4])
    out = MPPIAdapter(k).plan(r)
    assert np.isfinite(out.trajectory.states).all()
    if out.valid:
        assert not validate_trajectory(
            out.trajectory, r.terrain, [n.prediction], 1.15, r.previous_control
        )


def test_neighbor_input_changes_selected_plan_and_swept_crossing():
    k = kernel(terrain_dynamics=False)
    r = replace(low_input(), terrain=view())
    n = peer(2, np.tile([0.6, -0.5], (16, 1)))
    noise = np.random.default_rng(47).normal(size=(2, 1, 64, 15, 2)) * [0.35, 0.7]
    a = k.plan_batch([MPPIAdapter(k)], [r], noise=noise)[0]
    b = k.plan_batch([MPPIAdapter(k)], [replace(r, neighbors=(n,))], noise=noise)[0]
    assert np.max(abs(a.trajectory.controls - b.trajectory.controls)) > 1e-4
    states = np.zeros((1, 1, 16, 3))
    points = np.zeros((1, 1, 15, 9, 2))
    p = np.tile([0, 1], (16, 1))
    p[0] = [0, -1]
    crossing = NeighborBatch([replace(r, neighbors=(peer(2, p, 0.1),))], k.tensor)
    assert np.linalg.norm(p[0]) > 0.1 and np.linalg.norm(p[1]) > 0.1
    assert not crossing.safe(k.tensor(states), k.tensor(points)).any()


@pytest.mark.parametrize("goal", [(0.8, 0), (0.6, 0.6), (-0.8, 0)])
def test_given_legal_local_closed_loop(goal, record_property):
    k = kernel(samples=128)
    planner = MPPIAdapter(k)
    r = replace(low_input(goal), terrain=view())
    error0 = np.linalg.norm(r.intent.goal_xy - r.ego_pose[:2])
    collision = False
    for step in range(100):
        if step % 5 == 0:
            r = replace(r, intent=replace(r.intent, timestamp=r.now, expires_at=r.now + 1))
        out = planner.plan(r)
        assert out.valid
        assert not validate_trajectory(out.trajectory, r.terrain, [], 1.15, r.previous_control)
        u = out.trajectory.controls[0]
        feature = r.terrain.lookup(r.ego_pose[:2])[0][0]
        pose, velocity = dynamics(r.ego_pose, u, feature, 0.75)
        r = replace(
            r,
            ego_pose=pose,
            ego_velocity=velocity,
            previous_control=u,
            now=(step + 1) * 0.2,
            times=(step + 1 + np.arange(16)) * 0.2,
        )
        if np.linalg.norm(pose[:2] - r.intent.goal_xy) <= 0.2:
            break
    error = np.linalg.norm(r.ego_pose[:2] - r.intent.goal_xy)
    record_property("final_error_m", error)
    record_property("low_steps", step + 1)
    assert error <= 0.2 and error < error0 and not collision
    # Physical MOVE-to-HOLD stopping, independent from task dwell success.
    hold = replace(r, intent=LocalIntent.for_hold(r.ego_pose, r.now))
    for j in range(5):
        out = planner.plan(hold)
        u = out.trajectory.controls[0]
        assert (abs(u - hold.previous_control) <= np.array([0.3001, 0.6001])).all()
        pose, velocity = dynamics(hold.ego_pose, u, np.array([0, 0, 0, 0, 1]), 0.75)
        hold = replace(
            hold,
            ego_pose=pose,
            ego_velocity=velocity,
            previous_control=u,
            now=hold.now + 0.2,
            times=hold.times + 0.2,
        )
    assert abs(hold.ego_velocity).max() < 1e-8


def test_actual_sensor_environment_eight_instances_and_hidden_truth(tmp_path, record_property):
    torch.set_num_threads(1)
    torch.manual_seed(23)
    cfg = cfg_from_experiment("configs/experiment/t07_mppi_terrain_neighbor.yaml")
    cfg.simulation.num_envs = 2
    cfg.local_goal_contract.mppi_samples = 128
    env = LocalGoalEnvironment(
        cfg, tmp_path, solver_factory=lambda *a, **kw: pytest.fail("MPPI must not construct acados")
    )
    try:
        assert all(len(m.samples) > 0 for m in env.memories)
        from test_p0_learning_contracts import test_oracle_and_uncommunicated_truth_invariance_v2

        test_oracle_and_uncommunicated_truth_invariance_v2(env)
        before = env.poses().copy()
        action = PolicyAction(
            torch.tensor([0.35, 0.0]).expand(2, 4, 2).clone(), torch.zeros(2, 4, dtype=torch.bool)
        )
        transition = env.step(action, max_low_steps=1)
        assert env.planners[0].batch_kernel.batch_calls == 1
        assert len(transition.records[0]["planner_execution"]) == 8
        assert all(np.isfinite(p.states).all() for p in env.plans if p is not None)
        state = env.state_dict()
        env.load_state_dict(state)
        unchanged = deepcopy(env.planners[4].state_dict())
        env.reset([0])
        assert torch.equal(unchanged["rng_state"], env.planners[4].state_dict()["rng_state"])
        np.testing.assert_array_equal(unchanged["warm"], env.planners[4].warm)
        record_property("sensor_memory_counts", [len(m.samples) for m in env.memories])
        record_property(
            "actual_motion_m",
            np.linalg.norm(env.poses()[1, :, :2] - before[1, :, :2], axis=-1).tolist(),
        )
        record_property(
            "valid_outputs", sum(x["valid"] for x in transition.records[0]["planner_execution"])
        )
    finally:
        env.close()


def test_query_chunks_and_separate_kernels_pack_to_one_batch():
    r = replace(low_input(), terrain=view(risk=True))
    noise = np.random.default_rng(31).normal(size=(2, 1, 64, 15, 2)) * [0.35, 0.7]
    a = kernel(query_chunk=7)
    b = kernel(query_chunk=256)
    x = a.plan_batch([MPPIAdapter(a)], [r], noise=noise)[0]
    y = b.plan_batch([MPPIAdapter(b)], [r], noise=noise)[0]
    np.testing.assert_allclose(x.trajectory.controls, y.trajectory.controls, atol=1e-5, rtol=1e-5)
    c = kernel(query_chunk=256)
    requests = [r, replace(r, agent_id=1)]
    plan_batch([MPPIAdapter(b), MPPIAdapter(c)], requests)
    assert b.batch_calls == 2 and c.batch_calls == 0


def test_weighted_mean_is_rechecked_then_best_sample_used():
    terrain = view()
    snapshot = terrain.snapshot()
    memory = LocalTerrainMemory()
    features = snapshot.features.copy()
    xy = snapshot.xy
    features[(xy[:, 0] > 0.3) & (xy[:, 0] < 0.6) & (abs(xy[:, 1]) < 0.12), 4] = 0.1
    memory.observe(xy, features, 0)
    r = replace(low_input(), terrain=TerrainView(memory))
    k = kernel(iterations=1, terrain_dynamics=False)
    mean = k.seed_controls([r]).numpy()[0]
    raw = np.zeros((64, 15, 2))
    raw[..., 0] = 0.25
    raw[:32, :, 1] = 1.5
    raw[32:, :, 1] = -1.5
    noise = (raw - mean)[None, None]
    out = k.plan_batch([MPPIAdapter(k)], [r], noise=noise)[0]
    assert out.valid and out.feedback.candidate_rejection == "unknown_or_impassable"
    assert out.feedback.status == "feasible" and abs(out.trajectory.controls[0, 1]) > 0.5


def test_actual_mppi_control_invariant_to_hidden_oracle_and_peer_state(tmp_path):
    cfg = cfg_from_experiment("configs/experiment/t07_mppi_terrain_neighbor.yaml")
    cfg.simulation.num_envs = 1
    cfg.local_goal_contract.mppi_samples = 64
    env = LocalGoalEnvironment(
        cfg, tmp_path, solver_factory=lambda *a, **kw: pytest.fail("acados forbidden")
    )
    try:
        r = replace(
            low_input(),
            terrain=TerrainView(env.memories[0]),
            ego_pose=env.poses()[0, 0],
            neighbors=tuple(env.neighbors(0, 0, identified=True)),
        )
        p = env.planners[0]
        p.bind(r)
        saved = p.state_dict()
        before = p.plan(r)
        env.core.oracle_point.add_(1000)
        env.core.positions[0, 1:, :2].add_(3)
        p.load_state_dict(saved)
        after = p.plan(replace(r, neighbors=tuple(env.neighbors(0, 0, identified=True))))
        np.testing.assert_array_equal(before.trajectory.controls, after.trajectory.controls)
        assert before.valid == after.valid and before.fallback_reason == after.fallback_reason
    finally:
        env.close()
