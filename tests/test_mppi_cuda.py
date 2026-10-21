"""Actual CUDA contracts; a skipped case is not CUDA acceptance evidence."""

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest
import torch

from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.contracts import (
    DT,
    N,
    NeighborPrediction,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.execution import (
    validate_trajectory,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.memory import LocalTerrainMemory
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.mppi import (
    MPPIAdapter,
    MPPIConfig,
    MPPIKernel,
    importance_update,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.protocol import (
    IdentifiedNeighbor,
    LocalIntent,
    PlannerInput,
    TerrainView,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.tensor_views import TerrainBatch


@pytest.fixture(scope="module", autouse=True)
def actual_cuda():
    if not torch.cuda.is_available():
        pytest.skip("Actual CUDA device required; skip is not CUDA acceptance")
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.cuda.synchronize()
    torch.set_num_threads(previous_threads)


def kernel(device, *, query_chunk=1024, samples=64, dtype="float32"):
    return MPPIKernel(
        MPPIConfig(
            adapter="mppi_terrain_neighbor_stable_v2",
            device=device,
            samples=samples,
            dtype=dtype,
            query_chunk=query_chunk,
            terrain_cost=True,
            neighbor_cost=True,
            unknown_cost=True,
            terrain_dynamics=True,
        )
    )


def request(scene="flat", *, agent_id=0):
    axis = np.arange(-1.6, 1.61, 0.1)
    xy = np.stack(np.meshgrid(axis, axis), -1).reshape(-1, 2)
    features = np.zeros((len(xy), 5))
    features[:, 4] = 1
    if scene == "risk":
        features[xy[:, 1] < -0.02, 4] = 0.35
    if scene == "unknown":
        keep = xy[:, 0] <= 0.4
        xy, features = xy[keep], features[keep]
    memory = LocalTerrainMemory()
    memory.observe(xy, features, 0)
    neighbors = ()
    if scene in ("cross", "cross_offset"):
        t = np.arange(N + 1) * DT
        # Match the established crossing geometry in diagnose_planners.py.
        x, y = (0.45, 0.7) if scene == "cross" else (0.65, 0.9)
        positions = np.stack((np.full(N + 1, x), y - 0.5 * t), -1)
        prediction = NeighborPrediction(positions, np.full(N + 1, 0.46), 0, 1)
        neighbors = (IdentifiedNeighbor(2, prediction),)
    return PlannerInput(
        0,
        agent_id,
        np.zeros(3),
        np.zeros(2),
        LocalIntent(np.array([0.8, 0]), False, 0, 1),
        TerrainView(memory).conservative(),
        neighbors,
        np.zeros(2),
        0,
        np.arange(N + 1) * DT,
    )


@pytest.mark.parametrize("scene", ["flat", "risk", "cross", "cross_offset", "unknown"])
@pytest.mark.parametrize("samples", [64, 128, 512])
@pytest.mark.parametrize("seed", [23, 31, 47])
def test_cpu_cuda_same_injected_noise(scene, samples, seed, record_property):
    item = request(scene)
    noise = np.random.default_rng(seed).normal(size=(2, 1, samples, N, 2)) * [0.35, 0.7]
    planners = {
        device: MPPIAdapter(kernel(device, samples=samples), seed=seed)
        for device in ("cpu", "cuda")
    }
    outputs = {}
    for device, planner in planners.items():
        compute = planner.batch_kernel
        # Samples are exported to CPU by design; inspect the actual compute result.
        probe = compute.evaluate([item], compute.tensor(noise[0]))
        assert all(value.device.type == device for value in probe[:5])
        assert all(value.dtype == torch.float32 for value in probe[:3])
        assert probe[3].dtype == torch.float64
        outputs[device] = compute.plan_batch([planner], [item], noise=noise)[0]
        assert planner.last_samples["raw"].dtype == torch.float32
        assert planner.last_samples["weights"].dtype == torch.float64
    torch.cuda.synchronize()
    cpu, gpu = outputs["cpu"], outputs["cuda"]
    assert (cpu.valid, cpu.fallback_reason, cpu.feedback.status) == (
        gpu.valid,
        gpu.fallback_reason,
        gpu.feedback.status,
    )
    for name in ("states", "controls", "velocities"):
        np.testing.assert_allclose(
            getattr(cpu.trajectory, name), getattr(gpu.trajectory, name), atol=1e-5, rtol=1e-5
        )
    for name in ("states", "controls", "velocities", "cost", "score", "weights"):
        np.testing.assert_allclose(
            planners["cpu"].last_samples[name],
            planners["cuda"].last_samples[name],
            atol=1e-5,
            rtol=1e-5,
        )
    np.testing.assert_array_equal(
        planners["cpu"].last_samples["valid"], planners["cuda"].last_samples["valid"]
    )
    if gpu.valid:
        assert not validate_trajectory(
            gpu.trajectory,
            item.terrain,
            [p.prediction for p in item.neighbors],
            1.15,
            item.previous_control,
        )
    record_property("cuda_device", torch.cuda.get_device_name())
    record_property("samples", samples)
    record_property("seed", seed)
    record_property(
        "max_control_error", float(np.max(abs(cpu.trajectory.controls - gpu.trajectory.controls)))
    )


@pytest.mark.parametrize("chunk", [1, 7, 256, 1024, 2048])
def test_cuda_terrain_ties_support_boundary_and_empty_memory(chunk):
    memory = LocalTerrainMemory(support_radius=0.3)
    memory.observe(
        np.array([[-0.1, 0], [0.1, 0]]),
        np.array([[0, 0, 0, 0, 1], [0, 1, 0, 0, 0.1]]),
        0,
    )
    terrain = TerrainView(memory).conservative()
    empty = TerrainView(LocalTerrainMemory())
    # Include an exact tie, direct support, and points on either side of support.
    points = np.array([[0, 0], [-0.1, 0], [-0.1, 0.299], [-0.1, 0.301]])
    results = {}
    for device in ("cpu", "cuda"):
        compute = kernel(device)
        batch = TerrainBatch([terrain, empty], compute.tensor, chunk)
        result = batch.lookup(compute.tensor(np.stack([points, points])))
        assert batch.xy.device.type == device
        assert all(value.device.type == device for value in result)
        assert batch.max_query_block <= chunk
        results[device] = tuple(value.cpu().numpy() for value in result)
    np.testing.assert_allclose(results["cpu"][0], results["cuda"][0], atol=1e-5, rtol=1e-5)
    for index in (1, 2):
        np.testing.assert_array_equal(results["cpu"][index], results["cuda"][index])
    features, known, direct = results["cuda"]
    np.testing.assert_array_equal(known[0], [True, True, True, False])
    assert not known[1].any() and not direct[1].any()
    assert direct[0, 1] and not direct[0, 0]
    assert features[0, 0, 1] == 1 and features[0, 0, 4] == pytest.approx(0.1)
    for index, view in enumerate((terrain, empty)):
        expected = view.lookup(points)
        np.testing.assert_allclose(features[index], expected[0], atol=1e-5, rtol=1e-5)
        np.testing.assert_array_equal(known[index], expected[1])
        np.testing.assert_array_equal(direct[index], expected[2])


@pytest.mark.parametrize("scene", ["flat", "risk", "cross", "cross_offset", "unknown"])
def test_cuda_query_chunks_preserve_output_and_hard_validity(scene):
    item = request(scene)
    noise = np.random.default_rng(31).normal(size=(2, 1, 64, N, 2)) * [0.35, 0.7]
    reference = None
    for chunk in (256, 1024, 2048):
        compute = kernel("cuda", query_chunk=chunk)
        planner = MPPIAdapter(compute)
        output = compute.plan_batch([planner], [item], noise=noise)[0]
        if reference is None:
            reference = (output, planner.last_samples)
            continue
        previous, samples = reference
        assert (output.valid, output.fallback_reason, output.feedback.status) == (
            previous.valid,
            previous.fallback_reason,
            previous.feedback.status,
        )
        np.testing.assert_allclose(
            output.trajectory.controls, previous.trajectory.controls, atol=1e-5, rtol=1e-5
        )
        np.testing.assert_allclose(
            output.trajectory.states, previous.trajectory.states, atol=1e-5, rtol=1e-5
        )
        np.testing.assert_array_equal(planner.last_samples["valid"], samples["valid"])
        np.testing.assert_allclose(
            planner.last_samples["weights"], samples["weights"], atol=1e-5, rtol=1e-5
        )


def test_cuda_rng_identity_order_snapshot_and_isolated_reset():
    compute = kernel("cuda")
    requests = [request(agent_id=agent) for agent in range(2)]
    original = [MPPIAdapter(compute) for _ in requests]
    reordered = [MPPIAdapter(compute) for _ in requests]
    forward = compute.plan_batch(original, requests)
    reverse = compute.plan_batch(reordered[::-1], requests[::-1])
    for first, second in zip(forward, reverse[::-1]):
        np.testing.assert_allclose(
            first.trajectory.controls, second.trajectory.controls, atol=1e-5, rtol=1e-5
        )
    assert not torch.equal(original[0].rng.get_state(), original[1].rng.get_state())
    saved = original[1].state_dict()
    assert saved["rng_device"] == "cuda" and saved["warm"] is not None
    restored = MPPIAdapter(compute)
    restored.load_state_dict(saved)
    unchanged_rng = original[1].rng.get_state().clone()
    original[0].reset()
    assert torch.equal(original[1].rng.get_state(), unchanged_rng)
    following = replace(
        requests[1],
        now=DT,
        times=requests[1].times + DT,
        ego_pose=forward[1].trajectory.states[1],
        previous_control=forward[1].trajectory.controls[0],
    )
    continued, recovered = original[1].plan(following), restored.plan(following)
    np.testing.assert_array_equal(continued.trajectory.controls, recovered.trajectory.controls)
    np.testing.assert_array_equal(continued.trajectory.states, recovered.trajectory.states)
    assert torch.equal(original[1].rng.get_state(), restored.rng.get_state())
    restored.reset()
    assert restored.warm is None and restored.previous is None and restored.last_samples is None


@pytest.mark.parametrize("source_device,target_device", [("cpu", "cuda"), ("cuda", "cpu")])
def test_cross_device_snapshot_rejected_without_mutation(source_device, target_device):
    source, target = MPPIAdapter(kernel(source_device)), MPPIAdapter(kernel(target_device))
    source.bind(request())
    source.sample_noise(64)
    state = source.state_dict()
    before = target.state_dict()
    with pytest.raises(ValueError, match="Planner state mismatch"):
        target.load_state_dict(state)
    assert target.identity == before["identity"] and target.warm is None
    assert torch.equal(target.rng.get_state(), before["rng_state"])
    # Even a caller rewriting config metadata cannot bypass the RNG device boundary.
    forged = deepcopy(state)
    forged["configuration"] = before["configuration"]
    with pytest.raises(ValueError, match="Cross-device planner RNG"):
        target.load_state_dict(forged)
    assert torch.equal(target.rng.get_state(), before["rng_state"])


def test_cuda_same_noise_batch_single_reorder_and_cpu_batch():
    inputs = [request(scene, agent_id=i) for i, scene in enumerate(("flat", "cross_offset", "unknown"))]
    noise = np.random.default_rng(47).normal(size=(2, 3, 128, N, 2)) * [0.35, 0.7]
    gpu, cpu = kernel("cuda", samples=128), kernel("cpu", samples=128)
    batch = gpu.plan_batch([MPPIAdapter(gpu) for _ in inputs], inputs, noise=noise)
    reference = cpu.plan_batch([MPPIAdapter(cpu) for _ in inputs], inputs, noise=noise)
    reordered = gpu.plan_batch(
        [MPPIAdapter(gpu) for _ in inputs], inputs[::-1], noise=noise[:, ::-1].copy()
    )
    for i, item in enumerate(inputs):
        single = gpu.plan_batch([MPPIAdapter(gpu)], [item], noise=noise[:, i:i+1])[0]
        for other in (single, reference[i], reordered[-1-i]):
            assert batch[i].valid == other.valid and batch[i].feedback.status == other.feedback.status
            np.testing.assert_allclose(
                batch[i].trajectory.controls, other.trajectory.controls, atol=1e-5, rtol=1e-5
            )
            np.testing.assert_allclose(
                batch[i].trajectory.states, other.trajectory.states, atol=1e-5, rtol=1e-5
            )


def test_cuda_float64_reference_retains_float64_precision():
    item = request("cross_offset")
    noise = np.random.default_rng(23).normal(size=(2, 1, 64, N, 2)) * [0.35, 0.7]
    outputs = []
    for device in ("cpu", "cuda"):
        compute = kernel(device, dtype="float64")
        planner = MPPIAdapter(compute)
        outputs.append(compute.plan_batch([planner], [item], noise=noise)[0])
        assert planner.last_samples["raw"].dtype == torch.float64
        assert planner.last_samples["states"].dtype == torch.float64
    np.testing.assert_allclose(
        outputs[0].trajectory.controls, outputs[1].trajectory.controls, atol=1e-10, rtol=1e-10
    )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_stable_importance_float64_reference_and_all_invalid(device):
    rng = np.random.default_rng(31)
    mean = torch.tensor(rng.normal(size=(1, N, 2)), device=device, dtype=torch.float32)
    noise = torch.tensor(rng.normal(size=(1, 7, N, 2)), device=device, dtype=torch.float32)
    cost = torch.tensor([[1, 3, 2, 5, 4, 6, 7]], device=device, dtype=torch.float64)
    sigma = torch.tensor([0.35, 0.7], device=device, dtype=torch.float32)
    updated, weights, score, usable = importance_update(
        mean, noise, cost, torch.ones_like(cost, dtype=torch.bool), 1.0, sigma,
        accumulation_dtype=torch.float64,
    )
    m, e, c, s = (x.cpu().numpy().astype(np.float64) for x in (mean, noise, cost, sigma))
    expected_score = c + (m[:, None] * e / s**2).sum(axis=(-1, -2))
    expected_weights = np.exp(-(expected_score - expected_score.min(axis=-1, keepdims=True)))
    expected_weights /= expected_weights.sum(axis=-1, keepdims=True)
    expected_updated = (m + (expected_weights[..., None, None] * e).sum(axis=1)).astype(np.float32)
    assert updated.dtype == torch.float32 and weights.dtype == torch.float64 and usable.all()
    np.testing.assert_allclose(score.cpu(), expected_score, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(weights.cpu(), expected_weights, atol=1e-12, rtol=1e-12)
    np.testing.assert_array_equal(updated.cpu(), expected_updated)
    invalid_cost = torch.full_like(cost, torch.nan)
    noise[:] = torch.nan
    updated, weights, _, usable = importance_update(
        mean, noise, invalid_cost, torch.ones_like(cost, dtype=torch.bool), 1.0, sigma,
        accumulation_dtype=torch.float64,
    )
    assert torch.equal(updated, mean) and torch.count_nonzero(weights) == 0 and not usable.any()


def test_cuda_nonfinite_input_atomic_rejection_and_all_invalid_samples():
    compute = kernel("cuda")
    first, second = MPPIAdapter(compute), MPPIAdapter(compute)
    before = first.state_dict()
    invalid = replace(request(agent_id=1), ego_pose=np.array([np.nan, 0, 0]))
    with pytest.raises(ValueError):
        compute.plan_batch([first, second], [request(), invalid])
    assert first.identity is None and torch.equal(first.rng.get_state(), before["rng_state"])
    noise = np.full((2, 1, 64, N, 2), np.nan)
    output = compute.plan_batch([first], [request()], noise=noise)[0]
    assert not output.valid and output.fallback_reason == "all_samples_invalid"
    assert torch.isfinite(first.last_samples["weights"]).all()
    assert torch.count_nonzero(first.last_samples["weights"]) == 0
    assert np.isfinite(output.trajectory.states).all()


@pytest.mark.parametrize("expired", [False, True])
def test_cuda_hold_expiry_and_precision_state_rejection(expired):
    item = request()
    intent = LocalIntent.for_hold(item.ego_pose, 0)
    if expired:
        item = replace(item, now=1.2, times=item.times + 1.2)
    item = replace(item, intent=intent)
    results = [MPPIAdapter(kernel(device)).plan(item) for device in ("cpu", "cuda")]
    np.testing.assert_array_equal(results[0].trajectory.controls, results[1].trajectory.controls)
    assert results[0].valid == results[1].valid
    if expired:
        assert not results[1].valid and results[1].fallback_reason == "expired_intent"
    target = MPPIAdapter(kernel("cuda"))
    before = target.state_dict()
    incompatible = MPPIAdapter(kernel("cuda", dtype="float64")).state_dict()
    with pytest.raises(ValueError, match="configuration"):
        target.load_state_dict(incompatible)
    assert torch.equal(target.rng.get_state(), before["rng_state"])


def test_cuda_query_budget_b64_m4096_peak_memory_and_conservative_ties(record_property):
    axis = np.arange(-32, 32) * 0.1
    xy = np.stack(np.meshgrid(axis, axis), -1).reshape(-1, 2)
    features = np.zeros((len(xy), 5))
    features[:, 4] = 1
    features[::2, 1] = 0.5
    features[::2, 4] = 0.3
    memory = LocalTerrainMemory(capacity=4096)
    memory.observe(xy, features, 0)
    terrain = TerrainView(memory).conservative()
    assert len(terrain.snapshot().xy) == 4096
    points = np.tile(np.array([[0.05, 0.05], [0, 0], [-3.4, 0], [3.1, 3.1]]), (16, 1))
    compute = kernel("cuda")
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()
    query = compute.tensor(np.stack([points] * 64))
    bounded = TerrainBatch(
        [terrain] * 64, compute.tensor, 1024,
        memory_budget_bytes=compute.config.query_memory_budget,
    )
    actual = bounded.lookup(query)
    torch.cuda.synchronize()
    temporary_peak = torch.cuda.max_memory_allocated() - baseline
    assert bounded.chunk == 28 and bounded.max_query_block == 28
    assert temporary_peak < compute.config.query_memory_budget
    expected = terrain.lookup(points)
    for b in (0, 31, 63):
        np.testing.assert_allclose(actual[0][b].cpu(), expected[0], atol=1e-5, rtol=1e-5)
        np.testing.assert_array_equal(actual[1][b].cpu(), expected[1])
        np.testing.assert_array_equal(actual[2][b].cpu(), expected[2])
    assert actual[0][0, 0, 1].item() == 0.5
    assert actual[0][0, 0, 4].item() == pytest.approx(0.3)
    assert not actual[1][0, 2]
    record_property("batch_size", 64)
    record_property("memory_samples", 4096)
    record_property("effective_query_chunk", bounded.chunk)
    record_property("peak_extra_allocated_bytes", temporary_peak)
