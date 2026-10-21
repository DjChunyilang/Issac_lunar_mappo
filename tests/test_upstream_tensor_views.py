"""CPU/CUDA local pre-screen queries preserve the owned CPU view semantics."""

from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from _common import cfg_from_experiment
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.protocol import TerrainSnapshot
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_footprint import FootprintNeighbor, FootprintTerrain
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_planning import UpstreamIntent, UpstreamPlannerInput
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_tensor_views import LocalNeighborBatch, LocalTerrainBatch, PAIR_WORK_BYTES


@pytest.fixture(scope="module", autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(scope="module")
def spec():
    return cfg_from_experiment("configs/experiment/upstream_scale_s01_review_v2.yaml").spec


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda":
        assert torch.cuda.is_available(), "Required real CUDA query tests must run using .venv_terrain_cuda"
    return request.param


def view(spec, xy, features=None):
    xy = np.asarray(xy, dtype=float).reshape(-1, 2)
    if features is None:
        features = np.zeros((len(xy), 5)); features[:, 4] = 1.
    return FootprintTerrain(TerrainSnapshot(xy, np.asarray(features, dtype=float), np.zeros(len(xy)), spec.perception.support_radius), spec)


def check_reference(batch, views, queries):
    actual, known = batch.lookup(queries)
    actual, known = actual.cpu().numpy(), known.cpu().numpy()
    for b, terrain in enumerate(views):
        query = queries[b].detach().cpu().numpy()
        features, mask = terrain.lookup(query.reshape(-1, 2))
        reference = np.where(mask[:, None], features, 0.).reshape(*query.shape[:-1], 5).astype(actual.dtype)
        np.testing.assert_array_equal(known[b], mask.reshape(query.shape[:-1]))
        np.testing.assert_array_equal(actual[b], reference)
    return actual, known


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_unique_ties_unknown_and_batch_order_match_cpu_source(spec, device, dtype):
    rng = np.random.default_rng(31)
    xx, yy = np.meshgrid(np.arange(-1., 1.01, .2), np.arange(-1., 1.01, .2))
    xy = np.c_[xx.ravel(), yy.ravel()]
    features = rng.uniform(-.5, .5, (len(xy), 5))
    features[:, 4] = rng.uniform(.1, 1., len(xy))
    views = [view(spec, xy, features), view(spec, xy[::3], features[::3]), view(spec, [])]
    queries = np.stack([np.vstack([xy, xy + [.1, 0.], xy + [.1, .1], [[5., 5.]]])] * 3)
    tensor = torch.as_tensor(queries, device=device, dtype=dtype)
    first = check_reference(LocalTerrainBatch(views, device=device, chunk=17), views, tensor)
    order = [2, 0, 1]
    reordered = [views[i] for i in order]
    second = check_reference(LocalTerrainBatch(reordered, device=device, chunk=3), reordered, tensor[order])
    np.testing.assert_array_equal(second[0], first[0][order])


def test_support_boundary_tie_across_original_two_cell_gap(spec, device):
    terrain = view(spec, [[.31 - 5e-9, 0.], [.31 + 1e-9, 0.]], [[0., 0., 0., 1., 1.], [.2, .1, 0., 2., .1]])
    queries = torch.tensor([[[.155 - 1e-9, 0.]]], device=device, dtype=torch.float64)
    batch = LocalTerrainBatch([terrain], device=device)
    features, mask = check_reference(batch, [terrain], queries)
    assert mask.all() and features[0, 0, 4] == .1
    assert batch.ambiguous_queries == 1 and batch.cpu_tie_query_s > 0


def test_streaming_near_ties_do_not_chain_risk_beyond_final_minimum(spec, device):
    # One occupant per panel: d0~d1 and d1~d2, but d0 is not a tie of d2.
    terrain = view(spec, [[.1 + 1.5e-8, 0.], [.1 + .75e-8, 0.], [.1, 0.]],
                   [[10., 2., 0., 1., .01], [0., .1, 0., 2., .9], [0., 0., 0., 3., 1.]])
    batch = LocalTerrainBatch([terrain], device=device, memory_budget=4096)
    feature, mask = check_reference(batch, [terrain], torch.zeros((1, 1, 2), device=device, dtype=torch.float64))
    assert mask.all() and feature[0, 0, 0] == 0. and feature[0, 0, 4] == .9


def test_equal_magnitude_slopes_and_source_channel_use_cpu_tie_order(spec, device):
    xy = [[.1, 0.], [-.1, 0.], [0., .1], [0., -.1]]
    feature = [[0., .3, .4, 5., .9], [0., -.3, .4, 6., .8], [0., .4, .3, 7., .7], [0., -.4, -.3, 8., .6]]
    terrain = view(spec, xy, feature)
    batch = LocalTerrainBatch([terrain], device=device, memory_budget=4096)
    check_reference(batch, [terrain], torch.zeros((1, 1, 2), device=device, dtype=torch.float64))
    assert batch.ambiguous_queries == 1


def test_known_tolerance_edge_and_huge_finite_queries_do_not_wrap(spec, device):
    terrain = view(spec, [[0., 0.]])
    distances = [.155 - 1e-10, .155, .155 + .5e-10, .155 + 2e-10, 1e300, -1e300]
    queries = torch.tensor([[[x, 0.] for x in distances]], device=device, dtype=torch.float64)
    feature, known = LocalTerrainBatch([terrain], device=device).lookup(queries)
    np.testing.assert_array_equal(known.cpu().numpy(), [[True, True, True, False, False, False]])
    assert torch.isfinite(feature).all() and (feature[0, 3:] == 0).all()


def test_hash_multiplication_does_not_wrap_for_far_opposite_queries(spec, device):
    terrain = view(spec, [[1e8, 0.], [1e8, 1e8]])
    queries = torch.tensor([[[1e8, 0.], [-1e11, -1e11]]], device=device, dtype=torch.float64)
    features, known = LocalTerrainBatch([terrain], device=device).lookup(queries)
    assert known.tolist() == [[True, False]] and (features[0, 1] == 0).all()


@pytest.mark.parametrize("empty_memory", [False, True])
def test_empty_queries_and_memories_have_defined_shapes(spec, device, empty_memory):
    terrain = view(spec, [] if empty_memory else [[0., 0.]])
    batch = LocalTerrainBatch([terrain], device=device)
    features, known = batch.lookup(torch.empty((1, 0, 40, 2), device=device))
    assert features.shape == (1, 0, 40, 5) and known.shape == (1, 0, 40)
    if empty_memory:
        features, known = batch.lookup(torch.zeros((1, 2), device=device))
        assert features.shape == (1, 5) and not known.any() and not features.any()


def test_dense_coincident_buckets_respect_pair_memory_budget(spec, device, monkeypatch):
    count = 4096
    features = np.zeros((count, 5)); features[:, 4] = 1.
    features[-1] = [.15, .1, .1, 1., .2]
    terrain = view(spec, np.zeros((count, 2)), features)
    budget = 1024**2
    batch = LocalTerrainBatch([terrain], device=device, chunk=256, memory_budget=budget)
    def forbidden(*args, **kwargs):
        raise AssertionError("No query x complete-memory distance tensor is allowed")
    monkeypatch.setattr(torch, "cdist", forbidden)
    check_reference(batch, [terrain], torch.zeros((1, 3, 2), device=device, dtype=torch.float64))
    assert batch.max_pair_entries * PAIR_WORK_BYTES <= budget
    assert batch.max_pair_entries < 3 * 9 * count


def test_tiny_budget_also_chunks_queries(spec, device):
    terrain = view(spec, np.zeros((8, 2)))
    budget = 4096
    batch = LocalTerrainBatch([terrain], device=device, chunk=10000, memory_budget=budget)
    check_reference(batch, [terrain], torch.zeros((1, 12, 2), device=device, dtype=torch.float64))
    assert batch.max_pair_entries * PAIR_WORK_BYTES <= budget


@pytest.mark.parametrize("bad", ["nan", "shape", "dtype", "batch"])
def test_bad_tensor_queries_are_rejected(spec, device, bad):
    batch = LocalTerrainBatch([view(spec, [[0., 0.]])], device=device)
    query = torch.zeros((1, 2, 2), device=device)
    if bad == "nan":query[0, 0, 0] = float("nan")
    if bad == "shape":query = query[..., :1]
    if bad == "dtype":query = query.long()
    if bad == "batch":query = query.expand(2, 2, 2)
    with pytest.raises(ValueError, match="queries"):
        batch.lookup(query)


def test_explicit_cuda_alias_is_normalized_to_actual_device(spec):
    assert torch.cuda.is_available(), "Required real CUDA test"
    batch = LocalTerrainBatch([view(spec, [[0., 0.]])], device="cuda")
    assert batch.device == torch.device("cuda", torch.cuda.current_device())
    assert batch.lookup(torch.zeros((1, 1, 2), device="cuda:0"))[1].all()
    with pytest.raises(ValueError, match="queries"):
        batch.lookup(torch.zeros((1, 1, 2), device="cpu"))


def request(spec, peers=(), agent_id=0):
    return UpstreamPlannerInput(0, agent_id, np.zeros(3), np.zeros(2), UpstreamIntent([4., 0.], False, 0., 1.),
                                view(spec, [[0., 0.]]), tuple(peers), np.zeros(2), 0., np.arange(41) * .2,
                                (-100., -100., 100., 100.), spec)


def peer(spec, sender, x, radius=None):
    radius = 2 * spec.vehicle.effective_radius if radius is None else radius
    return FootprintNeighbor.for_spec(sender, [0., 8.], [[x, 0.], [x + .5, 0.]], [radius, radius + .1], spec)


def test_neighbor_sender_slots_sparse_batch_and_radius_match_numpy(spec, device):
    one, two = peer(spec, 1, 7.), peer(spec, 2, 9., 8.)
    inputs = [request(spec, [one, two]), request(spec, [two]), request(spec)]
    batch = LocalNeighborBatch(inputs, device=device, dtype=torch.float32)
    assert batch.senders.cpu().tolist() == [[1, 2], [2, -1], [-1, -1]]
    np.testing.assert_allclose(batch.radii[0, :, 0].cpu(), [one.at(t)[1] for t in inputs[0].times], atol=0, rtol=0)
    states = torch.zeros((3, 2, 41, 3), device=device, dtype=torch.float64)
    states[:, 1, :, 0] = 1.
    risk, safe = batch.risk_and_safe(states)
    for b, inp in enumerate(inputs):
        for k in range(2):
            total, allowed = 0., True
            for t, now in enumerate(inp.times):
                for neighbor in inp.neighbors:
                    xy, radius = neighbor.at(now)
                    distance = np.linalg.norm(np.array([k, 0.]) - xy)
                    total += 8 * max(0., (radius + .2 - distance) / .2)**2
                    allowed &= distance >= radius
            assert risk[b, k].item() == pytest.approx(total, rel=1e-12, abs=1e-12)
            assert safe[b, k].item() == allowed
    swapped = LocalNeighborBatch([request(spec, [two, one]), inputs[1], inputs[2]], device=device)
    new_risk, new_safe = swapped.risk_and_safe(states)
    torch.testing.assert_close(new_risk, risk, rtol=1e-12, atol=1e-12)
    assert torch.equal(new_safe, safe)


def test_neighbor_radius_is_not_rounded_down_in_float32_mode(spec, device):
    radius = 7. + 1e-8
    neighbor = peer(spec, 1, 7., radius)
    batch = LocalNeighborBatch([request(spec, [neighbor])], device=device, dtype=torch.float32)
    _, safe = batch.risk_and_safe(torch.zeros((1, 1, 41, 3), device=device, dtype=torch.float32))
    assert not safe.item() and batch.radii.dtype == torch.float64


def test_no_neighbor_zero_risk_and_empty_candidate_batch(spec, device):
    batch = LocalNeighborBatch([request(spec)], device=device)
    risk, safe = batch.risk_and_safe(torch.zeros((1, 0, 41, 3), device=device))
    assert risk.shape == safe.shape == (1, 0)
    risk, safe = batch.risk_and_safe(torch.zeros((1, 3, 41, 3), device=device))
    assert not risk.any() and safe.all()


def test_neighbor_rejects_forged_identity_and_bad_states(spec, device):
    inp = request(spec, [peer(spec, 1, 7.)])
    with pytest.raises(ValueError):
        replace(inp, neighbors=(inp.neighbors[0], inp.neighbors[0]))
    with pytest.raises(ValueError):
        replace(inp, neighbors=(replace(inp.neighbors[0], profile_fingerprint="other"),))
    with pytest.raises(ValueError, match="nonempty"):
        LocalNeighborBatch([], device=device)
    batch = LocalNeighborBatch([inp], device=device)
    for states in (torch.zeros((1, 41, 3), device=device), torch.zeros((1, 2, 16, 3), device=device),
                   torch.full((1, 2, 41, 3), float("inf"), device=device)):
        with pytest.raises(ValueError, match="states"):
            batch.risk_and_safe(states)


def test_persistent_rows_and_selection_preserve_source_order_with_empty_row(spec,device):
    rng=np.random.default_rng(191)
    points=rng.uniform(-2,2,(100,2))
    features=rng.uniform(-1,1,(100,5))
    views=[view(spec,points,features),view(spec,[]),view(spec,points+[5,-3],features)]
    rows=[LocalTerrainBatch([v],device=device) for v in views]
    packed=LocalTerrainBatch.from_rows(views,rows)
    original=LocalTerrainBatch(views,device=device)
    for key in ('keys','first','counts','xy','features'):
        assert torch.equal(getattr(packed,key),getattr(original,key)),key
    order=[2,1,0]
    queries=torch.tensor(np.stack([points+[5,-3],points,points]),device=device)
    check_reference(packed.select(order),[views[i] for i in order],queries)
    assert packed.select([0]).xy.data_ptr()==packed.xy.data_ptr()
    empty=packed.select([1])
    assert empty.empty
    check_reference(empty,[views[1]],queries[1:2])
