"""Explicit new execution gate, lazy backends and real grouped MPPI lifecycle."""

import builtins
from copy import deepcopy
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from _common import cfg_from_experiment
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_factory import (
    EXECUTION_VERSION, UpstreamPlannerPool, create_upstream_adapter, plan_batch,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_baselines import UpstreamPrimitiveInput
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_mppi import UpstreamMPPISettings
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_spec import parse_upstream_experiment
from test_upstream_baselines import request, terrain as terrain_fixture


@pytest.fixture(scope="module")
def cfg():
    return replace(cfg_from_experiment("configs/experiment/upstream_scale_s02.yaml"),
                   simulated_latency=0., mppi_samples=2, mppi_iterations=1)


@pytest.fixture(scope="module")
def terrain(cfg):
    return terrain_fixture.__wrapped__(cfg)


@pytest.fixture(scope="module", autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def inputs(cfg, terrain, identities=((0, 0), (1, 0))):
    base = request(cfg, terrain, cap=.3)
    return [replace(base, env_id=e, agent_id=a) for e, a in identities]


def test_s02_is_explicit_and_old_metadata_preserves_absent_execution_key():
    for file in ("upstream_scale_s00.yaml", "upstream_scale_s01.yaml", "upstream_scale_s01_review_v2.yaml"):
        old = cfg_from_experiment("configs/experiment/" + file)
        assert old.planner_execution_version == "not_connected"
        assert "planner_execution_version" not in old.metadata
        with pytest.raises(ValueError, match="explicit"):
            create_upstream_adapter(old)
    new = cfg_from_experiment("configs/experiment/upstream_scale_s02.yaml")
    assert new.planner_execution_version == EXECUTION_VERSION
    assert new.metadata["planner_execution_version"] == EXECUTION_VERSION
    assert new.mppi_samples == 512 and new.mppi_iterations == 2 and new.simulated_latency == .5
    assert new.training_paused
    disconnected = replace(new, planner_execution_version="not_connected")
    assert disconnected.metadata["solver_cache_key"] != new.metadata["solver_cache_key"]


def test_unknown_execution_version_is_rejected_by_parser(cfg):
    data = json.loads(cfg.configuration_json)
    data["upstream_scale"]["planner_execution_version"] = "upstream_planning_v99"
    with pytest.raises(ValueError, match="execution version"):
        parse_upstream_experiment(data, Path.cwd())


@pytest.mark.parametrize("override,reason", [
    ({"clock_mode": "realtime_measured_latency"}, "realtime"),
    ({"execution_review_version": "footprint_review_v1"}, "strict"),
    ({"terrain_query_version": "scalar_ties_v1"}, "strict"),
    ({"training_paused": False}, "pause"),
    ({"planner_adapter": "line_upstream_v1", "planner_device": "cuda"}, "CUDA"),
])
def test_factory_rejects_unsupported_execution_before_backend_construction(cfg, override, reason):
    with pytest.raises(ValueError, match=reason):
        create_upstream_adapter(replace(cfg, **override))


@pytest.mark.parametrize("adapter_id", ["line_upstream_v1", "quintic_upstream_v1",
                                      "primitive47_upstream_v1", "mppi_upstream_v1"])
def test_non_nmpc_backends_do_not_import_optional_solver(cfg, monkeypatch, adapter_id):
    original = builtins.__import__
    def reject_acados(name, *args, **kwargs):
        if "acados" in name or "upstream_nmpc" in name:
            raise AssertionError("Non-NMPC imported optional solver")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", reject_acados)
    adapter = create_upstream_adapter(replace(cfg, planner_adapter=adapter_id))
    assert adapter.adapter_id == adapter_id


def test_kernel_sharing_obeys_precision_sampling_settings_and_not_rng(cfg):
    cache = {}
    a = create_upstream_adapter(cfg, kernel_cache=cache)
    b = create_upstream_adapter(cfg, kernel_cache=cache)
    different = UpstreamMPPISettings(samples=2, iterations=1, dtype="float64")
    c = create_upstream_adapter(cfg, kernel_cache=cache, mppi_settings=different)
    assert a.batch_kernel is b.batch_kernel and a.generator is not b.generator
    assert c.batch_kernel is not a.batch_kernel and len(cache) == 2
    with pytest.raises(ValueError, match="configuration"):
        create_upstream_adapter(cfg, mppi_settings=replace(different, samples=4))


def test_unavailable_cuda_raises_without_fallback(cfg, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="no CPU fallback"):
        create_upstream_adapter(replace(cfg, planner_device="cuda"))


def test_empty_and_invalid_batches_do_not_bind_or_partly_advance(cfg, terrain):
    pool = UpstreamPlannerPool(cfg)
    assert pool.plan_batch([]) == [] and not pool._planners and not pool._kernels
    batch = inputs(cfg, terrain)
    with pytest.raises(ValueError, match="Duplicate"):
        pool.plan_batch([batch[0], batch[0]])
    assert not pool._planners
    a = create_upstream_adapter(cfg)
    b = create_upstream_adapter(replace(cfg, spec=replace(cfg.spec, rho_max=3.)))
    with pytest.raises(ValueError):
        plan_batch([a, b], batch)
    assert a.state_dict()["identity"] is None and b.state_dict()["identity"] is None


def test_mixed_baselines_native_and_mppi_preserve_input_order(cfg, terrain):
    ids = ("mppi_upstream_v1", "line_upstream_v1", "primitive47_upstream_v1", "quintic_upstream_v1")
    planners = [create_upstream_adapter(replace(cfg, planner_adapter=name)) for name in ids]
    batch = inputs(cfg, terrain, ((0, 0), (0, 1), (0, 2), (0, 3)))
    batch[2] = UpstreamPrimitiveInput.for_context(batch[2], 43)
    outputs = plan_batch(planners, batch)
    assert [o.diagnostics["adapter_id"] for o in outputs] == list(ids)
    assert all(o.valid for o in outputs)
    assert outputs[2].diagnostics["primitive_id"] == 43
    assert planners[0].batch_kernel.batch_calls == 1


def test_two_environment_eight_car_batch_is_one_real_kernel_call(cfg, terrain):
    pool = UpstreamPlannerPool(cfg)
    batch = inputs(cfg, terrain, tuple((e, a) for e in range(2) for a in range(4)))
    outputs = pool.plan_batch(batch)
    assert len(outputs) == 8 and all(o.valid for o in outputs)
    assert {o.diagnostics["batch_size"] for o in outputs} == {8}
    kernels = {id(p.batch_kernel): p.batch_kernel for p in pool._planners.values()}
    assert len(kernels) == 1 and next(iter(kernels.values())).batch_calls == 1
    assert len({id(p.generator) for p in pool._planners.values()}) == 8
    assert {p.state_dict()["identity"] for p in pool._planners.values()} == {(e, a) for e in range(2) for a in range(4)}


def test_restore_permutation_and_async_reset_preserve_other_environment(cfg, terrain):
    pool = UpstreamPlannerPool(cfg)
    batch = inputs(cfg, terrain)
    pool.plan_batch(batch)
    snapshot = pool.state_dict()
    restored = UpstreamPlannerPool(cfg); restored.load_state_dict(snapshot)
    left = pool.plan_batch(batch)
    right = restored.plan_batch(batch[::-1])[::-1]
    for a, b in zip(left, right, strict=True):
        np.testing.assert_allclose(a.trajectory.controls, b.trajectory.controls, atol=1e-5, rtol=1e-5)
    state_before = pool._planners[(1, 0)].state_dict()
    pool.reset(0)
    assert pool._planners[(0, 0)].state_dict()["identity"] is None
    state_after = pool._planners[(1, 0)].state_dict()
    torch.testing.assert_close(state_before["adapter_state"]["generator"], state_after["adapter_state"]["generator"], rtol=0, atol=0)
    np.testing.assert_array_equal(state_before["adapter_state"]["warm"], state_after["adapter_state"]["warm"])
    restored_after_reset = UpstreamPlannerPool(cfg); restored_after_reset.load_state_dict(pool.state_dict())
    assert restored_after_reset._planners[(0, 0)].state_dict()["identity"] is None


def test_invalid_late_restore_entry_does_not_replace_live_pool(cfg, terrain):
    pool = UpstreamPlannerPool(replace(cfg, planner_adapter="line_upstream_v1"))
    batch = inputs(cfg, terrain)
    pool.plan_batch(batch)
    previous = dict(pool._planners)
    bad = deepcopy(pool.state_dict())
    bad["planners"][-1]["state"]["adapter_id"] = "old_planner"
    with pytest.raises(ValueError, match="adapter"):
        pool.load_state_dict(bad)
    assert pool._planners == previous
    wrong = deepcopy(pool.state_dict()); wrong["contracts"].pop("planner_execution_version")
    with pytest.raises(ValueError, match="mismatch"):
        pool.load_state_dict(wrong)


def test_public_sample_export_is_detached_and_reports_only_recorded_iteration(cfg,terrain):
    pool=UpstreamPlannerPool(cfg)
    assert pool.export_diagnostics()['planners']==[]
    pool.plan_batch(inputs(cfg,terrain))
    records=pool.export_diagnostics()
    assert records['diagnostic_version']=='upstream_pool_diagnostics_v1'
    assert [r['identity'] for r in records['planners']]==[(0,0),(1,0)]
    samples=records['planners'][0]['samples']
    assert samples['raw_controls'].shape==(2,40,2) and samples['raw_controls'].device.type=='cpu'
    assert samples['total_iterations']==1 and samples['recorded_iteration']==0
    original=samples['raw_controls'].clone();samples['raw_controls'].fill_(900.)
    torch.testing.assert_close(pool.export_diagnostics()['planners'][0]['samples']['raw_controls'],original)
    state=pool.state_dict()
    assert 'samples' not in state['planners'][0]['state']
    pool.reset(0);after=pool.export_diagnostics()['planners']
    assert after[0]['samples'] is None and after[1]['samples'] is not None


def test_native_payload_type_mismatch_rejected_before_planning(cfg, terrain):
    native = create_upstream_adapter(replace(cfg, planner_adapter="primitive47_upstream_v1"))
    with pytest.raises(ValueError, match="payload"):
        plan_batch([native], inputs(cfg, terrain, ((0, 0),)))
    assert native.state_dict()["identity"] is None


def test_fresh_process_moves_with_acados_imports_actively_disabled():
    """Import blockers run before loading the package, rather than inspecting modules."""
    root = Path(__file__).resolve().parents[1]
    script = textwrap.dedent("""
        import builtins
        import importlib.abc
        import json
        import sys
        attempts = []
        original_import = builtins.__import__
        def guarded_import(name, *args, **kwargs):
            if 'acados' in name.lower():
                attempts.append(name)
                raise ImportError('acados deliberately unavailable in isolated process')
            return original_import(name, *args, **kwargs)
        class NoAcados(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if 'acados' in fullname.lower():
                    attempts.append(fullname)
                    raise ImportError('acados deliberately unavailable in isolated process')
                return None
        builtins.__import__ = guarded_import
        sys.meta_path.insert(0, NoAcados())
        try:
            import acados_template
            raise AssertionError('Import blocker was ineffective')
        except ImportError:
            pass
        assert attempts == ['acados_template']
        from dataclasses import replace
        import numpy as np
        import torch
        from _common import cfg_from_experiment
        from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.protocol import TerrainSnapshot
        from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_footprint import FootprintTerrain
        from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_planning import UpstreamIntent, UpstreamPlannerInput
        from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_factory import create_upstream_adapter
        torch.set_num_threads(1)
        old = cfg_from_experiment('configs/experiment/upstream_scale_s01_review_v2.yaml')
        try:
            create_upstream_adapter(old)
            raise AssertionError('Unconnected historical configuration was enabled')
        except ValueError as error:
            assert 'explicit upstream_planning_v1' in str(error)
        base = replace(cfg_from_experiment('configs/experiment/upstream_scale_s02.yaml'),
                       simulated_latency=0., mppi_samples=4)
        x, y = np.meshgrid(np.arange(-15., 15.01, .2), np.arange(-8., 8.01, .2), indexing='ij')
        xy = np.c_[x.ravel(), y.ravel()]
        features = np.zeros((len(xy), 5)); features[:, 4] = 1.
        terrain = FootprintTerrain(TerrainSnapshot(xy, features, np.zeros(len(xy)),
                                  base.spec.perception.support_radius), base.spec)
        request = UpstreamPlannerInput(0, 0, np.zeros(3), np.zeros(2),
            UpstreamIntent([4., 0.], False, 0., 1., speed_cap=.4), terrain, (), np.zeros(2),
            0., base.spec.grid.times(0.), (-100., -100., 100., 100.), base.spec)
        results = []
        for backend in ('line_upstream_v1', 'mppi_upstream_v1'):
            adapter = create_upstream_adapter(replace(base, planner_adapter=backend))
            output = adapter.plan(request)
            assert output.valid, output.fallback_reason
            assert output.trajectory.states[-1, 0] > .2
            assert output.diagnostics['information_reserve_verified']
            assert adapter.state_dict()['contracts']['planner_execution_version'] == 'upstream_planning_v1'
            results.append({'adapter':backend, 'valid':output.valid,
                            'progress_m':float(output.trajectory.states[-1, 0])})
        assert attempts == ['acados_template'], attempts
        print(json.dumps({'blocked_imports':attempts, 'results':results}))
    """)
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join((str(root / "scripts"), str(root / "source/lunar_rover_tasks")))
    result = subprocess.run([sys.executable, "-c", script], cwd=root, env=env,
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    evidence = json.loads(result.stdout.splitlines()[-1])
    assert evidence["blocked_imports"] == ["acados_template"]
    assert {r["adapter"] for r in evidence["results"]} == {"line_upstream_v1", "mppi_upstream_v1"}
    assert all(r["valid"] and r["progress_m"] > .2 for r in evidence["results"])
