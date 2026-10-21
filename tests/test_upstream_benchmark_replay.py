"""The benchmark restores cold/warm states without weakening vehicle isolation."""
from dataclasses import replace
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from benchmark_upstream_planners import replay_check
from upstream_local_closure import runtime_config
from upstream_s05_cases import formal_case
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_factory import create_upstream_adapter


@pytest.mark.parametrize('adapter_id', ['line_upstream_v1', 'mppi_upstream_v4', 'nmpc_upstream_v2'])
def test_cold_and_warm_replay_retains_vehicle_guard(adapter_id, tmp_path):
    torch.set_num_threads(1)
    cfg, _ = runtime_config('configs/experiment/upstream_scale_s05_benchmark_cpu_review_v2.yaml',
                            name='benchmark_restore_contract', seed=23, adapter=adapter_id, samples=16)
    case = formal_case(cfg, 'flat', 23, 0)
    adapter = create_upstream_adapter(cfg, cache_dir=tmp_path)
    comparison, evidence = replay_check(adapter, case.input)
    assert comparison['restore_equal']
    assert comparison['cold_restore']['equal'] and comparison['warm_restore']['equal']
    assert evidence['before']['identity'] is None
    assert evidence['warm']['identity'] == (case.input.env_id, case.input.agent_id)
    with pytest.raises(ValueError, match="another vehicle"):
        adapter.load_state_dict(evidence['before'])
    adapter.reset()
    other = replace(case.input, env_id=1,
                    actual_history=replace(case.input.actual_history, env_id=1))
    adapter.plan(other)
    with pytest.raises(ValueError, match="another vehicle"):
        adapter.load_state_dict(evidence['warm'])
