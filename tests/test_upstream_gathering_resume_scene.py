"""JSON scene replay accepts the original tuple caller, rejects changed seeds."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import evaluate_upstream_gathering as gathering


def test_real_npb_scene_is_prefetched_before_first_sensor_query():
    _, runtime, identities = gathering.terrain_for('npb', 23, 2)
    try:
        assert identities == ('npb_real_base_same_200m_development_window',)
        assert list(runtime.scene_bank.resident) == [0]
        query = runtime.scene_bank.query(torch.zeros((2, 4, 2)), runtime.scene_indices)
        assert query.metrics_valid.shape == (2, 4)
    finally:
        runtime.scene_bank.close()


def test_saved_scene_replays_original_seed_tuple_without_solver(tmp_path, monkeypatch):
    run = SimpleNamespace(path=tmp_path, config='configs/experiment/upstream_scale_s04_braking_lease_cuda_v5.yaml')
    (tmp_path / 'config').mkdir()

    class ReachedEnvironment(Exception):
        pass

    def stop_before_solver(*args, **kwargs):
        raise ReachedEnvironment

    monkeypatch.setattr(gathering, 'UpstreamEnvironment', stop_before_solver)
    # Real scenario/config generation on both calls, stopping before solver creation.
    for _ in range(2):
        with pytest.raises(ReachedEnvironment):
            gathering.execute_pair(run, 'compact', (23, 31), 'nmpc', True, 360)
    path = next((tmp_path / 'config').glob('*.json'))
    scene = json.loads(path.read_text())
    assert scene['layout_seeds'] == [23, 31]
    scene['layout_seeds'][1] = 47
    path.write_text(json.dumps(scene))
    with pytest.raises(ValueError, match='Frozen gathering scene/config changed'):
        gathering.execute_pair(run, 'compact', (23, 31), 'nmpc', True, 360)
