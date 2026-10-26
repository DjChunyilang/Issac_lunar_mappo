"""Read actual event files: physical steps, evaluation denominators and replay."""
import json
from pathlib import Path
import random
import sys

import numpy as np
import pytest
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from training_tensorboard import TrainingTensorBoard


def write(run, name, row):
    path = run / 'metrics' / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row))


def update(step, episodes=()):
    return {'actual_base_steps': step, 'sampling_updates': step // 400, 'episodes': list(episodes),
            'learning': {'epochs': [{'actor_loss': 1., 'entropy': None}, {'actor_loss': 3., 'entropy': 2.}],
                         'diagnostics': {'explained_variance_before_update': None}}}


def events(run):
    return EventAccumulator(str(run / 'tensorboard'), size_guidance={'scalars': 0}).Reload()


def test_live_flush_and_epoch_aggregation(tmp_path):
    write(tmp_path, 'update_000001.json', update(400))
    with TrainingTensorBoard(tmp_path) as log:
        ea = events(tmp_path)  # Must be readable while the writer is still open.
        assert [(e.step, e.value) for e in ea.Scalars('ppo/epoch_mean/actor_loss')] == [(400, 2.)]
        assert ea.Scalars('ppo/epoch_last/actor_loss')[0].value == 3.
        assert 'episode/raw_return_mean' not in ea.Tags()['scalars']
        assert 'ppo/diagnostics/explained_variance_before_update' not in ea.Tags()['scalars']
        log.sync()
        assert len(events(tmp_path).Scalars('ppo/epoch_mean/actor_loss')) == 1


def test_episode_and_evaluation_are_separate_with_all_cases_denominator(tmp_path):
    write(tmp_path, 'update_000001.json', update(400, [
        {'terminal_reason': 'success', 'raw_return': 10., 'reward_components': {'energy': -2}},
        {'terminal_reason': 'task_deadline', 'raw_return': -6., 'reward_components': {'energy': -4}}]))
    write(tmp_path, 'evaluation_final.json', {'training_base_steps': 400, 'actual_base_steps': 375,
          'results': {'parking_sample': {'successes': 1, 'initial': {'dmax': [10., 10., 10., 10.]},
                                       'episodes': [{'terminal_reason': 'success'}],
                                       'dmax_reduction': [1., 2., 3., 4.], 'full_episode_evaluation': False}}})
    with TrainingTensorBoard(tmp_path):
        ea = events(tmp_path)
        assert ea.Scalars('episode/raw_return_mean')[0].value == 2.
        assert ea.Scalars('episode/reward_components/energy_mean')[0].value == -3.
        assert ea.Scalars('episode/termination_fraction/success')[0].value == .5
        point = ea.Scalars('eval/short/parking_sample/success_rate')[0]
        assert (point.step, point.value) == (400, .25)  # Not 1/1 terminated episodes.
        assert ea.Scalars('eval/short/parking_sample/dmax_reduction/mean')[0].value == 2.5
        assert ea.Scalars('evaluation_budget/actual_base_steps')[0].value == 375


def test_reopen_rebuilds_once_and_recovers_json_written_before_crash(tmp_path):
    write(tmp_path, 'update_000001.json', update(400))
    with TrainingTensorBoard(tmp_path):
        pass
    write(tmp_path, 'update_000002.json', update(800))
    with TrainingTensorBoard(tmp_path) as log:
        write(tmp_path, 'update_000003.json', update(1200))
        log.sync()
    ea = events(tmp_path)
    assert [e.step for e in ea.Scalars('ppo/epoch_mean/actor_loss')] == [400, 800, 1200]


def test_logging_preserves_random_streams(tmp_path):
    write(tmp_path, 'update_000001.json', update(400))
    torch_state, numpy_state, python_state = torch.get_rng_state(), np.random.get_state(), random.getstate()
    with TrainingTensorBoard(tmp_path) as log:
        log.sync()
    assert torch.equal(torch.get_rng_state(), torch_state)
    assert np.array_equal(np.random.get_state()[1], numpy_state[1])
    assert np.random.get_state()[2:] == numpy_state[2:]
    assert random.getstate() == python_state


def test_nonfinite_metric_is_visible_error(tmp_path):
    write(tmp_path, 'update_000001.json', {'actual_base_steps': 400, 'hold_fraction': float('nan')})
    with pytest.raises(ValueError, match='Nonfinite TensorBoard metric'):
        TrainingTensorBoard(tmp_path)


def test_pinned_aliases_preserve_values_and_native_link(tmp_path):
    from urllib.parse import parse_qs, urlsplit
    from training_tensorboard import PINNED_SCALARS, pinned_url

    with TrainingTensorBoard(tmp_path) as log:
        for index, source in enumerate(PINNED_SCALARS):
            log.scalars(source, index / 10, 400)
        log.writer.flush()
        ea = events(tmp_path)
        for source, alias in PINNED_SCALARS.items():
            original, pinned = ea.Scalars(source)[0], ea.Scalars(alias)[0]
            assert (pinned.step, pinned.value) == (original.step, original.value)
        assert len([t for t in ea.Tags()['scalars'] if t.startswith('Main/')]) == 10
    url = urlsplit(pinned_url())
    query = parse_qs(url.query)
    assert url.fragment == 'timeseries'
    assert query['tagFilter'] == ['^Main/']
    assert json.loads(query['pinnedCards'][0]) == [
        {'plugin': 'scalars', 'tag': tag} for tag in PINNED_SCALARS.values()]
    assert (tmp_path / 'tensorboard/pinned_url.txt').read_text().strip() == pinned_url()
