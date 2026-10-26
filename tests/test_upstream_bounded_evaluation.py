"""Bounded evaluation reuses fixed panels without spending a full-run budget."""
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import upstream_curriculum_experiment as runner
import upstream_decision_evaluation as evaluation
from training_tensorboard import TrainingTensorBoard


def configuration(tmp_path):
    data = yaml.safe_load(Path('configs/training/upstream_1km_far_obstacle_input_v1.yaml').read_text())
    data.update(evaluation_mode='bounded_fixed_window_v1', evaluation_window_seconds=60.,
                final_evaluation_reservation=2400, evaluation_max_base_steps=4800)
    path = tmp_path / 'bounded.yaml'; path.write_text(yaml.safe_dump(data))
    return runner.load_curriculum(path)


def test_short_windows_preserve_training_deadline_and_legacy_budget(tmp_path):
    short = configuration(tmp_path)
    assert short['stages'][0]['deadline_s'] == 1500.
    assert runner.evaluation_panels(short, 0, 'initial') == [('far', 0, 60., False)]
    assert runner.evaluation_panels(short, 0, 'final', final=True) == [('far', 0, 60., False)]
    assert runner.evaluation_reservation(runner.evaluation_panels(short, 0, 'initial')) == 2400
    assert runner.evaluation_budget_upper_bound(short, 50000) == 4800
    assert runner.evaluation_budget_upper_bound(short, 2000000) == 4800
    assert not runner.convergence_report(tmp_path, short)['stable_convergence_screen_passed']
    old = runner.load_curriculum('configs/training/upstream_1km_far_obstacle_input_v1.yaml')
    assert runner.evaluation_panels(old, 0, 'initial') == [('far', 0, 1500., True)]
    assert runner.evaluation_panels(old, 0, 'final', final=True) == [('far', 0, 1500., True), ('unseen', 0, 1500., True)]
    assert runner.evaluation_budget_upper_bound(old, 2000000) == 360000


def test_periodic_continuation_reserves_all_trends_and_full_final(tmp_path):
    config = runner.load_curriculum('configs/training/upstream_1km_far_crater_reward_1m_v1.yaml')
    assert runner.periodic_window_evaluation(config)
    assert not runner.bounded_evaluation(config)  # Scheduled evaluations remain enabled.
    assert runner.evaluation_panels(config, 0, 'scheduled') == [('far', 0, 60., False)]
    final = runner.evaluation_panels(config, 0, 'final', final=True)
    assert final == [('far', 0, 60., False), ('full_far', 0, 1500., True)]
    assert runner.evaluation_reservation(final) == 62400
    assert runner.evaluation_budget_upper_bound(config, 1049871) == 88800
    assert not runner.convergence_report(tmp_path, config)['stable_convergence_screen_passed']
    old = runner.load_curriculum('configs/training/upstream_1km_far_crater_reward_v1.yaml')
    allowed = {'evaluation_mode', 'evaluation_every_base_steps', 'full_evaluation_at_base_steps',
               'evaluation_max_base_steps', 'final_evaluation_reservation'}
    assert {k for k in old if old[k] != config[k]} == allowed


def test_periodic_continuation_rejects_insufficient_final_reservation(tmp_path):
    data = runner.load_curriculum('configs/training/upstream_1km_far_crater_reward_1m_v1.yaml')
    data['final_evaluation_reservation'] = 2400
    path = tmp_path / 'invalid_periodic.yaml'; path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match='Invalid authorized fixed'):
        runner.load_curriculum(path)


def test_window_needs_explicit_mode_and_complete_policy_intervals(tmp_path):
    data = configuration(tmp_path)
    path = tmp_path / 'invalid.yaml'
    bad = deepcopy(data); del bad['evaluation_mode']; path.write_text(yaml.safe_dump(bad))
    with pytest.raises(ValueError, match='explicit bounded mode'):
        runner.load_curriculum(path)
    bad = deepcopy(data); bad['evaluation_window_seconds'] = 61.; path.write_text(yaml.safe_dump(bad))
    with pytest.raises(ValueError, match='bounded evaluation window'):
        runner.load_curriculum(path)


def test_censored_outcomes_and_actual_speed_use_each_cases_active_time():
    # Case 0 fails after one second while cases 1/2 run to the 60-second cut.
    # Frozen control values from case 0 never enter this distance/time measure.
    metrics = evaluation.bounded_outcome_metrics([5, 300, 300, 10], [True, False, False, True],
        [{'env_id': 0, 'terminal_reason': 'terrain_failure'}, {'env_id': 3, 'terminal_reason': 'success'}],
        np.repeat(np.array([.5, 30., 0., 1.])[:, None], 4, axis=1), .2)
    np.testing.assert_array_equal(metrics['observed_seconds'], [1., 60., 60., 2.])
    np.testing.assert_allclose(metrics['mean_actual_speed'], [.5, .5, 0., .5])
    assert metrics['mean_observed_seconds'] == 30.75
    assert metrics['censored'].tolist() == [False, True, True, False]
    assert metrics['censored_count'] == 2 and metrics['window_failure_fraction'] == .25
    assert metrics['window_failure_reasons'] == {'terrain_failure': 1}


def test_evaluator_accepts_live_window_ends_and_reserves_final_once(tmp_path, monkeypatch):
    import run_upstream_t11
    config = configuration(tmp_path)
    (tmp_path / 'metrics').mkdir()
    calls = []

    def make_collector(base, representation, seed, **kwargs):
        calls.append(('collector', seed, kwargs['num_envs'], kwargs['episode_seconds'], kwargs['curriculum_config']['scene_role']))
        spawn = SimpleNamespace(set_level=lambda level: None,
                                sample=lambda ids, replay: np.zeros((4, 4, 3)), deadlines=np.full(4, 1500.))
        env = SimpleNamespace(curriculum=spawn, task=SimpleNamespace(deadlines=np.zeros(4)),
                              reset=lambda ids, poses: None)
        return SimpleNamespace(env=env)

    def panel(collector, **kwargs):
        calls.append(('panel', kwargs['deterministic'], kwargs['seconds'], kwargs['bounded_metrics']))
        # This fixture performs zero environment steps. It models one ended
        # case and three live cases to exercise the evaluator's cut contract.
        result = evaluation.bounded_outcome_metrics([5, 300, 300, 300], [True, False, False, False],
            [{'env_id': 0, 'terminal_reason': 'terrain_failure', 'steps': 5}], np.zeros((4, 4)), .2)
        return {**result, 'initial': {'dmax': [400.] * 4}, 'dmax_reduction': [0.] * 4,
                'successes': 0, 'episodes': [{'terminal_reason': 'terrain_failure'}],
                'actual_base_steps': 905, 'trace': []}

    monkeypatch.setattr(run_upstream_t11, 'make_collector', make_collector)
    monkeypatch.setattr(evaluation, 'panel', panel)
    collector = SimpleNamespace(curriculum_progress={'level': 0}, policy=torch.nn.Linear(1, 1),
                                base_steps=0, learning_time=runner.timing_for(config))
    args = SimpleNamespace(config='configs/experiment/upstream_1km_crater_classes_v1.yaml')
    initial = runner.evaluate(tmp_path, 'initial', collector, args, config)
    collector.base_steps = 50000
    final = runner.evaluate(tmp_path, 'final', collector, args, config, final=True)
    for result in (initial, final):
        assert result['reserved_base_steps'] == 2400 and result['actual_base_steps'] == 1810
        assert not result['full_episode_evaluation']
        assert set(result['results']) == {'far_sample', 'far_mean'}
        assert all(row['censored_count'] == 3 for row in result['results'].values())
    assert [call[1] for call in calls if call[0] == 'collector'] == [104729] * 4
    assert all(call[2:] == (4, 1500., 'fixed_eval') for call in calls if call[0] == 'collector')
    before = len(calls)
    assert runner.evaluate(tmp_path, 'final', collector, args, config, final=True) == final
    assert len(calls) == before
    reserved = sum(json.loads(p.read_text())['reserved_base_steps'] for p in (tmp_path / 'metrics').glob('*_reservation.json'))
    assert reserved == 4800


def test_short_window_pins_do_not_claim_full_success(tmp_path):
    row = {'training_base_steps': 50000, 'actual_base_steps': 1810, 'reserved_base_steps': 2400,
           'results': {f'far_{mode}': {'full_episode_evaluation': False, 'initial': {'dmax': [400.] * 4},
               'successes': 0, 'mean_observed_seconds': 45.25, 'window_failure_fraction': .25,
               'dmax_reduction': [2., 4., 6., 8.]} for mode in ['sample', 'mean']}}
    with TrainingTensorBoard(tmp_path, pinned='bounded_far') as log:
        log.evaluation(row, 'evaluation_final')
    events = EventAccumulator(str(tmp_path / 'tensorboard')).Reload()
    assert events.Scalars('Main/ObservedTime_S')[-1].value == 45.25
    assert events.Scalars('Main/WindowFailure_M')[-1].value == .25
    assert events.Scalars('Main/Dmax_S')[-1].value == 5.
    assert not any(tag.startswith('eval/full/') or tag in ('Main/SR_S', 'Main/SR_M') for tag in events.Tags()['scalars'])
