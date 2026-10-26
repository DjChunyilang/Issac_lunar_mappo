"""Animation replay uses executed poses and verifies the original evaluation."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from upstream_decision_evaluation import EvaluationPoseCapture, _compare_evaluation_replay


def test_pose_capture_preserves_initial_terminal_and_asynchronous_masks():
    initial = np.zeros((2, 4, 3))
    capture = EvaluationPoseCapture(initial, .2)
    first = initial.copy()
    first[..., 0] = 1.
    capture({'active_envs': np.array([True, True]), 'before': initial, 'after': first})
    second = first.copy()
    second[0, :, 0] = 2.
    capture({'active_envs': np.array([True, False]), 'before': first, 'after': second})
    # Later mutations must not alter the recorded t=0 or terminal state.
    initial[:] = -10.
    first[:] = -20.
    second[:] = -30.
    histories = capture.arrays()
    assert [len(history) for history in histories] == [3, 2]
    np.testing.assert_array_equal(histories[0][:, 0, 0], [0., 1., 2.])
    np.testing.assert_array_equal(histories[1][:, 0, 0], [0., 1.])
    assert capture.dt == .2


def test_pose_capture_rejects_missing_step_or_reset():
    capture = EvaluationPoseCapture(np.zeros((1, 4, 3)), .2)
    with pytest.raises(ValueError, match='lost an executed pose'):
        capture({'active_envs': np.array([True]), 'before': np.ones((1, 4, 3)),
                 'after': np.ones((1, 4, 3))*2})


def test_replay_comparison_checks_discrete_endings_and_numeric_trace():
    reference = {'episodes': [{'env_id': 0, 'steps': 2, 'terminal_reason': 'terrain_failure'}],
                 'trace': [{'geometry': {'dmax': [50.]}, 'reward': [-2.], 'terminated': [True]}]}
    replay = {'episodes': [{'env_id': 0, 'steps': 2, 'terminal_reason': 'terrain_failure'}],
              'trace': [{'geometry': {'dmax': np.array([50.+1e-8])},
                         'reward': np.array([-2.]), 'terminated': np.array([True])}]}
    assert _compare_evaluation_replay(reference, replay)['matched']
    replay['episodes'][0]['steps'] = 3
    replay['trace'][0]['reward'][0] = -3.
    report = _compare_evaluation_replay(reference, replay)
    assert not report['matched']
    assert {row['field'] for row in report['mismatches']} == {
        'evaluation.episodes[0].steps', 'evaluation.trace[0].reward'}
