"""An invalid environment terminal is masked, not a corrupted entire rollout."""
from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from upstream_curriculum_experiment import learning_batch_problem
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.learning import duration_gae


def batch_with_invalid_terminal():
    valid = torch.tensor([[True, True], [False, True], [True, True]])
    ended = torch.tensor([[False, False], [True, False], [False, False]])
    return dict(obs=torch.zeros(3, 2, 4, 523), state=torch.zeros(3, 2, 1439),
        hidden=torch.zeros(3, 2, 4, 128), reward=torch.ones(3, 2), value=torch.zeros(3, 2),
        next_value=torch.zeros(3, 2), duration=torch.full((3, 2), 25),
        learning_valid=valid, terminated=ended, truncated=torch.zeros_like(ended),
        bootstrap_allowed=~ended, terminal_reason=torch.tensor([[0, 0], [5, 0], [0, 0]]),
        samples=[SimpleNamespace(log_prob=torch.zeros(2, 4)) for _ in range(3)])


def test_invalid_environment_terminal_is_excluded_and_breaks_credit_chain():
    b = batch_with_invalid_terminal()
    assert not b['learning_valid'][b['duration'] > 0].all()  # Reproduces the former rejection.
    assert learning_batch_problem(b) is None
    def gae(x):
        return duration_gae(x['reward'], x['value'], x['next_value'], x['duration'],
            x['terminated'], x['truncated'], bootstrap_allowed=x['bootstrap_allowed'],
            learning_valid=x['learning_valid'])
    a, r = gae(b)
    changed = deepcopy(b); changed['reward'][1, 0] = 1e9; changed['next_value'][1, 0] = 1e9
    other_a, other_r = gae(changed)
    torch.testing.assert_close(a, other_a, rtol=0, atol=0)
    torch.testing.assert_close(r, other_r, rtol=0, atol=0)
    assert a[1, 0] == r[1, 0] == 0 and a[0, 0] == 1


@pytest.mark.parametrize('field,value', [('terminal_reason', 6), ('terminated', False), ('bootstrap_allowed', True)])
def test_inconsistent_invalid_terminal_remains_fatal(field, value):
    b = batch_with_invalid_terminal(); b[field][1, 0] = value
    assert learning_batch_problem(b) == 'Invalid learning mask/terminal contract'


@pytest.mark.parametrize('field', ['obs', 'state', 'hidden', 'reward', 'value', 'next_value'])
def test_nonfinite_data_remains_fatal(field):
    b = batch_with_invalid_terminal(); b[field].reshape(-1)[0] = float('nan')
    assert learning_batch_problem(b) == 'Nonfinite learning data'


def test_nonfinite_action_probability_remains_fatal():
    b = batch_with_invalid_terminal(); b['samples'][0].log_prob[0, 0] = float('inf')
    assert learning_batch_problem(b) == 'Nonfinite learning data'
