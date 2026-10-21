"""The continuation must cover the fixed matrix and never pass a partial gate."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from continue_upstream_gathering import jobs, summarize


def rows():
    return [{'family': f, 'seed': seed, 'variant': v, 'safety': a, 'deadline': d,
             'successful': True, 'violations': [], 'failure_category': 'success'}
            for variant in ('nmpc', 'mppi512') for f, seeds, v, a, d in jobs(variant) for seed in seeds]


def test_complete_unique_matrix_and_simple_first():
    data = rows()
    assert len(data) == 144
    for variant in ('nmpc', 'mppi512'):
        first = jobs(variant)[:6]
        assert sum(len(j[1]) for j in first) == 9
        assert all(j[0] in ('compact', 'irregular', 'reverse') and j[3:] == (True, 360) for j in first)
    assert summarize(data)['s05_gathering_passed']


def test_partial_failure_and_duplicate_cannot_pass():
    data = rows()
    assert not summarize(data[:4])['simple_gate_18_passed']
    assert not summarize(data[:-1])['s05_gathering_passed']
    data[0]['successful'] = False
    assert not summarize(data)['s05_gathering_passed']
    with pytest.raises(ValueError, match='Duplicate'):
        summarize(data + data[:1])


def test_complex_failure_reported_and_unverified_not_safety_certification():
    data = rows()
    complex_row = next(r for r in data if r['family'] == 'npb')
    complex_row.update(successful=False, failure_category='environment_invalid', counts={'unverified_execution': 7})
    result = summarize(data)
    assert result['s05_gathering_passed']
    assert result['failure_categories']['environment_invalid'] == 1
    assert result['unverified_execution'] == 7
    assert not result['formal_training_ready']
