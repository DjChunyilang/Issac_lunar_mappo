from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from analyze_repeated_conflict_outcomes import extract_conflict_outcomes  # noqa: E402


def test_repeated_event_matches_delayed_collision_without_crossing_reset() -> None:
    shape = (8, 1, 1)
    predicted = torch.zeros(shape, dtype=torch.bool)
    repeated = torch.zeros(shape, dtype=torch.bool)
    near = torch.zeros(shape, dtype=torch.bool)
    collision = torch.zeros(shape, dtype=torch.bool)
    done = torch.zeros((8, 1), dtype=torch.bool)
    predicted[0, 0, 0] = True
    predicted[2:4, 0, 0] = True
    repeated[3, 0, 0] = True
    near[3, 0, 0] = True
    collision[5, 0, 0] = True
    done[5, 0] = True

    result = extract_conflict_outcomes(
        predicted=predicted,
        pair_repeated=repeated,
        near_active=near,
        collision=collision,
        done=done,
        outcome_window_steps=4,
        collision_lookback_steps=8,
    )
    assert result["nonrepeated_events"] == 1
    assert result["repeated_events"] == 1
    assert result["nonrepeated_collision_outcome_rate"] == pytest.approx(0.0)
    assert result["repeated_collision_outcome_rate"] == pytest.approx(1.0)
    assert result["nonrepeated_collision_outcomes"] == 0
    assert result["repeated_collision_outcomes"] == 1
    assert result["repeated_vs_nonrepeated_outcome_ratio"] is None
    assert result["repeated_vs_nonrepeated_ratio_is_infinite"] is True
    assert result["repeated_near_coverage_rate"] == pytest.approx(1.0)
    assert result["collision_involvement_events"] == 1
    assert result["collision_any_conflict_recall"] == pytest.approx(1.0)
    assert result["collision_repeated_conflict_recall"] == pytest.approx(1.0)


def test_incomplete_tail_event_is_right_censored() -> None:
    predicted = torch.tensor([[[False]], [[True]], [[True]]])
    zeros = torch.zeros_like(predicted)
    result = extract_conflict_outcomes(
        predicted=predicted,
        pair_repeated=zeros,
        near_active=zeros,
        collision=zeros,
        done=torch.zeros((3, 1), dtype=torch.bool),
        outcome_window_steps=4,
        collision_lookback_steps=8,
    )
    assert result["nonrepeated_events"] == 0
    assert result["repeated_events"] == 0
