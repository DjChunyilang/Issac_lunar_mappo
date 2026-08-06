from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from analyze_near_credit_lead_time import AgentLeadTimeTracker  # noqa: E402


def test_tracker_records_one_step_collision_lead_and_conflict_delay() -> None:
    tracker = AgentLeadTimeTracker(1, 2, "cpu")
    tracker.update(
        near_before=torch.tensor([[1.0, 1.0]]),
        near_after=torch.tensor([[0.6, 0.6]]),
        predicted_conflict_involvement=torch.tensor([[True, True]]),
        collision_involvement=torch.tensor([[False, False]]),
        done=torch.tensor([False]),
        near_distance=0.72,
    )
    tracker.update(
        near_before=torch.tensor([[0.6, 0.6]]),
        near_after=torch.tensor([[0.2, 0.2]]),
        predicted_conflict_involvement=torch.tensor([[True, True]]),
        collision_involvement=torch.tensor([[True, True]]),
        done=torch.tensor([True]),
        near_distance=0.72,
    )
    summary = tracker.summary()
    assert summary["collision_involvement_events"] == 2
    assert summary["collision_prior_near_fraction"] == pytest.approx(1.0)
    assert summary["collision_lead_ge_1_fraction"] == pytest.approx(1.0)
    assert summary["collision_lead_ge_2_fraction"] == pytest.approx(0.0)
    assert summary["collision_lead_steps"]["median"] == pytest.approx(1.0)
    assert summary["predicted_conflict_events"] == 2
    assert summary["predicted_conflict_near_at_onset_fraction"] == pytest.approx(0.0)
    assert summary["predicted_conflict_covered_fraction"] == pytest.approx(1.0)
    assert summary["predicted_conflict_coverage_delay_steps"]["median"] == pytest.approx(
        1.0
    )


def test_tracker_records_uncovered_resolved_conflict() -> None:
    tracker = AgentLeadTimeTracker(1, 1, "cpu")
    tracker.update(
        near_before=torch.tensor([[1.0]]),
        near_after=torch.tensor([[1.0]]),
        predicted_conflict_involvement=torch.tensor([[True]]),
        collision_involvement=torch.tensor([[False]]),
        done=torch.tensor([False]),
        near_distance=0.72,
    )
    tracker.update(
        near_before=torch.tensor([[1.0]]),
        near_after=torch.tensor([[1.0]]),
        predicted_conflict_involvement=torch.tensor([[False]]),
        collision_involvement=torch.tensor([[False]]),
        done=torch.tensor([False]),
        near_distance=0.72,
    )
    summary = tracker.summary()
    assert summary["predicted_conflict_events"] == 1
    assert summary["predicted_conflict_covered_fraction"] == pytest.approx(0.0)
