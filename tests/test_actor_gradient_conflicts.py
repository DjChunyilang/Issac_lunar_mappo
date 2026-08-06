from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from analyze_actor_gradient_conflicts import (  # noqa: E402
    agent_local_gather_progress,
    centered_safety_potential_credit,
    gradient_metrics,
    nearest_neighbor_indices,
    nearest_neighbor_safety_potential,
    project_auxiliary_against_primary,
)


def test_agent_local_gather_progress_uses_leave_one_out_centroid() -> None:
    before = torch.tensor([[[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])
    after = torch.tensor([[[-0.5, 0.0, 0.0], [0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]])
    progress = agent_local_gather_progress(before, after)
    assert progress[0, 0].item() == pytest.approx(0.75)
    assert progress[0, 1].item() == pytest.approx(0.0)
    assert progress[0, 2].item() == pytest.approx(0.75)


def test_nearest_neighbor_indices_use_lowest_sender_index_for_ties() -> None:
    positions = torch.tensor(
        [[[0.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [4.0, 0.0]]]
    )
    indices = nearest_neighbor_indices(positions)
    assert indices.tolist() == [[1, 0, 0, 2]]


def test_primary_projection_removes_only_conflicting_component() -> None:
    primary = torch.tensor([1.0, 0.0])
    auxiliary = torch.tensor([-2.0, 3.0])
    projected = project_auxiliary_against_primary(primary, auxiliary)
    assert projected.tolist() == pytest.approx([0.0, 3.0])
    assert torch.dot(primary, projected).item() == pytest.approx(0.0)


def test_primary_projection_leaves_aligned_auxiliary_unchanged() -> None:
    primary = torch.tensor([1.0, 0.0])
    auxiliary = torch.tensor([2.0, 3.0])
    projected = project_auxiliary_against_primary(primary, auxiliary)
    assert torch.equal(projected, auxiliary)


def test_gradient_metrics_preserve_primary_direction_after_projection() -> None:
    metrics = gradient_metrics(
        torch.tensor([1.0, 0.0]),
        torch.tensor([-2.0, 3.0]),
    )
    assert metrics["cosine"] < 0.0
    assert metrics["projected_primary_dot"] == pytest.approx(0.0)
    assert metrics["primary_vs_combined_cosine_at_scale_0_25"] > 0.75


def test_safety_potential_uses_existing_min_pairwise_margin() -> None:
    nearest = torch.tensor([[0.30, 0.42, 0.50]])
    potential = nearest_neighbor_safety_potential(nearest, 0.42)
    assert torch.allclose(potential, torch.tensor([[-0.12, 0.0, 0.0]]))


def test_safety_potential_credit_is_progress_signed_and_zero_sum() -> None:
    before = torch.tensor([[0.30, 0.30, 0.60, 0.60]])
    after = torch.tensor([[0.34, 0.28, 0.60, 0.60]])
    raw, centered = centered_safety_potential_credit(before, after, 0.42)
    assert torch.allclose(raw, torch.tensor([[0.04, -0.02, 0.0, 0.0]]))
    assert centered.sum(dim=1).abs().max().item() <= 1.0e-7
    assert centered[0, 0] > 0.0
    assert centered[0, 1] < 0.0
