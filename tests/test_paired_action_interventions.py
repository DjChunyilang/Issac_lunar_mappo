from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from analyze_paired_action_interventions import central_difference  # noqa: E402


def test_central_difference_uses_actual_clipped_action_spacing() -> None:
    minus_action = torch.tensor([-1.0, 0.2])
    plus_action = torch.tensor([-0.8, 0.6])
    minus_value = 3.0 * minus_action + 2.0
    plus_value = 3.0 * plus_action + 2.0
    derivative, valid = central_difference(
        minus_value, plus_value, minus_action, plus_action
    )
    assert valid.all()
    assert derivative.tolist() == pytest.approx([3.0, 3.0])


def test_central_difference_marks_zero_spacing_invalid() -> None:
    derivative, valid = central_difference(
        torch.tensor([1.0]),
        torch.tensor([1.0]),
        torch.tensor([1.0]),
        torch.tensor([1.0]),
    )
    assert not valid.item()
    assert derivative.item() == 0.0
