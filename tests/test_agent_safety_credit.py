from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from analyze_agent_safety_credit import credit_audit_spec, credit_statistics  # noqa: E402


def test_credit_statistics_report_activation_and_invariants() -> None:
    dataset = {
        "safe_distance": torch.tensor(0.42),
        "safety_raw_credits": torch.tensor([[0.1], [-0.1], [0.0], [0.0]]),
        "safety_centered_step_credits": torch.tensor(
            [[0.1], [-0.1], [0.0], [0.0]]
        ),
        "safety_credits": torch.tensor([[1.0], [-1.0], [0.5], [-0.5]]),
        "safety_active_rollout_fraction": torch.tensor(0.75),
        "safety_rollout_active_flags": torch.tensor([0.0, 1.0, 1.0, 1.0]),
        "safety_environment_rollout_active_fractions": torch.tensor(
            [0.0, 0.25, 0.50, 0.75]
        ),
        "safety_centered_zero_sum_max_abs": torch.tensor(0.0),
    }
    result = credit_statistics(dataset)
    assert result["safe_distance_m"] == pytest.approx(0.42)
    assert result["raw_active_fraction"] == pytest.approx(0.5)
    assert result["raw_positive_fraction"] == pytest.approx(0.25)
    assert result["raw_negative_fraction"] == pytest.approx(0.25)
    assert result["active_rollout_fraction"] == pytest.approx(0.75)
    assert result["first_active_rollout_index"] == 1
    assert result["environment_rollout_active_fraction_mean"] == pytest.approx(0.375)
    assert result["environment_rollout_active_fractions"] == pytest.approx(
        [0.0, 0.25, 0.50, 0.75]
    )
    assert result["centered_zero_sum_max_abs"] == pytest.approx(0.0)


def test_near_distance_credit_spec_uses_existing_reward_threshold() -> None:
    spec = credit_audit_spec(
        ROOT
        / "configs/experiment/exp125_decentralized_tiered_b0_pure_rl_relative_quintic.yaml",
        "near_distance",
    )
    assert spec.experiment_id == "exp133_agent_near_distance_credit"
    assert spec.distance_m == pytest.approx(0.72)
    assert spec.raw_active_min == pytest.approx(0.08)
    assert spec.first_active_rollout_index_max == 1
