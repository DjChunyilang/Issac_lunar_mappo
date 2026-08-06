from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from analyze_safety_aggregation_identifiability import (  # noqa: E402
    aggregation_gate,
    safety_aggregation_targets,
)


def test_safety_aggregation_reuses_identical_collision_term() -> None:
    nearest = torch.tensor([[0.50, 0.60, 0.90, 1.00], [0.80, 0.90, 1.00, 1.10]])
    collision = torch.tensor([True, False])
    targets = safety_aggregation_targets(
        nearest,
        collision,
        near_distance=0.72,
        near_coefficient=2.4,
        collision_coefficient=15.0,
        safety_weight=1.0,
    )
    expected_mean = -torch.tensor([15.0 + 2.4 * (0.22 + 0.12) / 4.0, 0.0])
    expected_worst = -torch.tensor([15.0 + 2.4 * 0.22, 0.0])
    torch.testing.assert_close(targets["safety_mean"], expected_mean)
    torch.testing.assert_close(targets["safety_worst_pair"], expected_worst)
    assert torch.all(targets["safety_worst_pair"] <= targets["safety_mean"])


def test_aggregation_gate_requires_every_seed_gain_delta() -> None:
    aggregate = {
        "safety_mean": {
            "mean_mse_improvement_fraction": 0.05,
            "minimum_seed_mse_improvement_fraction": 0.04,
        },
        "safety_worst_pair": {
            "mean_mse_improvement_fraction": 0.18,
            "minimum_seed_mse_improvement_fraction": 0.16,
        },
    }
    per_seed = {
        "1": {
            "active_rate": {"safety_worst_pair": 0.20},
            "components": {
                "safety_mean": {"mean_mse_improvement_fraction": 0.05},
                "safety_worst_pair": {"mean_mse_improvement_fraction": 0.17},
            }
        },
        "2": {
            "active_rate": {"safety_worst_pair": 0.18},
            "components": {
                "safety_mean": {"mean_mse_improvement_fraction": 0.04},
                "safety_worst_pair": {"mean_mse_improvement_fraction": 0.16},
            }
        },
    }
    passed = aggregation_gate(
        aggregate,
        per_seed,
        actor_unchanged=True,
        reconstruction_error=0.0,
    )
    assert passed["passed"]

    per_seed["2"]["components"]["safety_mean"][
        "mean_mse_improvement_fraction"
    ] = 0.07
    failed = aggregation_gate(
        aggregate,
        per_seed,
        actor_unchanged=True,
        reconstruction_error=0.0,
    )
    assert not failed["passed"]
    assert not failed["checks"]["worst_pair_every_seed_gain_delta_ge_0_10"]


def test_aggregation_gate_rejects_zero_activity_false_positive() -> None:
    aggregate = {
        "safety_mean": {
            "mean_mse_improvement_fraction": 0.05,
            "minimum_seed_mse_improvement_fraction": 0.05,
        },
        "safety_worst_pair": {
            "mean_mse_improvement_fraction": 0.50,
            "minimum_seed_mse_improvement_fraction": 0.50,
        },
    }
    per_seed = {
        "1": {
            "active_rate": {"safety_worst_pair": 0.0},
            "components": {
                "safety_mean": {"mean_mse_improvement_fraction": 0.05},
                "safety_worst_pair": {"mean_mse_improvement_fraction": 0.50},
            },
        }
    }
    gate = aggregation_gate(
        aggregate,
        per_seed,
        actor_unchanged=True,
        reconstruction_error=0.0,
    )
    assert not gate["passed"]
    assert not gate["checks"]["worst_pair_every_seed_active_rate_ge_0_05"]
