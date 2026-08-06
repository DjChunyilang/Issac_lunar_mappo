from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from analyze_collision_cost_value_feasibility import (  # noqa: E402
    binary_ranking_metrics,
    future_collision_labels,
    gate_decision,
)


def test_future_collision_labels_do_not_cross_episode_reset() -> None:
    collision = torch.zeros((7, 1), dtype=torch.bool)
    done = torch.zeros_like(collision)
    done[2, 0] = True
    collision[4, 0] = True
    done[4, 0] = True
    labels = future_collision_labels(collision, done, horizon=4)
    assert labels[:, 0].tolist() == [False, False, False, True]


def test_binary_ranking_metrics_are_exact_for_perfect_ordering() -> None:
    probability = torch.tensor([0.05, 0.9, 0.1, 0.8])
    target = torch.tensor([0, 1, 0, 1])
    metrics = binary_ranking_metrics(probability, target)
    assert metrics["auroc"] == pytest.approx(1.0)
    assert metrics["average_precision"] == pytest.approx(1.0)


def test_cost_value_gate_requires_every_validation_seed() -> None:
    result = {
        "training": {"collision_episodes": 40},
        "validation": {
            "40023": {
                "collision_episodes": 25,
                "positive_rate": 0.02,
                "mean_auroc": 0.80,
                "mean_average_precision": 0.10,
                "mean_brier_improvement_fraction": 0.20,
            },
            "41023": {
                "collision_episodes": 24,
                "positive_rate": 0.02,
                "mean_auroc": 0.78,
                "mean_average_precision": 0.08,
                "mean_brier_improvement_fraction": 0.18,
            },
        },
        "invariance": {
            "actor_digest_before": "same",
            "actor_digest_after": "same",
            "actor_probe_output_max_abs_change": 0.0,
        },
    }
    assert gate_decision(result)["passed"]
    result["validation"]["41023"]["mean_auroc"] = 0.74
    gate = gate_decision(result)
    assert not gate["passed"]
    assert not gate["checks"]["every_validation_seed_mean_auroc_ge_0_75"]
