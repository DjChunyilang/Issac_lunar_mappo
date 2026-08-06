from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from run_exp144_b0_checkpoint_trend_audit import (  # noqa: E402
    CHECKPOINTS,
    CONTRAST_SEEDS,
    EVAL_SEEDS,
    paired_gate,
)


def _passing_inputs() -> tuple[dict, list[dict]]:
    rows = []
    for checkpoint, late in ((CHECKPOINTS[0], False), (CHECKPOINTS[1], True)):
        for seed in EVAL_SEEDS:
            rows.append(
                {
                    "checkpoint": checkpoint,
                    "seed": seed,
                    "dmax_reduction_ratio": 0.20 if late else 0.30,
                    "success_rate": 0.05 if late else 0.01,
                    "collision_rate": 0.08 if late else 0.80,
                    "timeout_rate": 0.87 if late else 0.19,
                }
            )
    contrasts = []
    for checkpoint, late in ((CHECKPOINTS[0], False), (CHECKPOINTS[1], True)):
        for seed in CONTRAST_SEEDS:
            contrasts.append(
                {
                    "checkpoint_label": checkpoint,
                    "seed": seed,
                    "action_mse_normal_vs_zero_terrain": 0.012 if late else 0.006,
                    "path_risk_reduction_fraction": 0.008 if late else 0.001,
                }
            )
    return {"rows": rows}, contrasts


def test_paired_gate_allows_plan_only_when_geometry_and_terrain_trends_hold() -> None:
    gate = paired_gate(*_passing_inputs())
    assert gate["passed"]
    assert gate["decision"] == "allow_b0_depth_plan_only"
    assert gate["forty_m_authorized"] is False


def test_paired_gate_stops_when_terrain_does_not_improve() -> None:
    sweep, contrasts = _passing_inputs()
    for row in contrasts:
        if row["checkpoint_label"] == CHECKPOINTS[1]:
            row["action_mse_normal_vs_zero_terrain"] = 0.004
            row["path_risk_reduction_fraction"] = -0.003
    gate = paired_gate(sweep, contrasts)
    assert not gate["passed"]
    assert not gate["checks"]["terrain_action_mse_improved_in_two_of_three_seeds"]
    assert not gate["checks"]["terrain_path_risk_gain_mean_ge_0_005"]
    assert gate["decision"] == "stop_b0_depth_training_hypothesis"
