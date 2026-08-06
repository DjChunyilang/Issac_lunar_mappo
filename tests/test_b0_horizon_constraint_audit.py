from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from analyze_b0_horizon_and_constraint_competition import (  # noqa: E402
    build_audit,
    quarter_means,
)


def test_quarter_means_requires_exact_quarters() -> None:
    assert quarter_means([1.0, 3.0, 2.0, 4.0]) == [1.0, 3.0, 2.0, 4.0]
    assert quarter_means(list(map(float, range(8)))) == pytest.approx(
        [0.5, 2.5, 4.5, 6.5]
    )
    with pytest.raises(ValueError, match="divisible by four"):
        quarter_means([1.0, 2.0, 3.0])


def _passing_inputs() -> dict:
    b0_rows = []
    constraint_rows = []
    dmax_values = [5.0] * 4 + [4.5] * 4 + [4.0] * 4 + [3.2] * 4
    for index, dmax in enumerate(dmax_values, start=1):
        timestep = index * 32
        b0_rows.append(
            {
                "phase": "train",
                "timesteps": timestep,
                "dmax_mean": dmax,
                "success_done": max(0, index - 4),
            }
        )
        constraint_rows.append(
            {
                "phase": "train",
                "timesteps": timestep,
                "dmax_mean": dmax,
                "success_done": min(index, 5),
            }
        )
    history = []
    for update in range(1, 9):
        applied = 0.0 if update < 3 else 0.6 if update < 5 else 1.1
        history.append(
            {
                "update": float(update),
                "episode_equivalent_collision_rate": 1.2 if update < 3 else 0.1,
                "lagrangian_multiplier_applied": applied,
                "lagrangian_multiplier": applied,
                "cost_value_loss": 0.01,
            }
        )
    return {
        "b0_summary": {
            "timesteps": 2048,
            "training_diagnostics": {
                "policy_parameter_delta_l2": 1.0,
                "neighbor_encoder_parameter_delta_l2": 1.0,
                "terrain_encoder_parameter_delta_l2": 1.0,
                "post_training_action_std": 0.1,
            },
        },
        "b0_eval": {
            "evaluations": [
                {
                    "candidate_timestep": 1024,
                    "dmax_reduction_ratio": 0.30,
                    "success_rate": 0.01,
                    "collision_rate": 0.80,
                    "timeout_rate": 0.19,
                },
                {
                    "candidate_timestep": 2048,
                    "dmax_reduction_ratio": 0.20,
                    "success_rate": 0.05,
                    "collision_rate": 0.08,
                    "timeout_rate": 0.87,
                },
            ]
        },
        "b0_telemetry": b0_rows,
        "constraint_summary": {
            "training_diagnostics": {
                "collision_constraint_history": history,
                "lagrangian_multiplier": 1.1,
            }
        },
        "constraint_telemetry": constraint_rows,
        "curriculum_warmup_timesteps": 4096,
    }


def test_audit_allows_only_bounded_b0_depth_plan_when_all_checks_hold() -> None:
    result = build_audit(**_passing_inputs())
    assert result["passed"]
    assert result["decision"] == "allow_exp144_b0_12m_plan_only"
    assert result["forty_m_authorized"] is False
    assert result["evidence"]["constraint_rate_gt_one_fraction"] == pytest.approx(
        0.25
    )


def test_audit_stops_extension_when_late_b0_progress_is_absent() -> None:
    inputs = _passing_inputs()
    for row in inputs["b0_telemetry"][-4:]:
        row["dmax_mean"] = 4.0
    result = build_audit(**inputs)
    assert not result["passed"]
    assert not result["checks"]["last_quarter_dmax_improved_10pct"]
    assert result["decision"] == "stop_b0_depth_extension"
