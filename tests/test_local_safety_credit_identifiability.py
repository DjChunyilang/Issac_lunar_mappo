from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from analyze_local_safety_credit_identifiability import local_credit_gate  # noqa: E402


def _aggregate(raw: float, centered: float) -> dict:
    return {
        "near_raw_credit": {
            "mean_mse_improvement_fraction": raw,
            "minimum_seed_mse_improvement_fraction": raw,
        },
        "near_centered_credit": {
            "mean_mse_improvement_fraction": centered,
            "minimum_seed_mse_improvement_fraction": centered,
        },
        "repeated_conflict_involvement": {
            "mean_mse_improvement_fraction": 0.20,
            "minimum_seed_mse_improvement_fraction": 0.20,
        },
    }


def _validation(raw_active: float = 0.12, centered_active: float = 0.18) -> dict:
    return {
        "1": {
            "active_rate": {
                "near_raw_credit": raw_active,
                "near_centered_credit": centered_active,
            }
        },
        "2": {
            "active_rate": {
                "near_raw_credit": raw_active,
                "near_centered_credit": centered_active,
            }
        },
    }


def _exp134_checks() -> dict[str, bool]:
    return {
        "every_seed_collision_events_ge_20": True,
        "every_seed_collision_prior_near_fraction_ge_0_95": True,
        "every_seed_collision_lead_ge_1_fraction_ge_0_90": True,
        "every_seed_collision_lead_ge_2_fraction_ge_0_50": True,
        "every_seed_collision_lead_median_ge_2": True,
    }


def test_local_credit_gate_passes_only_with_all_existing_evidence() -> None:
    gate = local_credit_gate(
        aggregate=_aggregate(0.20, 0.18),
        validation=_validation(),
        exp134_checks=_exp134_checks(),
        zero_sum_error=0.0,
        actor_unchanged=True,
        actor_output_change=0.0,
    )
    assert gate["passed"]


def test_repeated_conflict_gain_does_not_replace_near_credit_gate() -> None:
    gate = local_credit_gate(
        aggregate=_aggregate(0.14, 0.14),
        validation=_validation(),
        exp134_checks=_exp134_checks(),
        zero_sum_error=0.0,
        actor_unchanged=True,
        actor_output_change=0.0,
    )
    assert not gate["passed"]
    assert not gate["checks"]["near_raw_mean_action_gain_ge_0_15"]
    assert not gate["checks"]["near_centered_mean_action_gain_ge_0_15"]
