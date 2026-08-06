from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from analyze_trajectory_execution_contract import trajectory_contract_gate  # noqa: E402


def _validation(utilization: float = 0.15) -> dict:
    row = {
        "timestamp_speed_violation_rate": 0.8,
        "required_to_declared_horizon_ratio": {"median": 4.0},
        "actual_path_utilization": {"median": utilization},
        "first_tracking_segment_fraction": {"median": 0.09},
    }
    return {"40023": row, "41023": {key: dict(value) if isinstance(value, dict) else value for key, value in row.items()}}


def test_contract_gate_confirms_only_cross_seed_systematic_mismatch() -> None:
    gate = trajectory_contract_gate(
        validation=_validation(),
        actor_parameters_unchanged=True,
        actor_output_change=0.0,
    )
    assert gate["mismatch_confirmed"]


def test_contract_gate_rejects_when_actual_execution_uses_most_of_path() -> None:
    gate = trajectory_contract_gate(
        validation=_validation(utilization=0.40),
        actor_parameters_unchanged=True,
        actor_output_change=0.0,
    )
    assert not gate["mismatch_confirmed"]
    assert not gate["checks"]["every_seed_actual_path_utilization_median_le_0_25"]
