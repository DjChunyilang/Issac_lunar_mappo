from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from analyze_nearest_pair_safety_action_coupling import pair_safety_gate  # noqa: E402


def _aggregate(value: float = 0.30) -> dict:
    return {
        "pair_vs_observation": {"mean": value, "minimum_seed": value},
        "own_given_neighbor": {"mean": value, "minimum_seed": value},
        "neighbor_given_own": {"mean": value, "minimum_seed": value},
        "neighbor_shuffle_degradation": {"mean": value, "minimum_seed": value},
    }


def _validation(active_rate: float = 0.16) -> dict:
    row = {"target_distribution": {"std": 0.0028, "active_rate": active_rate}}
    return {"40023": row, "41023": {"target_distribution": dict(row["target_distribution"])}}


def test_pair_safety_gate_passes_all_pre_registered_checks() -> None:
    gate = pair_safety_gate(
        aggregate=_aggregate(),
        validation=_validation(),
        actor_parameters_unchanged=True,
        actor_output_change=0.0,
        actor_observation_dim=101,
    )
    assert gate["passed"]


def test_pair_safety_gate_requires_independent_own_action_contribution() -> None:
    aggregate = _aggregate()
    aggregate["own_given_neighbor"]["minimum_seed"] = 0.149
    gate = pair_safety_gate(
        aggregate=aggregate,
        validation=_validation(),
        actor_parameters_unchanged=True,
        actor_output_change=0.0,
        actor_observation_dim=101,
    )
    assert not gate["passed"]
    assert not gate["checks"]["own_given_neighbor_every_seed_ge_0_15"]
