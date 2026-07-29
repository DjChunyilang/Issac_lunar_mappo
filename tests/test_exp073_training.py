from __future__ import annotations

import copy
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from _common import cfg_from_experiment, load_yaml  # noqa: E402


BASELINE = (
    ROOT
    / "configs"
    / "experiment"
    / "exp072_structured_bicycle_quintic_map25_robust_flat_oracle_slots.yaml"
)
CONFIG = (
    ROOT
    / "configs"
    / "experiment"
    / "exp073_structured_bicycle_quintic_map25_robust_flat_slots_radius42.yaml"
)


def _without_exp073_radius_delta(raw: dict) -> dict:
    normalized = copy.deepcopy(raw)
    normalized["experiment"]["name"] = "<experiment>"
    normalized["algorithm"]["training_semantics"] = "<semantics>"
    normalized["gather_point"].pop("execution_slot_radius")
    return normalized


def test_exp073_isolates_minimal_slot_radius_correction() -> None:
    baseline = load_yaml(BASELINE)
    exp073 = load_yaml(CONFIG)

    assert _without_exp073_radius_delta(exp073) == _without_exp073_radius_delta(
        baseline
    )

    cfg = cfg_from_experiment(CONFIG)
    radius = cfg.gather_point.execution_slot_radius
    assert radius == pytest.approx(0.42)
    assert math.sqrt(2.0) * radius == pytest.approx(0.5939696962)
    assert math.sqrt(2.0) * radius > cfg.success_thresholds.min_pairwise_distance
