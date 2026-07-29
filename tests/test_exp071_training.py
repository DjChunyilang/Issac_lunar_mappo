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
    / "exp069_structured_bicycle_quintic_map25_oracle_slots_hard_safety.yaml"
)
CONFIG = (
    ROOT
    / "configs"
    / "experiment"
    / "exp071_structured_bicycle_quintic_map25_oracle_slots_radius40.yaml"
)


def _without_exp071_radius_delta(raw: dict) -> dict:
    normalized = copy.deepcopy(raw)
    normalized["experiment"]["name"] = "<experiment>"
    normalized["algorithm"]["training_semantics"] = "<semantics>"
    normalized["gather_point"].pop("execution_slot_radius")
    return normalized


def test_exp071_isolates_the_tighter_safe_slot_formation() -> None:
    baseline = load_yaml(BASELINE)
    exp071 = load_yaml(CONFIG)

    assert _without_exp071_radius_delta(exp071) == _without_exp071_radius_delta(baseline)

    cfg = cfg_from_experiment(CONFIG)
    radius = cfg.gather_point.execution_slot_radius
    adjacent_spacing = math.sqrt(2.0) * radius
    assert radius == pytest.approx(0.40)
    assert adjacent_spacing > cfg.success_thresholds.min_pairwise_distance
    assert adjacent_spacing == pytest.approx(0.5656854249)
