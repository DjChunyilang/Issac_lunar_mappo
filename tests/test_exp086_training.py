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
    / "exp080_structured_bicycle_quintic_map25_strict_center_slots_radius41.yaml"
)
CONFIG = (
    ROOT
    / "configs"
    / "experiment"
    / "exp086_structured_bicycle_quintic_map25_flatness_gated_center_slots_radius39.yaml"
)


def _without_exp086_delta(raw: dict) -> dict:
    normalized = copy.deepcopy(raw)
    normalized["experiment"]["name"] = "<experiment>"
    normalized["algorithm"]["training_semantics"] = "<semantics>"
    control = normalized["low_level_control"]
    control["formation_center_activation_dmax_multiplier"] = 1.0
    control["formation_center_activation_dispersion_multiplier"] = 1.0
    control.pop("formation_center_correction_require_flatness_failure", None)
    normalized["gather_point"]["execution_slot_radius"] = 0.41
    return normalized


def test_exp086_isolates_flatness_gated_center_correction_and_slot_radius() -> None:
    baseline = load_yaml(BASELINE)
    exp086 = load_yaml(CONFIG)
    assert _without_exp086_delta(exp086) == _without_exp086_delta(baseline)

    cfg = cfg_from_experiment(CONFIG)
    control = cfg.low_level_control
    assert control.formation_center_correction_require_flatness_failure
    assert control.formation_center_activation_dmax_multiplier == pytest.approx(1.25)
    assert control.formation_center_activation_dispersion_multiplier == pytest.approx(1.25)
    assert cfg.gather_point.execution_slot_radius == pytest.approx(0.39)
    assert math.sqrt(2.0) * cfg.gather_point.execution_slot_radius > (
        cfg.success_thresholds.min_pairwise_distance
    )
