from __future__ import annotations

import copy
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
    / "exp078_structured_bicycle_quintic_map25_strict_terminal_center_correction.yaml"
)
CONFIG = (
    ROOT
    / "configs"
    / "experiment"
    / "exp079_structured_bicycle_quintic_map25_strict_center_correction_gain45.yaml"
)


def _without_exp079_gain_delta(raw: dict) -> dict:
    normalized = copy.deepcopy(raw)
    normalized["experiment"]["name"] = "<experiment>"
    normalized["algorithm"]["training_semantics"] = "<semantics>"
    normalized["low_level_control"]["formation_center_correction_gain"] = 0.55
    return normalized


def test_exp079_only_reduces_strict_terminal_correction_gain() -> None:
    baseline = load_yaml(BASELINE)
    exp079 = load_yaml(CONFIG)
    assert _without_exp079_gain_delta(exp079) == _without_exp079_gain_delta(baseline)
    assert cfg_from_experiment(CONFIG).low_level_control.formation_center_correction_gain == pytest.approx(
        0.45
    )
