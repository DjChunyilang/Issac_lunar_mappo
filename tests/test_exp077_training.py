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
    / "exp076_structured_bicycle_quintic_map25_terminal_center_correction.yaml"
)
CONFIG = (
    ROOT
    / "configs"
    / "experiment"
    / "exp077_structured_bicycle_quintic_map25_center_correction_limited.yaml"
)


def _without_exp077_magnitude_delta(raw: dict) -> dict:
    normalized = copy.deepcopy(raw)
    normalized["experiment"]["name"] = "<experiment>"
    normalized["algorithm"]["training_semantics"] = "<semantics>"
    control = normalized["low_level_control"]
    control["formation_center_correction_max_offset"] = 0.35
    control["formation_center_correction_gain"] = 0.55
    return normalized


def test_exp077_only_limits_center_correction_magnitude() -> None:
    baseline = load_yaml(BASELINE)
    exp077 = load_yaml(CONFIG)
    assert _without_exp077_magnitude_delta(exp077) == _without_exp077_magnitude_delta(
        baseline
    )

    control = cfg_from_experiment(CONFIG).low_level_control
    assert control.formation_center_correction_max_offset == pytest.approx(0.25)
    assert control.formation_center_correction_gain == pytest.approx(0.40)
