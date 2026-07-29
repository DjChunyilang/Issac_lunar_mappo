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
    / "exp078_structured_bicycle_quintic_map25_strict_terminal_center_correction.yaml"
)


def _without_exp078_activation_delta(raw: dict) -> dict:
    normalized = copy.deepcopy(raw)
    normalized["experiment"]["name"] = "<experiment>"
    normalized["algorithm"]["training_semantics"] = "<semantics>"
    control = normalized["low_level_control"]
    control["formation_center_activation_dmax_multiplier"] = 1.75
    control["formation_center_activation_dispersion_multiplier"] = 1.75
    return normalized


def test_exp078_only_requires_terminal_geometry_before_center_correction() -> None:
    baseline = load_yaml(BASELINE)
    exp078 = load_yaml(CONFIG)
    assert _without_exp078_activation_delta(exp078) == _without_exp078_activation_delta(
        baseline
    )

    control = cfg_from_experiment(CONFIG).low_level_control
    assert control.formation_center_activation_dmax_multiplier == pytest.approx(1.0)
    assert control.formation_center_activation_dispersion_multiplier == pytest.approx(1.0)
