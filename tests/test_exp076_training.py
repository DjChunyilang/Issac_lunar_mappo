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
    / "exp073_structured_bicycle_quintic_map25_robust_flat_slots_radius42.yaml"
)
CONFIG = (
    ROOT
    / "configs"
    / "experiment"
    / "exp076_structured_bicycle_quintic_map25_terminal_center_correction.yaml"
)


def _without_exp076_correction_delta(raw: dict) -> dict:
    normalized = copy.deepcopy(raw)
    normalized["experiment"]["name"] = "<experiment>"
    normalized["algorithm"]["training_semantics"] = "<semantics>"
    control = normalized["low_level_control"]
    control.pop("formation_center_correction_enabled", None)
    control.pop("formation_center_activation_dmax_multiplier", None)
    control.pop("formation_center_activation_dispersion_multiplier", None)
    control.pop("formation_center_correction_max_offset", None)
    control.pop("formation_center_correction_gain", None)
    return normalized


def test_exp076_isolates_terminal_formation_center_correction() -> None:
    baseline = load_yaml(BASELINE)
    exp076 = load_yaml(CONFIG)

    assert _without_exp076_correction_delta(exp076) == _without_exp076_correction_delta(
        baseline
    )

    cfg = cfg_from_experiment(CONFIG)
    control = cfg.low_level_control
    assert control.formation_center_correction_enabled
    assert control.formation_center_activation_dmax_multiplier == pytest.approx(1.75)
    assert control.formation_center_activation_dispersion_multiplier == pytest.approx(1.75)
    assert control.formation_center_correction_max_offset == pytest.approx(0.35)
    assert control.formation_center_correction_gain == pytest.approx(0.55)
