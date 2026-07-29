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
    / "exp080_structured_bicycle_quintic_map25_strict_center_slots_radius41.yaml"
)
CONFIG = (
    ROOT
    / "configs"
    / "experiment"
    / "exp083_structured_bicycle_quintic_map25_strict_light_slot_capture.yaml"
)


def _without_exp083_capture_delta(raw: dict) -> dict:
    normalized = copy.deepcopy(raw)
    normalized["experiment"]["name"] = "<experiment>"
    normalized["algorithm"]["training_semantics"] = "<semantics>"
    control = normalized["low_level_control"]
    control.pop("terminal_slot_capture_enabled", None)
    control.pop("terminal_slot_capture_dmax_multiplier", None)
    control.pop("terminal_slot_capture_dispersion_multiplier", None)
    control.pop("terminal_slot_capture_blend", None)
    return normalized


def test_exp083_isolates_strict_light_per_rover_slot_capture() -> None:
    baseline = load_yaml(BASELINE)
    exp083 = load_yaml(CONFIG)
    assert _without_exp083_capture_delta(exp083) == _without_exp083_capture_delta(
        baseline
    )

    control = cfg_from_experiment(CONFIG).low_level_control
    assert control.terminal_slot_capture_enabled
    assert control.terminal_slot_capture_dmax_multiplier == pytest.approx(1.0)
    assert control.terminal_slot_capture_dispersion_multiplier == pytest.approx(1.0)
    assert control.terminal_slot_capture_blend == pytest.approx(0.25)
