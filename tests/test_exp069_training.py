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
    / "exp067_structured_bicycle_quintic_map25_oracle_slots_bc.yaml"
)
CONFIG = (
    ROOT
    / "configs"
    / "experiment"
    / "exp069_structured_bicycle_quintic_map25_oracle_slots_hard_safety.yaml"
)


def _without_exp069_safety_delta(raw: dict) -> dict:
    normalized = copy.deepcopy(raw)
    normalized["experiment"]["name"] = "<experiment>"
    normalized["algorithm"]["training_semantics"] = "<semantics>"
    control = normalized["low_level_control"]
    for key in (
        "projection_activation_distance",
        "projection_stop_distance",
        "projection_horizon_s",
        "projection_strength",
        "projection_min_linear_scale",
        "projection_damp_nonclosing_near",
    ):
        control.pop(key)
    return normalized


def test_exp069_isolates_hard_directional_safety_shield() -> None:
    baseline = load_yaml(BASELINE)
    exp069 = load_yaml(CONFIG)

    assert _without_exp069_safety_delta(exp069) == _without_exp069_safety_delta(baseline)

    cfg = cfg_from_experiment(CONFIG)
    control = cfg.low_level_control
    assert control.safety_projection_enabled is True
    assert control.projection_activation_distance == pytest.approx(0.75)
    assert control.projection_stop_distance == pytest.approx(0.42)
    assert control.projection_horizon_s == pytest.approx(0.60)
    assert control.projection_strength == pytest.approx(1.0)
    assert control.projection_min_linear_scale == pytest.approx(0.0)
    assert control.projection_damp_nonclosing_near is True
    assert control.projection_directional_agent_scale is True
    assert control.projection_directional_agent_scale_mode == "mask"
