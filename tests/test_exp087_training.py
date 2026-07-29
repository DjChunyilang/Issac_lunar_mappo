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
    / "exp086_structured_bicycle_quintic_map25_flatness_gated_center_slots_radius39.yaml"
)
CONFIG = (
    ROOT
    / "configs"
    / "experiment"
    / "exp087_structured_bicycle_quintic_map25_flatness_gated_warmstart.yaml"
)


def _without_exp087_delta(raw: dict) -> dict:
    normalized = copy.deepcopy(raw)
    normalized["experiment"]["name"] = "<experiment>"
    normalized["experiment"]["checkpoint_interval"] = 1024
    algorithm = normalized["algorithm"]
    algorithm["training_semantics"] = "<semantics>"
    algorithm["learning_rate"] = 0.00010
    algorithm["bc_updates"] = 128
    algorithm.pop("init_checkpoint", None)
    return normalized


def test_exp087_isolates_checkpoint_initialized_low_rate_finetune() -> None:
    baseline = load_yaml(BASELINE)
    exp087 = load_yaml(CONFIG)
    assert _without_exp087_delta(exp087) == _without_exp087_delta(baseline)

    cfg = cfg_from_experiment(CONFIG)
    assert cfg.gather_point.execution_slot_radius == pytest.approx(0.39)
    assert cfg.low_level_control.formation_center_correction_require_flatness_failure
    assert exp087["algorithm"]["bc_updates"] == 0
    assert exp087["algorithm"]["learning_rate"] == pytest.approx(3.0e-5)
    assert str(exp087["algorithm"]["init_checkpoint"]).endswith("best.pt")
