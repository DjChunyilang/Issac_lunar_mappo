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
    / "exp078_structured_bicycle_quintic_map25_strict_terminal_center_correction.yaml"
)
CONFIG = (
    ROOT
    / "configs"
    / "experiment"
    / "exp080_structured_bicycle_quintic_map25_strict_center_slots_radius41.yaml"
)


def _without_exp080_radius_delta(raw: dict) -> dict:
    normalized = copy.deepcopy(raw)
    normalized["experiment"]["name"] = "<experiment>"
    normalized["algorithm"]["training_semantics"] = "<semantics>"
    normalized["gather_point"]["execution_slot_radius"] = 0.42
    return normalized


def test_exp080_only_reduces_slot_radius_to_recover_terminal_slack() -> None:
    baseline = load_yaml(BASELINE)
    exp080 = load_yaml(CONFIG)
    assert _without_exp080_radius_delta(exp080) == _without_exp080_radius_delta(
        baseline
    )

    cfg = cfg_from_experiment(CONFIG)
    assert cfg.gather_point.execution_slot_radius == pytest.approx(0.41)
    assert math.sqrt(2.0) * cfg.gather_point.execution_slot_radius > (
        cfg.success_thresholds.min_pairwise_distance
    )
