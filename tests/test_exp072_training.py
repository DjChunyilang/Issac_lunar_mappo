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
    / "exp069_structured_bicycle_quintic_map25_oracle_slots_hard_safety.yaml"
)
CONFIG = (
    ROOT
    / "configs"
    / "experiment"
    / "exp072_structured_bicycle_quintic_map25_robust_flat_oracle_slots.yaml"
)


def _without_exp072_flatness_margin_delta(raw: dict) -> dict:
    normalized = copy.deepcopy(raw)
    normalized["experiment"]["name"] = "<experiment>"
    normalized["algorithm"]["training_semantics"] = "<semantics>"
    normalized["gather_point"].pop("flatness_weight")
    return normalized


def test_exp072_isolates_robust_flatness_search_weight() -> None:
    baseline = load_yaml(BASELINE)
    exp072 = load_yaml(CONFIG)

    assert _without_exp072_flatness_margin_delta(exp072) == _without_exp072_flatness_margin_delta(
        baseline
    )

    cfg = cfg_from_experiment(CONFIG)
    assert cfg.gather_point.flatness_weight == pytest.approx(1.5)
    assert cfg.gather_point.flatness_weight > 0.25
