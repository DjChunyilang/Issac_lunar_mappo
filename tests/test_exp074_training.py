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
    / "exp074_structured_bicycle_quintic_map25_robust_envelope_slots_radius42.yaml"
)


def _without_exp074_search_delta(raw: dict) -> dict:
    normalized = copy.deepcopy(raw)
    normalized["experiment"]["name"] = "<experiment>"
    normalized["algorithm"]["training_semantics"] = "<semantics>"
    gather = normalized["gather_point"]
    gather.pop("robustness_radius", None)
    gather.pop("robustness_samples", None)
    gather["flatness_weight"] = 1.50
    return normalized


def test_exp074_isolates_the_robust_execution_envelope() -> None:
    baseline = load_yaml(BASELINE)
    exp074 = load_yaml(CONFIG)

    assert _without_exp074_search_delta(exp074) == _without_exp074_search_delta(
        baseline
    )

    cfg = cfg_from_experiment(CONFIG)
    assert cfg.gather_point.robustness_radius == pytest.approx(0.10)
    assert cfg.gather_point.robustness_samples == 8
    assert cfg.gather_point.flatness_weight == pytest.approx(0.25)
