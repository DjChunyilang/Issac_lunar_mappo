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
    / "exp063_structured_bicycle_quintic_map25_flatness_oracle_baseline.yaml"
)
CONFIG = (
    ROOT
    / "configs"
    / "experiment"
    / "exp064_structured_bicycle_quintic_map25_centroid_flatness_reward.yaml"
)


def _without_exp064_delta(raw: dict) -> dict:
    normalized = copy.deepcopy(raw)
    normalized["experiment"]["name"] = "<experiment>"
    normalized["algorithm"]["training_semantics"] = "<semantics>"
    normalized["reward"]["weights"].pop("flatness", None)
    for key in (
        "centroid_flatness_progress",
        "centroid_flatness_excess",
        "centroid_flatness_dmax_multiplier",
    ):
        normalized["reward"]["coefficients"].pop(key, None)
    return normalized


def test_exp064_isolates_actual_centroid_flatness_reward() -> None:
    baseline = load_yaml(BASELINE)
    exp064 = load_yaml(CONFIG)

    assert _without_exp064_delta(exp064) == _without_exp064_delta(baseline)

    cfg = cfg_from_experiment(CONFIG)
    baseline_cfg = cfg_from_experiment(BASELINE)
    assert cfg.reward_weights.flatness == pytest.approx(1.0)
    assert baseline_cfg.reward_weights.flatness == pytest.approx(0.0)
    assert cfg.reward_coefficients.centroid_flatness_progress == pytest.approx(2.0)
    assert cfg.reward_coefficients.centroid_flatness_excess == pytest.approx(0.02)
    assert cfg.reward_coefficients.centroid_flatness_dmax_multiplier == pytest.approx(2.0)


def test_exp064_preserves_decentralized_actor_and_oracle_contract() -> None:
    cfg = cfg_from_experiment(CONFIG)

    assert cfg.observation.schema_version == "ego_v3_local_terrain_grid"
    assert cfg.actor_obs_dim == 86
    assert cfg.critic_state_dim == 54
    assert cfg.gather_point.search_method == "terrain_aware_multiresolution"
    assert cfg.gather_point.require_flat_for_success
