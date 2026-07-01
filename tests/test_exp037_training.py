from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from _common import cfg_from_experiment, load_yaml  # noqa: E402


CONFIG = ROOT / "configs/experiment/exp037_randomized_terrain_directional_mask_timeout260.yaml"


def test_exp037_config_extends_episode_budget_only() -> None:
    raw = load_yaml(CONFIG)
    cfg = cfg_from_experiment(CONFIG)
    exp036 = cfg_from_experiment(
        ROOT / "configs/experiment/exp036_randomized_terrain_directional_mask_timeout_hold.yaml"
    )

    assert raw["experiment"]["name"] == "exp037_randomized_terrain_directional_mask_timeout260"
    assert raw["experiment"]["eval_steps"] == 260
    assert raw["simulation"]["episode_length_s"] == pytest.approx(52.0)
    assert cfg.simulation.max_episode_steps == 260
    assert raw["algorithm"]["mode"] == "pure_rl"
    assert raw["algorithm"]["update_mode"] == "shared_joint"
    assert raw["algorithm"]["bc_updates"] == 0
    assert cfg.planner.subgoal_filter.mode == exp036.planner.subgoal_filter.mode
    assert cfg.low_level_control.projection_activation_distance == pytest.approx(
        exp036.low_level_control.projection_activation_distance
    )
    assert cfg.low_level_control.projection_directional_agent_scale
    assert cfg.low_level_control.projection_directional_agent_scale_mode == "mask"
    assert cfg.reward_coefficients.timeout_penalty == pytest.approx(
        exp036.reward_coefficients.timeout_penalty
    )
    assert cfg.success_thresholds.hold_steps == exp036.success_thresholds.hold_steps
