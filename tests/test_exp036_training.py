from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from _common import cfg_from_experiment, load_yaml  # noqa: E402


CONFIG = ROOT / "configs/experiment/exp036_randomized_terrain_directional_mask_timeout_hold.yaml"


def test_exp036_config_keeps_exp035_safety_and_increases_timeout_hold_reward() -> None:
    raw = load_yaml(CONFIG)
    cfg = cfg_from_experiment(CONFIG)
    exp035 = cfg_from_experiment(
        ROOT / "configs/experiment/exp035_randomized_terrain_directional_mask_buffer.yaml"
    )

    assert raw["experiment"]["name"] == "exp036_randomized_terrain_directional_mask_timeout_hold"
    assert raw["algorithm"]["mode"] == "pure_rl"
    assert raw["algorithm"]["update_mode"] == "shared_joint"
    assert raw["algorithm"]["bc_updates"] == 0
    assert cfg.planner.subgoal_filter.mode == exp035.planner.subgoal_filter.mode
    assert cfg.low_level_control.projection_activation_distance == pytest.approx(
        exp035.low_level_control.projection_activation_distance
    )
    assert cfg.low_level_control.projection_stop_distance == pytest.approx(
        exp035.low_level_control.projection_stop_distance
    )
    assert cfg.low_level_control.projection_directional_agent_scale
    assert cfg.low_level_control.projection_directional_agent_scale_mode == "mask"
    assert cfg.reward_coefficients.dmax_progress > exp035.reward_coefficients.dmax_progress
    assert cfg.reward_coefficients.dispersion_progress > (
        exp035.reward_coefficients.dispersion_progress
    )
    assert cfg.reward_coefficients.success_hold_step > (
        exp035.reward_coefficients.success_hold_step
    )
    assert cfg.reward_coefficients.success_bonus > exp035.reward_coefficients.success_bonus
    assert cfg.reward_coefficients.timeout_penalty > exp035.reward_coefficients.timeout_penalty
    assert cfg.success_thresholds.hold_steps == exp035.success_thresholds.hold_steps
