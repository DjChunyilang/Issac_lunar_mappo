from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from _common import cfg_from_experiment, load_yaml  # noqa: E402


CONFIG = ROOT / "configs/experiment/exp033_randomized_terrain_directional_control_safety.yaml"


def test_exp033_config_uses_directional_agent_projection() -> None:
    raw = load_yaml(CONFIG)
    cfg = cfg_from_experiment(CONFIG)
    exp032 = cfg_from_experiment(
        ROOT / "configs/experiment/exp032_randomized_terrain_closing_control_safety.yaml"
    )

    assert raw["experiment"]["name"] == "exp033_randomized_terrain_directional_control_safety"
    assert raw["algorithm"]["mode"] == "pure_rl"
    assert raw["algorithm"]["update_mode"] == "shared_joint"
    assert raw["algorithm"]["bc_updates"] == 0
    assert cfg.planner.subgoal_filter.mode == exp032.planner.subgoal_filter.mode
    assert cfg.low_level_control.safety_projection_enabled
    assert not cfg.low_level_control.projection_damp_nonclosing_near
    assert cfg.low_level_control.projection_directional_agent_scale
    assert not cfg.low_level_control.success_zone_damping_enabled
    assert cfg.low_level_control.projection_activation_distance == pytest.approx(
        exp032.low_level_control.projection_activation_distance
    )
    assert cfg.low_level_control.projection_strength == pytest.approx(
        exp032.low_level_control.projection_strength
    )
    assert cfg.low_level_control.projection_min_linear_scale == pytest.approx(
        exp032.low_level_control.projection_min_linear_scale
    )
    assert cfg.success_thresholds.min_pairwise_distance == pytest.approx(0.42)
