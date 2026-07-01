from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from _common import cfg_from_experiment, load_yaml  # noqa: E402


CONFIG = ROOT / "configs/experiment/exp034_randomized_terrain_directional_mask_control_safety.yaml"


def test_exp034_config_uses_directional_mask_projection() -> None:
    raw = load_yaml(CONFIG)
    cfg = cfg_from_experiment(CONFIG)
    exp033 = cfg_from_experiment(
        ROOT / "configs/experiment/exp033_randomized_terrain_directional_control_safety.yaml"
    )

    assert raw["experiment"]["name"] == "exp034_randomized_terrain_directional_mask_control_safety"
    assert raw["algorithm"]["mode"] == "pure_rl"
    assert raw["algorithm"]["update_mode"] == "shared_joint"
    assert raw["algorithm"]["bc_updates"] == 0
    assert cfg.planner.subgoal_filter.mode == exp033.planner.subgoal_filter.mode
    assert cfg.low_level_control.safety_projection_enabled
    assert not cfg.low_level_control.projection_damp_nonclosing_near
    assert cfg.low_level_control.projection_directional_agent_scale
    assert cfg.low_level_control.projection_directional_agent_scale_mode == "mask"
    assert not cfg.low_level_control.success_zone_damping_enabled
    assert cfg.low_level_control.projection_activation_distance == pytest.approx(
        exp033.low_level_control.projection_activation_distance
    )
    assert cfg.low_level_control.projection_strength == pytest.approx(
        exp033.low_level_control.projection_strength
    )
    assert cfg.success_thresholds.min_pairwise_distance == pytest.approx(0.42)
