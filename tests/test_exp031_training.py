from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from _common import cfg_from_experiment, load_yaml  # noqa: E402
from lunar_rover_tasks.tasks.multi_rover_gathering.gathering_env import (  # noqa: E402
    MultiRoverGatheringCore,
)


CONFIG = ROOT / "configs/experiment/exp031_randomized_terrain_narrow_control_safety.yaml"


def test_exp031_config_narrows_control_safety_against_exp030() -> None:
    raw = load_yaml(CONFIG)
    cfg = cfg_from_experiment(CONFIG)
    exp030 = cfg_from_experiment(
        ROOT / "configs/experiment/exp030_randomized_terrain_control_safety.yaml"
    )

    assert raw["experiment"]["name"] == "exp031_randomized_terrain_narrow_control_safety"
    assert raw["algorithm"]["mode"] == "pure_rl"
    assert raw["algorithm"]["update_mode"] == "shared_joint"
    assert raw["algorithm"]["bc_updates"] == 0
    assert cfg.planner.subgoal_filter.mode == exp030.planner.subgoal_filter.mode
    assert cfg.reward_coefficients.success_hold_step == pytest.approx(
        exp030.reward_coefficients.success_hold_step
    )
    assert cfg.low_level_control.safety_projection_enabled
    assert not cfg.low_level_control.success_zone_damping_enabled
    assert (
        cfg.low_level_control.projection_activation_distance
        < exp030.low_level_control.projection_activation_distance
    )
    assert cfg.low_level_control.projection_strength < exp030.low_level_control.projection_strength
    assert (
        cfg.low_level_control.projection_min_linear_scale
        > exp030.low_level_control.projection_min_linear_scale
    )
    assert (
        cfg.safety.collision_distance
        < cfg.low_level_control.projection_stop_distance
        < cfg.success_thresholds.min_pairwise_distance
        < cfg.low_level_control.projection_activation_distance
    )


def test_exp031_step_reports_control_safety_metrics() -> None:
    cfg = cfg_from_experiment(CONFIG)
    cfg.simulation.device = "cpu"
    cfg.simulation.num_envs = 4
    env = MultiRoverGatheringCore(cfg)

    out = env.step(torch.zeros(cfg.simulation.num_envs, cfg.task.n_agents, 2))
    control_safety = out.info["control_safety"]

    assert control_safety["enabled"]
    assert not control_safety["success_zone_active"].any()
    assert control_safety["linear_scale"].shape == (cfg.simulation.num_envs, cfg.task.n_agents)
    assert torch.isfinite(control_safety["linear_scale"]).all()
    assert torch.isfinite(control_safety["pairwise_risk"]).all()
