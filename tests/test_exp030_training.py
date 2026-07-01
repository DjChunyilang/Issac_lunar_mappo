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


CONFIG = ROOT / "configs/experiment/exp030_randomized_terrain_control_safety.yaml"


def test_exp030_config_adds_control_safety_without_reward_escalation() -> None:
    raw = load_yaml(CONFIG)
    cfg = cfg_from_experiment(CONFIG)
    exp028 = cfg_from_experiment(ROOT / "configs/experiment/exp028_randomized_terrain_hold_reward.yaml")

    assert raw["experiment"]["name"] == "exp030_randomized_terrain_control_safety"
    assert raw["algorithm"]["mode"] == "pure_rl"
    assert raw["algorithm"]["update_mode"] == "shared_joint"
    assert raw["algorithm"]["bc_updates"] == 0
    assert cfg.planner.subgoal_filter.mode == exp028.planner.subgoal_filter.mode
    assert cfg.planner.subgoal_filter.mutual_path_collision_weight == pytest.approx(
        exp028.planner.subgoal_filter.mutual_path_collision_weight
    )
    assert cfg.reward_coefficients.inter_agent_collision == pytest.approx(
        exp028.reward_coefficients.inter_agent_collision
    )
    assert cfg.reward_coefficients.failure_penalty == pytest.approx(
        exp028.reward_coefficients.failure_penalty
    )
    assert cfg.reward_coefficients.timeout_penalty == pytest.approx(
        exp028.reward_coefficients.timeout_penalty
    )
    assert cfg.low_level_control.safety_projection_enabled
    assert cfg.low_level_control.success_zone_damping_enabled
    assert cfg.low_level_control.projection_activation_distance > cfg.success_thresholds.min_pairwise_distance
    assert (
        cfg.safety.collision_distance
        < cfg.low_level_control.projection_stop_distance
        < cfg.success_thresholds.min_pairwise_distance
    )


def test_exp030_step_reports_finite_control_safety_metrics() -> None:
    cfg = cfg_from_experiment(CONFIG)
    cfg.simulation.device = "cpu"
    cfg.simulation.num_envs = 4
    env = MultiRoverGatheringCore(cfg)

    out = env.step(torch.zeros(cfg.simulation.num_envs, cfg.task.n_agents, 2))
    control_safety = out.info["control_safety"]

    assert control_safety["enabled"]
    assert control_safety["linear_scale"].shape == (cfg.simulation.num_envs, cfg.task.n_agents)
    assert control_safety["applied"].shape == (cfg.simulation.num_envs, cfg.task.n_agents)
    for key in [
        "linear_scale",
        "raw_linear",
        "projected_linear",
        "pairwise_risk",
        "predicted_nearest_distance",
    ]:
        assert torch.isfinite(control_safety[key]).all(), key
    assert torch.all(control_safety["linear_scale"] >= 0.0)
    assert torch.all(control_safety["linear_scale"] <= 1.0)
