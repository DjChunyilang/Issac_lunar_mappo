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


CONFIG = ROOT / "configs/experiment/exp029_randomized_terrain_hold_reward_safe.yaml"


def test_exp029_config_combines_hold_reward_with_stronger_safety() -> None:
    raw = load_yaml(CONFIG)
    cfg = cfg_from_experiment(CONFIG)
    exp028 = cfg_from_experiment(ROOT / "configs/experiment/exp028_randomized_terrain_hold_reward.yaml")
    filter_cfg = cfg.planner.subgoal_filter

    assert raw["experiment"]["name"] == "exp029_randomized_terrain_hold_reward_safe"
    assert raw["algorithm"]["mode"] == "pure_rl"
    assert raw["algorithm"]["update_mode"] == "shared_joint"
    assert raw["algorithm"]["bc_updates"] == 0
    assert filter_cfg.mode == "terrain_safe_candidate_mutual_progress_curriculum"
    assert filter_cfg.mutual_path_collision_weight > exp028.planner.subgoal_filter.mutual_path_collision_weight
    assert filter_cfg.path_collision_weight > exp028.planner.subgoal_filter.path_collision_weight
    assert filter_cfg.endpoint_collision_weight > exp028.planner.subgoal_filter.endpoint_collision_weight
    assert cfg.reward_coefficients.success_hold_step == pytest.approx(
        exp028.reward_coefficients.success_hold_step
    )
    assert cfg.reward_coefficients.inter_agent_collision > exp028.reward_coefficients.inter_agent_collision
    assert cfg.reward_coefficients.failure_penalty > exp028.reward_coefficients.failure_penalty
    assert cfg.reward_coefficients.timeout_penalty == pytest.approx(
        exp028.reward_coefficients.timeout_penalty
    )


def test_exp029_step_reports_finite_safety_filter_metrics() -> None:
    cfg = cfg_from_experiment(CONFIG)
    cfg.simulation.device = "cpu"
    cfg.simulation.num_envs = 4
    env = MultiRoverGatheringCore(cfg)

    out = env.step(torch.zeros(cfg.simulation.num_envs, cfg.task.n_agents, 2))
    action_filter = out.info["action_filter"]

    assert action_filter["enabled"]
    assert action_filter["candidate_count"] == 28
    for key in [
        "raw_endpoint_collision_violation",
        "raw_path_collision_violation",
        "raw_mutual_path_collision_violation",
        "endpoint_collision_violation",
        "path_collision_violation",
        "mutual_path_collision_violation",
    ]:
        assert action_filter[key].shape == (cfg.simulation.num_envs, cfg.task.n_agents)
        assert torch.isfinite(action_filter[key]).all(), key
