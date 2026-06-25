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
from train_skrl_mappo import safe_progress_long_checkpoint_rank  # noqa: E402


CONFIG = ROOT / "configs/experiment/exp020_randomized_terrain_subgoal_filter.yaml"


def test_exp020_config_enables_subgoal_filter_contract() -> None:
    raw = load_yaml(CONFIG)
    cfg = cfg_from_experiment(CONFIG)

    assert raw["experiment"]["name"] == "exp020_randomized_terrain_subgoal_filter"
    assert raw["algorithm"]["mode"] == "pure_rl"
    assert raw["algorithm"]["update_mode"] == "shared_joint"
    assert raw["algorithm"]["bc_updates"] == 0
    assert cfg.planner.subgoal_filter.enabled
    assert cfg.planner.subgoal_filter.mode == "terrain_safe_candidate"
    assert cfg.planner.subgoal_filter.rho_scales == pytest.approx([0.65, 1.0])
    assert cfg.planner.subgoal_filter.beta_offsets_deg == pytest.approx(
        [-45.0, -22.5, 0.0, 22.5, 45.0]
    )
    assert cfg.planner.subgoal_filter.path_samples == 5
    assert cfg.success_thresholds.min_pairwise_distance == pytest.approx(0.42)
    assert cfg.safety.collision_distance < cfg.success_thresholds.min_pairwise_distance
    assert cfg.success_thresholds.min_pairwise_distance < cfg.success_thresholds.dmax


def test_exp020_step_reports_filter_metrics() -> None:
    cfg = cfg_from_experiment(CONFIG)
    cfg.simulation.device = "cpu"
    cfg.simulation.num_envs = 4
    env = MultiRoverGatheringCore(cfg)

    action = torch.zeros(cfg.simulation.num_envs, cfg.task.n_agents, 2)
    out = env.step(action)
    action_filter = out.info["action_filter"]

    assert action_filter["enabled"]
    assert action_filter["candidate_count"] == 10
    assert action_filter["raw_path_terrain_risk_mean"].shape == (
        cfg.simulation.num_envs,
        cfg.task.n_agents,
    )
    assert action_filter["filtered_path_terrain_risk_mean"].shape == (
        cfg.simulation.num_envs,
        cfg.task.n_agents,
    )
    assert action_filter["candidate_index"].shape == (
        cfg.simulation.num_envs,
        cfg.task.n_agents,
    )
    assert action_filter["candidate_index_histogram"].shape == (10,)
    for key, value in action_filter.items():
        if isinstance(value, torch.Tensor) and value.is_floating_point():
            assert torch.isfinite(value).all(), key


def test_safe_progress_long_prefers_progress_over_exp019_low_success_early_checkpoint() -> None:
    early_low_success = {
        "candidate_timestep": 1024,
        "dmax_reduction_ratio": 0.4293,
        "success_rate": 0.0195,
        "collision_rate": 0.0771,
        "timeout_rate": 0.9053,
    }
    later_progress = {
        "candidate_timestep": 10240,
        "dmax_reduction_ratio": 0.1552,
        "success_rate": 0.6201,
        "collision_rate": 0.1279,
        "timeout_rate": 0.2627,
    }

    best = min(
        [early_low_success, later_progress],
        key=safe_progress_long_checkpoint_rank,
    )

    assert best is later_progress


def test_safe_progress_long_strict_pass_still_wins() -> None:
    strict_pass = {
        "candidate_timestep": 2048,
        "dmax_reduction_ratio": 0.19,
        "success_rate": 0.91,
        "collision_rate": 0.01,
        "timeout_rate": 0.0,
    }
    trend_only = {
        "candidate_timestep": 10240,
        "dmax_reduction_ratio": 0.12,
        "success_rate": 0.80,
        "collision_rate": 0.04,
        "timeout_rate": 0.1,
    }

    best = min([trend_only, strict_pass], key=safe_progress_long_checkpoint_rank)

    assert best is strict_pass
