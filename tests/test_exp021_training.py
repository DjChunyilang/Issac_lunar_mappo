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
from train_skrl_mappo import balanced_progress_long_checkpoint_rank  # noqa: E402


CONFIG = ROOT / "configs/experiment/exp021_randomized_terrain_filter_curriculum.yaml"
EXP020_CONFIG = ROOT / "configs/experiment/exp020_randomized_terrain_subgoal_filter.yaml"


def test_exp021_config_enables_curriculum_filter_contract() -> None:
    raw = load_yaml(CONFIG)
    cfg = cfg_from_experiment(CONFIG)

    assert raw["experiment"]["name"] == "exp021_randomized_terrain_filter_curriculum"
    assert raw["algorithm"]["mode"] == "pure_rl"
    assert raw["algorithm"]["update_mode"] == "shared_joint"
    assert raw["algorithm"]["bc_updates"] == 0
    assert cfg.planner.subgoal_filter.enabled
    assert cfg.planner.subgoal_filter.mode == "terrain_safe_candidate_curriculum"
    assert cfg.planner.subgoal_filter.rho_scales == pytest.approx([0.85, 1.0])
    assert cfg.planner.subgoal_filter.beta_offsets_deg == pytest.approx(
        [-30.0, -15.0, 0.0, 15.0, 30.0]
    )
    assert cfg.planner.subgoal_filter.warmup_timesteps == 2048
    assert cfg.planner.subgoal_filter.ramp_timesteps == 4096
    assert cfg.planner.subgoal_filter.apply_probability_end == pytest.approx(0.60)
    assert cfg.planner.subgoal_filter.score_scale_start == pytest.approx(0.15)
    assert cfg.planner.subgoal_filter.score_scale_end == pytest.approx(0.75)
    assert cfg.planner.subgoal_filter.visible_neighbor_center_weight == pytest.approx(0.35)
    assert cfg.reward_coefficients.filter_raw_path_risk_cost == pytest.approx(0.30)
    assert cfg.reward_coefficients.filter_deviation_cost == pytest.approx(0.10)
    assert cfg.safety.collision_distance < cfg.success_thresholds.min_pairwise_distance
    assert cfg.success_thresholds.min_pairwise_distance < cfg.success_thresholds.dmax


def test_filter_auxiliary_reward_defaults_to_zero_for_exp020() -> None:
    cfg = cfg_from_experiment(EXP020_CONFIG)

    assert cfg.reward_coefficients.filter_raw_path_risk_cost == pytest.approx(0.0)
    assert cfg.reward_coefficients.filter_deviation_cost == pytest.approx(0.0)


def test_exp021_warmup_step_reports_filter_metrics_without_replacement() -> None:
    cfg = cfg_from_experiment(CONFIG)
    cfg.simulation.device = "cpu"
    cfg.simulation.num_envs = 4
    env = MultiRoverGatheringCore(cfg)

    action = torch.zeros(cfg.simulation.num_envs, cfg.task.n_agents, 2)
    out = env.step(action)
    action_filter = out.info["action_filter"]

    assert action_filter["enabled"]
    assert action_filter["candidate_count"] == 10
    assert action_filter["schedule_progress_step"] == 0
    assert action_filter["apply_probability"] == pytest.approx(0.0)
    assert not action_filter["applied"].any()
    assert action_filter["raw_score"].shape == (cfg.simulation.num_envs, cfg.task.n_agents)
    assert action_filter["suggested_score"].shape == (
        cfg.simulation.num_envs,
        cfg.task.n_agents,
    )
    assert torch.isfinite(out.info["reward_terms"].terrain).all()


def test_checkpoint_metadata_progress_can_force_eval_filter_progress() -> None:
    cfg = cfg_from_experiment(CONFIG)
    cfg.simulation.device = "cpu"
    cfg.simulation.num_envs = 2
    cfg.planner.subgoal_filter.progress_timestep_override = 6144
    cfg.planner.subgoal_filter.deterministic_eval = True
    env = MultiRoverGatheringCore(cfg)

    action = torch.zeros(cfg.simulation.num_envs, cfg.task.n_agents, 2)
    out = env.step(action)

    assert out.info["action_filter"]["schedule_progress_step"] == 6144
    assert out.info["action_filter"]["apply_probability"] == pytest.approx(0.60)
    assert out.info["action_filter"]["score_scale"] == pytest.approx(0.75)


def test_balanced_progress_long_rejects_success_zero_when_progress_exists() -> None:
    safe_but_no_progress = {
        "candidate_timestep": 10240,
        "dmax_reduction_ratio": 0.3752,
        "success_rate": 0.0,
        "collision_rate": 0.0498,
        "timeout_rate": 0.9506,
    }
    some_progress = {
        "candidate_timestep": 2048,
        "dmax_reduction_ratio": 0.28,
        "success_rate": 0.06,
        "collision_rate": 0.08,
        "timeout_rate": 0.80,
    }

    best = min(
        [safe_but_no_progress, some_progress],
        key=balanced_progress_long_checkpoint_rank,
    )

    assert best is some_progress


def test_balanced_progress_long_strict_pass_still_wins() -> None:
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

    best = min([trend_only, strict_pass], key=balanced_progress_long_checkpoint_rank)

    assert best is strict_pass
