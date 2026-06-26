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
from train_skrl_mappo import progress_preserving_long_checkpoint_rank  # noqa: E402


CONFIG = ROOT / "configs/experiment/exp023_randomized_terrain_soft_progress_filter.yaml"


def test_exp023_config_enables_soft_progress_filter_contract() -> None:
    raw = load_yaml(CONFIG)
    cfg = cfg_from_experiment(CONFIG)
    filter_cfg = cfg.planner.subgoal_filter

    assert raw["experiment"]["name"] == "exp023_randomized_terrain_soft_progress_filter"
    assert raw["algorithm"]["mode"] == "pure_rl"
    assert raw["algorithm"]["update_mode"] == "shared_joint"
    assert raw["algorithm"]["bc_updates"] == 0
    assert filter_cfg.enabled
    assert filter_cfg.mode == "terrain_safe_candidate_soft_progress_curriculum"
    assert filter_cfg.rho_scales == pytest.approx([0.75, 0.90, 1.0, 1.08])
    assert filter_cfg.beta_offsets_deg == pytest.approx(
        [-35.0, -20.0, -10.0, 0.0, 10.0, 20.0, 35.0]
    )
    assert len(filter_cfg.rho_scales) * len(filter_cfg.beta_offsets_deg) == 28
    assert not filter_cfg.hard_endpoint_near_filter
    assert not filter_cfg.hard_path_collision_filter
    assert not filter_cfg.hard_center_progress_filter
    assert not filter_cfg.safety_override_after_warmup
    assert filter_cfg.collision_override_after_warmup
    assert filter_cfg.apply_probability_end == pytest.approx(0.35)
    assert filter_cfg.score_scale_end == pytest.approx(0.55)
    assert filter_cfg.visible_neighbor_center_weight == pytest.approx(1.00)
    assert filter_cfg.center_progress_weight == pytest.approx(1.50)
    assert cfg.safety.near_distance == pytest.approx(0.85)
    assert cfg.reward_coefficients.near_distance == pytest.approx(6.0)
    assert cfg.reward_coefficients.inter_agent_collision == pytest.approx(90.0)
    assert cfg.reward_coefficients.failure_penalty == pytest.approx(50.0)
    assert cfg.reward_coefficients.filter_raw_path_risk_cost == pytest.approx(0.20)
    assert cfg.reward_coefficients.filter_deviation_cost == pytest.approx(0.05)
    assert cfg.safety.collision_distance < cfg.success_thresholds.min_pairwise_distance
    assert cfg.success_thresholds.min_pairwise_distance < cfg.success_thresholds.dmax


def test_exp023_warmup_step_reports_soft_progress_metrics_without_replacement() -> None:
    cfg = cfg_from_experiment(CONFIG)
    cfg.simulation.device = "cpu"
    cfg.simulation.num_envs = 4
    env = MultiRoverGatheringCore(cfg)

    action = torch.zeros(cfg.simulation.num_envs, cfg.task.n_agents, 2)
    out = env.step(action)
    action_filter = out.info["action_filter"]

    assert action_filter["enabled"]
    assert action_filter["candidate_count"] == 28
    assert action_filter["schedule_progress_step"] == 0
    assert action_filter["apply_probability"] == pytest.approx(0.0)
    assert action_filter["score_scale"] == pytest.approx(0.20)
    assert not action_filter["applied"].any()
    assert action_filter["raw_visible_center_cost"].shape == (
        cfg.simulation.num_envs,
        cfg.task.n_agents,
    )
    assert action_filter["center_progress_regression"].shape == (
        cfg.simulation.num_envs,
        cfg.task.n_agents,
    )
    assert isinstance(action_filter["collision_override_fraction"], float)
    for key, value in action_filter.items():
        if isinstance(value, torch.Tensor) and value.is_floating_point():
            assert torch.isfinite(value).all(), key


def test_exp023_eval_progress_sets_soft_schedule() -> None:
    cfg = cfg_from_experiment(CONFIG)
    cfg.simulation.device = "cpu"
    cfg.simulation.num_envs = 2
    cfg.planner.subgoal_filter.progress_timestep_override = 6144
    cfg.planner.subgoal_filter.deterministic_eval = True
    env = MultiRoverGatheringCore(cfg)

    action = torch.zeros(cfg.simulation.num_envs, cfg.task.n_agents, 2)
    out = env.step(action)
    action_filter = out.info["action_filter"]

    assert action_filter["schedule_progress_step"] == 6144
    assert action_filter["apply_probability"] == pytest.approx(0.35)
    assert action_filter["score_scale"] == pytest.approx(0.55)
    assert 0.0 <= action_filter["collision_override_fraction"] <= 1.0


def test_progress_preserving_long_keeps_progress_over_low_collision_standoff() -> None:
    exp022_style_standoff = {
        "candidate_timestep": 7168,
        "dmax_reduction_ratio": 0.47,
        "success_rate": 0.014,
        "collision_rate": 0.017,
        "timeout_rate": 0.97,
    }
    progress_candidate = {
        "candidate_timestep": 2048,
        "dmax_reduction_ratio": 0.28,
        "success_rate": 0.16,
        "collision_rate": 0.084,
        "timeout_rate": 0.72,
    }

    best = min(
        [exp022_style_standoff, progress_candidate],
        key=progress_preserving_long_checkpoint_rank,
    )

    assert best is progress_candidate


def test_progress_preserving_long_strict_pass_still_wins() -> None:
    strict_pass = {
        "candidate_timestep": 2048,
        "dmax_reduction_ratio": 0.19,
        "success_rate": 0.91,
        "collision_rate": 0.01,
        "timeout_rate": 0.0,
    }
    progress_only = {
        "candidate_timestep": 10240,
        "dmax_reduction_ratio": 0.12,
        "success_rate": 0.80,
        "collision_rate": 0.04,
        "timeout_rate": 0.1,
    }

    best = min([progress_only, strict_pass], key=progress_preserving_long_checkpoint_rank)

    assert best is strict_pass
