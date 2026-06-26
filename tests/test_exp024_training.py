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
from train_skrl_mappo import success_progress_long_checkpoint_rank  # noqa: E402


CONFIG = ROOT / "configs/experiment/exp024_randomized_terrain_mutual_path_filter.yaml"


def test_exp024_config_enables_mutual_path_filter_contract() -> None:
    raw = load_yaml(CONFIG)
    cfg = cfg_from_experiment(CONFIG)
    filter_cfg = cfg.planner.subgoal_filter

    assert raw["experiment"]["name"] == "exp024_randomized_terrain_mutual_path_filter"
    assert raw["algorithm"]["mode"] == "pure_rl"
    assert raw["algorithm"]["update_mode"] == "shared_joint"
    assert raw["algorithm"]["bc_updates"] == 0
    assert filter_cfg.enabled
    assert filter_cfg.mode == "terrain_safe_candidate_mutual_progress_curriculum"
    assert len(filter_cfg.rho_scales) * len(filter_cfg.beta_offsets_deg) == 28
    assert filter_cfg.mutual_path_near_weight == pytest.approx(1.50)
    assert filter_cfg.mutual_path_collision_weight == pytest.approx(900.0)
    assert not filter_cfg.hard_endpoint_near_filter
    assert not filter_cfg.hard_path_collision_filter
    assert not filter_cfg.hard_center_progress_filter
    assert not filter_cfg.safety_override_after_warmup
    assert filter_cfg.collision_override_after_warmup
    assert filter_cfg.apply_probability_end == pytest.approx(0.45)
    assert filter_cfg.score_scale_end == pytest.approx(0.60)
    assert cfg.safety.collision_distance < cfg.success_thresholds.min_pairwise_distance
    assert cfg.success_thresholds.min_pairwise_distance < cfg.success_thresholds.dmax


def test_exp024_warmup_step_reports_mutual_path_metrics() -> None:
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
    assert not action_filter["applied"].any()
    assert action_filter["raw_mutual_path_collision_violation"].shape == (
        cfg.simulation.num_envs,
        cfg.task.n_agents,
    )
    assert action_filter["mutual_path_collision_violation"].shape == (
        cfg.simulation.num_envs,
        cfg.task.n_agents,
    )
    for key, value in action_filter.items():
        if isinstance(value, torch.Tensor) and value.is_floating_point():
            assert torch.isfinite(value).all(), key


def test_exp024_eval_progress_sets_mutual_filter_schedule() -> None:
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
    assert action_filter["apply_probability"] == pytest.approx(0.45)
    assert action_filter["score_scale"] == pytest.approx(0.60)
    assert "raw_mutual_path_collision_violation" in action_filter
    assert "mutual_path_collision_violation" in action_filter


def test_success_progress_long_prefers_high_success_late_candidate() -> None:
    early_safe = {
        "candidate_timestep": 2048,
        "dmax_reduction_ratio": 0.2116,
        "success_rate": 0.2402,
        "collision_rate": 0.0381,
        "timeout_rate": 0.7266,
    }
    late_progress = {
        "candidate_timestep": 10240,
        "dmax_reduction_ratio": 0.1397,
        "success_rate": 0.8398,
        "collision_rate": 0.0674,
        "timeout_rate": 0.0947,
    }

    best = min(
        [early_safe, late_progress],
        key=success_progress_long_checkpoint_rank,
    )

    assert best is late_progress


def test_success_progress_long_strict_pass_still_wins() -> None:
    strict_pass = {
        "candidate_timestep": 2048,
        "dmax_reduction_ratio": 0.19,
        "success_rate": 0.91,
        "collision_rate": 0.01,
        "timeout_rate": 0.0,
    }
    late_progress = {
        "candidate_timestep": 10240,
        "dmax_reduction_ratio": 0.14,
        "success_rate": 0.84,
        "collision_rate": 0.067,
        "timeout_rate": 0.095,
    }

    best = min([late_progress, strict_pass], key=success_progress_long_checkpoint_rank)

    assert best is strict_pass
