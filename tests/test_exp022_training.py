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


CONFIG = ROOT / "configs/experiment/exp022_randomized_terrain_endpoint_safety_filter.yaml"


def test_exp022_config_enables_constrained_curriculum_filter_contract() -> None:
    raw = load_yaml(CONFIG)
    cfg = cfg_from_experiment(CONFIG)
    filter_cfg = cfg.planner.subgoal_filter

    assert raw["experiment"]["name"] == "exp022_randomized_terrain_endpoint_safety_filter"
    assert raw["algorithm"]["mode"] == "pure_rl"
    assert raw["algorithm"]["update_mode"] == "shared_joint"
    assert raw["algorithm"]["bc_updates"] == 0
    assert filter_cfg.enabled
    assert filter_cfg.mode == "terrain_safe_candidate_constrained_curriculum"
    assert filter_cfg.rho_scales == pytest.approx([0.45, 0.70, 0.90, 1.0])
    assert filter_cfg.beta_offsets_deg == pytest.approx(
        [-45.0, -30.0, -15.0, 0.0, 15.0, 30.0, 45.0]
    )
    assert len(filter_cfg.rho_scales) * len(filter_cfg.beta_offsets_deg) == 28
    assert filter_cfg.hard_endpoint_near_filter
    assert filter_cfg.hard_path_collision_filter
    assert filter_cfg.hard_center_progress_filter
    assert filter_cfg.safety_override_after_warmup
    assert filter_cfg.endpoint_safe_distance == pytest.approx(0.50)
    assert filter_cfg.path_safe_distance == pytest.approx(0.42)
    assert filter_cfg.apply_probability_end == pytest.approx(0.75)
    assert cfg.safety.collision_distance < cfg.success_thresholds.min_pairwise_distance
    assert cfg.success_thresholds.min_pairwise_distance < cfg.success_thresholds.dmax
    assert cfg.safety.near_distance == pytest.approx(0.95)
    assert cfg.reward_coefficients.near_distance == pytest.approx(8.0)
    assert cfg.reward_coefficients.inter_agent_collision == pytest.approx(120.0)
    assert cfg.reward_coefficients.failure_penalty == pytest.approx(60.0)


def test_exp022_step_reports_constrained_filter_metrics() -> None:
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
    assert action_filter["score_scale"] == pytest.approx(0.15)
    assert not action_filter["applied"].any()
    assert action_filter["path_collision_violation"].shape == (
        cfg.simulation.num_envs,
        cfg.task.n_agents,
    )
    assert action_filter["raw_path_collision_violation"].shape == (
        cfg.simulation.num_envs,
        cfg.task.n_agents,
    )
    assert action_filter["candidate_feasible"].shape == (
        cfg.simulation.num_envs,
        cfg.task.n_agents,
    )
    assert isinstance(action_filter["feasible_fraction"], float)
    assert isinstance(action_filter["safety_override_fraction"], float)
    assert action_filter["candidate_index_histogram"].shape == (28,)
    for key, value in action_filter.items():
        if isinstance(value, torch.Tensor) and value.is_floating_point():
            assert torch.isfinite(value).all(), key


def test_exp022_eval_progress_enables_safety_override_telemetry() -> None:
    cfg = cfg_from_experiment(CONFIG)
    cfg.simulation.device = "cpu"
    cfg.simulation.num_envs = 2
    cfg.planner.subgoal_filter.progress_timestep_override = 6144
    env = MultiRoverGatheringCore(cfg)

    action = torch.zeros(cfg.simulation.num_envs, cfg.task.n_agents, 2)
    out = env.step(action)
    action_filter = out.info["action_filter"]

    assert action_filter["schedule_progress_step"] == 6144
    assert action_filter["apply_probability"] == pytest.approx(0.75)
    assert action_filter["score_scale"] == pytest.approx(0.75)
    assert 0.0 <= action_filter["feasible_fraction"] <= 1.0
    assert 0.0 <= action_filter["safety_override_fraction"] <= 1.0
