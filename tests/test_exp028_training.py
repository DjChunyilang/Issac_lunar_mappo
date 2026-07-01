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


CONFIG = ROOT / "configs/experiment/exp028_randomized_terrain_hold_reward.yaml"


def test_exp028_config_keeps_dense_mutual_filter_and_strengthens_hold_reward() -> None:
    raw = load_yaml(CONFIG)
    cfg = cfg_from_experiment(CONFIG)
    filter_cfg = cfg.planner.subgoal_filter
    coeff = cfg.reward_coefficients

    assert raw["experiment"]["name"] == "exp028_randomized_terrain_hold_reward"
    assert raw["algorithm"]["mode"] == "pure_rl"
    assert raw["algorithm"]["update_mode"] == "shared_joint"
    assert raw["algorithm"]["bc_updates"] == 0
    assert filter_cfg.enabled
    assert filter_cfg.mode == "terrain_safe_candidate_mutual_progress_curriculum"
    assert filter_cfg.path_samples == 9
    assert len(filter_cfg.rho_scales) * len(filter_cfg.beta_offsets_deg) == 28
    assert filter_cfg.rho_scales == pytest.approx([0.60, 0.80, 1.0, 1.08])
    assert filter_cfg.hold_zone_dmax_multiplier == pytest.approx(0.0)
    assert filter_cfg.hold_zone_rho_weight == pytest.approx(0.0)
    assert coeff.success_hold_step == pytest.approx(4.0)
    assert coeff.success_bonus == pytest.approx(45.0)
    assert coeff.timeout_penalty == pytest.approx(18.0)
    assert coeff.inter_agent_collision == pytest.approx(90.0)


def test_exp028_success_hold_reward_is_stronger_than_exp025() -> None:
    cfg = cfg_from_experiment(CONFIG)
    exp025 = cfg_from_experiment(
        ROOT / "configs/experiment/exp025_randomized_terrain_dense_mutual_filter.yaml"
    )

    assert cfg.reward_coefficients.success_hold_step > exp025.reward_coefficients.success_hold_step
    assert cfg.reward_coefficients.timeout_penalty > exp025.reward_coefficients.timeout_penalty
    assert cfg.reward_coefficients.inter_agent_collision == pytest.approx(
        exp025.reward_coefficients.inter_agent_collision
    )


def test_exp028_step_reports_finite_dense_mutual_metrics() -> None:
    cfg = cfg_from_experiment(CONFIG)
    cfg.simulation.device = "cpu"
    cfg.simulation.num_envs = 4
    env = MultiRoverGatheringCore(cfg)

    out = env.step(torch.zeros(cfg.simulation.num_envs, cfg.task.n_agents, 2))
    action_filter = out.info["action_filter"]

    assert action_filter["enabled"]
    assert action_filter["candidate_count"] == 28
    assert action_filter["raw_mutual_path_collision_violation"].shape == (
        cfg.simulation.num_envs,
        cfg.task.n_agents,
    )
    for key, value in action_filter.items():
        if isinstance(value, torch.Tensor) and value.is_floating_point():
            assert torch.isfinite(value).all(), key
