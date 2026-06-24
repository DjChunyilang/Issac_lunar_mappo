from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from _common import cfg_from_experiment, load_yaml  # noqa: E402
from run_exp015_skrl_training import (  # noqa: E402
    SCREEN_THRESHOLDS,
    STRICT_THRESHOLDS,
    _suite_payload,
    stage_acceptance,
)


CONFIG = ROOT / "configs/experiment/exp015_skrl_weak_warmup_medium_soft.yaml"


def _summary(
    *,
    dmax: float = 0.25,
    success: float = 0.60,
    collision: float = 0.02,
    timeout: float = 0.40,
) -> dict:
    return {
        "observation_schema_version": "ego_v3_local_terrain_grid",
        "actor_obs_dim": 86,
        "critic_state_dim": 54,
        "timesteps": 1024,
        "env_steps": 2_097_152,
        "final_eval": {
            "dmax_reduction_ratio": dmax,
            "success_rate": success,
            "collision_rate": collision,
            "timeout_rate": timeout,
        },
        "training_diagnostics": {
            "policy_parameter_delta_l2": 0.1,
            "terrain_input_weight_delta_l2": 0.01,
            "bc_parameter_delta_l2": 0.2,
            "post_training_action_std": 0.05,
        },
    }


def test_exp015_config_matches_requested_training_contract() -> None:
    raw = load_yaml(CONFIG)
    cfg = cfg_from_experiment(CONFIG)
    exp = raw["experiment"]
    algo = raw["algorithm"]

    assert exp["num_envs"] == 2048
    assert exp["rollout_steps"] == 128
    assert exp["checkpoint_interval"] == 512
    assert cfg.simulation.max_episode_steps == 220
    assert cfg.planner.rho_max == pytest.approx(1.2)
    assert cfg.planner.beta_max == pytest.approx(0.78539816339)
    assert cfg.observation.schema_version == "ego_v3_local_terrain_grid"
    assert cfg.actor_obs_dim == 86
    assert cfg.critic_state_dim == 54
    assert algo["bc_updates"] == 20
    assert algo["bc_batch_size"] == 8192
    assert algo["bc_learning_rate"] == pytest.approx(1.0e-3)
    assert cfg.terrain.amplitude == pytest.approx(0.08)
    assert cfg.terrain.crater_min_radius == pytest.approx(0.40)
    assert cfg.terrain.crater_max_radius == pytest.approx(1.20)
    assert cfg.terrain.crater_depth_to_diameter == pytest.approx(0.10)
    assert cfg.terrain.crater_rim_height_to_diameter == pytest.approx(0.020)
    assert cfg.terrain.traversability_slope_scale == pytest.approx(0.55)
    assert cfg.terrain.slope_speed_scale == pytest.approx(0.90)
    assert cfg.terrain.min_speed_scale == pytest.approx(0.30)


def test_exp015_budget_uses_vector_timesteps() -> None:
    assert 1024 * 2048 == 2_097_152
    assert 4096 * 2048 == 8_388_608


def test_screen_gate_requires_metrics_and_training_signal() -> None:
    passing = stage_acceptance(_summary(), thresholds=SCREEN_THRESHOLDS)
    assert passing["passed"]

    no_terrain_update = _summary()
    no_terrain_update["training_diagnostics"]["terrain_input_weight_delta_l2"] = 0.0
    assert not stage_acceptance(no_terrain_update, thresholds=SCREEN_THRESHOLDS)["passed"]

    too_many_collisions = _summary(collision=0.031)
    assert not stage_acceptance(too_many_collisions, thresholds=SCREEN_THRESHOLDS)["passed"]


def test_formal_gate_uses_strict_thresholds_and_single_seed_language() -> None:
    formal_summary = _summary(dmax=0.19, success=0.91, collision=0.01, timeout=0.0)
    formal_summary["timesteps"] = 4096
    formal_summary["env_steps"] = 8_388_608
    screen_summary = _summary()
    screen_gate = stage_acceptance(screen_summary, thresholds=SCREEN_THRESHOLDS)
    formal_gate = stage_acceptance(formal_summary, thresholds=STRICT_THRESHOLDS)
    suite = _suite_payload(
        screen_summary=screen_summary,
        screen_gate=screen_gate,
        formal_summary=formal_summary,
        formal_gate=formal_gate,
    )

    assert formal_gate["passed"]
    assert suite["strict_passed"]
    assert suite["status"] == "single_seed_candidate"
    assert "single-seed candidate" in suite["conclusion"]
