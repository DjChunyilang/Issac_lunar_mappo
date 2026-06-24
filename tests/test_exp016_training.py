from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from _common import cfg_from_experiment, load_yaml  # noqa: E402
from run_exp016_shared_mappo_training import (  # noqa: E402
    SCREEN_THRESHOLDS,
    probe_acceptance,
    stage_acceptance,
)
from train_skrl_mappo import candidate_eval_seed, final_eval_seed  # noqa: E402


CONFIG = ROOT / "configs/experiment/exp016_shared_mappo_comm12.yaml"


def test_exp016_config_contract() -> None:
    raw = load_yaml(CONFIG)
    cfg = cfg_from_experiment(CONFIG)
    exp = raw["experiment"]
    algo = raw["algorithm"]

    assert exp["num_envs"] == 2048
    assert exp["rollout_steps"] == 64
    assert exp["checkpoint_interval"] == 256
    assert cfg.observation.communication_radius == pytest.approx(12.0)
    assert cfg.safety.near_distance == pytest.approx(0.75)
    assert cfg.reward_coefficients.success_hold_step == pytest.approx(1.0)
    assert cfg.reward_coefficients.timeout_penalty == pytest.approx(15.0)
    assert algo["update_mode"] == "shared_joint"
    assert algo["initial_log_std"] == pytest.approx(-1.0)
    assert algo["entropy_loss_scale"] == pytest.approx(0.002)
    assert algo["entropy_loss_scale_end"] == pytest.approx(0.0005)
    assert algo["bc_updates"] == 100
    assert algo["teacher_mode"] == "visible_local_centroid"
    assert algo["teacher_stop_radius"] == pytest.approx(0.54)
    assert algo["teacher_max_rho"] == pytest.approx(0.8)
    assert algo["bc_min_nearest_distance"] == pytest.approx(0.32)


def test_candidate_eval_seed_is_fixed_and_final_seed_is_independent() -> None:
    assert candidate_eval_seed(23, 1000) == 1023
    assert candidate_eval_seed(23, 1000) == candidate_eval_seed(23, 1000)
    assert final_eval_seed(23, 1000) == 11023


def test_probe_gate_requires_exact_joint_update_contract() -> None:
    summary = {
        "status": "ok",
        "update_mode": "shared_joint",
        "training_diagnostics": {
            "optimizer_count": 1,
            "joint_update_count": 2,
            "critic_update_count": 2,
            "policy_parameter_delta_l2": 0.1,
            "terrain_input_weight_delta_l2": 0.01,
            "post_training_action_std": 0.2,
        },
    }
    assert probe_acceptance(summary)["passed"]
    summary["training_diagnostics"]["optimizer_count"] = 4
    assert not probe_acceptance(summary)["passed"]


def test_exp016_screen_gate_includes_interface_and_communication_checks() -> None:
    summary = {
        "observation_schema_version": "ego_v3_local_terrain_grid",
        "actor_obs_dim": 86,
        "critic_state_dim": 54,
        "communication_radius": 12.0,
        "final_eval": {
            "dmax_reduction_ratio": 0.25,
            "success_rate": 0.6,
            "collision_rate": 0.02,
            "timeout_rate": 0.4,
        },
        "training_diagnostics": {
            "policy_parameter_delta_l2": 0.1,
            "terrain_input_weight_delta_l2": 0.01,
            "post_training_action_std": 0.2,
        },
    }
    assert stage_acceptance(summary, SCREEN_THRESHOLDS)["passed"]
    summary["communication_radius"] = 6.0
    assert not stage_acceptance(summary, SCREEN_THRESHOLDS)["passed"]
