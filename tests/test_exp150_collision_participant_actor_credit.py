from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from _common import cfg_from_experiment, load_yaml  # noqa: E402
from train_skrl_mappo import (  # noqa: E402
    collision_participant_centered_credit,
    install_actor_credit_rewards,
)
from run_exp150_collision_participant_credit_screen import component_gate  # noqa: E402
from lunar_rover_tasks.tasks.multi_rover_gathering.gathering_env import (  # noqa: E402
    MultiRoverGatheringSKRLEnv,
)


CONFIG = ROOT / "configs" / "experiment" / "exp150_collision_participant_actor_credit.yaml"
BASE_CONFIG = (
    ROOT
    / "configs"
    / "experiment"
    / "exp148_decentralized_b0_trajectory_time_consistent.yaml"
)


def test_exp150_changes_only_training_actor_credit_semantics() -> None:
    raw = load_yaml(CONFIG)
    cfg = cfg_from_experiment(CONFIG)
    base = cfg_from_experiment(BASE_CONFIG)
    algorithm = raw["algorithm"]
    assert raw["experiment"]["name"] == "exp150_collision_participant_actor_credit"
    assert algorithm["actor_credit_assignment"] == "collision_participant_centered"
    assert algorithm["actor_credit_scale"] == pytest.approx(0.25)
    assert algorithm["actor_credit_trace_lambda"] == pytest.approx(0.95)
    assert algorithm["actor_credit_gradient_mode"] == "additive_advantage"
    assert algorithm["collision_constraint_enabled"] is False
    assert algorithm["bc_updates"] == 0
    assert algorithm["init_checkpoint"] is None
    for section in (
        "observation",
        "reward_coefficients",
        "reward_weights",
        "safety",
        "success_thresholds",
        "planner",
        "trajectory_generator",
        "low_level_control",
    ):
        assert asdict(getattr(cfg, section)) == asdict(getattr(base, section))
    assert cfg.actor_obs_dim == 101


def test_single_collision_pair_credit_is_exactly_zero_sum() -> None:
    positions = torch.tensor(
        [[[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [3.0, 0.0, 0.0], [6.0, 0.0, 0.0]]]
    )
    result = collision_participant_centered_credit(
        positions,
        torch.tensor([True]),
        collision_distance=0.28,
        collision_penalty=155.0,
    )
    assert torch.equal(result["participants"], torch.tensor([[True, True, False, False]]))
    assert torch.allclose(result["policy"], torch.tensor([[-155.0, -155.0, 155.0, 155.0]]))
    assert torch.allclose(result["allocated"], torch.tensor([[-310.0, -310.0, 0.0, 0.0]]))
    assert float(result["zero_sum_error"].max()) == 0.0
    assert float(result["allocation_mean_error"].max()) == 0.0


def test_three_participant_collision_rescales_and_noncollision_is_zero() -> None:
    positions = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0], [4.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0], [6.0, 0.0, 0.0]],
        ]
    )
    result = collision_participant_centered_credit(
        positions,
        torch.tensor([True, False]),
        collision_distance=0.28,
        collision_penalty=155.0,
    )
    assert result["participant_count"].tolist() == [3, 0]
    assert torch.allclose(result["policy"].sum(dim=1), torch.zeros(2), atol=1.0e-5)
    assert torch.equal(result["policy"][1], torch.zeros(4))
    assert result["allocated"][0, 0] == pytest.approx(-155.0 * 4.0 / 3.0)
    assert result["allocated"][0, 3] == pytest.approx(0.0)


def test_collision_without_actual_participant_is_rejected() -> None:
    positions = torch.tensor(
        [[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0], [6.0, 0.0, 0.0]]]
    )
    with pytest.raises(RuntimeError, match="no actual participant"):
        collision_participant_centered_credit(
            positions,
            torch.tensor([True]),
            collision_distance=0.28,
            collision_penalty=155.0,
        )


def test_credit_wrapper_preserves_environment_reward_on_noncollision_step() -> None:
    cfg = cfg_from_experiment(CONFIG)
    cfg.simulation.device = "cpu"
    cfg.simulation.num_envs = 8
    env = MultiRoverGatheringSKRLEnv(cfg)
    install_actor_credit_rewards(env, assignment="collision_participant_centered")
    actions = {
        agent: torch.zeros((cfg.simulation.num_envs, 2))
        for agent in env.possible_agents
    }
    _, rewards, _, _, info = env.step(actions)
    reward_matrix = torch.stack([rewards[agent] for agent in env.possible_agents], dim=1)
    credit = info["actor_credit"]
    assert torch.equal(credit["policy"], torch.zeros_like(credit["policy"]))
    assert credit["policy_is_step_zero_sum"] is True
    assert torch.allclose(reward_matrix, reward_matrix[:, :1].expand_as(reward_matrix))
    assert torch.allclose(reward_matrix.mean(dim=1), info["reward_terms"].total)
    assert float(credit["team_reward_preservation_error"].max()) == 0.0


def test_exp150_component_gate_requires_original_b0_and_credit_invariants() -> None:
    summary = {
        "status": "ok",
        "training_diagnostics": {
            "policy_parameters_finite": True,
            "policy_parameter_delta_l2": 1.0,
            "neighbor_encoder_parameter_delta_l2": 1.0,
            "terrain_encoder_parameter_delta_l2": 1.0,
            "post_training_action_std": 0.1,
            "last_actor_credit_std": 1.0,
            "actor_credit_assignment": "collision_participant_centered",
            "collision_constraint_enabled": False,
        },
        "final_eval": {
            "dmax_reduction_ratio": 0.2,
            "success_rate": 0.1,
            "collision_rate": 0.08,
            "timeout_rate": 0.02,
        },
    }
    telemetry = []
    for index, dmax in enumerate((10.0, 10.0, 9.0, 8.0, 6.5, 6.4, 6.3, 6.2)):
        telemetry.append(
            {
                "phase": "train",
                "dmax_mean": dmax,
                "nan_flag": False,
                "success_done": 1 if index == 7 else 0,
                "actor_credit_active_rate": 0.01,
                "actor_credit_policy_step_sum_abs_max": 0.0,
                "actor_credit_team_reward_preservation_error": 0.0,
                "actor_credit_source_reconstruction_error": 0.0,
                "actor_credit_allocation_mean_error": 0.0,
            }
        )
    terrain = {
        "action_mse_normal_vs_zero_terrain": 0.03,
        "path_risk_reduction_fraction": 0.06,
    }
    repeated = {
        "per_seed": {
            "30023": {"failed_repeated_event_count": {"median": 3.0}},
            "31023": {"failed_repeated_event_count": {"median": 4.0}},
        }
    }
    result = component_gate(summary, telemetry, terrain, repeated)
    assert result["passed"]
    assert result["b0_gate_passed"]
    telemetry[-1]["actor_credit_policy_step_sum_abs_max"] = 2.0e-5
    failed = component_gate(summary, telemetry, terrain, repeated)
    assert not failed["passed"]
    assert not failed["checks"]["credit_step_zero_sum"]
