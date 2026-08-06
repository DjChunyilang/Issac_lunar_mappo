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
from run_exp140_local_near_credit_screen import component_gate  # noqa: E402
from shared_policy_mappo import normalized_agent_credit_traces  # noqa: E402
from train_skrl_mappo import install_actor_credit_rewards  # noqa: E402
from lunar_rover_tasks.tasks.multi_rover_gathering.gathering_env import (  # noqa: E402
    MultiRoverGatheringSKRLEnv,
)


CONFIG = (
    ROOT
    / "configs"
    / "experiment"
    / "exp140_agent_local_near_credit.yaml"
)


def test_exp140_changes_only_actor_credit_semantics() -> None:
    raw = load_yaml(CONFIG)
    cfg = cfg_from_experiment(CONFIG)
    assert raw["experiment"]["name"] == "exp140_agent_local_near_credit"
    assert raw["algorithm"]["actor_credit_assignment"] == "near_potential_local"
    assert raw["algorithm"]["actor_credit_scale"] == pytest.approx(0.25)
    assert raw["algorithm"]["actor_credit_trace_lambda"] == pytest.approx(0.95)
    assert raw["algorithm"]["actor_credit_gradient_mode"] == "additive_advantage"
    assert raw["algorithm"]["bc_updates"] == 0
    assert raw["algorithm"]["init_checkpoint"] is None
    assert cfg.actor_obs_dim == 101
    assert cfg.observation.schema_version == "ego_v8_decentralized_tiered"
    assert not cfg.planner.subgoal_filter.enabled
    assert not cfg.low_level_control.safety_projection_enabled


def test_agent_local_trace_is_jointly_normalized_without_step_centering() -> None:
    credits = torch.tensor(
        [
            [[[1.0]], [[0.5]], [[0.0]]],
            [[[1.0]], [[0.5]], [[0.0]]],
        ]
    )
    terminated = torch.zeros((3, 1, 1), dtype=torch.bool)
    truncated = torch.zeros_like(terminated)
    traces = normalized_agent_credit_traces(
        credits,
        terminated,
        truncated,
        discount_factor=0.99,
        trace_lambda=0.95,
        time_limit_bootstrap=False,
    )
    assert traces.shape == credits.shape
    assert float(traces.std()) == pytest.approx(1.0, abs=1.0e-6)
    # The two agents have identical responsibility and must retain identical
    # traces. Per-step cross-agent centering would erase this signal entirely.
    assert torch.allclose(traces[0], traces[1])
    assert not torch.allclose(traces, torch.zeros_like(traces))


def test_near_potential_local_credit_matches_existing_near_gap_and_preserves_reward() -> None:
    cfg = cfg_from_experiment(CONFIG)
    cfg.simulation.device = "cpu"
    cfg.simulation.num_envs = 8
    env = MultiRoverGatheringSKRLEnv(cfg)
    nearest_before = env.core.metrics.nearest_neighbor_distance.clone()
    install_actor_credit_rewards(env, assignment="near_potential_local")
    actions = {
        agent: torch.zeros((cfg.simulation.num_envs, 2))
        for agent in env.possible_agents
    }
    _, rewards, _, _, info = env.step(actions)
    nearest_after = info["metrics"].nearest_neighbor_distance
    near_distance = float(cfg.safety.near_distance)
    expected = -torch.relu(near_distance - nearest_after) + torch.relu(
        near_distance - nearest_before
    )
    credit = info["actor_credit"]
    reward_matrix = torch.stack([rewards[agent] for agent in env.possible_agents], dim=1)

    assert credit["assignment"] == "near_potential_local"
    assert credit["policy_is_step_zero_sum"] is False
    assert torch.allclose(credit["raw"], expected, atol=1.0e-7)
    assert torch.equal(credit["policy"], credit["raw"])
    assert torch.allclose(
        credit["centered"].sum(dim=1),
        torch.zeros(cfg.simulation.num_envs),
        atol=1.0e-7,
    )
    assert float(credit["source_reconstruction_error"].amax()) <= 1.0e-7
    assert torch.allclose(reward_matrix, reward_matrix[:, :1].expand_as(reward_matrix))
    assert torch.allclose(reward_matrix.mean(dim=1), info["reward_terms"].total)
    assert float(credit["team_reward_preservation_error"].amax()) == 0.0


def test_unknown_actor_credit_assignment_is_rejected() -> None:
    cfg = cfg_from_experiment(CONFIG)
    cfg.simulation.device = "cpu"
    cfg.simulation.num_envs = 2
    env = MultiRoverGatheringSKRLEnv(cfg)
    with pytest.raises(ValueError, match="actor_credit_assignment"):
        install_actor_credit_rewards(env, assignment="unknown")


def _passing_component_gate_inputs() -> tuple[dict, list[dict], dict, dict, dict]:
    diagnostics = {
        "policy_parameters_finite": True,
        "policy_parameter_delta_l2": 1.0,
        "neighbor_encoder_parameter_delta_l2": 1.0,
        "terrain_encoder_parameter_delta_l2": 1.0,
        "post_training_action_std": 0.1,
        "last_actor_credit_std": 1.0,
        "actor_credit_assignment": "near_potential_local",
    }
    summary = {
        "status": "ok",
        "training_diagnostics": diagnostics,
        "final_eval": {
            "dmax_reduction_ratio": 0.20,
            "success_rate": 0.04,
            "collision_rate": 0.06,
            "timeout_rate": 0.90,
        },
    }
    telemetry = []
    for index in range(8):
        telemetry.append(
            {
                "phase": "train",
                "dmax_mean": 5.0 if index < 2 else 3.5,
                "success_done": 1 if index == 7 else 0,
                "actor_credit_active_rate": 0.10,
                "actor_credit_team_reward_preservation_error": 0.0,
                "actor_credit_source_reconstruction_error": 0.0,
            }
        )
    repeated = {
        "per_seed": {
            "28023": {"failed_repeated_event_count": {"median": 13.0}},
            "29023": {"failed_repeated_event_count": {"median": 14.0}},
        }
    }
    baseline_summary = {
        "final_eval": {
            "dmax_reduction_ratio": 0.2047232687,
            "success_rate": 0.0517578125,
            "collision_rate": 0.0966796875,
            "timeout_rate": 0.8515625,
        }
    }
    baseline_repeated = {
        "per_seed": {
            "28023": {"failed_repeated_event_count": {"median": 17.0}},
            "29023": {"failed_repeated_event_count": {"median": 18.0}},
        }
    }
    return summary, telemetry, repeated, baseline_summary, baseline_repeated


def test_exp140_component_gate_requires_safety_improvement_and_preserves_gather() -> None:
    inputs = _passing_component_gate_inputs()
    result = component_gate(*inputs)
    assert result["passed"]
    assert result["checks"]["collision_reduced_30pct"]
    assert result["checks"]["every_seed_repeated_conflicts_reduced_20pct"]
    assert result["checks"]["success_drop_at_most_2pp"]
    assert result["thresholds"]["collision_rate"] == pytest.approx(
        0.0966796875 * 0.70
    )


def test_exp140_component_gate_stops_when_one_conflict_seed_does_not_improve() -> None:
    summary, telemetry, repeated, baseline_summary, baseline_repeated = (
        _passing_component_gate_inputs()
    )
    repeated["per_seed"]["29023"]["failed_repeated_event_count"]["median"] = 18.0
    result = component_gate(
        summary,
        telemetry,
        repeated,
        baseline_summary,
        baseline_repeated,
    )
    assert not result["passed"]
    assert not result["checks"]["every_seed_repeated_conflicts_reduced_20pct"]
