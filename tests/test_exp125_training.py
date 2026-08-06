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
from analyze_exp125_credit_assignment import (  # noqa: E402
    pearson_correlation,
    rank_correlation,
)
from _skrl_metadata import (  # noqa: E402
    CheckpointCompatibilityError,
    validate_checkpoint_compatibility,
)
from train_skrl_mappo import (  # noqa: E402
    build_skrl_mappo_models,
    install_actor_credit_rewards,
    observation_slices_metadata,
)
from shared_policy_mappo import normalized_centered_credit_traces  # noqa: E402
from run_exp125_b0_screen import screen_acceptance  # noqa: E402
from summarize_exp125_b0_screens import summarize  # noqa: E402
from lunar_rover_tasks.tasks.multi_rover_gathering.gathering_env import (  # noqa: E402
    MultiRoverGatheringCore,
    MultiRoverGatheringSKRLEnv,
)


CONFIG = (
    ROOT
    / "configs"
    / "experiment"
    / "exp125_decentralized_tiered_b0_pure_rl.yaml"
)
REWARD_FOCUS_CONFIG = (
    ROOT
    / "configs"
    / "experiment"
    / "exp125_decentralized_tiered_b0_pure_rl_reward_focus.yaml"
)
RELATIVE_QUINTIC_CONFIG = (
    ROOT
    / "configs"
    / "experiment"
    / "exp125_decentralized_tiered_b0_pure_rl_relative_quintic.yaml"
)
RELATIVE_ONLY_CONFIG = (
    ROOT
    / "configs"
    / "experiment"
    / "exp125_decentralized_tiered_b0_pure_rl_relative_only.yaml"
)
CENTERED_CREDIT_CONFIG = (
    ROOT
    / "configs"
    / "experiment"
    / "exp126_decentralized_b0_centered_terrain_credit.yaml"
)


def test_exp125_is_strict_decentralized_pure_rl_baseline() -> None:
    raw = load_yaml(CONFIG)
    cfg = cfg_from_experiment(CONFIG)
    assert cfg.observation.schema_version == "ego_v8_decentralized_tiered"
    assert cfg.observation.communication_radius == pytest.approx(12.0)
    assert cfg.observation.effective_neighbor_dim == 12
    assert cfg.actor_obs_dim == 101
    assert cfg.simulation.episode_length_s == pytest.approx(96.0)
    assert cfg.simulation.max_episode_steps == 480
    assert not cfg.task.explicit_goal_in_execution
    assert not cfg.task.dynamic_terminal_slot_goal_enabled
    assert not cfg.planner.subgoal_filter.enabled
    assert not cfg.low_level_control.safety_projection_enabled
    assert not cfg.low_level_control.projection_directional_agent_scale
    assert not cfg.low_level_control.success_zone_damping_enabled
    assert not cfg.low_level_control.formation_center_correction_enabled
    assert not cfg.low_level_control.terminal_slot_capture_enabled
    assert not cfg.low_level_control.flat_geometry_capture_enabled
    assert raw["algorithm"]["bc_updates"] == 0
    assert raw["algorithm"]["init_checkpoint"] is None
    assert raw["reward"]["weights"]["flatness"] == pytest.approx(1.0)
    assert raw["reward"]["coefficients"]["centroid_flatness_progress"] > 0.0


def test_branched_v5_consumes_101_dim_slices() -> None:
    cfg = cfg_from_experiment(CONFIG)
    cfg.simulation.device = "cpu"
    cfg.simulation.num_envs = 2
    env = MultiRoverGatheringSKRLEnv(cfg)
    models = build_skrl_mappo_models(
        env,
        shared_actor=True,
        centralized_critic=True,
        actor_architecture="branched_v5",
        critic_architecture="structured_v1",
    )
    policy = models[env.possible_agents[0]]["policy"]
    observations, _ = env.core.get_observations()
    means, _ = policy.compute(
        {"observations": observations.reshape(-1, cfg.actor_obs_dim)},
        role="policy",
    )
    assert policy.neighbor_encoder[0].in_features == 36
    assert policy.trunk[0].in_features == 160
    assert means.shape == (cfg.simulation.num_envs * cfg.task.n_agents, 2)
    assert torch.isfinite(means).all()
    assert observation_slices_metadata(101)["neighbors"] == {
        "start": 10,
        "end": 46,
        "dim": 36,
    }


def test_old_checkpoint_dimensions_are_rejected() -> None:
    cfg = cfg_from_experiment(CONFIG)
    checkpoint = {
        "metadata": {
            "observation_schema_version": "ego_v3_local_terrain_grid",
            "actor_obs_dim": 86,
            "critic_state_dim": 54,
        }
    }
    with pytest.raises(CheckpointCompatibilityError, match="incompatible"):
        validate_checkpoint_compatibility(checkpoint, cfg)


def test_oracle_and_slots_do_not_change_execution_command() -> None:
    cfg_a = cfg_from_experiment(CONFIG)
    cfg_b = cfg_from_experiment(CONFIG)
    for cfg in (cfg_a, cfg_b):
        cfg.simulation.device = "cpu"
        cfg.simulation.num_envs = 2

    reference = MultiRoverGatheringCore(cfg_a)
    perturbed = MultiRoverGatheringCore(cfg_b)
    assert torch.equal(reference.positions, perturbed.positions)
    assert torch.equal(reference.yaws, perturbed.yaws)

    # These centralized quantities may affect training rewards and the critic,
    # but the v8 execution path must not consume them.
    perturbed.oracle_point.add_(37.0)
    perturbed.gather_slot_points.sub_(19.0)
    perturbed.execution_slot_points.mul_(0.0).add_(11.0)

    action = torch.tensor(
        [[[0.2, -0.3], [-0.1, 0.4], [0.5, 0.1], [-0.4, -0.2]]],
        dtype=torch.float32,
    ).expand(2, -1, -1).clone()
    reference_step = reference.step(action)
    perturbed_step = perturbed.step(action)

    assert torch.equal(
        reference_step.info["trajectory"].points,
        perturbed_step.info["trajectory"].points,
    )
    assert torch.equal(
        reference_step.info["control"].packed,
        perturbed_step.info["control"].packed,
    )
    assert torch.equal(reference.positions, perturbed.positions)
    assert torch.equal(reference.yaws, perturbed.yaws)


def test_b0_screen_gate_requires_all_declared_signals() -> None:
    summary = {
        "status": "ok",
        "training_diagnostics": {
            "policy_parameters_finite": True,
            "policy_parameter_delta_l2": 0.3,
            "neighbor_encoder_parameter_delta_l2": 0.1,
            "terrain_encoder_parameter_delta_l2": 0.2,
            "post_training_action_std": 0.15,
        },
        "final_eval": {
            "dmax_reduction_ratio": 0.4,
            "success_rate": 0.02,
            "collision_rate": 0.05,
            "timeout_rate": 0.93,
        },
    }
    telemetry = [
        {"phase": "train", "dmax_mean": value, "nan_flag": False}
        for value in (10.0, 10.0, 9.0, 8.0, 7.5, 7.2, 6.9, 6.8)
    ]
    terrain_contrast = {
        "action_mse_normal_vs_zero_terrain": 0.021,
        "path_risk_reduction_fraction": 0.051,
    }

    gate = screen_acceptance(summary, telemetry, terrain_contrast)
    assert gate["passed"]
    assert gate["checks"]["training_dmax_reduced_30pct"]

    terrain_contrast["action_mse_normal_vs_zero_terrain"] = 0.02
    failed = screen_acceptance(summary, telemetry, terrain_contrast)
    assert not failed["passed"]
    assert not failed["checks"]["terrain_action_mse_gt_0_02"]


def test_c2_screen_gate_requires_projection_invariants() -> None:
    summary = {
        "status": "ok",
        "training_diagnostics": {
            "policy_parameters_finite": True,
            "policy_parameter_delta_l2": 0.3,
            "neighbor_encoder_parameter_delta_l2": 0.1,
            "terrain_encoder_parameter_delta_l2": 0.2,
            "post_training_action_std": 0.15,
            "actor_credit_assignment": "terrain_relative_centered",
            "last_actor_credit_std": 1.0,
            "actor_credit_gradient_mode": "primary_projected_norm_cap",
            "last_actor_gradient_conflict_fraction": 0.45,
            "last_actor_gradient_projected_dot_min": -1.0e-9,
            "last_actor_gradient_combined_cosine_min": 0.971,
        },
        "final_eval": {"success_rate": 0.02, "collision_rate": 0.05},
    }
    telemetry = [
        {
            "phase": "train",
            "dmax_mean": value,
            "nan_flag": False,
            "actor_credit_team_reward_preservation_error": 0.0,
        }
        for value in (10.0, 10.0, 9.0, 8.0, 7.5, 7.2, 6.9, 6.8)
    ]
    contrast = {
        "action_mse_normal_vs_zero_terrain": 0.021,
        "path_risk_reduction_fraction": 0.051,
    }
    gate = screen_acceptance(summary, telemetry, contrast)
    assert gate["passed"]
    assert gate["checks"]["gradient_projection_active"]
    summary["training_diagnostics"][
        "last_actor_gradient_combined_cosine_min"
    ] = 0.969
    failed = screen_acceptance(summary, telemetry, contrast)
    assert not failed["passed"]
    assert not failed["checks"]["gradient_projection_primary_alignment"]


def test_exp125_reward_focus_only_reallocates_existing_reward_terms() -> None:
    raw = load_yaml(REWARD_FOCUS_CONFIG)
    cfg = cfg_from_experiment(REWARD_FOCUS_CONFIG)
    assert raw["algorithm"]["training_semantics"].endswith("reward_focus")
    assert cfg.observation.schema_version == "ego_v8_decentralized_tiered"
    assert cfg.actor_obs_dim == 101
    assert not cfg.planner.subgoal_filter.enabled
    assert not cfg.low_level_control.safety_projection_enabled
    assert cfg.reward_weights.terrain == pytest.approx(0.30)
    assert cfg.reward_coefficients.slope_cost == 0.0
    assert cfg.reward_coefficients.terrain_cost == 0.0
    assert cfg.reward_coefficients.terrain_speed_loss_cost == 0.0
    assert cfg.reward_coefficients.terrain_height_change_cost == 0.0
    assert cfg.reward_coefficients.path_terrain_mean_cost > 0.0
    assert cfg.reward_coefficients.path_terrain_max_cost > 0.0
    assert cfg.reward_coefficients.centroid_flatness_progress == pytest.approx(0.50)


def test_exp125_relative_quintic_keeps_execution_baseline_unchanged() -> None:
    raw = load_yaml(RELATIVE_QUINTIC_CONFIG)
    cfg = cfg_from_experiment(RELATIVE_QUINTIC_CONFIG)
    assert raw["algorithm"]["training_semantics"].endswith("relative_quintic")
    assert cfg.observation.schema_version == "ego_v8_decentralized_tiered"
    assert cfg.actor_obs_dim == 101
    assert not cfg.planner.subgoal_filter.enabled
    assert not cfg.low_level_control.safety_projection_enabled
    assert cfg.reward_coefficients.path_terrain_mean_cost == 0.0
    assert cfg.reward_coefficients.path_terrain_max_cost == 0.0
    assert cfg.reward_coefficients.path_terrain_relative_cost == pytest.approx(2.0)


def test_exp125_relative_only_removes_absolute_terrain_baseline() -> None:
    cfg = cfg_from_experiment(RELATIVE_ONLY_CONFIG)
    assert cfg.reward_weights.terrain == pytest.approx(1.0)
    assert cfg.reward_coefficients.path_terrain_relative_cost == pytest.approx(5.0)
    assert cfg.reward_coefficients.slope_cost == 0.0
    assert cfg.reward_coefficients.terrain_cost == 0.0
    assert cfg.reward_coefficients.subgoal_terrain_cost == 0.0
    assert cfg.reward_coefficients.path_height_change_cost == 0.0
    assert cfg.reward_coefficients.centroid_flatness_progress > 0.0
    assert not cfg.planner.subgoal_filter.enabled


def test_screen_comparison_preserves_stop_decision() -> None:
    records = [
        {
            "run": "baseline",
            "passed": False,
            "training_dmax_reduction": 0.28,
            "eval_dmax_reduction_ratio": 0.24,
            "eval_success_rate": 0.02,
            "terrain_path_risk_reduction_fraction": -0.01,
        },
        {
            "run": "relative",
            "passed": False,
            "training_dmax_reduction": 0.27,
            "eval_dmax_reduction_ratio": 0.20,
            "eval_success_rate": 0.05,
            "terrain_path_risk_reduction_fraction": 0.01,
        },
    ]
    comparison = summarize(records)
    assert comparison["completed_screen_count"] == 2
    assert comparison["passed_screen_count"] == 0
    assert comparison["decision"] == "stop_before_40m"
    assert comparison["best"]["eval_success_rate"]["run"] == "relative"


def test_credit_correlation_helpers_detect_direction_and_degeneracy() -> None:
    values = torch.tensor([3.0, 1.0, 4.0, 2.0])
    assert pearson_correlation(values, 2.0 * values) == pytest.approx(1.0)
    assert rank_correlation(values, -values) == pytest.approx(-1.0)
    assert pearson_correlation(values, torch.ones_like(values)) is None
    assert rank_correlation(values, torch.ones_like(values)) is None
    assert rank_correlation(
        torch.tensor([0.0, 0.0, 1.0, 1.0]),
        torch.tensor([1.0, 1.0, 2.0, 2.0]),
    ) == pytest.approx(1.0)


def test_exp126_changes_only_actor_credit_semantics() -> None:
    raw = load_yaml(CENTERED_CREDIT_CONFIG)
    cfg = cfg_from_experiment(CENTERED_CREDIT_CONFIG)
    assert cfg.observation.schema_version == "ego_v8_decentralized_tiered"
    assert cfg.actor_obs_dim == 101
    assert not cfg.planner.subgoal_filter.enabled
    assert not cfg.low_level_control.safety_projection_enabled
    assert raw["algorithm"]["bc_updates"] == 0
    assert raw["algorithm"]["init_checkpoint"] is None
    assert raw["algorithm"]["actor_credit_assignment"] == "terrain_relative_centered"
    assert raw["algorithm"]["actor_credit_scale"] == pytest.approx(0.25)
    assert raw["algorithm"]["actor_credit_trace_lambda"] == pytest.approx(0.95)


def test_centered_terrain_credit_preserves_team_reward() -> None:
    cfg = cfg_from_experiment(CENTERED_CREDIT_CONFIG)
    cfg.simulation.device = "cpu"
    cfg.simulation.num_envs = 4
    env = MultiRoverGatheringSKRLEnv(cfg)
    install_actor_credit_rewards(env, assignment="terrain_relative_centered")
    actions = {
        agent: torch.zeros((cfg.simulation.num_envs, 2))
        for agent in env.possible_agents
    }
    _, rewards, _, _, info = env.step(actions)
    reward_matrix = torch.stack([rewards[agent] for agent in env.possible_agents], dim=1)
    centered = info["actor_credit"]["centered"]
    assert torch.allclose(centered.sum(dim=1), torch.zeros(cfg.simulation.num_envs))
    assert torch.allclose(reward_matrix, reward_matrix[:, :1].expand_as(reward_matrix))
    assert torch.allclose(reward_matrix.mean(dim=1), info["reward_terms"].total)
    assert float(info["actor_credit"]["team_reward_preservation_error"].amax()) <= 1.0e-6


def test_centered_credit_trace_preserves_agent_allocation() -> None:
    credits = torch.tensor(
        [
            [[[1.0]], [[0.5]], [[-0.5]]],
            [[[-1.0]], [[-0.5]], [[0.5]]],
        ]
    )
    terminated = torch.zeros((3, 1, 1), dtype=torch.bool)
    truncated = torch.zeros_like(terminated)
    traces = normalized_centered_credit_traces(
        credits,
        terminated,
        truncated,
        discount_factor=0.99,
        trace_lambda=0.95,
        time_limit_bootstrap=False,
    )
    assert traces.shape == credits.shape
    assert torch.allclose(traces.sum(dim=0), torch.zeros_like(traces[0]), atol=1.0e-6)
    assert float(traces.std()) == pytest.approx(1.0, abs=1.0e-6)
