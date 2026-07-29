from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from _common import cfg_from_experiment  # noqa: E402
from _skrl_metadata import observation_interface_metadata  # noqa: E402
from evaluate_proxy_policy import evaluate_checkpoint  # noqa: E402
from physx_jackal_common import (  # noqa: E402
    JackalSkidSteerController,
    TrackingControllerCfg,
    TrackingControllerState,
    compute_chassis_servo_command,
    compute_tracking_command,
    generate_reference_path,
    nearest_path_index,
    tracking_acceptance,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.gathering_env import MultiRoverGatheringCore  # noqa: E402
from lunar_rover_tasks.tasks.multi_rover_gathering.gathering_env_cfg import make_debug_cfg  # noqa: E402
from lunar_rover_tasks.tasks.multi_rover_gathering.oracle import (  # noqa: E402
    compute_mean_oracle_distance,
)
from run_proxy_convergence_suite import build_strict_acceptance  # noqa: E402
from train import Actor, Critic  # noqa: E402
from train_proxy_convergence import (  # noqa: E402
    Rollout,
    _checkpoint_candidate_allowed,
    _is_better_checkpoint_candidate,
    _randomize_bc_state,
    ppo_update,
    run_behavior_cloning,
    scripted_gather_action,
    strict_acceptance,
)


def test_cfg_from_experiment_parses_reward_control_and_success_thresholds(tmp_path: Path) -> None:
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "experiment": {"num_envs": 3, "device": "cpu"},
                "low_level_control": {"max_linear_speed": 0.7},
                "reward": {
                    "weights": {"energy": 0.123},
                    "coefficients": {
                        "dmax_level": 0.25,
                        "success_hold_step": 1.75,
                        "success_bonus": 12.0,
                    },
                },
                "safety": {"near_distance": 0.95},
                "terrain": {
                    "type": "lunar_crater_proxy",
                    "dynamics_enabled": True,
                    "slope_speed_scale": 1.25,
                    "min_speed_scale": 0.25,
                    "crater_count": 5,
                    "crater_min_radius": 0.4,
                    "crater_max_radius": 1.1,
                    "crater_depth_to_diameter": 0.07,
                    "crater_rim_height_to_diameter": 0.02,
                    "crater_field_size": 8.0,
                    "crater_seed": 19,
                    "randomize_per_reset": True,
                    "random_translation_m": 1.5,
                    "random_yaw_rad": 3.14,
                    "amplitude_scale_min": 0.9,
                    "amplitude_scale_max": 1.1,
                    "crater_radius_scale_min": 0.85,
                    "crater_radius_scale_max": 1.15,
                    "crater_depth_scale_min": 0.8,
                    "crater_depth_scale_max": 1.2,
                },
                "success_thresholds": {"dmax": 0.9, "hold_steps": 4},
            }
        ),
        encoding="utf-8",
    )
    cfg = cfg_from_experiment(config_path)
    assert cfg.simulation.num_envs == 3
    assert cfg.low_level_control.max_linear_speed == 0.7
    assert cfg.reward_weights.energy == 0.123
    assert cfg.reward_coefficients.dmax_level == 0.25
    assert cfg.reward_coefficients.success_hold_step == 1.75
    assert cfg.reward_coefficients.success_bonus == 12.0
    assert cfg.safety.near_distance == 0.95
    assert cfg.terrain.type == "lunar_crater_proxy"
    assert cfg.terrain.dynamics_enabled is True
    assert cfg.terrain.slope_speed_scale == 1.25
    assert cfg.terrain.min_speed_scale == 0.25
    assert cfg.terrain.crater_count == 5
    assert cfg.terrain.crater_min_radius == 0.4
    assert cfg.terrain.crater_max_radius == 1.1
    assert cfg.terrain.crater_depth_to_diameter == 0.07
    assert cfg.terrain.crater_rim_height_to_diameter == 0.02
    assert cfg.terrain.crater_field_size == 8.0
    assert cfg.terrain.crater_seed == 19
    assert cfg.terrain.randomize_per_reset is True
    assert cfg.terrain.random_translation_m == 1.5
    assert cfg.terrain.random_yaw_rad == 3.14
    assert cfg.terrain.amplitude_scale_min == 0.9
    assert cfg.terrain.amplitude_scale_max == 1.1
    assert cfg.terrain.crater_radius_scale_min == 0.85
    assert cfg.terrain.crater_radius_scale_max == 1.15
    assert cfg.terrain.crater_depth_scale_min == 0.8
    assert cfg.terrain.crater_depth_scale_max == 1.2
    assert cfg.success_thresholds.dmax == 0.9
    assert cfg.success_thresholds.hold_steps == 4


def test_behavior_cloning_reduces_scripted_action_mse() -> None:
    cfg = make_debug_cfg(num_envs=4, device="cpu")
    env = MultiRoverGatheringCore(cfg)
    actor = Actor(cfg.actor_obs_dim)
    actor_obs, _ = env.get_observations()
    target = scripted_gather_action(env).reshape(-1, 2)
    obs = actor_obs.reshape(-1, actor_obs.shape[-1])
    with torch.no_grad():
        before = torch.nn.functional.mse_loss(actor(obs).mean, target)
    run_behavior_cloning(actor, cfg, steps=20, batch_size=64, learning_rate=5.0e-3)
    with torch.no_grad():
        after = torch.nn.functional.mse_loss(actor(obs).mean, target)
    assert after < before


def test_safety_aware_teacher_reduces_rho_near_centroid() -> None:
    cfg = make_debug_cfg(num_envs=1, device="cpu")
    env = MultiRoverGatheringCore(cfg)
    env.positions[0, :, :2] = torch.tensor(
        [[0.20, 0.0], [0.0, 0.20], [-0.20, 0.0], [0.0, -0.20]],
        dtype=torch.float32,
    )
    env.yaws.zero_()
    direct = scripted_gather_action(env, safety_aware=False)
    safe = scripted_gather_action(env, stop_radius=0.45, slow_distance=0.4, safety_aware=True)
    direct_rho = 0.5 * (direct[..., 0] + 1.0) * cfg.planner.rho_max
    safe_rho = 0.5 * (safe[..., 0] + 1.0) * cfg.planner.rho_max
    assert safe_rho.max() < direct_rho.max()


def test_bc_randomized_state_refreshes_terrain_aware_oracle() -> None:
    cfg = make_debug_cfg(num_envs=4, device="cpu")
    cfg.gather_point.coarse_grid_size = 3
    cfg.gather_point.refinement_grid_size = 3
    cfg.gather_point.refinement_levels = 0
    cfg.gather_point.flatness_rings = 1
    cfg.gather_point.flatness_samples_per_ring = 4
    cfg.gather_point.path_samples = 2
    env = MultiRoverGatheringCore(cfg)

    with patch.object(
        env,
        "refresh_oracle_point",
        wraps=env.refresh_oracle_point,
    ) as refresh_oracle:
        _randomize_bc_state(env)

    refresh_oracle.assert_called_once_with()
    actual_oracle = env.oracle_point.clone()
    actual_objective = env.oracle_search_objective.clone()
    expected = env.refresh_oracle_point()
    expected_prev_distance = compute_mean_oracle_distance(env.positions, expected.point)
    assert torch.allclose(actual_oracle, expected.point)
    assert torch.allclose(actual_objective, expected.objective)
    assert torch.allclose(env.prev_mean_oracle_distance, expected_prev_distance)


def test_evaluate_proxy_policy_outputs_finite_ratio(tmp_path: Path) -> None:
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "experiment": {"num_envs": 2, "device": "cpu"},
                "simulation": {"episode_length_s": 2.0},
                "planner": {
                    "subgoal_filter": {
                        "mode": "terrain_safe_candidate_hold_progress_curriculum",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    cfg = cfg_from_experiment(config_path)
    actor = Actor(cfg.actor_obs_dim)
    critic = Critic(cfg.critic_state_dim)
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "actor": actor.state_dict(),
            "critic": critic.state_dict(),
            "cfg": {},
            "metadata": observation_interface_metadata(cfg),
        },
        checkpoint_path,
    )
    result = evaluate_checkpoint(
        config_path,
        checkpoint_path,
        device="cpu",
        num_envs=2,
        steps=3,
        output=tmp_path / "eval.json",
        filter_progress_override=7,
    )
    assert result["status"] == "ok"
    assert result["filter_progress_override"] == 7
    assert result["filter_progress_timestep"] == 7
    assert result["initial_dmax"] > 0.0
    assert torch.isfinite(torch.tensor(result["dmax_reduction_ratio"]))
    assert "min_nearest_distance" in result
    assert "near_violation_rate" in result
    assert "collision_episode_ids" in result
    for key in (
        "dmax_ok_rate",
        "dispersion_ok_rate",
        "speed_ok_rate",
        "flatness_ok_rate",
        "gather_point_is_flat_rate",
        "instant_success_rate",
        "max_success_hold_count_mean",
        "final_success_hold_count_mean",
        "final_mean_speed",
        "final_gather_point_height_range_mean",
        "final_gather_point_max_slope_mean",
    ):
        assert key in result
        assert torch.isfinite(torch.tensor(result[key]))
    assert result["oracle_search"]["method"] == "terrain_aware_multiresolution"
    assert 0.0 <= result["oracle_search"]["feasible_rate"] <= 1.0
    assert torch.isfinite(torch.tensor(result["oracle_search"]["objective_mean"]))
    assert "hold_count_histogram" in result
    assert "timeout_episode_metrics" in result
    assert result["timeout_episode_metrics"]["count"] >= 0
    for key in (
        "mean_terrain_height",
        "terrain_height_range",
        "mean_roughness",
        "max_roughness",
        "min_traversability",
        "mean_terrain_speed_scale",
    ):
        assert key in result
        assert torch.isfinite(torch.tensor(result[key]))
    assert Path(result["artifact"]).exists()


def test_strict_acceptance_and_suite_summary() -> None:
    passing = {
        "dmax_reduction_ratio": 0.1,
        "success_rate": 0.95,
        "collision_rate": 0.0,
        "timeout_rate": 0.0,
        "phase": "ppo",
    }
    failing = {
        "dmax_reduction_ratio": 0.3,
        "success_rate": 0.95,
        "collision_rate": 0.0,
        "timeout_rate": 0.0,
        "phase": "bc",
    }
    assert strict_acceptance(passing)["passed"]
    assert not strict_acceptance(failing)["passed"]
    assert strict_acceptance(passing, required_phase="ppo")["passed"]
    phase_fail = {**passing, "phase": "bc"}
    assert not strict_acceptance(phase_fail, required_phase="ppo")["passed"]
    suite = build_strict_acceptance(
        [
            {
                "mode": "bc_ppo",
                "seed": 23,
                "run_name": "bc_ppo_seed_23",
                "checkpoint_path": "a.pt",
                "best_metrics": passing,
                "strict_acceptance": strict_acceptance(passing),
            },
            {
                "mode": "bc_ppo",
                "seed": 31,
                "run_name": "bc_ppo_seed_31",
                "checkpoint_path": "b.pt",
                "best_metrics": passing,
                "strict_acceptance": strict_acceptance(passing),
            },
        ]
    )
    assert suite["passed"]
    assert len(suite["seeds"]) == 2


def test_jackal_skid_steer_controller_maps_and_clips_wheels() -> None:
    controller = JackalSkidSteerController(
        wheel_radius=0.1,
        track_width=0.4,
        max_linear_speed=1.0,
        max_angular_speed=2.0,
        max_wheel_speed=8.0,
    )

    straight = controller.forward([0.5, 0.0])
    turn = controller.forward([0.0, 1.0])
    arc = controller.forward([0.4, -0.5])
    clipped = controller.forward([10.0, 10.0])

    assert straight.tolist() == [5.0, 5.0, 5.0, 5.0]
    assert turn.tolist() == [-2.0, 2.0, -2.0, 2.0]
    assert arc[0] > arc[1]
    assert turn[2] == turn[0]
    assert turn[3] == turn[1]
    assert float(abs(clipped).max()) <= 8.0


def test_reference_paths_are_finite_and_monotonic() -> None:
    for profile in ("straight", "circle", "sine", "double_lane_change"):
        path = generate_reference_path(profile, samples=32)
        assert path.points_xy.shape == (32, 2)
        assert path.yaws.shape == (32,)
        assert path.cumulative_s.shape == (32,)
        assert torch.isfinite(torch.tensor(path.points_xy)).all()
        assert torch.isfinite(torch.tensor(path.yaws)).all()
        assert torch.isfinite(torch.tensor(path.cumulative_s)).all()
        assert torch.all(torch.diff(torch.tensor(path.cumulative_s)) >= 0.0)
        assert path.length_m > 0.0


def test_double_lane_change_path_shifts_out_and_back() -> None:
    path = generate_reference_path("double_lane_change", samples=128)

    assert abs(float(path.points_xy[0, 1])) < 1.0e-9
    assert abs(float(path.points_xy[-1, 1])) < 1.0e-9
    assert float(path.points_xy[:, 1].max()) > 0.45
    assert path.length_m > 6.0


def test_circular_path_progress_search_does_not_wrap_to_finish() -> None:
    path = generate_reference_path("circle", samples=128)
    near_start = path.points_xy[0] + torch.tensor([0.01, 0.0]).numpy()

    limited_idx = nearest_path_index(path, near_start, start_index=0, end_index=32)

    assert limited_idx < 32


def test_stanley_keeps_terminal_crawl_before_endpoint() -> None:
    path = generate_reference_path("straight", samples=128)
    cfg = TrackingControllerCfg(
        mode="stanley_pid",
        target_speed_mps=0.25,
        speed_kp=0.0,
        yaw_rate_kp=0.0,
        max_linear_accel_mps2=100.0,
        max_angular_accel_radps2=100.0,
    )

    _, details = compute_tracking_command(
        path,
        path.points_xy[-5],
        float(path.yaws[-5]),
        cfg,
        progress_index=len(path.points_xy) - 8,
        controller_state=TrackingControllerState(),
        dt=0.01,
    )

    assert 0.0 < details["reference_linear_mps"] <= cfg.target_speed_mps


def test_stanley_controller_corrects_cross_track_direction() -> None:
    path = generate_reference_path("straight", samples=64)
    cfg = TrackingControllerCfg(
        mode="stanley_pid",
        heading_gain=0.0,
        stanley_gain=1.0,
        softening_speed_mps=0.2,
        target_speed_mps=0.3,
        max_angular_accel_radps2=100.0,
    )

    left_command, left_details = compute_tracking_command(path, [-2.0, 0.2], 0.0, cfg, dt=0.2)
    right_command, right_details = compute_tracking_command(path, [-2.0, -0.2], 0.0, cfg, dt=0.2)

    assert left_details["signed_cross_track_m"] > 0.0
    assert left_command[1] < 0.0
    assert right_details["signed_cross_track_m"] < 0.0
    assert right_command[1] > 0.0


def test_stanley_speed_pid_limits_acceleration_and_resets() -> None:
    path = generate_reference_path("straight", samples=64)
    cfg = TrackingControllerCfg(
        mode="stanley_pid",
        target_speed_mps=0.5,
        speed_kp=0.5,
        speed_ki=0.1,
        max_linear_servo_correction_mps=0.05,
        max_linear_accel_mps2=1.0,
        max_angular_accel_radps2=100.0,
    )
    state = TrackingControllerState()

    first, first_details = compute_tracking_command(
        path,
        [-2.6, 0.0],
        0.0,
        cfg,
        controller_state=state,
        dt=0.2,
    )
    second, _ = compute_tracking_command(
        path,
        [-2.6, 0.0],
        0.0,
        cfg,
        controller_state=state,
        dt=0.2,
    )

    assert 0.0 < first[0] <= 0.2
    assert second[0] <= first[0] + 0.2 + 1.0e-9
    assert first_details["measured_speed_mps"] == 0.0
    assert abs(first_details["linear_servo_correction_mps"]) == 0.0
    state.reset()
    assert state.last_xy is None
    assert state.last_yaw is None
    assert state.previous_linear_mps == 0.0
    assert state.measured_yaw_rate_radps == 0.0


def test_chassis_servo_step_command_uses_measured_velocity_errors() -> None:
    cfg = TrackingControllerCfg(
        mode="stanley_pid",
        speed_kp=0.5,
        speed_ki=0.0,
        yaw_rate_kp=0.5,
        yaw_rate_ki=0.0,
        max_linear_servo_correction_mps=1.0,
        max_angular_servo_correction_radps=1.0,
        max_linear_accel_mps2=100.0,
        max_angular_accel_radps2=100.0,
    )
    state = TrackingControllerState()

    first_command, first_details = compute_chassis_servo_command(
        [0.4, 0.5],
        [0.0, 0.0],
        0.0,
        cfg,
        controller_state=state,
        dt=0.1,
    )
    second_command, second_details = compute_chassis_servo_command(
        [0.4, 0.5],
        [0.0, 0.0],
        0.0,
        cfg,
        controller_state=state,
        dt=0.1,
    )

    assert not first_details["measurement_valid"]
    assert first_command.tolist() == [0.4, 0.5]
    assert second_details["measurement_valid"]
    assert second_details["measured_speed_mps"] == 0.0
    assert second_details["measured_yaw_rate_radps"] == 0.0
    assert second_command[0] > second_details["reference_linear_mps"]
    assert second_command[1] > second_details["reference_angular_radps"]


def test_stanley_chassis_velocity_observation_projects_forward_speed_and_yaw_rate() -> None:
    path = generate_reference_path("straight", samples=64)
    cfg = TrackingControllerCfg(
        mode="stanley_pid",
        target_speed_mps=0.0,
        heading_gain=0.0,
        stanley_gain=0.0,
        speed_kp=0.0,
        yaw_rate_kp=0.0,
        max_linear_accel_mps2=100.0,
        max_angular_accel_radps2=100.0,
    )
    state = TrackingControllerState()

    _, first = compute_tracking_command(path, [-2.6, 0.0], 0.0, cfg, controller_state=state, dt=0.2)
    _, second = compute_tracking_command(path, [-2.4, 0.0], 0.2, cfg, controller_state=state, dt=0.2)

    assert first["measured_speed_mps"] == 0.0
    assert first["measured_yaw_rate_radps"] == 0.0
    assert abs(second["measured_speed_mps"] - 0.9800665778) < 1.0e-6
    assert abs(second["measured_yaw_rate_radps"] - 1.0) < 1.0e-6


def test_stanley_chassis_velocity_observation_ignores_sideways_motion() -> None:
    path = generate_reference_path("straight", samples=64)
    cfg = TrackingControllerCfg(
        mode="stanley_pid",
        target_speed_mps=0.0,
        heading_gain=0.0,
        stanley_gain=0.0,
        speed_kp=0.0,
        yaw_rate_kp=0.0,
        max_linear_accel_mps2=100.0,
        max_angular_accel_radps2=100.0,
    )
    state = TrackingControllerState()

    compute_tracking_command(path, [-2.6, 0.0], 0.0, cfg, controller_state=state, dt=0.2)
    _, details = compute_tracking_command(path, [-2.6, 0.2], 0.0, cfg, controller_state=state, dt=0.2)

    assert abs(details["measured_speed_mps"]) < 1.0e-9


def test_yaw_rate_servo_increases_and_decreases_angular_command() -> None:
    path = generate_reference_path("straight", samples=64)
    cfg = TrackingControllerCfg(
        mode="stanley_pid",
        target_speed_mps=0.0,
        heading_gain=1.0,
        stanley_gain=0.0,
        curvature_feedforward_gain=0.0,
        speed_kp=0.0,
        yaw_rate_kp=0.5,
        yaw_rate_ki=0.0,
        max_angular_servo_correction_radps=10.0,
        max_linear_accel_mps2=100.0,
        max_angular_accel_radps2=100.0,
    )

    slow_state = TrackingControllerState()
    compute_tracking_command(path, [-2.6, 0.0], -0.4, cfg, controller_state=slow_state, dt=0.2)
    slow_command, slow_details = compute_tracking_command(
        path,
        [-2.6, 0.0],
        -0.4,
        cfg,
        controller_state=slow_state,
        dt=0.2,
    )
    assert slow_details["measured_yaw_rate_radps"] == 0.0
    assert slow_command[1] > slow_details["reference_angular_radps"]

    fast_state = TrackingControllerState()
    compute_tracking_command(path, [-2.6, 0.0], -0.6, cfg, controller_state=fast_state, dt=0.2)
    fast_command, fast_details = compute_tracking_command(
        path,
        [-2.6, 0.0],
        0.0,
        cfg,
        controller_state=fast_state,
        dt=0.2,
    )
    assert fast_details["measured_yaw_rate_radps"] > fast_details["reference_angular_radps"]
    assert fast_command[1] < fast_details["reference_angular_radps"]


def test_yaw_rate_servo_correction_is_clipped() -> None:
    path = generate_reference_path("straight", samples=64)
    cfg = TrackingControllerCfg(
        mode="stanley_pid",
        target_speed_mps=0.0,
        heading_gain=2.0,
        stanley_gain=0.0,
        speed_kp=0.0,
        yaw_rate_kp=10.0,
        yaw_rate_ki=1.0,
        max_angular_servo_correction_radps=0.25,
        max_linear_accel_mps2=100.0,
        max_angular_accel_radps2=100.0,
    )
    state = TrackingControllerState()

    compute_tracking_command(path, [-2.6, 0.0], -1.0, cfg, controller_state=state, dt=0.2)
    _, details = compute_tracking_command(path, [-2.6, 0.0], -1.0, cfg, controller_state=state, dt=0.2)

    assert abs(details["angular_servo_correction_radps"]) <= 0.25 + 1.0e-9


def test_tracking_acceptance_uses_error_completion_and_tilt() -> None:
    passing = tracking_acceptance(
        {
            "rmse_cross_track_m": 0.07,
            "max_cross_track_m": 0.17,
            "path_completion_ratio": 0.995,
            "max_tilt_deg": 3.0,
        },
        "flat",
    )
    high_error = tracking_acceptance(
        {
            "rmse_cross_track_m": 0.09,
            "max_cross_track_m": 0.17,
            "path_completion_ratio": 0.995,
            "max_tilt_deg": 3.0,
        },
        "flat",
    )
    low_completion = tracking_acceptance(
        {
            "rmse_cross_track_m": 0.07,
            "max_cross_track_m": 0.17,
            "path_completion_ratio": 0.5,
            "max_tilt_deg": 3.0,
        },
        "flat",
    )

    assert passing["passed"]
    assert not high_error["passed"]
    assert not low_completion["passed"]


def test_strong_terrain_acceptance_is_diagnostic_only() -> None:
    result = tracking_acceptance(
        {
            "rmse_cross_track_m": 2.0,
            "max_cross_track_m": 3.0,
            "path_completion_ratio": 0.1,
            "max_tilt_deg": 40.0,
        },
        "strong_lunar_crater",
    )

    assert result["diagnostic_only"]
    assert result["passed"]
    assert not result["diagnostic_passed"]


def test_required_ppo_checkpoint_filter_rejects_bc_candidates() -> None:
    assert _checkpoint_candidate_allowed("ppo", "ppo", "ppo")
    assert _checkpoint_candidate_allowed("ppo", "all", "ppo")
    assert not _checkpoint_candidate_allowed("bc", "all", "ppo")
    assert not _checkpoint_candidate_allowed("initial", "all", "ppo")


def test_strict_checkpoint_candidate_is_not_replaced_by_lower_dmax_failure() -> None:
    strict_candidate = {
        "dmax_reduction_ratio": 0.16,
        "success_rate": 1.0,
        "collision_rate": 0.0,
        "timeout_rate": 0.0,
        "phase": "ppo",
    }
    lower_dmax_failure = {
        "dmax_reduction_ratio": 0.14,
        "success_rate": 0.99,
        "collision_rate": 0.0,
        "timeout_rate": 0.01,
        "phase": "ppo",
    }
    safer_strict_candidate = {
        "dmax_reduction_ratio": 0.17,
        "success_rate": 1.0,
        "collision_rate": 0.0,
        "timeout_rate": 0.0,
        "phase": "ppo",
    }
    assert _is_better_checkpoint_candidate(strict_candidate, lower_dmax_failure, "ppo")
    assert not _is_better_checkpoint_candidate(lower_dmax_failure, strict_candidate, "ppo")
    assert _is_better_checkpoint_candidate(safer_strict_candidate, strict_candidate, "ppo") is False


def test_ppo_update_outputs_health_metrics() -> None:
    cfg = make_debug_cfg(num_envs=2, device="cpu")
    actor = Actor(cfg.actor_obs_dim)
    critic = Critic(cfg.critic_state_dim)
    rollout_steps = 2
    obs = torch.randn(rollout_steps, cfg.simulation.num_envs, cfg.task.n_agents, cfg.actor_obs_dim)
    states = torch.randn(rollout_steps, cfg.simulation.num_envs, cfg.critic_state_dim)
    with torch.no_grad():
        flat_obs = obs.reshape(-1, cfg.actor_obs_dim)
        dist = actor(flat_obs)
        flat_actions = dist.sample()
        log_probs = dist.log_prob(flat_actions).sum(dim=-1).view(
            rollout_steps,
            cfg.simulation.num_envs,
            cfg.task.n_agents,
        )
    rollout = Rollout(
        obs=obs,
        states=states,
        actions=flat_actions.view(rollout_steps, cfg.simulation.num_envs, cfg.task.n_agents, 2),
        teacher_actions=torch.zeros(rollout_steps, cfg.simulation.num_envs, cfg.task.n_agents, 2),
        log_probs=log_probs,
        rewards=torch.randn(rollout_steps, cfg.simulation.num_envs),
        dones=torch.zeros(rollout_steps, cfg.simulation.num_envs),
        values=torch.randn(rollout_steps, cfg.simulation.num_envs),
        returns=torch.randn(rollout_steps, cfg.simulation.num_envs),
        advantages=torch.randn(rollout_steps, cfg.simulation.num_envs),
    )
    optimizer = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=1.0e-4)
    metrics = ppo_update(
        rollout,
        actor,
        critic,
        optimizer,
        clip_epsilon=0.2,
        ppo_epochs=1,
        mini_batches=2,
        value_loss_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        scripted_teacher_coef=0.1,
    )
    for key in ("approx_kl", "clip_fraction", "explained_variance", "scripted_teacher_loss"):
        assert key in metrics
        assert torch.isfinite(torch.tensor(metrics[key]))
    assert 0.0 <= metrics["clip_fraction"] <= 1.0
