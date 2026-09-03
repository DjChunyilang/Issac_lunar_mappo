from __future__ import annotations

from dataclasses import fields

import torch

from lunar_rover_tasks.tasks.multi_rover_gathering.gathering_env_cfg import (
    MultiRoverGatheringEnvCfg,
    RewardCoefficientsCfg,
    RewardWeightsCfg,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.metrics import compute_team_metrics
from lunar_rover_tasks.tasks.multi_rover_gathering.reward import (
    RewardTerms,
    compute_centroid_flatness_cost,
    compute_centroid_flatness_reward,
    compute_gather_reward,
    compute_oracle_reward,
    compute_safety_reward,
    compute_terrain_reward,
    compute_terminal_reward,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.termination import DoneFlags


def test_gather_reward_positive_when_team_contracts() -> None:
    cfg = MultiRoverGatheringEnvCfg()
    previous = torch.tensor([[[-2.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, -2.0, 0.0], [0.0, 2.0, 0.0]]])
    current = 0.5 * previous
    velocities = torch.zeros(1, 4, 2)
    reward = compute_gather_reward(
        compute_team_metrics(previous, velocities),
        compute_team_metrics(current, velocities),
        cfg,
    )
    assert reward.item() > 0.0


def test_oracle_reward_positive_when_mean_distance_decreases() -> None:
    cfg = MultiRoverGatheringEnvCfg()
    positions = torch.tensor([[[0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [-0.5, 0.0, 0.0], [0.0, -0.5, 0.0]]])
    oracle = torch.zeros(1, 3)
    reward, mean_distance = compute_oracle_reward(positions, oracle, torch.tensor([2.0]), cfg)
    assert reward.item() > 0.0
    assert mean_distance.item() < 2.0


def test_oracle_reward_accepts_per_rover_execution_slots() -> None:
    cfg = MultiRoverGatheringEnvCfg()
    positions = torch.tensor(
        [[[0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [-0.5, 0.0, 0.0], [0.0, -0.5, 0.0]]]
    )
    slots = positions.clone()

    reward, mean_distance = compute_oracle_reward(
        positions,
        slots,
        torch.tensor([1.0]),
        cfg,
    )

    assert reward.item() > 0.0
    assert torch.allclose(mean_distance, torch.zeros_like(mean_distance))


def test_oracle_reward_masks_infeasible_environments_but_updates_distance() -> None:
    cfg = MultiRoverGatheringEnvCfg()
    positions = torch.tensor(
        [
            [[0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [-0.5, 0.0, 0.0], [0.0, -0.5, 0.0]],
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
        ]
    )
    oracle = torch.zeros(2, 3)
    previous_distance = torch.tensor([2.0, 2.0])

    reward, mean_distance = compute_oracle_reward(
        positions,
        oracle,
        previous_distance,
        cfg,
        oracle_feasible=torch.tensor([True, False]),
    )

    assert reward[0] > 0.0
    assert reward[1] == 0.0
    assert torch.allclose(mean_distance, torch.tensor([0.5, 1.0]))


def test_centroid_flatness_cost_matches_the_hard_gate_boundary() -> None:
    cfg = MultiRoverGatheringEnvCfg()
    height_range = torch.tensor([0.18, 0.10, 0.181, 0.10])
    max_slope = torch.tensor([0.20, 0.25, 0.20, 0.251])

    cost = compute_centroid_flatness_cost(height_range, max_slope, cfg)
    hard_gate = (
        (height_range <= cfg.gather_point.max_height_range)
        & (max_slope <= cfg.gather_point.max_slope)
    )

    assert torch.equal(cost <= 1.0, hard_gate)
    assert torch.all(cost >= 0.0)
    assert torch.all(cost <= 3.0)


def test_centroid_flatness_reward_tracks_progress_and_geometric_activation() -> None:
    cfg = MultiRoverGatheringEnvCfg()
    cfg.reward_coefficients.centroid_flatness_progress = 2.0
    cfg.reward_coefficients.centroid_flatness_excess = 0.02
    cfg.reward_coefficients.centroid_flatness_dmax_multiplier = 2.0
    previous = torch.tensor([1.5, 1.5, 1.5])
    current = torch.tensor([1.4, 1.6, 1.4])
    dmax = torch.tensor(
        [
            cfg.success_thresholds.dmax,
            cfg.success_thresholds.dmax,
            2.0 * cfg.success_thresholds.dmax,
        ]
    )

    reward, progress, activation = compute_centroid_flatness_reward(
        previous,
        current,
        dmax,
        dmax,
        cfg,
    )

    assert reward[0] > 0.0
    assert reward[1] < 0.0
    assert reward[2] == 0.0
    assert torch.allclose(progress[:2], previous[:2] - current[:2])
    assert progress[2] == 0.0
    assert torch.allclose(activation, torch.tensor([1.0, 1.0, 0.0]))


def test_default_centroid_flatness_shaping_is_behaviorally_disabled() -> None:
    cfg = MultiRoverGatheringEnvCfg()
    reward, _, _ = compute_centroid_flatness_reward(
        torch.tensor([2.0]),
        torch.tensor([1.5]),
        torch.tensor([cfg.success_thresholds.dmax]),
        torch.tensor([cfg.success_thresholds.dmax]),
        cfg,
    )

    assert cfg.reward_weights.flatness == 0.0
    assert torch.allclose(reward, torch.zeros_like(reward))


def test_centroid_flatness_potential_closes_activation_boundary_cycle() -> None:
    cfg = MultiRoverGatheringEnvCfg()
    cfg.reward_coefficients.centroid_flatness_progress = 2.0
    cfg.reward_coefficients.centroid_flatness_excess = 0.0
    near = torch.tensor([cfg.success_thresholds.dmax])
    far = torch.tensor(
        [
            cfg.reward_coefficients.centroid_flatness_dmax_multiplier
            * cfg.success_thresholds.dmax
        ]
    )
    rough = torch.tensor([2.0])
    flat = torch.tensor([0.0])

    rough_near_to_flat_near, _, _ = compute_centroid_flatness_reward(
        rough,
        flat,
        near,
        near,
        cfg,
    )
    flat_near_to_rough_far, _, _ = compute_centroid_flatness_reward(
        flat,
        rough,
        near,
        far,
        cfg,
    )
    rough_far_to_rough_near, _, _ = compute_centroid_flatness_reward(
        rough,
        rough,
        far,
        near,
        cfg,
    )

    cycle_reward = (
        rough_near_to_flat_near
        + flat_near_to_rough_far
        + rough_far_to_rough_near
    )
    assert torch.allclose(cycle_reward, torch.zeros_like(cycle_reward))


def test_default_level_shaping_preserves_existing_gather_formula() -> None:
    cfg = MultiRoverGatheringEnvCfg()
    previous = torch.tensor([[[-2.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, -2.0, 0.0], [0.0, 2.0, 0.0]]])
    current = 0.5 * previous
    velocities = torch.zeros(1, 4, 2)
    prev_metrics = compute_team_metrics(previous, velocities)
    metrics = compute_team_metrics(current, velocities)
    reward = compute_gather_reward(prev_metrics, metrics, cfg)
    expected = (
        cfg.reward_coefficients.dmax_progress * (prev_metrics.dmax - metrics.dmax)
        + cfg.reward_coefficients.dispersion_progress * (prev_metrics.dispersion - metrics.dispersion)
    )
    assert torch.allclose(reward, expected)


def test_terminal_reward_uses_configured_bonus_and_penalty() -> None:
    cfg = MultiRoverGatheringEnvCfg()
    cfg.reward_coefficients.success_bonus = 3.5
    cfg.reward_coefficients.failure_penalty = 4.5
    flags = DoneFlags(
        success=torch.tensor([True, False]),
        collision=torch.tensor([False, True]),
        out_of_bounds=torch.tensor([False, False]),
        timeout=torch.tensor([False, False]),
        terminated=torch.tensor([True, True]),
        truncated=torch.tensor([False, False]),
        done=torch.tensor([True, True]),
    )
    reward = compute_terminal_reward(flags, cfg)
    assert torch.allclose(reward, torch.tensor([3.5, -4.5]))


def test_timeout_penalty_only_applies_to_pure_timeout_truncation() -> None:
    cfg = MultiRoverGatheringEnvCfg()
    cfg.reward_coefficients.failure_penalty = 4.5
    cfg.reward_coefficients.timeout_penalty = 2.0
    flags = DoneFlags(
        success=torch.tensor([False, False, False]),
        collision=torch.tensor([False, True, False]),
        out_of_bounds=torch.tensor([False, False, True]),
        timeout=torch.tensor([True, True, True]),
        terminated=torch.tensor([False, True, True]),
        truncated=torch.tensor([True, False, False]),
        done=torch.tensor([True, True, True]),
    )

    reward = compute_terminal_reward(flags, cfg)

    assert torch.allclose(reward, torch.tensor([-2.0, -4.5, -4.5]))


def test_terminal_pairwise_gap_penalty_only_applies_near_success_geometry() -> None:
    cfg = MultiRoverGatheringEnvCfg()
    cfg.reward_coefficients.near_distance = 0.0
    cfg.reward_coefficients.inter_agent_collision = 0.0
    cfg.reward_coefficients.terminal_pairwise_gap = 4.0
    cfg.success_thresholds.min_pairwise_distance = 0.42
    done = DoneFlags(
        success=torch.tensor([False, False, False]),
        collision=torch.tensor([False, False, False]),
        out_of_bounds=torch.tensor([False, False, False]),
        timeout=torch.tensor([False, False, False]),
        terminated=torch.tensor([False, False, False]),
        truncated=torch.tensor([False, False, False]),
        done=torch.tensor([False, False, False]),
    )
    positions = torch.tensor(
        [
            [[-0.15, 0.0, 0.0], [0.20, 0.0, 0.0], [0.0, 0.22, 0.0], [0.0, -0.22, 0.0]],
            [[-0.50, 0.0, 0.0], [0.50, 0.0, 0.0], [0.0, 0.50, 0.0], [0.0, -0.50, 0.0]],
            [[-3.0, 0.0, 0.0], [-2.65, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 3.0, 0.0]],
        ],
        dtype=torch.float32,
    )
    metrics = compute_team_metrics(positions, torch.zeros(3, 4, 2))

    reward = compute_safety_reward(positions, metrics, done, cfg)

    assert reward[0] < 0.0
    assert torch.allclose(reward[1], torch.tensor(0.0))
    assert torch.allclose(reward[2], torch.tensor(0.0))


def test_default_terrain_reward_is_zero_even_with_rough_features() -> None:
    cfg = MultiRoverGatheringEnvCfg()
    positions = torch.zeros(1, 4, 3)
    terrain_features = torch.tensor(
        [[[0.0, 0.2, 0.1, 0.4, 0.6], [0.0, 0.1, 0.0, 0.2, 0.8], [0.0, 0.0, 0.1, 0.1, 0.9], [0.0, 0.2, 0.2, 0.5, 0.5]]]
    )
    reward = compute_terrain_reward(terrain_features, cfg, positions)
    assert torch.allclose(reward, torch.zeros_like(reward))


def test_positive_terrain_coefficients_penalize_rough_low_traversability_features() -> None:
    cfg = MultiRoverGatheringEnvCfg()
    cfg.reward_coefficients.slope_cost = 0.5
    cfg.reward_coefficients.terrain_cost = 1.0
    positions = torch.zeros(2, 4, 3)
    flat_features = torch.tensor(
        [[[0.0, 0.0, 0.0, 0.0, 1.0]] * 4],
        dtype=torch.float32,
    )
    rough_features = torch.tensor(
        [[[0.0, 0.2, 0.1, 0.4, 0.6]] * 4],
        dtype=torch.float32,
    )
    terrain_features = torch.cat((flat_features, rough_features), dim=0)
    reward = compute_terrain_reward(terrain_features, cfg, positions)
    assert torch.allclose(reward[0], torch.tensor(0.0))
    assert reward[1] < reward[0]


def test_relative_trajectory_risk_rewards_safer_than_straight_reference() -> None:
    cfg = MultiRoverGatheringEnvCfg()
    cfg.reward_coefficients.path_terrain_relative_cost = 2.0
    positions = torch.zeros(2, 4, 3)
    terrain_features = torch.zeros(2, 4, 5)
    selected = torch.tensor([[0.20] * 4, [0.50] * 4])
    reference = torch.tensor([[0.50] * 4, [0.20] * 4])

    reward = compute_terrain_reward(
        terrain_features,
        cfg,
        positions,
        path_terrain_risk_mean=selected,
        path_terrain_reference_risk_mean=reference,
    )

    assert reward[0] > 0.0
    assert reward[1] < 0.0
    assert torch.allclose(reward[0], -reward[1])


def test_terrain_reward_penalizes_risky_subgoal_speed_loss_and_height_change() -> None:
    cfg = MultiRoverGatheringEnvCfg()
    cfg.reward_coefficients.subgoal_terrain_cost = 0.5
    cfg.reward_coefficients.terrain_speed_loss_cost = 0.75
    cfg.reward_coefficients.terrain_height_change_cost = 1.25
    positions = torch.zeros(2, 4, 3)
    underfoot = torch.tensor([[[0.0, 0.0, 0.0, 0.1, 0.9]] * 4] * 2)
    subgoal = torch.tensor(
        [
            [[0.0, 0.0, 0.0, 0.1, 0.95]] * 4,
            [[0.0, 0.0, 0.0, 0.8, 0.20]] * 4,
        ]
    )
    speed_scale = torch.tensor([[0.95] * 4, [0.30] * 4])
    height_delta = torch.tensor([[0.01] * 4, [0.20] * 4])

    reward = compute_terrain_reward(
        underfoot,
        cfg,
        positions,
        subgoal_terrain_features=subgoal,
        terrain_speed_scale=speed_scale,
        height_delta=height_delta,
    )

    assert reward[1] < reward[0]


def test_terrain_reward_penalizes_path_risk_and_path_height_change() -> None:
    cfg = MultiRoverGatheringEnvCfg()
    cfg.reward_coefficients.path_terrain_mean_cost = 0.6
    cfg.reward_coefficients.path_terrain_max_cost = 0.4
    cfg.reward_coefficients.path_height_change_cost = 0.2
    positions = torch.zeros(2, 4, 3)
    underfoot = torch.tensor([[[0.0, 0.0, 0.0, 0.1, 0.9]] * 4] * 2)
    path_risk_mean = torch.tensor([[0.05] * 4, [0.60] * 4])
    path_risk_max = torch.tensor([[0.10] * 4, [0.90] * 4])
    path_height = torch.tensor([[0.01] * 4, [0.30] * 4])

    reward = compute_terrain_reward(
        underfoot,
        cfg,
        positions,
        path_terrain_risk_mean=path_risk_mean,
        path_terrain_risk_max=path_risk_max,
        path_height_change_mean=path_height,
    )

    assert reward[1] < reward[0]


def test_reward_config_keys_match_consumed_reward_terms() -> None:
    weight_keys = {field.name for field in fields(RewardWeightsCfg)}
    term_keys = {field.name for field in fields(RewardTerms)} - {"success_hold", "total"}
    assert weight_keys == term_keys

    consumed_coefficients = {
        "action_consistency",
        "centroid_flatness_dmax_multiplier",
        "centroid_flatness_excess",
        "centroid_flatness_progress",
        "dispersion_level",
        "dispersion_progress",
        "dmax_level",
        "dmax_progress",
        "dstc_belief_progress",
        "dstc_commit_bonus",
        "dstc_site_distance_progress",
        "failure_penalty",
        "filter_deviation_cost",
        "filter_raw_path_risk_cost",
        "inter_agent_collision",
        "near_distance",
        "oracle_mean_distance_progress",
        "path_length",
        "path_height_change_cost",
        "path_terrain_max_cost",
        "path_terrain_mean_cost",
        "path_terrain_relative_cost",
        "slope_cost",
        "subgoal_stagnation",
        "subgoal_terrain_cost",
        "subgoal_turn",
        "success_bonus",
        "success_hold_step",
        "terminal_pairwise_dispersion_multiplier",
        "terminal_pairwise_dmax_multiplier",
        "terminal_pairwise_gap",
        "timeout_penalty",
        "terrain_cost",
        "terrain_height_change_cost",
        "terrain_speed_loss_cost",
        "turn_cost",
    }
    assert {field.name for field in fields(RewardCoefficientsCfg)} == consumed_coefficients
    assert "obstacle_collision" not in consumed_coefficients
