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
    compute_gather_reward,
    compute_oracle_reward,
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


def test_reward_config_keys_match_consumed_reward_terms() -> None:
    weight_keys = {field.name for field in fields(RewardWeightsCfg)}
    term_keys = {field.name for field in fields(RewardTerms)} - {"success_hold", "total"}
    assert weight_keys == term_keys

    consumed_coefficients = {
        "action_consistency",
        "dispersion_level",
        "dispersion_progress",
        "dmax_level",
        "dmax_progress",
        "failure_penalty",
        "inter_agent_collision",
        "near_distance",
        "oracle_mean_distance_progress",
        "path_length",
        "slope_cost",
        "subgoal_stagnation",
        "subgoal_terrain_cost",
        "subgoal_turn",
        "success_bonus",
        "success_hold_step",
        "timeout_penalty",
        "terrain_cost",
        "terrain_height_change_cost",
        "terrain_speed_loss_cost",
        "turn_cost",
    }
    assert {field.name for field in fields(RewardCoefficientsCfg)} == consumed_coefficients
    assert "obstacle_collision" not in consumed_coefficients
