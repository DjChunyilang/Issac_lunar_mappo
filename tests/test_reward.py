from __future__ import annotations

import torch

from lunar_rover_tasks.tasks.multi_rover_gathering.gathering_env_cfg import MultiRoverGatheringEnvCfg
from lunar_rover_tasks.tasks.multi_rover_gathering.metrics import compute_team_metrics
from lunar_rover_tasks.tasks.multi_rover_gathering.reward import (
    compute_gather_reward,
    compute_oracle_reward,
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
