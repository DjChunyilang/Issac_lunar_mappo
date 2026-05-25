from __future__ import annotations

import torch

from lunar_rover_tasks.tasks.multi_rover_gathering.gathering_env_cfg import MultiRoverGatheringEnvCfg
from lunar_rover_tasks.tasks.multi_rover_gathering.metrics import compute_team_metrics
from lunar_rover_tasks.tasks.multi_rover_gathering.reward import compute_gather_reward, compute_oracle_reward


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

