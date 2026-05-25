from __future__ import annotations

import torch

from lunar_rover_tasks.tasks.multi_rover_gathering.gathering_env_cfg import SafetyCfg, SuccessThresholdsCfg
from lunar_rover_tasks.tasks.multi_rover_gathering.metrics import compute_team_metrics
from lunar_rover_tasks.tasks.multi_rover_gathering.termination import compute_done


def test_success_requires_hold_steps() -> None:
    positions = torch.tensor(
        [[[-0.1, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, -0.1, 0.0]]]
    )
    velocities = torch.zeros(1, 4, 2)
    metrics = compute_team_metrics(positions, velocities)
    thresholds = SuccessThresholdsCfg(dmax=1.0, dispersion=1.0, speed=0.1, hold_steps=2)
    flags, hold = compute_done(
        positions,
        velocities,
        metrics,
        torch.zeros(1, dtype=torch.long),
        torch.zeros(1, dtype=torch.long),
        100,
        thresholds,
        SafetyCfg(collision_distance=0.01),
    )
    assert not bool(flags.success.item())
    flags, hold = compute_done(
        positions,
        velocities,
        metrics,
        hold,
        torch.zeros(1, dtype=torch.long),
        100,
        thresholds,
        SafetyCfg(collision_distance=0.01),
    )
    assert bool(flags.success.item())


def test_collision_failure() -> None:
    positions = torch.zeros(1, 4, 3)
    velocities = torch.zeros(1, 4, 2)
    metrics = compute_team_metrics(positions, velocities)
    flags, _ = compute_done(
        positions,
        velocities,
        metrics,
        torch.zeros(1, dtype=torch.long),
        torch.zeros(1, dtype=torch.long),
        100,
        SuccessThresholdsCfg(),
        SafetyCfg(collision_distance=0.5),
    )
    assert bool(flags.collision.item())
    assert bool(flags.terminated.item())

