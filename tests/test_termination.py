from __future__ import annotations

import torch

from lunar_rover_tasks.tasks.multi_rover_gathering.gathering_env_cfg import SafetyCfg, SuccessThresholdsCfg
from lunar_rover_tasks.tasks.multi_rover_gathering.metrics import compute_team_metrics
from lunar_rover_tasks.tasks.multi_rover_gathering.termination import compute_done, compute_success_gates


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


def test_success_gate_diagnostics_match_hold_logic() -> None:
    positions = torch.tensor(
        [
            [[-0.2, 0.0, 0.0], [0.2, 0.0, 0.0], [0.0, 0.2, 0.0], [0.0, -0.2, 0.0]],
            [[-0.2, 0.0, 0.0], [0.2, 0.0, 0.0], [0.0, 0.2, 0.0], [0.0, -0.2, 0.0]],
            [[-2.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, -2.0, 0.0]],
        ]
    )
    velocities = torch.tensor(
        [
            [[0.01, 0.0], [0.0, 0.01], [0.0, 0.0], [0.0, 0.0]],
            [[0.20, 0.0], [0.0, 0.20], [0.0, 0.0], [0.0, 0.0]],
            [[0.01, 0.0], [0.0, 0.01], [0.0, 0.0], [0.0, 0.0]],
        ]
    )
    thresholds = SuccessThresholdsCfg(dmax=1.0, dispersion=1.0, speed=0.1, hold_steps=2)
    metrics = compute_team_metrics(positions, velocities)

    gates = compute_success_gates(metrics, velocities, thresholds)
    assert gates.dmax_ok.tolist() == [True, True, False]
    assert gates.dispersion_ok.tolist() == [True, True, False]
    assert gates.speed_ok.tolist() == [True, False, True]
    assert gates.min_pairwise_ok.tolist() == [True, True, True]
    assert gates.instant_success.tolist() == [True, False, False]

    _, hold = compute_done(
        positions,
        velocities,
        metrics,
        torch.ones(3, dtype=torch.long),
        torch.zeros(3, dtype=torch.long),
        100,
        thresholds,
        SafetyCfg(collision_distance=0.01),
    )
    assert hold.tolist() == [2, 0, 0]


def test_success_requires_min_pairwise_distance_when_configured() -> None:
    positions = torch.tensor(
        [
            [[-0.20, 0.0, 0.0], [0.20, 0.0, 0.0], [0.0, 0.20, 0.0], [0.0, -0.20, 0.0]],
            [[-0.35, 0.0, 0.0], [0.35, 0.0, 0.0], [0.0, 0.35, 0.0], [0.0, -0.35, 0.0]],
        ]
    )
    velocities = torch.zeros(2, 4, 2)
    metrics = compute_team_metrics(positions, velocities)
    thresholds = SuccessThresholdsCfg(
        dmax=1.0,
        dispersion=1.0,
        speed=0.1,
        hold_steps=1,
        min_pairwise_distance=0.42,
    )

    gates = compute_success_gates(metrics, velocities, thresholds)
    assert gates.dmax_ok.tolist() == [True, True]
    assert gates.min_pairwise_ok.tolist() == [False, True]
    assert gates.instant_success.tolist() == [False, True]

    flags, hold = compute_done(
        positions,
        velocities,
        metrics,
        torch.zeros(2, dtype=torch.long),
        torch.zeros(2, dtype=torch.long),
        100,
        thresholds,
        SafetyCfg(collision_distance=0.28),
    )
    assert flags.success.tolist() == [False, True]
    assert hold.tolist() == [0, 1]
