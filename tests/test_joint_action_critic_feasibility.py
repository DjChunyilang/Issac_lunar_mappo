from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from analyze_joint_action_critic_feasibility import (  # noqa: E402
    compute_n_step_targets,
    fit_diagnostic_critic,
    regression_metrics,
)


def test_n_step_targets_stop_at_episode_boundary() -> None:
    rewards = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    done = torch.tensor([[False], [True], [False], [False]])
    next_values = torch.tensor([[10.0], [20.0], [30.0], [40.0]])
    targets = compute_n_step_targets(
        rewards,
        done,
        next_values,
        horizon=2,
        gamma=0.5,
    )
    assert targets[:, 0].tolist() == pytest.approx(
        [
            1.0 + 0.5 * 2.0,
            2.0,
            3.0 + 0.5 * 4.0 + 0.25 * 40.0,
        ]
    )


def test_joint_action_features_explain_action_dependent_target() -> None:
    generator = torch.Generator().manual_seed(13)
    samples = 1024
    states = torch.randn((samples, 4), generator=generator)
    actions = torch.randn((samples, 2), generator=generator)
    targets = 0.1 * states[:, 0] + 1.5 * actions[:, 0] - 0.8 * actions[:, 1]
    train = slice(0, 768)
    validation = slice(768, None)

    state_critic = fit_diagnostic_critic(
        states[train],
        targets[train],
        device=torch.device("cpu"),
        seed=5,
        epochs=25,
        batch_size=128,
        learning_rate=1.0e-3,
        hidden_dim=32,
    )
    joint_critic = fit_diagnostic_critic(
        torch.cat((states[train], actions[train]), dim=-1),
        targets[train],
        device=torch.device("cpu"),
        seed=5,
        epochs=25,
        batch_size=128,
        learning_rate=1.0e-3,
        hidden_dim=32,
    )
    state_metrics = regression_metrics(
        state_critic.predict(states[validation]), targets[validation]
    )
    joint_metrics = regression_metrics(
        joint_critic.predict(torch.cat((states[validation], actions[validation]), dim=-1)),
        targets[validation],
    )
    assert joint_metrics["mse"] < 0.15 * state_metrics["mse"]
    assert joint_metrics["r2"] > 0.90


def test_n_step_target_rejects_invalid_horizon() -> None:
    values = torch.zeros((3, 2))
    done = torch.zeros((3, 2), dtype=torch.bool)
    with pytest.raises(ValueError, match="horizon"):
        compute_n_step_targets(values, done, values, horizon=4, gamma=0.99)
