from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from analyze_reward_component_identifiability import (  # noqa: E402
    component_metrics,
    fit_multi_output_regressor,
    identifiability_targets,
)


def test_multi_output_joint_features_identify_action_dependent_component() -> None:
    generator = torch.Generator().manual_seed(31)
    states = torch.randn((1024, 5), generator=generator)
    actions = torch.randn((1024, 3), generator=generator)
    targets = torch.stack(
        (
            states[:, 0] + 0.2 * states[:, 1],
            1.4 * actions[:, 0] - 0.7 * actions[:, 2],
        ),
        dim=-1,
    )
    train = slice(0, 768)
    validation = slice(768, None)
    state_model = fit_multi_output_regressor(
        states[train],
        targets[train],
        device=torch.device("cpu"),
        seed=11,
        epochs=25,
        batch_size=128,
        learning_rate=1.0e-3,
        hidden_dim=32,
    )
    joint_model = fit_multi_output_regressor(
        torch.cat((states[train], actions[train]), dim=-1),
        targets[train],
        device=torch.device("cpu"),
        seed=11,
        epochs=25,
        batch_size=128,
        learning_rate=1.0e-3,
        hidden_dim=32,
    )
    state_prediction = state_model.predict(states[validation])
    joint_prediction = joint_model.predict(
        torch.cat((states[validation], actions[validation]), dim=-1)
    )
    state_action_term = component_metrics(
        state_prediction[:, 1], targets[validation, 1]
    )
    joint_action_term = component_metrics(
        joint_prediction[:, 1], targets[validation, 1]
    )
    assert joint_action_term["mse"] < 0.15 * state_action_term["mse"]
    assert joint_action_term["r2"] > 0.90


def test_identifiability_targets_include_existing_diagnostics() -> None:
    class Dataset:
        reward_terms = {"gather": torch.tensor([1.0, 2.0])}
        relative_path_risk = torch.tensor([[1.0, 3.0], [2.0, 4.0]])
        conflict_involvement = torch.tensor([[0.0, 2.0], [1.0, 1.0]])
        collision_involvement = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
        nearest_distance_change = torch.tensor([[0.2, 0.4], [-0.2, 0.0]])

    targets = identifiability_targets(
        Dataset(),
        (
            "gather",
            "diagnostic_relative_path_risk",
            "diagnostic_predicted_conflict_involvement",
            "diagnostic_collision_involvement",
            "diagnostic_nearest_distance_change",
        ),
    )
    assert targets.tolist() == [
        [1.0, 2.0, 1.0, 0.0, pytest.approx(0.3)],
        [2.0, 3.0, 1.0, 0.5, pytest.approx(-0.1)],
    ]
