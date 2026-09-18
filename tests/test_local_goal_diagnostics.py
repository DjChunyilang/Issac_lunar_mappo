"""Document current failure mechanisms, not acceptance of planner readiness."""

import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.contracts import NeighborPrediction
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.environment import (
    LocalGoalEnvironment,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.memory import LocalTerrainMemory
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.planner import NMPCPlanner


def test_observed_support_has_gap_between_fine_and_medium_forward_samples():
    memory = LocalTerrainMemory()
    points = np.array([[0.8, 0.0], [1.2, 0.0]])
    features = np.zeros((2, 5))
    features[:, 4] = 1.0
    memory.observe(points, features, 0.0)
    assert memory.safe([[0.8, 0.0], [1.2, 0.0]]).all()
    assert not memory.safe([[1.0, 0.0]])[0]


def test_fresh_stationary_full_plan_allows_nearby_terminal_but_expiry_does_not():
    env = LocalGoalEnvironment.__new__(LocalGoalEnvironment)
    env.a = 4
    env.core = SimpleNamespace(step_count=torch.tensor([1]))
    separation = float(np.sqrt(2) * 0.6)
    position = np.array([separation, 0.0])
    env.received = {
        (0, 0, 1): {
            "timestamp": 0.2,
            "expires_at": 0.8,
            "full": True,
            "position": position,
            "velocity": np.zeros(2),
            "plan": (np.arange(16) * 0.2, np.tile(position, (16, 1))),
        }
    }
    peer = env.neighbors(0, 0)[0]
    assert peer.radii[0] < separation  # Present separation is not the problem.
    np.testing.assert_allclose(peer.radii[-1], 0.70)
    assert separation > peer.radii[-1]
    message = env.received[(0, 0, 1)]
    _, xy = message["plan"]
    xy[-1, 0] += 0.01
    np.testing.assert_allclose(env.neighbors(0, 0)[0].radii[-1], 0.93)
    message["plan"] = None
    np.testing.assert_allclose(env.neighbors(0, 0)[0].radii[-1], 0.91)
    message["stop_intent"] = True
    np.testing.assert_allclose(env.neighbors(0, 0)[0].radii[-1], 0.70)
    env.core.step_count[:] = 5
    stale = env.neighbors(0, 0)[0]
    assert stale.radii[-1] > separation


@pytest.mark.parametrize("direction", [1, -1])
def test_verified_prefix_moves_then_stops_without_crossing_unknown_or_neighbor(direction):
    memory = LocalTerrainMemory()
    xy = np.stack(np.meshgrid(np.arange(-0.4, 0.81, 0.2), np.arange(-0.8, 0.81, 0.2)), -1).reshape(
        -1, 2
    )
    features = np.zeros((len(xy), 5))
    features[:, 4] = 1
    memory.observe(xy, features, 0)
    planner = NMPCPlanner(SimpleNamespace(reset=lambda: None), 0.75)
    controls = np.zeros((15, 2))
    controls[:, 0] = direction * np.minimum(np.arange(1, 16) * 0.3, 1.15)
    result, length = planner.stopped_prefix(
        np.zeros(3), controls, memory, [], 1.15, np.zeros(2), 0, time.monotonic() + 2
    )
    assert 0 < length < 15
    assert direction * result.states[1, 0] > 0 and abs(result.states[-1, 0]) < 0.955
    np.testing.assert_allclose(result.controls[-1], 0, atol=1e-6)
    assert not planner.validate(result, memory, [], 1.15, np.zeros(2))
    replay = planner.rollout(np.zeros(3), result.controls, memory, 0)
    np.testing.assert_allclose(replay.states, result.states)
    neighbor = NeighborPrediction(np.tile([0.2, 0], (16, 1)), np.full(16, 0.46), 0, 1)
    rejected, _ = planner.stopped_prefix(
        np.zeros(3), controls, memory, [neighbor], 1.15, np.zeros(2), 0, time.monotonic() + 2
    )
    assert rejected is None


def test_sparse_payload_cannot_enable_stop_intent_uncertainty():
    env = LocalGoalEnvironment.__new__(LocalGoalEnvironment)
    env.a = 4
    env.core = SimpleNamespace(step_count=torch.tensor([0]))
    env.received = {
        (0, 0, 1): {
            "timestamp": 0.0,
            "expires_at": 0.6,
            "full": False,
            "stop_intent": True,
            "position": np.array([0.85, 0]),
            "velocity": np.zeros(2),
            "plan": None,
        }
    }
    assert env.neighbors(0, 0)[0].radii[-1] > 3
