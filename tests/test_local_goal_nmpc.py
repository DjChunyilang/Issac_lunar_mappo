"""New-route semantic regressions; passing these is not convergence evidence."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from _common import cfg_from_experiment
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.contracts import (
    DT,
    LocalGoalRequest,
    N,
    NeighborPrediction,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.environment import (
    LocalGoalEnvironment,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.learning import (
    RecurrentPolicy,
    bounded_log_prob,
    duration_gae,
    ppo_update,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.memory import LocalTerrainMemory
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.planner import (
    NMPCPlanner,
    dynamics,
    track_trajectory,
)


class FailingSolver:
    """Dependency injection for failure contracts ONLY, never a training backend."""

    def reset(self):
        pass

    def set(self, *args):
        pass

    def constraints_set(self, *args):
        pass

    def solve(self):
        return 4


def flat_memory():
    memory = LocalTerrainMemory(capacity=4096)
    xy = np.stack(np.meshgrid(np.arange(-2, 2.01, 0.1), np.arange(-2, 2.01, 0.1)), -1).reshape(
        -1, 2
    )
    features = np.zeros((len(xy), 5))
    features[:, 4] = 1
    memory.observe(xy, features, 0)
    return memory


def test_goal_is_frozen_and_stop_keeps_heading_request():
    pose = np.array([2.0, 3.0, np.pi / 2])
    goal = LocalGoalRequest.from_action([1, 0, 0.5, -1], pose, 4.0)
    np.testing.assert_allclose(goal.target[:2], [2.0, 4.6])
    pose[:] = 0
    np.testing.assert_allclose(goal.target[:2], [2.0, 4.6])
    assert goal.speed_cap == 0 and goal.expires_at == 5
    with pytest.raises(ValueError):
        LocalGoalRequest.from_action([2, 0, 0, 0], pose, 0)


def test_memory_bounded_unknown_not_free_interpolation_not_observed():
    memory = LocalTerrainMemory(capacity=2)
    feature = np.array([[0, 0, 0, 0, 1.0]] * 3)
    memory.observe(np.array([[0, 0], [1, 0], [2, 0.0]]), feature, 0)
    assert len(memory.samples) == 2
    _, known, direct = memory.lookup([[2.05, 0], [9, 0]])
    assert known.tolist() == [True, False] and not direct.any()
    assert not memory.safe([[0, 0], [9, 0]]).any()


def test_signed_motion_spin_and_generic_tracking():
    feature = np.array([0, 0, 0, 0, 1])
    state, velocity = dynamics(np.zeros(3), np.array([-0.4, 0]), feature, 0.75)
    assert state[0] < 0 and velocity[0] < 0
    spun, _ = dynamics(np.zeros(3), np.array([0, 1]), feature, 0.75)
    np.testing.assert_allclose(spun, [0, 0, 0.2])
    planner = NMPCPlanner(FailingSolver(), 0.75)
    plan = planner.rollout(np.zeros(3), np.tile([-0.2, 0.1], (N, 1)), flat_memory(), 0)
    np.testing.assert_allclose(track_trajectory(np.zeros(3), plan), [-0.2, 0.1])


def test_failure_brakes_and_old_plan_is_revalidated():
    planner = NMPCPlanner(FailingSolver(), 0.75)
    memory = flat_memory()
    goal = LocalGoalRequest(np.array([1.0, 0, 0]), 1.0, 0, 1)
    plan, feedback = planner.plan(np.zeros(3), goal, memory, [], np.array([0.5, 0]), 0)
    assert feedback.emergency_brake and feedback.reason == "solver_failure"
    assert plan.controls[0, 0] == pytest.approx(0.2)
    assert not plan.controls[1:].any()
    planner.previous = planner.rollout(np.zeros(3), np.zeros((N, 2)), memory, 0)
    _, feedback = planner.plan(np.zeros(3), goal, memory, [], np.zeros(2), 0.2)
    assert feedback.used_previous
    planner.previous = plan
    memory.reset()
    _, feedback = planner.plan(np.zeros(3), goal, memory, [], np.zeros(2), 0.4)
    assert feedback.emergency_brake and not feedback.used_previous


def test_swept_crossing_rejected_between_nodes():
    planner = NMPCPlanner(FailingSolver(), 0.75)
    plan = planner.rollout(np.zeros(3), np.zeros((N, 2)), flat_memory(), 0)
    xy = np.tile([1.0, 0.0], (N + 1, 1))
    xy[0] = [-1.0, 0.0]
    peer = NeighborPrediction(xy, np.full(N + 1, 0.46), 0, 1)
    assert (
        planner.validate(plan, flat_memory(), [peer], 1.0, np.zeros(2))
        == "neighbor_swept_collision"
    )


def test_solver_exception_and_nonfinite_output_fail_closed():
    class RaisingSolver(FailingSolver):
        def solve(self):
            raise RuntimeError("injected solve failure")

    class NanSolver(FailingSolver):
        def solve(self):
            return 0

        def get(self, *args):
            return np.array([np.nan, 0.0])

    goal = LocalGoalRequest(np.array([1.0, 0, 0]), 1.0, 0, 1)
    for solver, reason in [(RaisingSolver(), "solver_exception"), (NanSolver(), "nonfinite")]:
        planner = NMPCPlanner(solver, 0.75)
        trajectory, feedback = planner.plan(np.zeros(3), goal, flat_memory(), [], np.zeros(2), 0)
        assert feedback.emergency_brake and feedback.reason == reason
        assert np.isfinite(trajectory.states).all()


def test_log_probability_matches_transformed_distribution_and_request():
    normal = torch.distributions.Normal(torch.zeros(4), torch.ones(4))
    latent = torch.tensor([0.5, -0.7, 1.0, -1.5])
    transformed = torch.distributions.TransformedDistribution(
        normal, [torch.distributions.TanhTransform(cache_size=1)]
    )
    torch.testing.assert_close(
        bounded_log_prob(normal, latent), transformed.log_prob(latent.tanh()).sum()
    )
    assert torch.isfinite(bounded_log_prob(normal, torch.tensor([50.0, -50.0, 0.0, 0.0])))
    policy = RecurrentPolicy(990)
    obs = torch.zeros(1, 4, 410)
    hidden = torch.zeros(1, 4, 128)
    action, z, old, next_hidden = policy.sample(obs, hidden)
    normal, _ = policy.distribution(obs, hidden)
    torch.testing.assert_close(old, bounded_log_prob(normal, z))
    assert action.shape == (1, 4, 4) and torch.all(action.abs() <= 1)
    # Planner controls are a separate two-dimensional quantity and never enter PPO.
    assert z.shape[-1] == 4 and next_hidden.shape[-1] == 128


def test_duration_gae_timeout_bootstraps_terminal_and_stops_trace():
    reward = torch.tensor([[1.0], [2.0], [3.0]])
    value = torch.zeros_like(reward)
    terminal_value = torch.tensor([[10.0], [20.0], [30.0]])
    duration = torch.tensor([[5], [2], [1]])
    terminated = torch.tensor([[False], [True], [False]])
    truncated = torch.tensor([[True], [False], [False]])
    adv, _ = duration_gae(
        reward, value, terminal_value, duration, terminated, truncated, gamma=0.9, lam=0.8
    )
    assert adv[0].item() == pytest.approx(1 + 0.9**5 * 10)
    assert adv[1].item() == 2
    assert adv[2].item() == pytest.approx(3 + 0.9 * 30)


@pytest.fixture
def env():
    cfg = cfg_from_experiment(ROOT / "configs/experiment/exp167_local_goal_nmpc.yaml")
    cfg.simulation.num_envs = 2
    env = LocalGoalEnvironment(cfg, "unused", solver_factory=lambda *a, **k: FailingSolver())
    yield env
    env.close()


def test_macro_timeout_and_no_cross_episode_state(env):
    env.core.step_count[:] = env.core.max_episode_steps - 2
    old_positions = env.core.positions.clone()
    step = env.step(np.zeros((2, 4, 4)))
    assert step.duration.tolist() == [2, 2]
    assert step.truncated.all() and not step.terminated.any()
    assert env.core.step_count.tolist() == [0, 0]
    assert not torch.equal(step.terminal_critic, step.critic)
    assert not torch.equal(old_positions, env.core.positions)
    assert all(planner.previous is None for planner in env.planners)
    assert all(f.status == "reset" for f in env.feedback)


def test_sparse_expiry_and_uncertainty(env):
    env.received = {
        (0, 0, 1): {
            "position": np.array([2.0, 0]),
            "velocity": np.array([1.0, 0]),
            "timestamp": 0.0,
            "expires_at": 0.6,
            "version": 1,
            "full": True,
            "goal": np.array([4.0, 0]),
            "plan": (np.arange(N + 1) * DT, np.tile([3.0, 0.0], (N + 1, 1))),
        }
    }
    fresh = env.neighbors(0, 0)[0]
    env.core.step_count[0] = 10
    stale = env.neighbors(0, 0)[0]
    np.testing.assert_allclose(stale.positions, np.tile([2.0, 0.0], (N + 1, 1)))
    assert (stale.radii > fresh.radii).all() and np.all(np.diff(stale.radii) > 0)


def test_planner_has_no_global_terrain_reference_and_legacy_is_unchanged(env):
    for planner in env.planners:
        assert not hasattr(planner, "core") and not hasattr(planner, "terrain_runtime")
    source = (
        ROOT
        / "source/lunar_rover_tasks/lunar_rover_tasks/tasks/multi_rover_gathering/local_goal/planner.py"
    ).read_text()
    assert "query_terrain" not in source
    assert env.core.cfg.success_thresholds.min_pairwise_distance == 0.42
    assert env.core.cfg.success_thresholds.hold_steps == 8
    cfg = cfg_from_experiment(
        ROOT / "configs/experiment/exp156_differential_multiscale_ablation.yaml"
    )
    assert cfg.planner.action_type == "differential_trajectory_primitives"


def test_checkpoint_environment_restores_local_state(env):
    env.step(np.zeros((2, 4, 4)), max_low_steps=1)
    saved = env.state_dict()
    positions = env.core.positions.clone()
    env.reset([0])
    env.load_state_dict(saved)
    torch.testing.assert_close(env.core.positions, positions)
    assert env.core.step_count.tolist() == [1, 1]


def test_reward_aggregation_respects_each_environment_duration(env, monkeypatch):
    original = env._physics

    def unit_reward(commands, active):
        _, done, gates = original(commands, active)
        return torch.ones(env.e), done, gates

    monkeypatch.setattr(env, "_physics", unit_reward)
    env.core.positions[0, 1] = env.core.positions[0, 0]
    step = env.step(np.zeros((2, 4, 4)))
    assert step.duration.tolist() == [1, 5]
    assert step.terminated.tolist() == [True, False]
    torch.testing.assert_close(step.reward, torch.tensor([1.0, sum(0.99**k for k in range(5))]))
    assert env.core.step_count.tolist() == [0, 5]


def test_oracle_and_undelivered_peer_state_do_not_reach_local_actor(env):
    actor, _ = env.observe()
    env.core.oracle_point.add_(1000.0)
    actor_again, _ = env.observe()
    torch.testing.assert_close(actor, actor_again)
    before = env.neighbors(0, 0)
    env.core.positions[0, 1, :2].add_(3.0)
    # No communication advance: the live peer change must be invisible to NMPC.
    after = env.neighbors(0, 0)
    for left, right in zip(before, after):
        np.testing.assert_array_equal(left.positions, right.positions)
        np.testing.assert_array_equal(left.radii, right.radii)


def test_memory_lru_array_storage_preserves_latest_direct_samples():
    memory = LocalTerrainMemory(capacity=4)
    for index in range(20):
        point = np.array([[float(index), 0.0]])
        feature = np.array([[index, 0.0, 0.0, 0.0, 1.0]])
        memory.observe(point, feature, index)
        values, known, direct = memory.lookup(point)
        assert known[0] and direct[0] and values[0, 0] == index
    assert len(memory.samples) == 4
    for index in range(16, 20):
        values, known, direct = memory.lookup([[index, 0.0]])
        assert known[0] and direct[0] and values[0, 0] == index


def test_free_space_is_not_a_time_indexed_previous_path():
    memory = flat_memory()
    disks = memory.free_disks([0.0, 0.0])
    for point in ([0.0, 0.0], [-0.4, 0.0], [0.4, 0.0]):
        assert np.any(np.linalg.norm(disks[:, :2] - point, axis=1) <= disks[:, 2])
    assert not np.any(np.linalg.norm(disks[:, :2] - [10.0, 0.0], axis=1) <= disks[:, 2])


def test_critic_gradient_not_hard_clipped():
    policy = RecurrentPolicy(990)
    with torch.no_grad():
        policy.critic[-1].bias.fill_(1e4)
    loss = (policy.value(torch.zeros(2, 990)) - torch.ones(2)).square().mean()
    loss.backward()
    assert policy.critic[-1].bias.grad.abs().item() > 100


def test_recurrent_ppo_sequence_and_termination_mask():
    torch.set_num_threads(1)
    policy = RecurrentPolicy(990)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
    obs = torch.randn(17, 1, 4, 410) * 0.05
    state = torch.randn(17, 1, 990) * 0.05
    hidden = torch.zeros(1, 4, 128)
    h, latent, log = [], [], []
    terminated = torch.zeros(17, 1, dtype=torch.bool)
    terminated[7] = True
    with torch.no_grad():
        for t in range(17):
            h.append(hidden.clone())
            _, z, p, hidden = policy.sample(obs[t], hidden)
            latent.append(z)
            log.append(p)
            if terminated[t]:
                hidden.zero_()
    assert h[8].count_nonzero() == 0
    assert h[16].count_nonzero() > 0
    batch = {
        "obs": obs,
        "state": state,
        "hidden": torch.stack(h),
        "latent": torch.stack(latent),
        "log_prob": torch.stack(log),
        "reward": torch.ones(17, 1),
        "duration": torch.full((17, 1), 5),
        "terminated": terminated,
        "truncated": torch.zeros_like(terminated),
        "value": torch.zeros(17, 1),
        "next_value": torch.zeros(17, 1),
    }
    before = policy.gru.weight_hh.clone()
    metrics = ppo_update(policy, optimizer, batch, epochs=1)
    assert np.isfinite(list(metrics.values())).all()
    assert not torch.equal(before, policy.gru.weight_hh)


@pytest.mark.parametrize(
    "action,expected",
    [
        ([0.6, 0, 0, 0], "forward"),
        ([-0.6, 0, 0, 0], "reverse"),
        ([0, 0, 0.6, -1], "spin"),
        ([0, 0, 0, -1], "stop"),
    ],
)
def test_real_acados_first_control_and_slew(tmp_path, action, expected):
    import os

    if not os.environ.get("ACADOS_SOURCE_DIR"):
        pytest.skip("Optional real-acados integration test: use the documented environment")
    from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.planner import build_solver

    planner = NMPCPlanner(build_solver(tmp_path, 0.75), 0.75, deadline=10.0)
    goal = LocalGoalRequest.from_action(action, np.zeros(3), 0)
    trajectory, feedback = planner.plan(np.zeros(3), goal, flat_memory(), [], np.zeros(2), 0)
    assert not feedback.emergency_brake, feedback
    v, w = trajectory.controls[0]
    assert abs(v) <= 0.3001 and abs(w) <= 0.6001
    if expected == "forward":
        assert v > 0
    if expected == "reverse":
        assert v < 0
    if expected == "spin":
        assert abs(v) < 1e-5 and w > 0
    if expected == "stop":
        assert abs(v) < 1e-5 and abs(w) < 1e-5
    if expected in ("stop", "spin"):
        assert feedback.blocked_seconds == 0


def test_real_acados_can_reverse_a_previous_forward_plan(tmp_path):
    import os

    if not os.environ.get("ACADOS_SOURCE_DIR"):
        pytest.skip("Optional real-acados integration test")
    from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.planner import build_solver

    planner = NMPCPlanner(build_solver(tmp_path, 0.75), 0.75, deadline=10.0)
    memory = flat_memory()
    first, feedback = planner.plan(
        np.zeros(3),
        LocalGoalRequest.from_action([0.6, 0, 0, 0], np.zeros(3), 0),
        memory,
        [],
        np.zeros(2),
        0,
    )
    assert not feedback.emergency_brake
    pose = first.states[1]
    second, feedback = planner.plan(
        pose,
        LocalGoalRequest.from_action([-0.6, 0, 0, 0], pose, 0.2),
        memory,
        [],
        first.controls[0],
        0.2,
    )
    assert not feedback.emergency_brake, feedback
    assert second.states[-1, 0] < pose[0] - 0.1
    assert second.controls[:, 0].min() < -0.05
