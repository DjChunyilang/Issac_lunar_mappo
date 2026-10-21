"""Finite task v2: analytic targets, information identity, return accounting and compatibility."""

from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from _common import cfg_from_experiment, load_yaml
from train_local_goal_nmpc import (
    implementation_sources,
    validate_checkpoint_contract,
    validate_training_gate,
)
from test_local_goal_nmpc import FailingSolver, flat_memory
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.contracts import (
    LocalGoalRequest,
    contract_metadata,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.environment import (
    LocalGoalEnvironment,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.learning import (
    RecurrentPolicy,
    bootstrap_values,
    duration_gae,
    ppo_update,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.planner import (
    NMPCPlanner,
    build_solver,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.metrics import compute_team_metrics
from lunar_rover_tasks.tasks.multi_rover_gathering.reward import (
    RewardTerms,
    apply_finite_task_reward,
    finite_task_potentials,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.termination import TerminalReason, compute_done


@pytest.fixture
def cfg():
    torch.set_num_threads(1)
    torch.manual_seed(23)
    cfg = cfg_from_experiment(ROOT / "configs/experiment/p0_contract_v2.yaml")
    cfg.simulation.num_envs = 2
    return cfg


@pytest.fixture
def env(cfg):
    instance = LocalGoalEnvironment(cfg, "unused", solver_factory=lambda *a, **k: FailingSolver())
    yield instance
    instance.close()


def terminal(
    cfg,
    *,
    hold=24,
    step=480,
    spin=0.0,
    flat=True,
    invalid=False,
    collision=False,
    bounds=False,
    version="finite_task_v2",
):
    positions = torch.tensor([[[-0.5, -0.5, 0], [-0.5, 0.5, 0], [0.5, -0.5, 0], [0.5, 0.5, 0]]])
    if collision:
        positions[:, 1] = positions[:, 0]
    if bounds:
        positions[..., 0] += 100
    velocities = torch.zeros(1, 4, 2)
    metrics = compute_team_metrics(positions, velocities)
    return compute_done(
        positions,
        velocities,
        metrics,
        torch.tensor([hold]),
        torch.tensor([step]),
        480,
        cfg.success_thresholds,
        cfg.safety,
        flatness_ok=torch.tensor([flat]),
        terrain_unknown=torch.tensor([invalid]),
        angular_velocities=torch.full((1, 4), spin),
        task_contract_version=version,
    )


@pytest.mark.parametrize(
    "kind,kwargs",
    [
        (TerminalReason.SUCCESS, {}),
        (TerminalReason.TASK_DEADLINE, {"flat": False}),
        (TerminalReason.COLLISION, {"collision": True}),
        (TerminalReason.OUT_OF_BOUNDS, {"bounds": True}),
        (TerminalReason.INVALID_TERRAIN, {"invalid": True, "collision": True}),
    ],
)
def test_terminal_precedence_and_masks(cfg, kind, kwargs):
    done, count = terminal(cfg, **kwargs)
    assert done.terminal_reason.item() == kind
    assert done.terminated.item() == (kind != TerminalReason.INVALID_TERRAIN)
    assert done.truncated.item() == (kind == TerminalReason.INVALID_TERRAIN)
    assert not done.bootstrap_allowed.item()
    assert done.learning_valid.item() == (kind != TerminalReason.INVALID_TERRAIN)
    assert done.success.item() == (kind == TerminalReason.SUCCESS)
    assert done.timeout.item() == (kind == TerminalReason.TASK_DEADLINE)
    if kind != TerminalReason.SUCCESS:
        assert count.item() == 0


def test_spin_breaks_dwell_and_legacy_gate_is_preserved(cfg):
    done, count = terminal(cfg, spin=0.1001, step=100)
    assert not done.done.item() and count.item() == 0
    done, count = terminal(cfg, spin=0.1, step=100)
    assert done.success.item() and count.item() == 25
    done, _ = terminal(cfg, spin=1.0, version="legacy_v1")
    assert done.success.item()
    done, _ = terminal(cfg, flat=False, version="legacy_v1")
    assert done.truncated.item() and done.bootstrap_allowed.item()


def test_exact_deadline_and_rollout_targets():
    reward = torch.tensor([[-80.0, -80.0, float("nan")]])
    value = torch.tensor([[0.0, 0.0, float("nan")]])
    successor = torch.tensor([[float("nan"), 10.0, float("nan")]])
    terminated = torch.tensor([[True, False, False]])
    truncated = torch.tensor([[False, True, True]])
    valid = torch.tensor([[True, True, False]])
    adv, returns = duration_gae(
        reward,
        value,
        successor,
        torch.tensor([[5, 5, 2]]),
        terminated,
        truncated,
        bootstrap_allowed=torch.tensor([[False, True, False]]),
        learning_valid=valid,
    )
    torch.testing.assert_close(returns, torch.tensor([[-80.0, -80 + 0.99**5 * 10, 0.0]]))
    assert torch.isfinite(adv).all()


def test_invalid_transition_breaks_trace_and_never_calls_critic():
    class Guard:
        def value(self, state):
            assert torch.isfinite(state).all()
            return state[:, 0]

    states = torch.tensor([[float("nan")], [4.0]])
    values = bootstrap_values(Guard(), states, torch.tensor([False, True]))
    assert values.tolist() == [0.0, 4.0]
    rewards = torch.tensor([[1.0], [float("nan")], [100.0]])
    valid = torch.tensor([[True], [False], [True]])
    adv, _ = duration_gae(
        rewards,
        torch.zeros_like(rewards),
        torch.zeros_like(rewards),
        torch.ones_like(rewards),
        torch.zeros_like(valid),
        ~valid,
        learning_valid=valid,
    )
    assert adv[:, 0].tolist() == [1.0, 0.0, 100.0]


def test_clock_inputs_and_model_versions(env):
    actor, critic = env.observe()
    assert actor.shape == (2, 4, 411) and critic.shape == (2, 991)
    assert actor[..., -1].eq(1).all() and critic[..., -1].eq(1).all()
    env.core.step_count[0] = env.core.max_episode_steps // 2
    later, state = env.observe()
    assert later[0, :, -1].eq(0.5).all() and state[0, -1] == 0.5
    assert later[1, :, -1].eq(1).all()
    policy = RecurrentPolicy(critic.shape[-1], observation_version="local_goal_clock_v2")
    action, latent, log, _ = policy.sample(later, torch.zeros(2, 4, 128))
    assert action.shape == latent.shape == (2, 4, 4)
    assert torch.isfinite(log).all() and torch.isfinite(policy.value(state)).all()
    with pytest.raises(RuntimeError):
        policy.load_state_dict(RecurrentPolicy(990).state_dict())
    # Validity-aware terrain follows the same dynamic clock layout, without map files.
    real_policy = RecurrentPolicy(
        1439, terrain_channels=3, observation_version="local_goal_clock_v2"
    )
    assert real_policy.encode(torch.zeros(2, 4, 523)).shape == (2, 4, 128)


def test_neighbor_slots_follow_sender_across_distance_crossing(env, monkeypatch):
    monkeypatch.setattr(env, "_transport", lambda poses: None)
    for e in range(2):
        for a in range(4):
            for sender in range(4):
                if sender != a:
                    point = np.array([10.0 + sender, 20.0 + sender])
                    env.received[e, a, sender] = {
                        "timestamp": 0.0,
                        "expires_at": 1.0,
                        "version": 1,
                        "full": True,
                        "goal": point,
                        "plan": (np.array([0.0, 2.0]), np.stack([point, point])),
                    }
    for reversed_order in (False, True):
        # Cached relative positions control distance ordering without reading live peer truth.
        for sender in (1, 2, 3):
            env.core.communication_cache.features[0, 0, sender, :2] = torch.tensor(
                [float(4 - sender if reversed_order else sender), 0.0]
            )
        obs, _ = env.observe()
        senders = env.core.last_communication_snapshot.sender_indices[0, 0].tolist()
        assert senders == ([3, 2, 1] if reversed_order else [1, 2, 3])
        for slot, sender in enumerate(senders):
            values = obs[0, 0, 295 + 85 + slot * 10 : 295 + 95 + slot * 10]
            torch.testing.assert_close(values[4:6], torch.tensor([10.0 + sender, 20.0 + sender]))
            torch.testing.assert_close(values[6:8], values[4:6])
            torch.testing.assert_close(values[8:10], values[4:6])


@pytest.mark.parametrize("mode", ["missing", "expired", "sparse", "unsupported", "invalid_sender"])
def test_neighbor_permissions(env, monkeypatch, mode):
    monkeypatch.setattr(env, "_transport", lambda poses: None)
    snapshot = env.core.last_communication_snapshot
    sender = int(snapshot.sender_indices[0, 0, 0])
    key = (0, 0, sender)
    message = env.received[key]
    message["goal"] = np.array([100.0, 200.0])
    message["plan"] = (np.array([0.0, 2.0]), np.array([[100.0, 200.0], [100.0, 200.0]]))
    if mode == "missing":
        env.received.pop(key)
    elif mode == "expired":
        message["expires_at"] = -1.0
    elif mode == "sparse":
        env.core.communication_cache.full[0, 0, sender] = False
    elif mode == "unsupported":
        message["version"] = 999
    else:
        env.core.communication_cache.valid[0, 0, sender] = False
    obs, _ = env.observe()
    senders = env.core.last_communication_snapshot.sender_indices[0, 0].tolist()
    slot = senders.index(sender) if sender in senders else senders.index(-1)
    payload = obs[0, 0, 295 + 89 + slot * 10 : 295 + 95 + slot * 10]
    assert payload.eq(0).all()


def test_oracle_and_uncommunicated_truth_invariance_v2(env):
    obs, _ = env.observe()
    env.core.oracle_point.add_(1000)
    again, _ = env.observe()
    torch.testing.assert_close(obs, again)
    before = env.neighbors(0, 0)
    env.core.positions[0, 1, :2].add_(3)
    after = env.neighbors(0, 0)
    for a, b in zip(before, after):
        np.testing.assert_array_equal(a.positions, b.positions)
        np.testing.assert_array_equal(a.radii, b.radii)


def test_mid_macro_deadline_and_async_reset(env):
    env.core.step_count[0] = env.core.max_episode_steps - 2
    step = env.step(np.zeros((2, 4, 4)))
    assert step.duration.tolist() == [2, 5]
    assert step.terminated.tolist() == [True, False]
    assert not step.truncated.any()
    assert step.bootstrap_allowed.tolist() == [False, True]
    assert step.terminal_reason.tolist() == [TerminalReason.TASK_DEADLINE, TerminalReason.NONE]
    assert step.terminal_actor[0, :, -1].eq(0).all()
    assert step.actor[0, :, -1].eq(1).all()
    assert env.core.step_count.tolist() == [0, 5]
    assert env.episode_steps.tolist() == [0, 5]
    assert step.episodes[0]["timeout"]


def test_returns_persist_across_rollouts_and_reset_per_env(env, monkeypatch):
    env.core.cfg.simulation.episode_length_s = 1.0
    original = env._physics

    def constant(commands, active):
        _, done, gates = original(commands, active)
        env.last_reward_terms = {"energy": np.full(2, 2.0), "total": np.full(2, 2.0)}
        return torch.full((2,), 2.0), done, gates

    monkeypatch.setattr(env, "_physics", constant)
    first = env.step(np.zeros((2, 4, 4)), max_low_steps=2)
    assert first.bootstrap_allowed.all() and not first.terminated.any()
    saved = env.state_dict()
    env.reset([0])
    assert env.episode_steps.tolist() == [0, 2]
    env.load_state_dict(saved)
    final = env.step(np.zeros((2, 4, 4)), max_low_steps=5)
    assert final.duration.tolist() == [3, 3]
    for episode in final.episodes:
        assert episode["raw_return"] == 10.0
        assert episode["discounted_return"] == pytest.approx(2 * sum(0.99**t for t in range(5)))
        assert episode["reward_terms"]["energy"] == 10.0
        assert episode["duration_low_steps"] == 5
    assert env.episode_steps.tolist() == [0, 0]


def test_invalid_terrain_macro_has_no_learning_or_bootstrap(env, monkeypatch):
    integrate = env.core._integrate

    def invalid(commands):
        integrate(commands)
        env.core.last_terrain_unknown[0] = True

    monkeypatch.setattr(env.core, "_integrate", invalid)
    step = env.step(np.zeros((2, 4, 4)))
    assert step.learning_valid.tolist() == [False, True]
    assert step.bootstrap_allowed.tolist() == [False, True]
    assert step.truncated.tolist() == [True, False]
    assert not step.episodes[0]["timeout"]
    assert step.episodes[0]["terminal_reason"] == "invalid_terrain"
    assert step.episodes[0]["reward_terms"]["terminal"] == 0


def test_contract_metadata_checkpoint_and_pause_gate(cfg, env):
    contracts = contract_metadata(cfg)
    with pytest.raises(ValueError, match="contract mismatch"):
        validate_checkpoint_contract({}, contracts)
    validate_checkpoint_contract({"contracts": contracts}, contracts)
    for key in contracts:
        wrong = dict(contracts, **{key: "wrong"})
        with pytest.raises(ValueError):
            validate_checkpoint_contract({"contracts": wrong}, contracts)
    saved = env.state_dict()
    del saved["task_contract"]
    with pytest.raises(ValueError):
        env.load_state_dict(saved)
    with pytest.raises(ValueError, match="training remains paused"):
        validate_training_gate(load_yaml(ROOT / "configs/experiment/p0_contract_v2.yaml"))
    with pytest.raises(ValueError, match="requires LocalGoalEnvironment"):
        env.core.step(torch.zeros(2, 4, dtype=torch.long))
    assert {"reward.py", "termination.py", "communication.py", "observation.py", "_common.py"} <= {
        p.name for p in implementation_sources()
    }
    cfg.local_goal_contract.observation_version = "local_goal_v1"
    with pytest.raises(ValueError):
        contract_metadata(cfg)


def blank_terms():
    return RewardTerms(**{key: torch.zeros(1) for key in RewardTerms.__dataclass_fields__})


def geometry(dmax, dispersion):
    # Use real TeamMetrics and replace only controlled audit variables.
    base = compute_team_metrics(torch.zeros(1, 4, 3), torch.zeros(1, 4, 2))
    return replace(
        base, dmax=torch.tensor([float(dmax)]), dispersion=torch.tensor([float(dispersion)])
    )


def test_potential_boundary_and_interrupted_hold_no_extra_profit(cfg):
    # Same start/end and length; interior path and interrupted dwell may differ.
    paths = [
        [(4.0, 1.2, 0), (3.0, 1.0, 0), (2.0, 0.75, 12), (2.0, 0.75, 25)],
        [(4.0, 1.2, 0), (2.0, 0.75, 24), (4.0, 1.2, 0), (2.0, 0.75, 25)],
    ]
    # Isolate shaping from unchanged level costs and terminal events.
    cfg.reward_coefficients.dmax_level = 0
    cfg.reward_coefficients.dispersion_level = 0
    cfg.reward_coefficients.centroid_flatness_excess = 0
    totals = []
    for path in paths:
        potentials = [
            finite_task_potentials(geometry(d, s), torch.ones(1), torch.tensor([h]), cfg)
            for d, s, h in path
        ]
        accumulated = {key: 0.0 for key in potentials[0]}
        for t in range(len(path) - 1):
            done, _ = terminal(cfg, step=100, hold=24 if t == 2 else 0)
            d0, s0, h0 = path[t]
            d1, s1, h1 = path[t + 1]
            terms = apply_finite_task_reward(
                blank_terms(),
                geometry(d0, s0),
                geometry(d1, s1),
                torch.ones(1),
                torch.ones(1),
                torch.tensor([h0]),
                torch.tensor([h1]),
                done,
                cfg,
            )
            for key in accumulated:
                accumulated[key] += 0.99**t * getattr(terms, key).item()
        for key, value in accumulated.items():
            assert value == pytest.approx(-potentials[0][key].item(), abs=1e-5)
        totals.append(accumulated)
    assert totals[0] == pytest.approx(totals[1], abs=1e-5)


@pytest.mark.parametrize("hold_steps", [8, 25])
def test_dwell_shaping_budget_independent_of_duration(cfg, hold_steps):
    cfg.success_thresholds.hold_steps = hold_steps
    metrics = geometry(2, 0.75)
    result = 0.0
    for t in range(hold_steps):
        done, _ = terminal(cfg, hold=t, step=t)
        terms = apply_finite_task_reward(
            blank_terms(),
            metrics,
            metrics,
            torch.ones(1),
            torch.ones(1),
            torch.tensor([t]),
            torch.tensor([t + 1]),
            done,
            cfg,
        )
        result += 0.99**t * terms.success_hold.item()
    assert result == pytest.approx(cfg.reward_coefficients.success_hold_step, abs=1e-5)


@pytest.mark.parametrize(
    "reason,expected",
    [
        (TerminalReason.SUCCESS, 130.0),
        (TerminalReason.TASK_DEADLINE, -80.0),
        (TerminalReason.COLLISION, -55.0),
        (TerminalReason.OUT_OF_BOUNDS, -55.0),
        (TerminalReason.INVALID_TERRAIN, 0.0),
    ],
)
def test_terminal_reward_uses_reason(cfg, reason, expected):
    done, _ = terminal(cfg)
    invalid = reason == TerminalReason.INVALID_TERRAIN
    done = replace(
        done,
        terminal_reason=torch.tensor([reason]),
        terminated=torch.tensor([not invalid]),
        truncated=torch.tensor([invalid]),
        learning_valid=torch.tensor([not invalid]),
    )
    terms = apply_finite_task_reward(
        blank_terms(),
        geometry(2, 0.75),
        geometry(2, 0.75),
        torch.ones(1),
        torch.ones(1),
        torch.tensor([0]),
        torch.tensor([0]),
        done,
        cfg,
    )
    assert terms.terminal.item() == expected
    if invalid:
        assert terms.total.item() == 0


def test_hold_request_is_frozen_and_does_not_declare_success(env):
    pose = np.array([1.0, 2.0, 0.4])
    request = LocalGoalRequest.for_hold(pose, 3.0)
    pose[:] = 99.0
    np.testing.assert_array_equal(request.target, [1.0, 2.0, 0.4])
    assert request.speed_cap == 0 and request.expires_at == 4
    step = env.step(np.zeros((2, 4, 4)), hold_mask=np.ones((2, 4), dtype=bool), max_low_steps=1)
    assert not step.terminated.any() and not step.episodes
    assert all(g.speed_cap == 0 for g in env.goals)
    assert step.records[0]["hold_request"] == [[True] * 4] * 2


def test_masked_ppo_nan_rows_do_not_poison_gradients():
    torch.set_num_threads(1)
    torch.manual_seed(12)
    policy = RecurrentPolicy(991, observation_version="local_goal_clock_v2")
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
    obs = torch.zeros(3, 1, 4, 411)
    hidden = torch.zeros(3, 1, 4, 128)
    latent, logs = [], []
    with torch.no_grad():
        for t in range(3):
            _, z, log, _ = policy.sample(obs[t], hidden[t])
            latent.append(z)
            logs.append(log)
    batch = {
        "obs": obs,
        "hidden": hidden,
        "state": torch.zeros(3, 1, 991),
        "latent": torch.stack(latent),
        "log_prob": torch.stack(logs),
        "reward": torch.tensor([[1.0], [0.0], [3.0]]),
        "duration": torch.ones(3, 1),
        "value": torch.zeros(3, 1),
        "next_value": torch.zeros(3, 1),
        "terminated": torch.tensor([[False], [False], [True]]),
        "truncated": torch.tensor([[False], [True], [False]]),
        "learning_valid": torch.tensor([[True], [False], [True]]),
        "bootstrap_allowed": torch.tensor([[True], [False], [False]]),
    }
    for key in ("obs", "state", "hidden", "latent", "log_prob", "reward", "value", "next_value"):
        batch[key][1] = float("nan")
    metrics = ppo_update(policy, optimizer, batch, epochs=1)
    assert all(np.isfinite(list(metrics.values())))
    assert all(torch.isfinite(p).all() for p in policy.parameters())


def test_real_acados_hold_decelerates_with_slew_limits(tmp_path):
    import os

    if not os.environ.get("ACADOS_SOURCE_DIR"):
        pytest.skip("Use documented acados environment for acceptance")
    planner = NMPCPlanner(build_solver(tmp_path, 0.75), 0.75, deadline=10.0)
    pose, previous = np.zeros(3), np.array([0.6, 0.8])
    target = LocalGoalRequest.for_hold(pose, 0.0)
    trajectory, feedback = planner.plan(pose, target, flat_memory(), [], previous, 0.0)
    assert np.isfinite(trajectory.controls).all()
    assert abs(trajectory.controls[0, 0] - previous[0]) <= 0.30001
    assert abs(trajectory.controls[0, 1] - previous[1]) <= 0.60001
    assert abs(trajectory.controls[-1, 0]) < 0.05
    assert abs(trajectory.controls[-1, 1]) < 0.1
    # A brake fallback is legitimate execution feedback, never task success.
    assert feedback.status in ("feasible", "feasible_iteration_limit", "feasible_prefix", "brake")


def test_new_config_training_cli_stops_before_creating_run(tmp_path):
    import subprocess

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/train_local_goal_nmpc.py"),
            "--config",
            str(ROOT / "configs/experiment/p0_contract_v2.yaml"),
            "--run-dir",
            str(tmp_path / "must_not_exist"),
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert result.returncode != 0
    assert "training remains paused" in result.stderr
    assert not (tmp_path / "must_not_exist").exists()


def test_null_and_false_planner_metadata_rejected(cfg):
    cfg.local_goal_contract.task_contract_version = None
    cfg.local_goal_contract.observation_version = None
    with pytest.raises(ValueError):
        contract_metadata(cfg)
    cfg.local_goal_contract.task_contract_version = "finite_task_v2"
    cfg.local_goal_contract.observation_version = "local_goal_clock_v2"
    cfg.local_goal_contract.planner_revision = "mppi"
    with pytest.raises(ValueError, match="planner revision"):
        contract_metadata(cfg)
