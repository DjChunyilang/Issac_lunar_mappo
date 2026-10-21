"""T04 conditional action/probability contracts; no formal environment training."""

from copy import deepcopy
from dataclasses import replace
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch
from torch.distributions import Normal, TransformedDistribution
from torch.distributions.transforms import AffineTransform, TanhTransform

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from _common import cfg_from_experiment, load_yaml
from train_local_goal_nmpc import (
    build_policy,
    resolved_contracts,
    sample_for_rollout,
    stack_rollout,
    validate_checkpoint_contract,
    validate_evaluation_gate,
    validate_training_gate,
)
from test_planner_protocol import EchoSolver, StationaryTestAdapter
from test_local_goal_nmpc import flat_memory
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.actions import (
    ACTION_VERSION,
    ROLLOUT_VERSION,
    HybridSample,
    PolicyAction,
    PolarHoldDistribution,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.contracts import (
    DT,
    N,
    contract_metadata,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.environment import (
    LocalGoalEnvironment,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.learning import (
    RecurrentPolicy,
    ppo_update,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.planner import (
    NMPCPlanner,
    build_solver,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.protocol import (
    NMPCAdapter,
    PlannerInput,
    TerrainView,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.metrics import compute_team_metrics
from lunar_rover_tasks.tasks.multi_rover_gathering.termination import compute_done


CONFIG = ROOT / "configs/experiment/t04_polar_hold.yaml"


def distribution(count=8, *, dtype=torch.float64, mean=0.0, std=0.6, logits=0.0):
    return PolarHoldDistribution(
        torch.full((count, 2), mean, dtype=dtype),
        torch.full((count, 2), std, dtype=dtype),
        torch.full((count,), logits, dtype=dtype),
    )


def make_policy():
    torch.set_num_threads(1)
    return RecurrentPolicy(
        991, observation_version="local_intent_clock_v3", action_version=ACTION_VERSION
    )


def synthetic_batch(policy, mode="mixed", length=3):
    torch.manual_seed(23)
    rows = []
    hidden = torch.randn(2, 4, 128) * 0.1
    for t in range(length):
        obs, state = torch.randn(2, 4, 411) * 0.05, torch.randn(2, 991) * 0.05
        with torch.no_grad():
            dist, following = policy.distribution(obs, hidden)
            if mode in ("hold", "move"):
                hold = torch.full((2, 4), mode == "hold", dtype=torch.bool)
                latent = torch.zeros(2, 4, 2) if mode == "hold" else dist.normal.sample()
                physical = torch.where(hold[..., None], 0.0, dist.transform(latent))
                sample = HybridSample(
                    PolicyAction(physical, hold), latent, dist.log_prob(latent, hold), following
                )
            else:
                sample = policy.sample(obs, hidden)
            ended = torch.tensor([t == 0, False])
            rows.append(
                {
                    "obs": obs,
                    "state": state,
                    "hidden": hidden.clone(),
                    **sample.rollout_fields(),
                    "reward": torch.tensor([-1.0 + t, 0.5]),
                    "value": policy.value(state),
                    "next_value": torch.zeros(2),
                    "duration": torch.tensor([1, 5]),
                    "terminated": ended,
                    "truncated": torch.zeros(2, dtype=torch.bool),
                    "bootstrap_allowed": ~ended,
                    "learning_valid": torch.ones(2, dtype=torch.bool),
                }
            )
            hidden = sample.hidden.clone()
            hidden[ended] = 0
    return stack_rollout(rows, policy)


@pytest.fixture
def env():
    torch.set_num_threads(1)
    torch.manual_seed(23)
    cfg = cfg_from_experiment(CONFIG)
    cfg.simulation.num_envs = 2
    instance = LocalGoalEnvironment(cfg, "unused", solver_factory=lambda *a, **kw: EchoSolver(4))
    yield instance
    instance.close()


def test_polar_decode_bounds_freeze_and_se2():
    poses = np.array([[[1.0, 2.0, np.pi / 2], [-1.0, 3.0, -0.2], [0.0, 0.0, 0.0]]])
    action = PolicyAction(
        np.array([[[1.6, 0.0], [0.0, np.pi], [0.0, 0.0]]]), np.array([[False, False, True]])
    )
    intents = action.decode(poses, [4.0], 1.6, np.pi)
    np.testing.assert_allclose(intents[0].goal_xy, [1, 3.6])
    assert intents[0].terminal_yaw is None and intents[0].speed_cap is None
    assert not intents[1].hold and intents[2].hold  # rho=0 is not HOLD
    assert intents[2].expires_at == 5
    poses[:] = 100
    np.testing.assert_allclose(intents[0].goal_xy, [1, 3.6])
    angle = 0.8
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    base = np.array([[[0.1, 0.2, 0.3]]])
    shifted = base.copy()
    shifted[0, 0, :2] = rot @ base[0, 0, :2] + [4, -2]
    shifted[..., 2] += angle
    move = PolicyAction(np.array([[[1.0, -0.7]]]), np.array([[False]]))
    a = move.decode(base, [0.0], 1.6, np.pi)[0]
    b = move.decode(shifted, [0.0], 1.6, np.pi)[0]
    np.testing.assert_allclose(b.goal_xy, rot @ a.goal_xy + [4, -2])


@pytest.mark.parametrize(
    "physical,hold",
    [
        ([[[1.7, 0]]], [[False]]),
        ([[[-0.1, 0]]], [[False]]),
        ([[[0, 4]]], [[False]]),
        ([[[float("nan"), 0]]], [[False]]),
        ([[[1, 0]]], [[True]]),
        ([[[0, 0]]], [[1]]),
        ([[0, 0]], [[False]]),
    ],
)
def test_invalid_action_rejected_before_environment_mutation(env, physical, hold):
    before = env.core.positions.clone()
    intents = env.intents.copy()
    physical, hold = np.array(physical), np.array(hold)
    if physical.shape == (1, 1, 2):
        physical = np.tile(physical, (2, 4, 1))
        hold = np.tile(hold, (2, 4))
    with pytest.raises(ValueError):
        env.step(PolicyAction(physical, hold), max_low_steps=1)
    assert env.intents == intents
    torch.testing.assert_close(before, env.core.positions)


def test_log_prob_matches_independent_transformed_distribution(record_property):
    d = distribution()
    latent = torch.linspace(-2, 2, 16, dtype=torch.float64).reshape(8, 2)
    hold = torch.tensor([True, False] * 4)
    scale = torch.tensor([0.8, math.pi], dtype=torch.float64)
    offset = torch.tensor([0.8, 0.0], dtype=torch.float64)
    reference = TransformedDistribution(d.normal, [TanhTransform(), AffineTransform(offset, scale)])
    physical = d.transform(latent)
    expected = torch.where(
        hold,
        d.branch.log_prob(hold.double()),
        d.branch.log_prob(hold.double()) + reference.log_prob(physical).sum(-1),
    )
    actual = d.log_prob(latent, hold)
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-8)
    error = float((actual - expected).abs().max())
    record_property("max_analytic_log_prob_error", error)


def test_probability_normalization_in_physical_coordinates(record_property):
    # Midpoint quadrature, independent inverse transform and explicit density Jacobians.
    n = 700
    rho = (torch.arange(n, dtype=torch.float64) + 0.5) * 1.6 / n
    beta = -math.pi + (torch.arange(n, dtype=torch.float64) + 0.5) * 2 * math.pi / n
    z0, z1 = torch.atanh(2 * rho / 1.6 - 1), torch.atanh(beta / math.pi)
    norm = Normal(torch.tensor(0.0, dtype=torch.float64), torch.tensor(0.6, dtype=torch.float64))
    f0 = norm.log_prob(z0).exp() / (0.8 * (1 - (2 * rho / 1.6 - 1) ** 2))
    f1 = norm.log_prob(z1).exp() / (math.pi * (1 - (beta / math.pi) ** 2))
    integral = f0.sum() * 1.6 / n * f1.sum() * 2 * math.pi / n
    p = 0.37
    total = p + (1 - p) * integral
    assert abs(float(total) - 1) < 1e-5
    record_property("mixture_mass_error", abs(float(total) - 1))


@pytest.mark.parametrize("logit", [-100.0, 0.0, 100.0])
def test_extreme_latent_and_logits_finite(logit):
    d = distribution(logits=logit)
    z = torch.tensor([[20.0, -20.0]] * 8, dtype=torch.float64)
    hold = torch.tensor([True, False] * 4)
    assert torch.isfinite(d.log_prob(z, hold)).all()
    assert torch.isfinite(d.entropy()).all()
    action, latent, logs = d.sample(deterministic=True)
    assert torch.equal(action.hold, torch.full((8,), logit > 0))
    assert torch.all(latent[action.hold] == 0) and torch.all(action.physical[action.hold] == 0)
    torch.testing.assert_close(logs, d.log_prob(latent, action.hold), atol=0, rtol=0)


def test_hold_log_prob_ignores_unused_continuous_and_has_no_continuous_gradient():
    mean = torch.zeros(3, 2, dtype=torch.float64, requires_grad=True)
    logstd = torch.zeros(2, dtype=torch.float64, requires_grad=True)
    logits = torch.zeros(3, dtype=torch.float64, requires_grad=True)
    d = PolarHoldDistribution(mean, logstd.exp(), logits)
    hold = torch.ones(3, dtype=torch.bool)
    result = d.log_prob(torch.full((3, 2), float("nan"), dtype=torch.float64), hold)
    result.sum().backward()
    assert mean.grad is None and logstd.grad is None
    assert logits.grad is not None and (logits.grad != 0).all()


def test_move_and_entropy_gradients_and_mixture_entropy(record_property):
    torch.manual_seed(23)
    count = 65536
    mean = torch.zeros(count, 2, dtype=torch.float64, requires_grad=True)
    logits = torch.full((count,), 0.3, dtype=torch.float64, requires_grad=True)
    d = PolarHoldDistribution(mean, torch.ones_like(mean) * 0.6, logits)
    estimates = d.entropy()
    # Independent Gauss-Hermite expectation of transformed conditional entropy.
    nodes, weights = np.polynomial.hermite.hermgauss(128)
    z = np.sqrt(2) * 0.6 * nodes
    logjac = 2 * (np.log(2) - z - np.logaddexp(0, -2 * z))
    conditional = 2 * (
        0.5 * math.log(2 * math.pi * math.e * 0.6**2) + float(weights @ logjac) / np.sqrt(np.pi)
    ) + math.log(0.8 * math.pi)
    p = 1 / (1 + math.exp(-0.3))
    expected = -p * math.log(p) - (1 - p) * math.log(1 - p) + (1 - p) * conditional
    error = abs(float(estimates.mean().detach()) - expected)
    tolerance = max(5 * float(estimates.std().detach()) / math.sqrt(count), 1e-3)
    assert error < tolerance
    estimates.mean().backward()
    assert torch.isfinite(mean.grad).all() and torch.isfinite(logits.grad).all()
    record_property("entropy_error", error)
    record_property("entropy_tolerance", tolerance)
    d = distribution()
    d.normal.loc.requires_grad_()
    d.log_prob(
        torch.ones(8, 2, dtype=torch.float64) * 0.4, torch.zeros(8, dtype=torch.bool)
    ).sum().backward()
    assert (d.normal.loc.grad != 0).all()


@pytest.mark.parametrize("mode", ["hold", "move", "mixed"])
def test_ppo_update_and_ratio_one_for_saved_branches(mode, record_property):
    policy = make_policy()
    batch = synthetic_batch(policy, mode)
    hidden = batch["hidden"][0]
    logs = []
    for t in range(len(batch["reward"])):
        if t:
            hidden = hidden * (~batch["terminated"][t - 1])[:, None, None]
        dist, hidden = policy.distribution(batch["obs"][t], hidden)
        logs.append(dist.log_prob(batch["latent"][t], batch["hold"][t]))
    log = torch.stack(logs)
    torch.testing.assert_close(log, batch["log_prob"], atol=1e-6, rtol=1e-6)
    ratio_error = float(((log - batch["log_prob"]).exp() - 1).abs().max().detach())
    record_property("max_replay_ratio_error", ratio_error)
    before = policy.mean.weight.detach().clone()
    stats = ppo_update(
        policy, torch.optim.Adam(policy.parameters(), lr=1e-4), batch, epochs=1, entropy_weight=0
    )
    assert all(np.isfinite(v) for v in stats.values())
    if mode == "hold":
        torch.testing.assert_close(before, policy.mean.weight, atol=0, rtol=0)
    else:
        assert not torch.equal(before, policy.mean.weight)


def test_invalid_nan_rows_and_all_invalid_do_not_update():
    policy = make_policy()
    batch = synthetic_batch(policy)
    batch["learning_valid"][1, 0] = False
    for key in (
        "obs",
        "state",
        "hidden",
        "latent",
        "action",
        "log_prob",
        "reward",
        "value",
        "next_value",
    ):
        batch[key][1, 0] = float("nan")
    stats = ppo_update(policy, torch.optim.Adam(policy.parameters()), batch, epochs=1)
    assert all(np.isfinite(v) for v in stats.values())
    batch["learning_valid"][:] = False
    before = deepcopy(policy.state_dict())
    ppo_update(policy, torch.optim.Adam(policy.parameters()), batch, epochs=1)
    for k, v in policy.state_dict().items():
        torch.testing.assert_close(before[k], v, atol=0, rtol=0)


def test_rollout_version_and_action_tampering_rejected():
    policy = make_policy()
    batch = synthetic_batch(policy, "move")
    optimizer = torch.optim.Adam(policy.parameters())
    for broken in ("version", "physical", "branch"):
        b = deepcopy(batch)
        if broken == "version":
            b.pop("rollout_version")
        elif broken == "physical":
            b["action"][0, 0, 0, 0] += 0.1
        else:
            b["motion_active"][0, 0, 0] = False
        with pytest.raises(ValueError):
            ppo_update(policy, optimizer, b, epochs=1)


class RecordingAdapter(StationaryTestAdapter):
    def __init__(self):
        self.requests = []

    def plan(self, item):
        self.requests.append(item)
        return super().plan(item)


def test_environment_move_uses_true_intent_and_observation_schema(env):
    actor, state = env.observe()
    assert actor.shape == (2, 4, 411) and state.shape == (2, 991)
    assert actor[:, :, 12:14].eq(0).all()
    assert actor[:, :, -116:-111].eq(0).all()
    env.planners = [RecordingAdapter() for _ in env.planners]
    action = PolicyAction(np.tile([1.0, 0.2], (2, 4, 1)), np.zeros((2, 4), bool))
    step = env.step(action, max_low_steps=2)
    for planner in env.planners:
        first, second = planner.requests
        assert first.intent is second.intent
        assert first.intent.terminal_yaw is None and first.intent.speed_cap is None
    assert all(g is None for g in env.goals)
    assert env.core.committed_reference_speed.eq(0).all()
    assert env.core.committed_planned_yaw_delta.eq(0).all()
    extra = step.actor[:, :, -116:-106]
    assert extra[:, :, 2].eq(0).all() and extra[:, :, 3].eq(1).all()
    torch.testing.assert_close(extra[:, :, 4], torch.full((2, 4), 0.6))
    assert step.records[0]["request"]["version"] == ACTION_VERSION
    saved = env.state_dict()
    env.reset([0])
    env.load_state_dict(saved)
    assert all(i is not None for i in env.intents)
    with pytest.raises(ValueError):
        env.step(action, hold_mask=np.zeros((2, 4), bool))


def test_hold_and_move_dispatch_without_invented_constraints(env):
    move = PolicyAction(np.tile([1.0, 0.0], (2, 4, 1)), np.zeros((2, 4), bool))
    result = env.step(move, max_low_steps=1)
    assert all(
        r["fallback_reason"] == "unsupported_intent" for r in result.records[0]["planner_execution"]
    )
    hold = PolicyAction(np.zeros((2, 4, 2)), np.ones((2, 4), bool))
    result = env.step(hold, max_low_steps=1)
    assert all(
        r["fallback_reason"] != "unsupported_intent" for r in result.records[0]["planner_execution"]
    )
    assert not any(ep["success"] for ep in result.episodes)
    assert result.records[0]["hold_request"] == [[True] * 4] * 2


def test_metadata_builder_checkpoints_and_evaluation_gate(env):
    raw = load_yaml(CONFIG)
    assert resolved_contracts(raw) == env.contracts
    assert env.contracts["rollout_version"] == ROLLOUT_VERSION
    policy = build_policy(env)
    sample = policy.sample(*[env.observe()[0], torch.zeros(2, 4, 128)])
    assert isinstance(sample, HybridSample) and sample.latent.shape == (2, 4, 2)
    for name in ("exp167_local_goal_nmpc", "p0_contract_v2", "t03_planner_protocol"):
        cfg = cfg_from_experiment(ROOT / f"configs/experiment/{name}.yaml")
        old = contract_metadata(cfg)
        with pytest.raises(ValueError):
            validate_checkpoint_contract({"contracts": old}, env.contracts)
    other = dict(env.contracts, rho_max=1.2)
    with pytest.raises(ValueError):
        validate_checkpoint_contract({"contracts": other}, env.contracts)
    with pytest.raises(ValueError, match="T05"):
        validate_evaluation_gate(raw)
    with pytest.raises(ValueError, match="paused"):
        validate_training_gate(raw)
    t03 = load_yaml(ROOT / "configs/experiment/t03_planner_protocol.yaml")
    assert resolved_contracts(t03)["planner_adapter"] == "nmpc_bridge_v1"
    validate_evaluation_gate(t03)


@pytest.mark.parametrize(
    "key,value",
    [
        ("action_version", "bad"),
        ("observation_version", "local_goal_clock_v2"),
        ("planner_protocol", "legacy_direct"),
        ("rho_max", 0),
        ("beta_max", 4),
    ],
)
def test_incompatible_config_rejected(env, key, value):
    cfg = deepcopy(env.core.cfg)
    setattr(cfg.local_goal_contract, key, value)
    with pytest.raises(ValueError):
        contract_metadata(cfg)


def test_cli_refuses_training_and_full_eval_before_creating_run(tmp_path):
    for extra in ([], ["--evaluate-only"]):
        run = tmp_path / ("eval" if extra else "train")
        result = subprocess.run(
            [
                str(ROOT / ".venv_isaaclab/bin/python"),
                str(ROOT / "scripts/train_local_goal_nmpc.py"),
                "--config",
                str(CONFIG),
                "--run-dir",
                str(run),
                *extra,
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0 and ("T05" if extra else "paused") in result.stderr
        assert not run.exists()


def test_sender_slot_and_permission_invariance(env, monkeypatch):
    from test_p0_learning_contracts import (
        test_neighbor_slots_follow_sender_across_distance_crossing,
    )
    from test_p0_learning_contracts import test_oracle_and_uncommunicated_truth_invariance_v2

    test_neighbor_slots_follow_sender_across_distance_crossing(env, monkeypatch)
    test_oracle_and_uncommunicated_truth_invariance_v2(env)


@pytest.mark.parametrize("mode", ["missing", "expired", "sparse", "unsupported", "invalid_sender"])
def test_hybrid_neighbor_permissions(env, monkeypatch, mode):
    from test_p0_learning_contracts import test_neighbor_permissions

    test_neighbor_permissions(env, monkeypatch, mode)


def test_stop_publication_requires_actual_linear_and_angular_stop(env):
    hold = PolicyAction(np.zeros((2, 4, 2)), np.ones((2, 4), bool))
    env.intents = hold.decode(env.poses(), np.zeros(2), 1.6, np.pi)
    env.core.velocities_xy[:] = 0
    for angular, linear, expected in ((0.2, 0.0, False), (0.0, 0.06, False), (0.1, 0.05, True)):
        env.received.clear()
        env.core.angular_velocities[:] = angular
        env.core.velocities_xy[:, :, 0] = linear
        env._transport(env.poses())
        full = [m for m in env.received.values() if m["full"]]
        assert full and all(m["stop_intent"] == expected for m in full)


def test_hybrid_async_deadline_reset_and_clock(env):
    from test_p0_learning_contracts import test_mid_macro_deadline_and_async_reset

    # Same actual environment gate, but supply structured actions.
    env.core.step_count[0] = env.core.max_episode_steps - 2
    result = env.step(PolicyAction(np.zeros((2, 4, 2)), np.ones((2, 4), bool)))
    assert result.duration.tolist() == [2, 5] and result.bootstrap_allowed.tolist() == [False, True]
    assert result.actor[0, :, -1].eq(1).all() and result.terminal_actor[0, :, -1].eq(0).all()
    assert all(i is None for i in env.intents[:4]) and all(i is not None for i in env.intents[4:])


def test_real_acados_policy_hold_decelerates_and_dwell_is_physical(tmp_path, record_property):
    assert os.environ.get("ACADOS_SOURCE_DIR"), "Required T04 actual solver validation cannot skip"
    policy = make_policy()
    with torch.no_grad():
        policy.hold_logit.weight.zero_()
        policy.hold_logit.bias.fill_(10)
    sample = policy.sample(torch.zeros(1, 4, 411), torch.zeros(1, 4, 128), deterministic=True)
    assert sample.action.hold.all()
    adapter = NMPCAdapter(NMPCPlanner(build_solver(tmp_path, 0.75), 0.75, deadline=10))
    pose = np.zeros(3)
    previous = np.array([0.8, 0.5])
    memory = TerrainView(flat_memory())
    intent = sample.action.numpy().decode(np.zeros((1, 4, 3)), [0.0], 1.6, np.pi)[0]
    trace = []
    for k in range(5):
        now = k * DT
        item = PlannerInput(
            0, 0, pose, previous, intent, memory, (), previous, now, now + np.arange(N + 1) * DT
        )
        out = adapter.plan(item)
        command = out.nominal_controls[0]
        assert np.all(np.abs(command - previous) <= np.array([0.3001, 0.6001]))
        trace.append(
            {
                "step": k,
                "command": command.tolist(),
                "velocity": out.trajectory.velocities[0].tolist(),
                "valid": out.valid,
                "reason": out.fallback_reason,
            }
        )
        pose, previous = out.predicted_states[1], command.copy()
    assert np.max(np.abs(previous)) < 1e-5
    cfg = cfg_from_experiment(CONFIG)
    positions = torch.tensor([[[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.5, 0.5, 0.0]]])
    velocity = torch.zeros(1, 4, 2)
    common = dict(positions=positions, velocities_xy=velocity)
    metrics = compute_team_metrics(positions, velocity)
    for angular, expected_hold, success in ((0.2, 0, False), (0.0, 25, True)):
        done, hold = compute_done(
            positions,
            velocity,
            metrics,
            torch.tensor([24]),
            torch.tensor([25]),
            480,
            cfg.success_thresholds,
            cfg.safety,
            flatness_ok=torch.tensor([True]),
            task_contract_version="finite_task_v2",
            angular_velocities=torch.full((1, 4), angular),
        )
        assert hold.item() == expected_hold and done.success.item() == success
    record_property("hold_trace", __import__("json").dumps(trace))


def test_gru_sequence_boundary_uses_saved_hidden_and_done_resets():
    policy = make_policy()
    batch = synthetic_batch(policy, length=18)
    seen = []

    def capture(module, args):
        seen.append(args[1].detach().clone().reshape(2, 4, 128))

    handle = policy.gru.register_forward_pre_hook(capture)
    try:
        ppo_update(policy, torch.optim.Adam(policy.parameters(), lr=1e-4), batch, epochs=1)
    finally:
        handle.remove()
    assert len(seen) == 18
    torch.testing.assert_close(seen[0], batch["hidden"][0], atol=0, rtol=0)
    torch.testing.assert_close(seen[16], batch["hidden"][16], atol=0, rtol=0)
    assert seen[1][0].eq(0).all()
    assert seen[1][1].abs().sum() > 0 and seen[16].abs().sum() > 0


def test_hold_branch_never_samples_a_continuous_latent(monkeypatch):
    d = distribution(logits=100)

    def forbidden(*args, **kwargs):
        raise AssertionError("Normal.sample called on HOLD-only batch")

    monkeypatch.setattr(Normal, "sample", forbidden)
    action, latent, log = d.sample()
    assert action.hold.all() and latent.eq(0).all() and torch.isfinite(log).all()


def test_audit_entry_reports_test_only_and_rejects_overwrite(tmp_path):
    from audit_polar_hold import audit

    result = audit(CONFIG)
    assert result["move"]["test_only"] and result["training_interactions"] == 0
    assert result["synthetic_optimizer_updates"] == 1
    assert result["analytic_log_prob_max_error"] < 1e-10
    output = tmp_path / "existing.json"
    output.write_text("preserve")
    run = subprocess.run(
        [
            str(ROOT / ".venv_isaaclab/bin/python"),
            str(ROOT / "scripts/audit_polar_hold.py"),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
    )
    assert run.returncode != 0 and output.read_text() == "preserve"


def test_runner_rejects_wrong_sample_version(monkeypatch):
    policy = make_policy()
    obs, hidden = torch.zeros(2, 4, 411), torch.zeros(2, 4, 128)
    sample = policy.sample(obs, hidden)
    for broken in (
        replace(sample, rollout_version="old"),
        replace(sample, action=replace(sample.action, version="old")),
    ):
        monkeypatch.setattr(policy, "sample", lambda *args, value=broken, **kwargs: value)
        with pytest.raises(ValueError, match="version"):
            sample_for_rollout(policy, obs, hidden)
