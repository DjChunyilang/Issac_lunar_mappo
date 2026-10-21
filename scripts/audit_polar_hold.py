#!/usr/bin/env python3
"""T04 contract audit: test-only MOVE adapter, optional real HOLD, synthetic PPO only."""

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import torch
from torch.distributions import Normal, TransformedDistribution
from torch.distributions.transforms import AffineTransform, TanhTransform

from _common import ROOT, cfg_from_experiment, load_yaml
from train_local_goal_nmpc import (
    build_policy,
    implementation_hash,
    sample_for_rollout,
    stack_rollout,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.actions import (
    ACTION_VERSION,
    PolicyAction,
    PolarHoldDistribution,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.contracts import (
    DT,
    N,
    PlannerFeedback,
    TimedTrajectory,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.environment import (
    LocalGoalEnvironment,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.learning import ppo_update
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.memory import LocalTerrainMemory
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.planner import (
    NMPCPlanner,
    build_solver,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.protocol import (
    NMPCAdapter,
    PlannerInput,
    PlannerOutput,
    TerrainView,
)


class NoSolve:
    def reset(self):
        pass


class TestOnlyIntentAdapter:
    """Records MOVE input and returns a stationary fixture, never an operational baseline."""

    test_only = True

    def __init__(self):
        self.last_intent = None

    def plan(self, item):
        item.validate()
        self.last_intent = item.intent
        return PlannerOutput(
            TimedTrajectory(
                np.tile(item.ego_pose, (N + 1, 1)),
                np.zeros((N, 2)),
                np.zeros((N, 2)),
                item.times.copy(),
                item.now,
            ),
            PlannerFeedback(status="feasible"),
            True,
        )

    def reset(self):
        self.last_intent = None

    def state_dict(self):
        return {"test_only": True, "last_intent": deepcopy(self.last_intent)}

    def load_state_dict(self, state):
        if state.get("test_only") is not True:
            raise ValueError("Test-only adapter state mismatch")
        self.last_intent = deepcopy(state["last_intent"])


def synthetic_rollout(policy, actor_dim, critic_dim, *, steps=3):
    """Fixed small synthetic tensors; no simulator transitions or learned-task claims."""
    rows = []
    hidden = torch.zeros(2, 4, 128)
    for t in range(steps):
        obs, state = torch.randn(2, 4, actor_dim) * 0.05, torch.randn(2, critic_dim) * 0.05
        with torch.no_grad():
            _, following, fields = sample_for_rollout(policy, obs, hidden)
            ended = torch.tensor([t == 0, False])
            rows.append(
                {
                    "obs": obs,
                    "state": state,
                    "hidden": hidden.clone(),
                    **fields,
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
            hidden = following.clone()
            hidden[ended] = 0
    return stack_rollout(rows, policy)


def real_hold(policy, cache):
    policy = deepcopy(policy)
    with torch.no_grad():
        # Deterministic diagnostic head setup, not a trained HOLD policy.
        policy.hold_logit.weight.zero_()
        policy.hold_logit.bias.fill_(10)
        sample = policy.sample(torch.zeros(1, 4, 411), torch.zeros(1, 4, 128), deterministic=True)
    assert sample.action.hold.all()
    memory = LocalTerrainMemory()
    xy = np.stack(np.meshgrid(np.arange(-2, 2.01, 0.1), np.arange(-2, 2.01, 0.1)), -1).reshape(
        -1, 2
    )
    features = np.zeros((len(xy), 5))
    features[:, 4] = 1
    memory.observe(xy, features, 0)
    view = TerrainView(memory)
    adapter = NMPCAdapter(NMPCPlanner(build_solver(cache, 0.75), 0.75, deadline=10))
    intent = sample.action.numpy().decode(
        np.zeros((1, 4, 3)), [0.0], policy.rho_max, policy.beta_max
    )[0]
    pose, previous = np.zeros(3), np.array([0.8, 0.5])
    trace = []
    for k in range(5):
        now = k * DT
        item = PlannerInput(
            0, 0, pose, previous, intent, view, (), previous, now, now + np.arange(N + 1) * DT
        )
        output = adapter.plan(item)
        control = output.nominal_controls[0]
        assert np.all(np.abs(control - previous) <= [0.3001, 0.6001])
        trace.append(
            {
                "time": now,
                "control": control.tolist(),
                "pose": output.predicted_states[1].tolist(),
                "actual_velocity": output.trajectory.velocities[0].tolist(),
                "valid": output.valid,
                "fallback_reason": output.fallback_reason,
            }
        )
        pose, previous = output.predicted_states[1], control.copy()
    assert np.max(np.abs(previous)) < 1e-5
    return {
        "executed": True,
        "policy_head": "fixed diagnostic HOLD logits",
        "flat_sensor_memory": True,
        "solver_deadline_s": 10.0,
        "scope": "single-rover limited braking; no task success claim",
        "trace": trace,
    }


def audit(config, *, real_acados=False, cache=None):
    cfg = cfg_from_experiment(config)
    if cfg.local_goal_contract.action_version != ACTION_VERSION:
        raise ValueError("Audit requires polar_hold_v2")
    if cfg.terrain.type == "lunar_scene_pack":
        raise ValueError(
            "This bounded audit uses synthetic maps; real-map schema is covered by tests"
        )
    torch.set_num_threads(1)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    cfg.simulation.num_envs = 1
    env = LocalGoalEnvironment(cfg, "unused", solver_factory=lambda *a, **k: NoSolve())
    try:
        actor, critic = env.observe()
        policy = build_policy(env)
        # Float64 reference uses an independent Torch transformed distribution.
        z = torch.tensor([[0.2, -0.4], [0.5, 0.1]], dtype=torch.float64)
        d = PolarHoldDistribution(
            torch.zeros_like(z),
            torch.ones_like(z) * 0.6,
            torch.zeros(2, dtype=torch.float64),
            policy.rho_max,
            policy.beta_max,
        )
        hold = torch.tensor([True, False])
        scale = torch.tensor([policy.rho_max / 2, policy.beta_max], dtype=torch.float64)
        offset = torch.tensor([policy.rho_max / 2, 0], dtype=torch.float64)
        reference = TransformedDistribution(
            Normal(torch.zeros_like(z), torch.ones_like(z) * 0.6),
            [TanhTransform(), AffineTransform(offset, scale)],
        )
        expected = torch.where(
            hold,
            d.branch.log_prob(hold.double()),
            d.branch.log_prob(hold.double()) + reference.log_prob(d.transform(z)).sum(-1),
        )
        error = float((d.log_prob(z, hold) - expected).abs().max())
        assert error < 1e-10
        env.planners = [TestOnlyIntentAdapter() for _ in env.planners]
        move = PolicyAction(np.tile([1.0, 0.2], (1, 4, 1)), np.zeros((1, 4), bool))
        step = env.step(move, max_low_steps=1)
        assert all(
            p.last_intent.terminal_yaw is None and p.last_intent.speed_cap is None
            for p in env.planners
        )
        probe_policy = deepcopy(policy)
        batch = synthetic_rollout(probe_policy, actor.shape[-1], critic.shape[-1])
        losses = ppo_update(
            probe_policy, torch.optim.Adam(probe_policy.parameters(), lr=1e-4), batch, epochs=1
        )
        assert all(np.isfinite(v) for v in losses.values())
        result = {
            "scope": "T04 contract diagnostics, not complete Gate A or task performance",
            "training_interactions": 0,
            "synthetic_optimizer_updates": 1,
            "seed": cfg.seed,
            "contracts": env.contracts,
            "actor_dim": actor.shape[-1],
            "critic_dim": critic.shape[-1],
            "analytic_log_prob_max_error": error,
            "synthetic_ppo": losses,
            "move": {
                "test_only": True,
                "planner": "TestOnlyIntentAdapter",
                "records": step.records,
            },
            "real_hold": real_hold(policy, cache) if real_acados else {"executed": False},
            "implementation_sha256": implementation_hash(),
            "audit_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "resolved_config_sha256": hashlib.sha256(
                json.dumps(load_yaml(config), sort_keys=True).encode()
            ).hexdigest(),
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
        }
        return result
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/experiment/t04_polar_hold.yaml"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--real-acados", action="store_true")
    parser.add_argument("--cache", type=Path, default=ROOT / ".venv_isaaclab/t04_hold_solver")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite an existing audit")
    report = audit(args.config, real_acados=args.real_acados, cache=args.cache)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "contracts": report["contracts"],
                "analytic_error": report["analytic_log_prob_max_error"],
                "real_hold": report["real_hold"]["executed"],
            }
        )
    )


if __name__ == "__main__":
    main()
