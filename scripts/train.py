#!/usr/bin/env python
"""First-stage centralized-critic/shared-actor training smoke test.

The script requires SKRL to import when configured, then runs a compact MAPPO-style smoke trainer
against the proxy environment. It validates the first-stage data contract without claiming that
the proxy dynamics are the final Isaac Sim articulation environment.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from _common import cfg_from_experiment, ensure_output_dir, load_yaml
from lunar_rover_tasks.tasks.multi_rover_gathering.gathering_env import MultiRoverGatheringCore


class Actor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, action_dim),
        )
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.4))

    def forward(self, obs: torch.Tensor) -> torch.distributions.Normal:
        mean = torch.tanh(self.net(obs))
        std = torch.exp(self.log_std).expand_as(mean)
        return torch.distributions.Normal(mean, std)


class Critic(nn.Module):
    def __init__(self, state_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state).squeeze(-1)


@dataclass(slots=True)
class Rollout:
    obs: torch.Tensor
    states: torch.Tensor
    actions: torch.Tensor
    log_probs: torch.Tensor
    rewards: torch.Tensor
    values: torch.Tensor
    returns: torch.Tensor


def check_skrl_import(require: bool) -> str:
    if not require:
        return "not-required"
    try:
        import skrl
    except Exception as exc:
        raise RuntimeError("SKRL import failed; Isaac Lab rl[skrl] installation is incomplete") from exc
    return getattr(skrl, "__version__", "installed")


def collect_rollout(
    env: MultiRoverGatheringCore,
    actor: Actor,
    critic: Critic,
    rollout_steps: int,
    gamma: float,
) -> Rollout:
    obs_items = []
    state_items = []
    action_items = []
    log_prob_items = []
    reward_items = []
    value_items = []
    actor_obs, critic_state = env.get_observations()
    for _ in range(rollout_steps):
        flat_obs = actor_obs.reshape(-1, actor_obs.shape[-1])
        dist = actor(flat_obs)
        flat_action = torch.clamp(dist.sample(), -1.0, 1.0)
        flat_log_prob = dist.log_prob(flat_action).sum(dim=-1)
        action = flat_action.view(env.num_envs, env.n_agents, 2).detach()
        value = critic(critic_state)
        out = env.step(action)
        reward = out.rewards.mean(dim=-1)

        obs_items.append(actor_obs.detach())
        state_items.append(critic_state.detach())
        action_items.append(action)
        log_prob_items.append(flat_log_prob.detach().view(env.num_envs, env.n_agents).mean(dim=-1))
        reward_items.append(reward.detach())
        value_items.append(value.detach())
        actor_obs = out.actor_obs
        critic_state = out.critic_state

    rewards = torch.stack(reward_items)
    values = torch.stack(value_items)
    returns = torch.zeros_like(rewards)
    running = critic(critic_state).detach()
    for step in reversed(range(rollout_steps)):
        running = rewards[step] + gamma * running
        returns[step] = running
    return Rollout(
        obs=torch.stack(obs_items),
        states=torch.stack(state_items),
        actions=torch.stack(action_items),
        log_probs=torch.stack(log_prob_items),
        rewards=rewards,
        values=values,
        returns=returns.detach(),
    )


def update(
    rollout: Rollout,
    actor: Actor,
    critic: Critic,
    optimizer: torch.optim.Optimizer,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
) -> dict[str, float]:
    obs = rollout.obs.reshape(-1, rollout.obs.shape[-1])
    states = rollout.states.reshape(-1, rollout.states.shape[-1])
    actions = rollout.actions.reshape(-1, rollout.actions.shape[-1])
    env_actions = rollout.actions.reshape(rollout.actions.shape[0], rollout.actions.shape[1], -1, 2)
    del env_actions
    returns = rollout.returns.reshape(-1)
    values = critic(states)
    advantages = (returns - values.detach())
    advantages = (advantages - advantages.mean()) / (advantages.std().clamp_min(1.0e-6))

    dist = actor(obs)
    agent_log_probs = dist.log_prob(actions).sum(dim=-1)
    grouped_log_probs = agent_log_probs.view(rollout.actions.shape[0], rollout.actions.shape[1], -1).mean(
        dim=-1
    )
    policy_loss = -(grouped_log_probs.reshape(-1) * advantages).mean()
    value_loss = F.mse_loss(values, returns)
    entropy = dist.entropy().sum(dim=-1).mean()
    loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(list(actor.parameters()) + list(critic.parameters()), 1.0)
    optimizer.step()
    return {
        "loss": float(loss.detach().cpu()),
        "policy_loss": float(policy_loss.detach().cpu()),
        "value_loss": float(value_loss.detach().cpu()),
        "entropy": float(entropy.detach().cpu()),
        "mean_reward": float(rollout.rewards.mean().detach().cpu()),
        "mean_return": float(rollout.returns.mean().detach().cpu()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment/exp_001_minimal.yaml")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    raw_cfg = load_yaml(args.config)
    exp = raw_cfg.get("experiment", {})
    algo = raw_cfg.get("algorithm", {})
    cfg = cfg_from_experiment(args.config)
    if args.device is not None:
        cfg.simulation.device = args.device

    torch.manual_seed(cfg.seed)
    skrl_version = check_skrl_import(bool(exp.get("require_skrl_import", True)))
    env = MultiRoverGatheringCore(cfg)
    actor = Actor(cfg.actor_obs_dim).to(env.device)
    critic = Critic(cfg.critic_state_dim).to(env.device)
    optimizer = torch.optim.Adam(
        list(actor.parameters()) + list(critic.parameters()),
        lr=float(algo.get("learning_rate", 5.0e-4)),
    )

    total_steps = int(exp.get("total_steps", 512))
    rollout_steps = int(exp.get("rollout_steps", 32))
    updates = max(1, total_steps // rollout_steps)
    metrics = []
    for update_id in range(updates):
        rollout = collect_rollout(
            env,
            actor,
            critic,
            rollout_steps,
            gamma=float(algo.get("gamma", 0.99)),
        )
        train_metrics = update(rollout, actor, critic, optimizer)
        train_metrics["update"] = update_id
        if not all(torch.isfinite(torch.tensor(v)) for v in train_metrics.values()):
            raise RuntimeError(f"non-finite training metric at update {update_id}: {train_metrics}")
        metrics.append(train_metrics)

    log_dir = ensure_output_dir(exp.get("log_dir", "outputs/logs/exp_001_minimal"))
    checkpoint_dir = ensure_output_dir(exp.get("checkpoint_dir", "outputs/checkpoints"))
    log_path = log_dir / "train_metrics.json"
    with log_path.open("w", encoding="utf-8") as stream:
        json.dump({"skrl_version": skrl_version, "metrics": metrics}, stream, indent=2)

    checkpoint_path = checkpoint_dir / f"{exp.get('name', 'exp_001_minimal_proxy')}.pt"
    torch.save(
        {
            "actor": actor.state_dict(),
            "critic": critic.state_dict(),
            "cfg": raw_cfg,
            "skrl_version": skrl_version,
        },
        checkpoint_path,
    )
    print(
        yaml.safe_dump(
            {
                "status": "ok",
                "skrl_version": skrl_version,
                "updates": updates,
                "last_metrics": metrics[-1],
                "log_path": str(log_path),
                "checkpoint_path": str(checkpoint_path),
            },
            sort_keys=False,
        )
    )


def _dispatch() -> None:
    import sys

    backend = "skrl"
    if "--backend" in sys.argv:
        index = sys.argv.index("--backend")
        try:
            backend = sys.argv[index + 1]
        except IndexError as exc:
            raise SystemExit("--backend requires one of: skrl, smoke") from exc
        del sys.argv[index : index + 2]
    if backend == "skrl":
        from train_skrl_mappo import main as skrl_main

        skrl_main()
    elif backend == "smoke":
        main()
    else:
        raise SystemExit(f"Unknown backend {backend!r}; expected 'skrl' or 'smoke'")


if __name__ == "__main__":
    _dispatch()
