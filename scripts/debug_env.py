#!/usr/bin/env python
"""Run a random-action environment smoke test."""

from __future__ import annotations

import argparse
import json

import torch

from _common import cfg_from_experiment
from lunar_rover_tasks.tasks.multi_rover_gathering.gathering_env import MultiRoverGatheringCore


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment/exp_001_minimal.yaml")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    cfg = cfg_from_experiment(args.config)
    if args.device is not None:
        cfg.simulation.device = args.device
    env = MultiRoverGatheringCore(cfg)
    actor_obs, critic_state = env.get_observations()
    assert actor_obs.shape == (cfg.simulation.num_envs, cfg.task.n_agents, cfg.actor_obs_dim)
    assert critic_state.shape == (cfg.simulation.num_envs, cfg.critic_state_dim)
    reward_mean = 0.0
    done_count = 0
    for _ in range(args.steps):
        out = env.step(env.random_actions())
        reward_mean += float(out.rewards.mean().detach().cpu())
        done_count += int((out.terminated | out.truncated).sum().detach().cpu())
        if not torch.isfinite(out.actor_obs).all():
            raise RuntimeError("actor_obs contains non-finite values")
        if not torch.isfinite(out.critic_state).all():
            raise RuntimeError("critic_state contains non-finite values")
        if not torch.isfinite(out.rewards).all():
            raise RuntimeError("rewards contains non-finite values")
    summary = {
        "steps": args.steps,
        "num_envs": cfg.simulation.num_envs,
        "n_agents": cfg.task.n_agents,
        "actor_obs_shape": list(out.actor_obs.shape),
        "critic_state_shape": list(out.critic_state.shape),
        "mean_reward": reward_mean / args.steps,
        "done_count": done_count,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

