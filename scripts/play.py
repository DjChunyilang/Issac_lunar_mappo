#!/usr/bin/env python
"""Run a saved first-stage policy checkpoint deterministically."""

from __future__ import annotations

import argparse
import json

import gymnasium as gym
import torch

from _common import cfg_from_experiment, load_yaml
from _skrl_metadata import validate_checkpoint_compatibility
from lunar_rover_tasks.tasks.multi_rover_gathering.gathering_env import MultiRoverGatheringCore
from train import Actor
from train_skrl_mappo import (
    SKRLPolicy,
    normalize_actor_architecture,
    normalize_critic_architecture,
)


def _expected_architectures(raw_cfg: dict | None) -> tuple[str | None, str | None]:
    if raw_cfg is None:
        return None, None
    algorithm = raw_cfg.get("algorithm", {})
    if not isinstance(algorithm, dict):
        algorithm = {}
    return (
        normalize_actor_architecture(algorithm.get("actor_architecture", "mlp_v1")),
        normalize_critic_architecture(algorithm.get("critic_architecture", "mlp_v1")),
    )


def _load_policy_players(checkpoint: dict, cfg, device, raw_cfg: dict | None = None):
    expected_actor_architecture, expected_critic_architecture = _expected_architectures(raw_cfg)
    metadata = validate_checkpoint_compatibility(
        checkpoint,
        cfg,
        expected_actor_architecture=expected_actor_architecture,
        expected_critic_architecture=expected_critic_architecture,
    )
    if "actor" in checkpoint:
        actor = Actor(cfg.actor_obs_dim).to(device)
        actor.load_state_dict(checkpoint["actor"])
        actor.eval()

        def act(actor_obs):
            return actor(actor_obs).mean

        return act, "smoke"

    agent_ids = [f"rover_{i}" for i in range(cfg.task.n_agents)]
    if not all(agent_id in checkpoint for agent_id in agent_ids):
        raise KeyError(
            "Unsupported checkpoint format. Expected either a smoke checkpoint with key "
            "'actor' or an SKRL checkpoint with rover_0..rover_N policy entries."
        )

    obs_space = gym.spaces.Box(
        low=-float("inf"),
        high=float("inf"),
        shape=(cfg.actor_obs_dim,),
        dtype=float,
    )
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=float)
    actor_architecture = str(metadata.get("actor_architecture", "mlp_v1"))
    policies = []
    for agent_id in agent_ids:
        policy = SKRLPolicy(
            obs_space,
            action_space,
            device,
            architecture=actor_architecture,
        ).to(device)
        policy.load_state_dict(checkpoint[agent_id]["policy"])
        policy.eval()
        policies.append(policy)

    def act(actor_obs):
        actions = []
        for index, policy in enumerate(policies):
            mean, _ = policy.compute({"observations": actor_obs[:, index, :]}, role="policy")
            actions.append(mean)
        return torch.stack(actions, dim=1)

    return act, "skrl.mappo"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment/exp_001_minimal.yaml")
    parser.add_argument("--checkpoint", default="outputs/checkpoints/exp_001_minimal_proxy.pt")
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()

    raw_cfg = load_yaml(args.config)
    cfg = cfg_from_experiment(args.config)
    map_location = torch.device(cfg.simulation.device)
    if map_location.type == "cuda" and not torch.cuda.is_available():
        map_location = torch.device("cpu")
    checkpoint = torch.load(args.checkpoint, map_location=map_location)
    metadata = checkpoint.get("metadata", {}) if isinstance(checkpoint, dict) else {}
    if cfg.planner.subgoal_filter.mode in {
        "terrain_safe_candidate_curriculum",
        "terrain_safe_candidate_constrained_curriculum",
        "terrain_safe_candidate_soft_progress_curriculum",
        "terrain_safe_candidate_mutual_progress_curriculum",
        "terrain_safe_candidate_hold_progress_curriculum",
    }:
        cfg.planner.subgoal_filter.progress_timestep_override = int(metadata.get("timesteps", 0))
        cfg.planner.subgoal_filter.deterministic_eval = True
    env = MultiRoverGatheringCore(cfg)
    act, backend = _load_policy_players(checkpoint, cfg, env.device, raw_cfg=raw_cfg)
    actor_obs, _ = env.get_observations()
    rewards = []
    for _ in range(args.steps):
        with torch.no_grad():
            action = act(actor_obs)
        out = env.step(action)
        actor_obs = out.actor_obs
        rewards.append(float(out.rewards.mean().detach().cpu()))
    print(
        json.dumps(
            {
                "backend": backend,
                "steps": args.steps,
                "mean_reward": sum(rewards) / len(rewards),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
