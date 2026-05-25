#!/usr/bin/env python
"""Run a short SKRL MAPPO training job on the first-stage proxy environment."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import yaml

from _common import cfg_from_experiment, ensure_output_dir, load_yaml
from lunar_rover_tasks.tasks.multi_rover_gathering.gathering_env import MultiRoverGatheringSKRLEnv

from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.multi_agents.torch.mappo import MAPPO
from skrl.trainers.torch import SequentialTrainer


class SKRLPolicy(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device):
        Model.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
        GaussianMixin.__init__(self, clip_actions=True, clip_log_std=True, reduction="sum")
        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 128),
            nn.ELU(),
            nn.Linear(128, 128),
            nn.ELU(),
            nn.Linear(128, self.num_actions),
        )
        self.log_std_parameter = nn.Parameter(torch.full((self.num_actions,), -0.5))

    def compute(self, inputs, role):
        mean = torch.tanh(self.net(inputs["observations"]))
        return mean, {"log_std": self.log_std_parameter.expand_as(mean)}


class SKRLValue(DeterministicMixin, Model):
    def __init__(self, observation_space, state_space, action_space, device):
        Model.__init__(
            self,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
        )
        DeterministicMixin.__init__(self)
        self.net = nn.Sequential(
            nn.Linear(self.num_states, 128),
            nn.ELU(),
            nn.Linear(128, 128),
            nn.ELU(),
            nn.Linear(128, 1),
        )

    def compute(self, inputs, role):
        return self.net(inputs["states"]), {}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment/exp_001_minimal.yaml")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--timesteps", type=int, default=128)
    args = parser.parse_args()

    raw_cfg = load_yaml(args.config)
    exp = raw_cfg.get("experiment", {})
    algo = raw_cfg.get("algorithm", {})
    cfg = cfg_from_experiment(args.config)
    cfg.simulation.device = args.device

    env = MultiRoverGatheringSKRLEnv(cfg)
    wrapped_env = wrap_env(env, wrapper="isaaclab-multi-agent", verbose=False)
    possible_agents = env.possible_agents
    empty_kwargs = {uid: {} for uid in possible_agents}

    models = {}
    memories = {}
    for agent in possible_agents:
        models[agent] = {
            "policy": SKRLPolicy(env.observation_spaces[agent], env.action_spaces[agent], env.device),
            "value": SKRLValue(
                env.observation_spaces[agent],
                env.state_space,
                env.action_spaces[agent],
                env.device,
            ),
        }
        memories[agent] = RandomMemory(
            memory_size=int(exp.get("rollout_steps", 32)),
            num_envs=env.num_envs,
            device=env.device,
        )

    agent = MAPPO(
        possible_agents=possible_agents,
        models=models,
        memories=memories,
        observation_spaces=env.observation_spaces,
        state_spaces=env.state_spaces,
        action_spaces=env.action_spaces,
        device=env.device,
        cfg={
            "rollouts": int(exp.get("rollout_steps", 32)),
            "learning_epochs": 1,
            "mini_batches": 1,
            "discount_factor": float(algo.get("gamma", 0.99)),
            "learning_rate": float(algo.get("learning_rate", 5.0e-4)),
            "learning_rate_scheduler_kwargs": empty_kwargs,
            "observation_preprocessor_kwargs": empty_kwargs,
            "state_preprocessor_kwargs": empty_kwargs,
            "value_preprocessor_kwargs": empty_kwargs,
            "entropy_loss_scale": 0.01,
            "value_loss_scale": 0.5,
            "random_timesteps": 0,
            "learning_starts": 0,
        },
    )
    trainer = SequentialTrainer(
        env=wrapped_env,
        agents=agent,
        cfg={
            "timesteps": args.timesteps,
            "headless": True,
            "disable_progressbar": True,
            "close_environment_at_exit": False,
        },
    )
    trainer.train()

    checkpoint_dir = ensure_output_dir(exp.get("checkpoint_dir", "outputs/checkpoints"))
    checkpoint_path = checkpoint_dir / "exp_001_minimal_skrl_mappo.pt"
    torch.save({uid: {k: v.state_dict() for k, v in models[uid].items()} for uid in possible_agents}, checkpoint_path)
    print(
        yaml.safe_dump(
            {
                "status": "ok",
                "backend": "skrl.mappo",
                "timesteps": args.timesteps,
                "checkpoint_path": str(checkpoint_path),
            },
            sort_keys=False,
        )
    )


if __name__ == "__main__":
    main()
