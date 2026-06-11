#!/usr/bin/env python
"""Run a short SKRL MAPPO smoke job on the first-stage proxy environment."""

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


TRAINING_SEMANTICS = "skrl_mappo_smoke"


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


def parse_bool_config(value, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return bool(value)


def _validate_homogeneous_spaces(env: MultiRoverGatheringSKRLEnv) -> None:
    first = env.possible_agents[0]
    obs_shape = env.observation_spaces[first].shape
    action_shape = env.action_spaces[first].shape
    for agent_id in env.possible_agents[1:]:
        if env.observation_spaces[agent_id].shape != obs_shape:
            raise ValueError("Shared actor requires homogeneous agent observation spaces.")
        if env.action_spaces[agent_id].shape != action_shape:
            raise ValueError("Shared actor requires homogeneous agent action spaces.")


def build_skrl_mappo_models(
    env: MultiRoverGatheringSKRLEnv,
    *,
    shared_actor: bool = True,
    centralized_critic: bool = True,
    shared_value: bool = True,
) -> dict[str, dict[str, Model]]:
    """Build MAPPO models with project CTDE semantics.

    Policies consume per-agent local observations. Values consume the centralized state returned
    by ``env.state()`` / ``env.state_space``.
    """
    if not centralized_critic:
        raise ValueError("This project only wires SKRL MAPPO with a centralized critic state.")
    if shared_actor or shared_value:
        _validate_homogeneous_spaces(env)

    first_agent = env.possible_agents[0]
    shared_policy = (
        SKRLPolicy(
            env.observation_spaces[first_agent],
            env.action_spaces[first_agent],
            env.device,
        )
        if shared_actor
        else None
    )
    shared_critic = (
        SKRLValue(
            env.observation_spaces[first_agent],
            env.state_space,
            env.action_spaces[first_agent],
            env.device,
        )
        if shared_value
        else None
    )

    models: dict[str, dict[str, Model]] = {}
    for agent_id in env.possible_agents:
        models[agent_id] = {
            "policy": shared_policy
            if shared_actor
            else SKRLPolicy(
                env.observation_spaces[agent_id],
                env.action_spaces[agent_id],
                env.device,
            ),
            "value": shared_critic
            if shared_value
            else SKRLValue(
                env.observation_spaces[agent_id],
                env.state_space,
                env.action_spaces[agent_id],
                env.device,
            ),
        }
    return models


def build_skrl_mappo_memories(
    env: MultiRoverGatheringSKRLEnv,
    *,
    rollout_steps: int,
) -> dict[str, RandomMemory]:
    return {
        agent_id: RandomMemory(
            memory_size=rollout_steps,
            num_envs=env.num_envs,
            device=env.device,
        )
        for agent_id in env.possible_agents
    }


def skrl_mappo_checkpoint_payload(
    models: dict[str, dict[str, Model]],
    possible_agents: list[str],
    *,
    raw_cfg: dict,
    shared_actor: bool,
    centralized_critic: bool,
    shared_value: bool,
    timesteps: int,
) -> dict:
    payload = {
        agent_id: {
            "policy": models[agent_id]["policy"].state_dict(),
            "value": models[agent_id]["value"].state_dict(),
        }
        for agent_id in possible_agents
    }
    payload["metadata"] = {
        "training_semantics": TRAINING_SEMANTICS,
        "backend": "skrl.mappo",
        "shared_actor": shared_actor,
        "centralized_critic": centralized_critic,
        "shared_value": shared_value,
        "timesteps": timesteps,
    }
    payload["cfg"] = raw_cfg
    return payload


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
    shared_actor = parse_bool_config(algo.get("shared_actor"), default=True)
    centralized_critic = parse_bool_config(algo.get("centralized_critic"), default=True)
    shared_value = parse_bool_config(algo.get("shared_value"), default=True)

    models = build_skrl_mappo_models(
        env,
        shared_actor=shared_actor,
        centralized_critic=centralized_critic,
        shared_value=shared_value,
    )
    memories = build_skrl_mappo_memories(env, rollout_steps=int(exp.get("rollout_steps", 32)))

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
    torch.save(
        skrl_mappo_checkpoint_payload(
            models,
            possible_agents,
            raw_cfg=raw_cfg,
            shared_actor=shared_actor,
            centralized_critic=centralized_critic,
            shared_value=shared_value,
            timesteps=args.timesteps,
        ),
        checkpoint_path,
    )
    print(
        yaml.safe_dump(
            {
                "status": "ok",
                "backend": "skrl.mappo",
                "training_semantics": TRAINING_SEMANTICS,
                "shared_actor": shared_actor,
                "centralized_critic": centralized_critic,
                "shared_value": shared_value,
                "timesteps": args.timesteps,
                "checkpoint_path": str(checkpoint_path),
            },
            sort_keys=False,
        )
    )


if __name__ == "__main__":
    main()
