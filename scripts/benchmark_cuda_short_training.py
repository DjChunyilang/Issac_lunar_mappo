#!/usr/bin/env python
"""Benchmark a short SKRL-MAPPO run when CUDA is visible."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from _common import ROOT, cfg_from_experiment, load_yaml
from _skrl_metadata import resolve_training_semantics
from lunar_rover_tasks.tasks.multi_rover_gathering.gathering_env import MultiRoverGatheringSKRLEnv


def _resolve_output_root(path: str | Path) -> Path:
    output = Path(path)
    if not output.is_absolute():
        output = ROOT / output
    output.mkdir(parents=True, exist_ok=True)
    return output


def _write_result(result: dict, output_root: str | Path) -> dict:
    root = _resolve_output_root(output_root)
    log_dir = root / "logs" / "cuda_short_training"
    log_dir.mkdir(parents=True, exist_ok=True)
    result["artifact"] = str(log_dir / "cuda_benchmark.json")
    with (log_dir / "cuda_benchmark.json").open("w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    return result


def _cuda_device_index(device: str) -> int:
    parsed = torch.device(device)
    return 0 if parsed.index is None else parsed.index


def _cuda_unavailable_result(config: str, device: str, timesteps: int, output_root: str | Path) -> dict:
    result = {
        "status": "cuda_unavailable",
        "config": config,
        "device_requested": device,
        "timesteps": timesteps,
        "cuda_available": False,
        "device_count": torch.cuda.device_count(),
        "torch_version": torch.__version__,
        "message": "CUDA is not visible to this Python process; no training benchmark was run.",
    }
    return _write_result(result, output_root)


def run_cuda_short_training(
    config: str,
    device: str,
    timesteps: int,
    output_root: str | Path = "outputs",
) -> dict:
    requested_device = torch.device(device)
    if requested_device.type == "cuda" and not torch.cuda.is_available():
        return _cuda_unavailable_result(config, device, timesteps, output_root)

    from skrl.envs.wrappers.torch import wrap_env
    from skrl.multi_agents.torch.mappo import MAPPO
    from skrl.trainers.torch import SequentialTrainer
    from train_skrl_mappo import parse_bool_config, build_skrl_mappo_memories, build_skrl_mappo_models

    raw_cfg = load_yaml(config)
    exp = raw_cfg.get("experiment", {})
    algo = raw_cfg.get("algorithm", {})
    training_semantics = resolve_training_semantics(raw_cfg)
    cfg = cfg_from_experiment(config)
    cfg.simulation.device = device

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
            "timesteps": timesteps,
            "headless": True,
            "disable_progressbar": True,
            "close_environment_at_exit": False,
        },
    )

    peak_memory_mb = None
    if env.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(env.device)
        torch.cuda.synchronize(env.device)
    start = time.perf_counter()
    trainer.train()
    if env.device.type == "cuda":
        torch.cuda.synchronize(env.device)
        peak_memory_mb = torch.cuda.max_memory_allocated(env.device) / (1024.0 * 1024.0)
    wall_time_s = time.perf_counter() - start

    env_steps = timesteps * env.num_envs
    agent_steps = env_steps * env.num_agents
    result = {
        "status": "ok",
        "config": config,
        "device_requested": device,
        "device_used": str(env.device),
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
        "gpu_name": (
            torch.cuda.get_device_name(_cuda_device_index(device))
            if requested_device.type == "cuda" and torch.cuda.is_available()
            else None
        ),
        "torch_version": torch.__version__,
        "num_envs": env.num_envs,
        "n_agents": env.num_agents,
        "timesteps": timesteps,
        "env_steps": env_steps,
        "agent_steps": agent_steps,
        "training_semantics": training_semantics,
        "shared_actor": shared_actor,
        "centralized_critic": centralized_critic,
        "shared_value": shared_value,
        "wall_time_s": wall_time_s,
        "env_steps_per_s": env_steps / wall_time_s,
        "agent_steps_per_s": agent_steps / wall_time_s,
        "estimated_seconds_per_1m_env_steps": 1_000_000.0 / (env_steps / wall_time_s),
        "peak_cuda_memory_mb": peak_memory_mb,
    }
    return _write_result(result, output_root)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment/exp_001_minimal.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--timesteps", type=int, default=128)
    parser.add_argument("--out-dir", default="outputs")
    args = parser.parse_args()

    result = run_cuda_short_training(args.config, args.device, args.timesteps, args.out_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
