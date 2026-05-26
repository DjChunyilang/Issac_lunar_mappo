#!/usr/bin/env python
"""Evaluate a proxy gathering policy without rendering."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from _common import ROOT, cfg_from_experiment
from lunar_rover_tasks.tasks.multi_rover_gathering.gathering_env import MultiRoverGatheringCore
from play import _load_policy_players


def _resolve_path(path: str | Path | None) -> Path | None:
    if path is None:
        return None
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = ROOT / resolved
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def evaluate_checkpoint(
    config: str | Path,
    checkpoint: str | Path,
    device: str | None = None,
    num_envs: int = 256,
    steps: int = 100,
    seed: int | None = None,
    output: str | Path | None = None,
) -> dict:
    cfg = cfg_from_experiment(config)
    cfg.simulation.num_envs = num_envs
    if device is not None:
        cfg.simulation.device = device
    if seed is not None:
        cfg.seed = seed

    env = MultiRoverGatheringCore(cfg)
    checkpoint_data = torch.load(checkpoint, map_location=env.device)
    act, backend = _load_policy_players(checkpoint_data, cfg, env.device)
    actor_obs, _ = env.get_observations()

    initial_dmax = env.metrics.dmax.detach().clone()
    initial_dispersion = env.metrics.dispersion.detach().clone()
    final_dmax = initial_dmax.clone()
    final_dispersion = initial_dispersion.clone()
    active = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    success_seen = torch.zeros_like(active)
    collision_seen = torch.zeros_like(active)
    timeout_seen = torch.zeros_like(active)
    done_step = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)
    reward_sum = torch.tensor(0.0, device=env.device)
    reward_count = torch.tensor(0.0, device=env.device)

    for step_id in range(steps):
        active_before = active.clone()
        with torch.no_grad():
            action = act(actor_obs)
        step_output = env.step(action)
        actor_obs = step_output.actor_obs
        metrics = step_output.info["metrics"]
        done = step_output.info["done"]

        final_dmax = torch.where(active_before, metrics.dmax, final_dmax)
        final_dispersion = torch.where(active_before, metrics.dispersion, final_dispersion)
        per_env_reward = step_output.rewards.mean(dim=-1)
        reward_sum = reward_sum + per_env_reward[active_before].sum()
        reward_count = reward_count + active_before.float().sum()

        first_done = done.done & active_before
        done_step = torch.where(first_done, torch.full_like(done_step, step_id + 1), done_step)
        success_seen = success_seen | (done.success & active_before)
        collision_seen = collision_seen | (done.collision & active_before)
        timeout_seen = timeout_seen | (done.timeout & active_before)
        active = active & ~done.done
        if not active.any():
            break

    initial_dmax_mean = initial_dmax.mean()
    final_dmax_mean = final_dmax.mean()
    result = {
        "status": "ok",
        "backend": backend,
        "config": str(config),
        "checkpoint": str(checkpoint),
        "device": str(env.device),
        "num_envs": env.num_envs,
        "steps": steps,
        "initial_dmax": float(initial_dmax_mean.detach().cpu()),
        "final_dmax": float(final_dmax_mean.detach().cpu()),
        "dmax_reduction_ratio": float((final_dmax_mean / initial_dmax_mean.clamp_min(1.0e-6)).detach().cpu()),
        "initial_dispersion": float(initial_dispersion.mean().detach().cpu()),
        "final_dispersion": float(final_dispersion.mean().detach().cpu()),
        "mean_reward": float((reward_sum / reward_count.clamp_min(1.0)).detach().cpu()),
        "success_rate": float(success_seen.float().mean().detach().cpu()),
        "collision_rate": float(collision_seen.float().mean().detach().cpu()),
        "timeout_rate": float(timeout_seen.float().mean().detach().cpu()),
        "finished_rate": float((~active).float().mean().detach().cpu()),
        "mean_done_step": float(done_step[done_step > 0].float().mean().detach().cpu())
        if (done_step > 0).any()
        else None,
    }
    output_path = _resolve_path(output)
    if output_path is not None:
        with output_path.open("w", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2)
        result["artifact"] = str(output_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment/exp_001_minimal.yaml")
    parser.add_argument("--checkpoint", default="outputs/checkpoints/exp_001_minimal_proxy.pt")
    parser.add_argument("--device", default=None)
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    result = evaluate_checkpoint(
        args.config,
        args.checkpoint,
        device=args.device,
        num_envs=args.num_envs,
        steps=args.steps,
        seed=args.seed,
        output=args.output,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
