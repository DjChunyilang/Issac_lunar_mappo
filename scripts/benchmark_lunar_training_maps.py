#!/usr/bin/env python3
"""No-learning execution checks; never constructs an optimizer or updates weights."""

import argparse
import hashlib
import json
import os
import resource
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
import torch
from _common import cfg_from_experiment
from lunar_rover_tasks.tasks.multi_rover_gathering.gathering_env import MultiRoverGatheringCore
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.environment import (
    LocalGoalEnvironment,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.learning import RecurrentPolicy


def checksum(policy):
    return hashlib.sha256(
        b"".join(p.detach().cpu().numpy().tobytes() for p in policy.parameters())
    ).hexdigest()


def quantiles(times):
    return dict(zip(("p50_ms", "p95_ms", "max_ms"), np.quantile(times, [0.5, 0.95, 1]) * 1000))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/terrain/npb_interface_diagnostic.yaml")
    p.add_argument("--scene", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--mode", choices=["proxy", "nmpc"], required=True)
    p.add_argument("--device", choices=["cpu", "cuda"], required=True)
    p.add_argument("--num-envs", type=int, nargs="+", required=True)
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--solver-cache", type=Path, default=Path(".venv_isaaclab/local_goal_solver_v1"))
    a = p.parse_args()
    if a.steps < 2 or min(a.num_envs) < 1:
        p.error("Positive environment count and >=2 steps required")
    if a.output.exists():
        raise FileExistsError("Use a fresh benchmark output path")
    torch.set_num_threads(1)
    torch.manual_seed(23)
    np.random.seed(23)
    if a.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no silent CPU fallback")

    def sync():
        if a.device == "cuda":
            torch.cuda.synchronize()

    results = []
    for count in a.num_envs:
        cfg = cfg_from_experiment(a.config)
        cfg.simulation.num_envs = count
        cfg.simulation.device = a.device if a.mode == "proxy" else "cpu"
        cfg.terrain.raster_device = a.device
        cfg.terrain.scene_manifests = [str(a.scene.resolve())]
        sync()
        start = time.perf_counter()
        env = (
            MultiRoverGatheringCore(cfg)
            if a.mode == "proxy"
            else LocalGoalEnvironment(cfg, a.solver_cache, policy_device=a.device)
        )
        sync()
        load = time.perf_counter() - start
        core = env if a.mode == "proxy" else env.core
        samples = defaultdict(list)

        def instrument(obj, name, label, sample_sink=samples):
            original = getattr(obj, name)

            def wrapped(*args, **kwargs):
                sync()
                t = time.perf_counter()
                out = original(*args, **kwargs)
                sync()
                sample_sink[label].append(time.perf_counter() - t)
                return out

            setattr(obj, name, wrapped)

        bank = core.terrain_runtime.scene_bank
        instrument(bank, "query", "query_nested")
        instrument(core, "get_observations", "observation_nested")
        instrument(core, "_integrate", "dynamics_nested")
        instrument(core, "_advance_communication", "communication_nested")
        if a.mode == "proxy":
            obs, state = core.get_observations()
            actor_obs = torch.cat([obs, torch.zeros(count, 4, 115, device=a.device)], -1)
        else:
            actor_obs, state = env.observe()
        policy = RecurrentPolicy(state.shape[-1], terrain_channels=3).to(a.device)
        hidden = torch.zeros(count, 4, 128, device=a.device)
        initial = checksum(policy)
        for step in range(a.steps):
            sync()
            t = time.perf_counter()
            with torch.no_grad():
                action, _, _, hidden = policy.sample(actor_obs, hidden, deterministic=True)
                policy.value(state)
            sync()
            samples["actor_critic"].append(time.perf_counter() - t)
            sync()
            t = time.perf_counter()
            if a.mode == "proxy":
                # Deterministic primitive requests validate the existing vectorized
                # proxy, not a replacement NMPC controller or learning algorithm.
                action_shape = (
                    (count, 4)
                    if cfg.planner.action_type
                    in {"spatiotemporal_primitives", "differential_trajectory_primitives"}
                    else (count, 4, 2)
                )
                out = core.step(torch.zeros(*action_shape, device=a.device))
                actor_obs = torch.cat(
                    [out.actor_obs, torch.zeros(count, 4, 115, device=a.device)], -1
                )
                state = out.critic_state
                reward = out.rewards
            else:
                out = env.step(action, max_low_steps=1)
                actor_obs, state, reward = out.actor, out.critic, out.reward
            sync()
            samples["environment_step"].append(time.perf_counter() - t)
            if not (
                torch.isfinite(actor_obs).all()
                and torch.isfinite(state).all()
                and torch.isfinite(reward).all()
            ):
                raise RuntimeError("Nonfinite transition")
        sync()
        t = time.perf_counter()
        # Fixed diagnostic backward only; no optimizer, rollout loss or parameter update.
        normal, _ = policy.distribution(actor_obs.detach(), torch.zeros_like(hidden))
        (normal.mean.square().mean() + policy.value(state.detach()).square().mean()).backward()
        sync()
        backward = time.perf_counter() - t
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in policy.parameters())
        assert checksum(policy) == initial
        record = {
            "mode": a.mode,
            "device": a.device,
            "num_envs": count,
            "agents": 4,
            "steps": a.steps,
            "learning_updates": 0,
            "parameters_unchanged": True,
            "load_reset_seconds": load,
            "stages": {k: dict(calls=len(v), **quantiles(v)) for k, v in samples.items()},
            "fixed_diagnostic_backward_ms": backward * 1000,
            "joint_four_rover_steps_per_second": count / np.mean(samples["environment_step"]),
            "rss_process_peak_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            "map_cache_mib": bank.bytes / 1024**2,
            "terrain_contract": bank.contract(),
            "actor_shape": list(actor_obs.shape),
            "critic_shape": list(state.shape),
            "invalid_terrain_agents_last": int(core.last_terrain_unknown.sum()),
            "blocked_terrain_agents_last": int(core.last_terrain_blocked.sum()),
        }
        if a.device == "cuda":
            record["cuda_peak_allocated_mib"] = torch.cuda.max_memory_allocated() / 1024**2
            record["cuda_peak_reserved_mib"] = torch.cuda.max_memory_reserved() / 1024**2
        if a.mode == "nmpc":
            record["planner"] = [asdict(f) for f in env.metrics]
            # Snapshot excludes raster tensors, and validates immutable scene identity.
            snapshot = env.state_dict()
            env.load_state_dict(snapshot)
            record["snapshot_restore"] = True
            env.close()
        bank.close()
        results.append(record)
        print(
            json.dumps(
                {k: v for k, v in record.items() if k not in ("planner", "terrain_contract")}
            ),
            flush=True,
        )
        del env, core, policy, bank
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(
        json.dumps(
            {
                "torch_version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "config": a.config,
                "scene": str(a.scene),
                "results": results,
                "timing_note": "Instrumented synchronized timings; nested stages overlap. CPU solver persists in NMPC mode; not a training throughput estimate.",
            },
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
