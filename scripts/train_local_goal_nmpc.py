#!/usr/bin/env python3
"""Bounded exp167 runner. No implicit long training; optional evaluate-only mode."""

import argparse
import ast
import gzip
import hashlib
import json
import os
import random
import resource
import sys
import time
from collections import Counter
from copy import deepcopy
from pathlib import Path

# Must precede numerical-library imports. acados itself is built without OpenMP.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import numpy as np
import torch
import yaml
from _common import ROOT, cfg_from_experiment, load_yaml
from generate_exp156_scenario_manifest import DISTANCES, TOPOLOGIES, scenario_snapshot
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal import VERSION
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.environment import (
    LocalGoalEnvironment,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.learning import (
    RecurrentPolicy,
    ppo_update,
)


def implementation_hash():
    directory = (
        ROOT / "source/lunar_rover_tasks/lunar_rover_tasks/tasks/multi_rover_gathering/local_goal"
    )
    sources = [Path(__file__), *sorted(directory.glob("*.py"))]
    return hashlib.sha256(
        "\n".join(ast.dump(ast.parse(p.read_text())) for p in sources).encode()
    ).hexdigest()


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def planner_statistics(feedback):
    if not feedback:
        return {}
    times = np.array([f.solve_seconds for f in feedback])
    return {
        "calls": len(feedback),
        "solve_seconds_quantiles": dict(
            zip(
                ["p50", "p90", "p95", "p99", "max"],
                np.quantile(times, [0.5, 0.9, 0.95, 0.99, 1]).tolist(),
            )
        ),
        "status": dict(Counter(f.status for f in feedback)),
        "reasons": dict(Counter(f.reason for f in feedback if f.reason)),
        "failure_ratio": sum(bool(f.reason) for f in feedback) / len(feedback),
        "blocked_ratio": sum(f.blocked_seconds > 0 for f in feedback) / len(feedback),
        "solver_nonzero_ratio": sum(f.solver_status != 0 for f in feedback) / len(feedback),
        "emergency_brake_ratio": sum(f.emergency_brake for f in feedback) / len(feedback),
        "prefix_ratio": sum(f.prefix_steps > 0 for f in feedback) / len(feedback),
        "candidate_rejections": dict(
            Counter(f.candidate_rejection for f in feedback if f.candidate_rejection)
        ),
        "terrain_refresh_ratio": sum(f.solver_passes > 1 for f in feedback) / len(feedback),
        "model_error_m_quantiles": dict(
            zip(
                ["p50", "p95", "max"],
                np.quantile([f.model_error_m for f in feedback], [0.5, 0.95, 1]).tolist(),
            )
        ),
    }


def make_cfg(raw, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return cfg_from_experiment(path)


def evaluate(policy, raw, run, cache, phase, episodes_per_cell):
    # Core scenario construction seeds global libraries. Evaluation must not
    # change the seed-23 training sampling stream.
    saved = (torch.get_rng_state(), np.random.get_state(), random.getstate())
    try:
        return _evaluate(policy, raw, run, cache, phase, episodes_per_cell)
    finally:
        torch.set_rng_state(saved[0])
        np.random.set_state(saved[1])
        random.setstate(saved[2])


def _evaluate(policy, raw, run, cache, phase, episodes_per_cell):
    policy_hash = hashlib.sha256(
        b"".join(v.detach().numpy().tobytes() for v in policy.state_dict().values())
    ).hexdigest()
    report_path = (
        run / "metrics" / ("baseline_eval.json" if phase == "before" else "final_eval_proxy.json")
    )
    report = {
        "version": VERSION,
        "phase": phase,
        "policy_sha256": policy_hash,
        "episodes_per_cell": episodes_per_cell,
        "cells": {},
    }
    if report_path.exists():
        previous = json.loads(report_path.read_text())
        if (
            previous.get("policy_sha256") == policy_hash
            and previous["episodes_per_cell"] == episodes_per_cell
        ):
            report = previous
    for cell_index, (distance, topology) in enumerate(
        (d, t) for d in DISTANCES for t in TOPOLOGIES
    ):
        cell = f"{distance}_{topology.lower()}"
        if cell in report["cells"] and len(report["cells"][cell]["episodes"]) == episodes_per_cell:
            continue
        rows, feedback, snapshots = [], [], []
        cell_start = time.monotonic()
        for batch_index in range((episodes_per_cell + 3) // 4):
            count = min(4, episodes_per_cell - batch_index * 4)
            cell_raw = deepcopy(raw)
            cell_raw["experiment"].update(
                seed=230000 + cell_index * 1000 + batch_index, num_envs=count
            )
            cell_raw["initial_state"].update(DISTANCES[distance])
            cell_raw["terrain"].update(TOPOLOGIES[topology])
            if topology == "Bottleneck":
                cell_raw["terrain"]["crater_count"] = 30
            cfg = make_cfg(cell_raw, run / "config" / f"{cell}_{batch_index}.yaml")
            env = LocalGoalEnvironment(cfg, cache)
            snapshots.extend(scenario_snapshot(env.core))
            obs, _ = env.observe()
            hidden = torch.zeros(count, 4, 128)
            completed = set()
            trace = []
            macro_steps = 0
            while len(completed) < count:
                if macro_steps % 10 == 0:
                    write_json(
                        run / "metrics" / "progress.json",
                        {
                            "phase": phase,
                            "cell": cell,
                            "batch": batch_index + 1,
                            "batches": (episodes_per_cell + 3) // 4,
                            "low_steps": env.core.step_count.tolist(),
                            "completed_in_batch": len(completed),
                            "updated_unix": time.time(),
                        },
                    )
                with torch.no_grad():
                    action, _, _, hidden = policy.sample(obs, hidden, deterministic=True)
                step = env.step(
                    action.numpy(), enabled=torch.tensor([i not in completed for i in range(count)])
                )
                obs = step.actor
                if batch_index == 0:
                    trace.extend(step.records)
                for episode in step.episodes:
                    if episode["env"] not in completed:
                        completed.add(episode["env"])
                        episode["scenario_id"] = batch_index * 4 + episode["env"]
                        rows.append(episode)
                hidden[step.terminated | step.truncated] = 0
                macro_steps += 1
            feedback.extend(env.metrics)
            env.close()
            if batch_index == 0:
                with gzip.open(
                    run / "metrics" / f"{phase}_{cell}_trajectories.json.gz", "wt"
                ) as stream:
                    json.dump(trace, stream)
        digest = hashlib.sha256(json.dumps(snapshots, sort_keys=True).encode()).hexdigest()
        freeze = run / "config" / f"{cell}_frozen_scenarios.json"
        if freeze.exists():
            if json.loads(freeze.read_text())["sha256"] != digest:
                raise RuntimeError("Before/after frozen scenario mismatch")
        else:
            write_json(freeze, {"sha256": digest, "scenarios": snapshots})
        report["cells"][cell] = {
            "episodes": rows,
            "scene_sha256": digest,
            "seconds": time.monotonic() - cell_start,
            **{
                key + "_rate": sum(row[key] for row in rows) / len(rows)
                for key in ("success", "collision", "timeout")
            },
            "planner": planner_statistics(feedback),
        }
        write_json(report_path, report)
        print(
            json.dumps(
                {
                    "phase": phase,
                    "cell": cell,
                    "success_rate": report["cells"][cell]["success_rate"],
                    "seconds": report["cells"][cell]["seconds"],
                }
            ),
            flush=True,
        )
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment/exp167_local_goal_nmpc.yaml")
    parser.add_argument("--output-layout", choices=["run"], default="run")
    parser.add_argument("--run-name", default="pilot_seed23_100k_cpu")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--max-interactions", type=int, default=100000)
    parser.add_argument("--max-seconds", type=float, default=7200)
    parser.add_argument("--eval-episodes", type=int, default=16)
    parser.add_argument(
        "--skip-eval", action="store_true", help="Development smoke only, not a pilot delivery"
    )
    args = parser.parse_args()
    if not 0 < args.max_interactions <= 100000 or not 0 < args.max_seconds <= 7200:
        parser.error("Budget may only be reduced, never extended")
    if not 0 < args.eval_episodes <= 16:
        parser.error("First round permits at most 16 episodes per evaluation cell")
    torch.set_num_threads(1)
    torch.manual_seed(23)
    np.random.seed(23)
    raw = load_yaml(args.config)
    expected = {
        "version": VERSION,
        "planner_revision": "prefix_consistency_v2",
        "decision_steps": 5,
        "horizon_steps": 15,
        "planning_dt": 0.2,
        "displacement_limit": 1.6,
        "speed_limit": 1.15,
        "hidden_size": 128,
        "sequence_length": 16,
        "planner_threads": 4,
        "solver_threads": 1,
    }
    if any(raw.get("local_goal", {}).get(key) != value for key, value in expected.items()):
        raise ValueError("Unsupported local_goal configuration: first-round interface is frozen")
    if raw["algorithm"]["gamma"] != 0.99 or raw["algorithm"]["gae_lambda"] != 0.95:
        raise ValueError("First-round duration-aware discount configuration is frozen")
    args.max_interactions = min(args.max_interactions, raw["local_goal"]["interaction_budget"])
    args.max_seconds = min(args.max_seconds, raw["local_goal"]["wall_seconds"])
    code_hash = implementation_hash()
    run = (
        args.run_dir or ROOT / "outputs" / "runs" / "exp167_local_goal_nmpc" / args.run_name
    ).resolve()
    if (run / "checkpoints" / "latest.pt").exists() and not (args.resume or args.evaluate_only):
        parser.error("Existing run: use --resume or a new --run-name; refusing overwrite")
    for directory in ("config", "metrics", "checkpoints"):
        (run / directory).mkdir(parents=True, exist_ok=True)
    cfg = make_cfg(raw, run / "config" / "experiment.yaml")
    cache = ROOT / ".venv_isaaclab" / "local_goal_solver_v1"
    env = LocalGoalEnvironment(cfg, cache)
    obs, state = env.observe()
    policy = RecurrentPolicy(state.shape[-1])
    optimizer = torch.optim.Adam(policy.parameters(), lr=raw["algorithm"]["learning_rate"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    interactions, elapsed, updates = 0, 0.0, 0
    hidden = torch.zeros(env.e, 4, 128)
    resume = args.resume or (run / "checkpoints" / "latest.pt" if args.evaluate_only else None)
    if resume:
        checkpoint = torch.load(resume, map_location="cpu", weights_only=False)
        if checkpoint["version"] != VERSION or checkpoint["config"] != raw:
            raise ValueError(
                "Checkpoint schema/config mismatch; legacy checkpoints are not supported"
            )
        if checkpoint.get("implementation_sha256") != code_hash:
            raise ValueError(
                "Implementation changed since checkpoint: do not silently mix algorithm versions"
            )
        policy.load_state_dict(checkpoint["policy"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        interactions, elapsed, updates = (
            checkpoint["interactions"],
            checkpoint["elapsed"],
            checkpoint["updates"],
        )
        env.load_state_dict(checkpoint["environment"])
        hidden = checkpoint["hidden"]
        torch.set_rng_state(checkpoint["torch_rng"])
        np.random.set_state(checkpoint["numpy_rng"])
        if "python_rng" in checkpoint:
            random.setstate(checkpoint["python_rng"])
        args.max_interactions = min(
            args.max_interactions, checkpoint.get("budget_interactions", 100000)
        )
        args.max_seconds = min(args.max_seconds, checkpoint.get("budget_seconds", 7200))
        obs, state = env.observe()
    manifest = {
        "version": VERSION,
        "implementation_sha256": code_hash,
        "command": sys.argv,
        "device": "cpu",
        "torch": torch.__version__,
        "config": "config/experiment.yaml",
        "checkpoint": "checkpoints/latest.pt",
        "budget_interactions": args.max_interactions,
        "budget_seconds": args.max_seconds,
        "eval_episodes_per_cell": args.eval_episodes,
        "metrics": [
            "metrics/baseline_eval.json",
            "metrics/final_eval_proxy.json",
            "metrics/summary.json",
        ],
        "status": "running",
        "evaluation_excluded_from_training_budget": True,
    }
    write_json(run / "run_manifest.json", manifest)
    if args.evaluate_only:
        env.close()
        evaluate(policy, raw, run, cache, "after", args.eval_episodes)
        return
    baseline_path = run / "metrics" / "baseline_eval.json"
    baseline_complete = (
        baseline_path.exists() and len(json.loads(baseline_path.read_text()).get("cells", {})) == 6
    )
    if not args.skip_eval and not baseline_complete:
        initial_path = run / "checkpoints" / "initial.pt"
        if not initial_path.exists():
            torch.save(
                {"version": VERSION, "policy": policy.state_dict(), "config": raw}, initial_path
            )
        saved_training_rng = torch.get_rng_state()
        initial_policy = RecurrentPolicy(state.shape[-1])
        initial_policy.load_state_dict(torch.load(initial_path, weights_only=False)["policy"])
        # Construction itself consumes RNG; preserve the exact training stream.
        evaluate(initial_policy, raw, run, cache, "before", args.eval_episodes)
        torch.set_rng_state(saved_training_rng)
    training_start = time.monotonic()
    deadline = training_start + max(0, args.max_seconds - elapsed)
    starting_interactions = interactions
    episodes = []
    feedback = []
    while interactions + env.e <= args.max_interactions and time.monotonic() < deadline:
        buffer = []
        for _ in range(32):
            remaining = args.max_interactions - interactions
            if remaining < env.e or time.monotonic() >= deadline:
                break
            with torch.no_grad():
                action, latent, log, following_hidden = policy.sample(obs, hidden)
                value = policy.value(state)
            step = env.step(
                action.numpy(), max_low_steps=min(5, remaining // env.e), deadline=deadline
            )
            with torch.no_grad():
                terminal_value = policy.value(step.terminal_critic)
            buffer.append(
                {
                    "obs": obs,
                    "state": state,
                    "hidden": hidden,
                    "latent": latent,
                    "log_prob": log,
                    "value": value,
                    "next_value": terminal_value,
                    "reward": step.reward,
                    "duration": step.duration,
                    "terminated": step.terminated,
                    "truncated": step.truncated,
                }
            )
            hidden = following_hidden
            hidden[step.terminated | step.truncated] = 0
            obs, state = step.actor, step.critic
            interactions += int(step.duration.sum())
            episodes.extend(step.episodes)
            # Keep one rollout's full request / trajectory / actual-control trace.
            if updates == 0:
                with gzip.open(run / "metrics" / "training_first_rollout.jsonl.gz", "at") as stream:
                    for record in step.records:
                        stream.write(json.dumps(record) + "\n")
        if not buffer:
            break
        batch = {key: torch.stack([row[key] for row in buffer]) for key in buffer[0]}
        losses = ppo_update(
            policy,
            optimizer,
            batch,
            epochs=raw["algorithm"]["ppo_epochs"],
            clip=raw["algorithm"]["clip_epsilon"],
            entropy_weight=raw["algorithm"]["entropy_loss_scale"],
        )
        scheduler.step()
        updates += 1
        feedback.extend(env.metrics)
        env.metrics.clear()
        consumed = elapsed + time.monotonic() - training_start
        checkpoint = {
            "version": VERSION,
            "implementation_sha256": code_hash,
            "config": raw,
            "policy": policy.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "interactions": interactions,
            "elapsed": consumed,
            "updates": updates,
            "environment": env.state_dict(),
            "hidden": hidden,
            "torch_rng": torch.get_rng_state(),
            "numpy_rng": np.random.get_state(),
            "python_rng": random.getstate(),
            "budget_interactions": args.max_interactions,
            "budget_seconds": args.max_seconds,
        }
        torch.save(checkpoint, run / "checkpoints" / "latest.tmp")
        (run / "checkpoints" / "latest.tmp").replace(run / "checkpoints" / "latest.pt")
        row = {
            "update": updates,
            "interactions": interactions,
            "training_seconds": consumed,
            **losses,
        }
        with (run / "metrics" / "train_metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
        write_json(
            run / "metrics" / "summary.json",
            {
                **row,
                "episodes": episodes,
                "planner": planner_statistics(feedback),
                "env_interactions_per_second": (interactions - starting_interactions)
                / max(time.monotonic() - training_start, 1e-6),
                "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                "stop_reason": "running",
                "formal_convergence_claim": False,
            },
        )
    env.close()
    summary_path = run / "metrics" / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        summary["stop_reason"] = (
            "interaction_budget" if interactions + env.e > args.max_interactions else "wall_budget"
        )
        write_json(summary_path, summary)
    if not args.skip_eval:
        evaluate(policy, raw, run, cache, "after", args.eval_episodes)
    manifest["status"] = (
        "complete_smoke" if args.skip_eval or args.eval_episodes != 16 else "complete_pilot"
    )
    write_json(run / "run_manifest.json", manifest)


if __name__ == "__main__":
    main()
