#!/usr/bin/env python3
"""Fixed-goal closed-loop diagnosis only: no policy, optimizer, or constraint changes."""

import argparse
import gzip
import json
import os
import sys
import time
from copy import deepcopy
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import torch
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.contracts import rotation, wrap
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.environment import (
    LocalGoalEnvironment,
)
from train_local_goal_nmpc import (
    ROOT,
    TOPOLOGIES,
    implementation_hash,
    load_yaml,
    make_cfg,
    planner_statistics,
    write_json,
)


def constraints(env, e, a, cap):
    """Read-only necessary conditions; never feed oracle information to the planner."""
    pose = env.poses()[e, a]
    peers = env.neighbors(e, a)
    memory = env.memories[e * 4 + a]
    disks = memory.free_disks(pose[:2])
    margin = [float(np.linalg.norm(pose[:2] - n.positions[0]) - n.radii[0]) for n in peers]
    # Any endpoint is inside a cap*3 reachable ball (slope only reduces speed).
    impossible = [
        bool(np.linalg.norm(pose[:2] - n.positions[-1]) + cap * 3 < n.radii[-1]) for n in peers
    ]
    return {
        "current_terrain_safe": bool(memory.safe(pose[:2])[0]),
        "current_disk_margin": float(
            np.max(disks[:, 2] - np.linalg.norm(disks[:, :2] - pose[:2], axis=1))
        ),
        "neighbor_initial_margins": margin,
        "terminal_reachable_ball_excluded": any(impossible),
        "cap_slew_conflict": bool(abs(env.commands[e, a, 0]) > cap + 0.3001),
        "messages": [
            {
                "sender": s,
                "full": m["full"],
                "age": float(env.core.step_count[e]) * 0.2 - m["timestamp"],
                "has_plan": m["plan"] is not None,
            }
            for (ee, aa, s), m in env.received.items()
            if ee == e and aa == a
        ],
    }


def witness(env, e, a, sign, turn=0.0):
    """A rate-limited straight motion followed by stop, checked using received data only."""
    i = e * 4 + a
    u = np.zeros((15, 2))
    u[:10, 0] = sign * 0.2
    u[:10, 1] = turn
    plan = env.planners[i].rollout(env.poses()[e, a], u, env.memories[i], 0.0)
    reason = env.planners[i].validate(plan, env.memories[i], env.neighbors(e, a), 0.4, np.zeros(2))
    disks = env.memories[i].free_disks(env.poses()[e, a, :2])
    disk_margin = np.max(
        disks[None, :, 2] - np.linalg.norm(plan.states[:, None, :2] - disks[None, :, :2], axis=-1),
        axis=1,
    )
    node_margin = [
        float(np.min(np.linalg.norm(plan.states[:, :2] - n.positions, axis=1) - n.radii))
        for n in env.neighbors(e, a)
    ]
    return plan.states[-1].copy(), {
        "validator_reason": reason,
        "min_disk_margin": float(disk_margin.min()),
        "neighbor_node_margins": node_margin,
        "certified": not reason
        and bool((disk_margin >= -1e-4).all())
        and all(m >= -1e-4 for m in node_margin),
        "controls": u.tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--seconds", type=int, default=12, choices=range(1, 25))
    parser.add_argument(
        "--topologies", nargs="+", default=["Open", "Mixed", "Bottleneck"], choices=list(TOPOLOGIES)
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["forward", "reverse", "cap_drop"],
        choices=[
            "forward",
            "reverse",
            "cap_drop",
            "arc",
            "long_forward",
            "cap_drop_fast",
            "close_stop",
        ],
    )
    args = parser.parse_args()
    run = args.run_dir.resolve()
    if (run / "run_manifest.json").exists():
        parser.error("Refusing to overwrite an existing diagnosis")
    torch.set_num_threads(1)
    raw = load_yaml(ROOT / "configs/experiment/exp167_local_goal_nmpc.yaml")
    manifest = {
        "status": "running_diagnosis",
        "training_updates": 0,
        "implementation_sha256": implementation_hash(),
        "command": sys.argv,
        "deadline_seconds": 0.15,
        "decision_seconds": 1,
        "num_envs": 4,
        "planner_threads": 4,
        "seconds_per_case": args.seconds,
        "goal_source": "fixed endpoint of sensor-only straight-motion witness; not task solution",
        "metrics": ["metrics/summary.json", "metrics/trace.jsonl.gz"],
    }
    write_json(run / "run_manifest.json", manifest)
    (run / "metrics").mkdir(parents=True, exist_ok=True)
    results = {}
    with gzip.open(run / "metrics" / "trace.jsonl.gz", "wt") as stream:
        for topology in args.topologies:
            for mode in args.modes:
                cell = topology.lower() + "_" + mode
                config = deepcopy(raw)
                config["experiment"].update(
                    seed=230000 + list(TOPOLOGIES).index(topology) * 1000, num_envs=4
                )
                config["terrain"].update(TOPOLOGIES[topology])
                if topology == "Bottleneck":
                    config["terrain"]["crater_count"] = 30
                if mode == "close_stop":
                    config["initial_state"].update(
                        spawn_radius_min=0.6, spawn_radius_max=0.6, jitter_std=0.0
                    )
                cfg = make_cfg(config, run / "config" / (cell + ".yaml"))
                env = LocalGoalEnvironment(cfg, ROOT / ".venv_isaaclab" / "local_goal_solver_v1")
                start = env.poses().copy()
                goals = start.copy()
                witnesses = []
                for e in range(4):
                    for a in range(4):
                        goals[e, a], evidence = witness(
                            env, e, a, -1 if mode == "reverse" else 1, 0.5 if mode == "arc" else 0.0
                        )
                        if mode in ("long_forward", "cap_drop_fast"):
                            goals[e, a] = start[e, a] + np.array([1.2, 0.0, 0.0])
                            evidence["certified_short_prefix_only"] = evidence.pop("certified")
                            evidence["certified"] = False
                        if mode == "close_stop":
                            goals[e, a] = start[e, a]
                            evidence["certified_short_prefix_only"] = evidence.pop("certified")
                            evidence["certified"] = False
                        witnesses.append({"env": e, "agent": a, **evidence})
                initial = [constraints(env, e, a, 0.4) for e in range(4) for a in range(4)]
                active = torch.ones(4, dtype=torch.bool)
                episodes, checks, errors, movement = [], [], [], []
                tick = time.monotonic()
                try:
                    for second in range(args.seconds):
                        pose = env.poses()
                        cap = 1.15 if mode == "cap_drop_fast" else 0.4
                        if mode in ("cap_drop", "cap_drop_fast") and second == 1:
                            cap = 0.0
                        if mode == "close_stop":
                            cap = 0.0
                        action = np.zeros((4, 4, 4))
                        for e in range(4):
                            for a in range(4):
                                action[e, a, :2] = (
                                    rotation(-pose[e, a, 2])
                                    @ (goals[e, a, :2] - pose[e, a, :2])
                                    / 1.6
                                )
                                action[e, a, 2] = wrap(goals[e, a, 2] - pose[e, a, 2]) / np.pi
                                action[e, a, 3] = 2 * cap / 1.15 - 1
                        # Same 1 s goal-refresh cadence; endpoints remain fixed in odometry.
                        before = [constraints(env, e, a, cap) for e in range(4) for a in range(4)]
                        step = env.step(np.clip(action, -1, 1), enabled=active)
                        active &= ~(step.terminated | step.truncated)
                        episodes.extend(step.episodes)
                        checks.extend(before)
                        error = np.linalg.norm(env.poses()[..., :2] - goals[..., :2], axis=-1)
                        errors.append(error.tolist())
                        movement.append(
                            np.linalg.norm(env.poses()[..., :2] - start[..., :2], axis=-1).tolist()
                        )
                        stream.write(
                            json.dumps(
                                {
                                    "case": cell,
                                    "second": second,
                                    "goals": goals.tolist(),
                                    "preconditions": before,
                                    "goal_errors": error.tolist(),
                                    "records": step.records,
                                }
                            )
                            + "\n"
                        )
                        if not active.any():
                            break
                    final = env.poses()
                    results[cell] = {
                        "seconds_wall": time.monotonic() - tick,
                        "initial": initial,
                        "witnesses": witnesses,
                        "goals": goals.tolist(),
                        "initial_distance": np.linalg.norm(
                            goals[..., :2] - start[..., :2], axis=-1
                        ).tolist(),
                        "final_error": np.linalg.norm(
                            goals[..., :2] - final[..., :2], axis=-1
                        ).tolist(),
                        "net_displacement": np.linalg.norm(
                            final[..., :2] - start[..., :2], axis=-1
                        ).tolist(),
                        "error_history": errors,
                        "displacement_history": movement,
                        "planner": planner_statistics(env.metrics),
                        "episodes": episodes,
                        "necessary_condition_counts": {
                            "samples": len(checks),
                            "initial_neighbor_violation": sum(
                                any(v < -1e-4 for v in c["neighbor_initial_margins"])
                                for c in checks
                            ),
                            "terminal_reachable_ball_excluded": sum(
                                c["terminal_reachable_ball_excluded"] for c in checks
                            ),
                            "cap_slew_conflict": sum(c["cap_slew_conflict"] for c in checks),
                        },
                    }
                    write_json(run / "metrics" / "summary.json", results)
                    print(
                        json.dumps(
                            {
                                "case": cell,
                                "certified_witnesses": sum(w["certified"] for w in witnesses),
                                "reached": int(
                                    (np.array(results[cell]["final_error"]) < 0.1).sum()
                                ),
                                "failure_ratio": results[cell]["planner"]["failure_ratio"],
                            }
                        ),
                        flush=True,
                    )
                finally:
                    env.close()
    manifest["status"] = "complete_diagnosis"
    write_json(run / "run_manifest.json", manifest)


if __name__ == "__main__":
    main()
