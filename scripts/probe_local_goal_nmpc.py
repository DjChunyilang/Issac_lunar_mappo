#!/usr/bin/env python3
"""Frozen-state counterexamples; never execute controls or train a policy."""

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

from diagnose_local_goal_nmpc import (
    ROOT,
    LocalGoalEnvironment,
    implementation_hash,
    load_yaml,
    make_cfg,
    np,
    torch,
    witness,
    write_json,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.contracts import LocalGoalRequest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    if (run / "run_manifest.json").exists():
        parser.error("Refusing overwrite")
    torch.set_num_threads(1)
    raw = load_yaml(ROOT / "configs/experiment/exp167_local_goal_nmpc.yaml")
    raw["experiment"].update(seed=230000, num_envs=4)
    env = LocalGoalEnvironment(
        make_cfg(raw, run / "config/experiment.yaml"), ROOT / ".venv_isaaclab/local_goal_solver_v1"
    )
    rows = []
    try:
        for cap, warm in [(0.4, False), (1.15, False), (1.15, True)]:
            pending = []
            for e in range(4):
                for a in range(4):
                    i = e * 4 + a
                    planner = env.planners[i]
                    planner.reset()
                    _, certificate = witness(env, e, a, 1)
                    if warm:
                        planner.previous = planner.rollout(
                            env.poses()[e, a],
                            np.array(certificate["controls"]),
                            env.memories[i],
                            0.0,
                        )
                    goal = LocalGoalRequest(np.array([1.2, 0.0, 0.0]), cap, 0.0, 1.0)
                    future = env.pool.submit(
                        planner.plan,
                        env.poses()[e, a],
                        goal,
                        env.memories[i],
                        env.neighbors(e, a),
                        np.zeros(2),
                        0.0,
                    )
                    pending.append((e, a, certificate, future))
            for e, a, certificate, future in pending:
                executed, feedback = future.result()
                i = e * 4 + a
                planner, memory = env.planners[i], env.memories[i]
                controls = np.stack([planner.solver.get(k, "u") for k in range(15)])
                candidate = planner.rollout(env.poses()[e, a], controls, memory, 0.0)
                shooting = np.stack([planner.solver.get(k, "x")[:3] for k in range(16)])
                points = np.concatenate(
                    [
                        (1 - alpha) * candidate.states[:-1, :2] + alpha * candidate.states[1:, :2]
                        for alpha in np.linspace(0, 1, 9)
                    ]
                )
                features, known, _ = memory.lookup(points)
                bad = ~memory.safe(points)
                disks = memory.free_disks(np.zeros(2))
                node_margin = np.max(
                    disks[None, :, 2]
                    - np.linalg.norm(candidate.states[:, None, :2] - disks[None, :, :2], axis=-1),
                    axis=1,
                )
                shooting_margin = np.max(
                    disks[None, :, 2]
                    - np.linalg.norm(shooting[:, None, :2] - disks[None, :, :2], axis=-1),
                    axis=1,
                )
                rows.append(
                    {
                        "env": e,
                        "agent": a,
                        "cap": cap,
                        "known_feasible_warm_start": warm,
                        "short_motion_certificate": certificate,
                        "feedback": asdict(feedback),
                        "candidate_controls": controls.tolist(),
                        "candidate_states": candidate.states.tolist(),
                        "candidate_node_disk_margin_min": float(node_margin.min()),
                        "solver_node_disk_margin_min": float(shooting_margin.min()),
                        "shooting_rollout_max_xy_difference": float(
                            np.linalg.norm(shooting[:, :2] - candidate.states[:, :2], axis=1).max()
                        ),
                        "bad_sample_count": int(bad.sum()),
                        "unknown_sample_count": int((~known).sum()),
                        "bad_sample_examples": points[bad][:8].tolist(),
                        "bad_sample_features": features[bad][:8].tolist(),
                        "executed_first_control": executed.controls[0].tolist(),
                        "executed_validation_reason": planner.validate(
                            executed, memory, env.neighbors(e, a), cap, np.zeros(2)
                        ),
                        "executed_rollout_error_m": float(
                            np.linalg.norm(
                                planner.rollout(
                                    env.poses()[e, a], executed.controls, memory, 0.0
                                ).states[:, :2]
                                - executed.states[:, :2],
                                axis=1,
                            ).max()
                        ),
                    }
                )
    finally:
        env.close()
    write_json(run / "metrics/frozen_probe.json", rows)
    write_json(
        run / "run_manifest.json",
        {
            "status": "complete_diagnosis",
            "implementation_sha256": implementation_hash(),
            "command": sys.argv,
            "training_updates": 0,
            "environment_interactions": 0,
            "deadline_seconds": 0.15,
            "metrics": ["metrics/frozen_probe.json"],
        },
    )
    print("Frozen probes complete:", len(rows))


if __name__ == "__main__":
    main()
