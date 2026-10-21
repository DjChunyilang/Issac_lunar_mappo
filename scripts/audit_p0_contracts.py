#!/usr/bin/env python3
"""Zero-training finite_task_v2 probes. No solver calls and no checkpoint writes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import torch

from _common import ROOT, cfg_from_experiment, load_yaml
from train_local_goal_nmpc import implementation_hash
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.environment import (
    LocalGoalEnvironment,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.learning import duration_gae
from lunar_rover_tasks.tasks.multi_rover_gathering.metrics import compute_team_metrics
from lunar_rover_tasks.tasks.multi_rover_gathering.termination import compute_done


class NoSolve:
    """Observe-only fixture: absence of solve() makes accidental planning fail."""

    def reset(self):
        pass


def audit(config):
    raw = load_yaml(config)
    cfg = cfg_from_experiment(config)
    if cfg.local_goal_contract.task_contract_version != "finite_task_v2":
        raise ValueError("This audit requires finite_task_v2")
    cfg.simulation.num_envs = 1
    cfg.simulation.device = "cpu"
    torch.set_num_threads(1)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    env = LocalGoalEnvironment(cfg, "unused", solver_factory=lambda *a, **k: NoSolve())
    try:
        actor, critic = env.observe()
        initial_clock = float(actor[0, 0, -1])
        env.core.step_count[:] = env.core.max_episode_steps // 2
        later, state = env.observe()
        assert initial_clock == 1.0 and later[0, 0, -1] == 0.5 and state[0, -1] == 0.5
        positions = env.core.positions
        velocities = torch.zeros_like(env.core.velocities_xy)
        done, _ = compute_done(
            positions,
            velocities,
            compute_team_metrics(positions, velocities),
            torch.zeros(1, dtype=torch.long),
            torch.tensor([env.core.max_episode_steps]),
            env.core.max_episode_steps,
            cfg.success_thresholds,
            cfg.safety,
            flatness_ok=torch.tensor([False]),
            task_contract_version="finite_task_v2",
            angular_velocities=torch.zeros(1, 4),
        )
        assert done.timeout.item() and done.terminated.item() and not done.bootstrap_allowed.item()
        _, target = duration_gae(
            torch.tensor([[-80.0]]),
            torch.zeros(1, 1),
            torch.tensor([[10.0]]),
            torch.tensor([[5]]),
            done.terminated[None],
            done.truncated[None],
            bootstrap_allowed=done.bootstrap_allowed[None],
            learning_valid=done.learning_valid[None],
        )
        assert target.item() == -80.0
        return {
            "scope": "analytic targets and live observation construction; zero training/solver steps",
            "seed": cfg.seed,
            "config": str(config.relative_to(ROOT)),
            "resolved_config_sha256": hashlib.sha256(
                json.dumps(raw, sort_keys=True).encode()
            ).hexdigest(),
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "implementation_sha256": implementation_hash(),
            "audit_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "contracts": env.contracts,
            "actor_dim": actor.shape[-1],
            "critic_dim": critic.shape[-1],
            "clock": {"initial": initial_clock, "half_deadline": float(later[0, 0, -1])},
            "neighbor_sender_indices": env.core.last_communication_snapshot.sender_indices.tolist(),
            "deadline": {
                "terminated": True,
                "truncated": False,
                "bootstrap_allowed": False,
                "illustrative_macro_reward": -80.0,
                "next_value": 10.0,
                "duration_low_steps": 5,
                "return_target": float(target.item()),
            },
            "checks_passed": True,
            "formal_convergence_claim": False,
        }
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/experiment/p0_contract_v2.yaml"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; preserve previous evidence and select a new path")
    result = audit(args.config.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
