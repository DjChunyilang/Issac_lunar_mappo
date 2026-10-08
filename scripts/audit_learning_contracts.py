#!/usr/bin/env python
"""Read-only learning-contract probes: no optimizer, solver, rollout, or checkpoint.

Synthetic component returns are algebraic counterexamples, not feasible rover
trajectories or evidence that a trained policy exploits the reward.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from _common import ROOT, cfg_from_experiment, load_yaml
from lunar_rover_tasks.tasks.multi_rover_gathering.gathering_env import MultiRoverGatheringCore
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.contracts import DT, LocalGoalRequest
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.learning import duration_gae
from lunar_rover_tasks.tasks.multi_rover_gathering.reward import compute_success_hold_reward
from lunar_rover_tasks.tasks.multi_rover_gathering.termination import compute_done


def discounted_sum(values, gamma):
    return sum(gamma**i * float(value) for i, value in enumerate(values))


def audit(config: Path):
    raw = load_yaml(config)
    if raw.get("local_goal", {}).get("version") != "local_goal_nmpc_v1":
        raise ValueError("This audit describes local_goal_nmpc_v1 only")
    if raw.get("terrain", {}).get("type") == "lunar_scene_pack":
        raise ValueError("Use the legacy exp167 contract; this audit does not evaluate lunar maps")
    cfg = cfg_from_experiment(config)
    cfg.simulation.num_envs = 1
    cfg.simulation.device = "cpu"
    torch.manual_seed(23)
    np.random.seed(23)
    torch.set_num_threads(1)
    core = MultiRoverGatheringCore(cfg)
    gamma, lam = raw["algorithm"]["gamma"], raw["algorithm"]["gae_lambda"]
    steps = raw["local_goal"]["decision_steps"]

    # Freeze sensor and communication caches. Isolate clock / hold inputs, not
    # elapsed-time side effects of a real trajectory.
    actor0, critic0 = core.get_observations()
    core.step_count[:] = core.max_episode_steps - 1
    actor1, critic1 = core.get_observations()
    core.success_hold_count[:] = cfg.success_thresholds.hold_steps - 1
    actor2, critic2 = core.get_observations()
    alias = {
        "scope": "base observations, frozen physical state and message caches",
        "actor_clock_max_difference": float((actor1 - actor0).abs().max()),
        "critic_clock_max_difference": float((critic1 - critic0).abs().max()),
        "actor_hold_max_difference": float((actor2 - actor1).abs().max()),
        "critic_hold_max_difference": float((critic2 - critic1).abs().max()),
        "base_actor_dim": actor0.shape[-1],
        "base_critic_dim": critic0.shape[-1],
        "local_goal_actor_dim": actor0.shape[-1] + 115,
        "local_goal_critic_dim": critic0.shape[-1] + 40,
        "base_neighbor_sender_order": core.last_communication_snapshot.sender_indices.tolist(),
        "extra_neighbor_sender_order": [[j for j in range(4) if j != i] for i in range(4)],
    }

    # Actual production termination function on a nonsuccessful static state.
    done, _ = compute_done(
        core.positions,
        core.velocities_xy,
        core.metrics,
        torch.zeros(1, dtype=torch.long),
        torch.full((1,), core.max_episode_steps, dtype=torch.long),
        core.max_episode_steps,
        cfg.success_thresholds,
        cfg.safety,
        flatness_ok=torch.zeros(1, dtype=torch.bool),
    )
    _, timeout_return = duration_gae(
        torch.tensor([[-80.0]]),
        torch.zeros(1, 1),
        torch.tensor([[10.0]]),
        torch.tensor([[steps]]),
        done.terminated[None],
        done.truncated[None],
        gamma=gamma,
        lam=lam,
    )
    timeout = {
        "terminated": bool(done.terminated[0]),
        "truncated": bool(done.truncated[0]),
        "synthetic_macro_reward": -80.0,
        "synthetic_next_value": 10.0,
        "return_from_production_gae": float(timeout_return[0, 0]),
        "finite_task_terminal_return": -80.0,
        "note": "-80 is an illustrative macro reward, not a reconstructed real timeout",
    }

    hold = {}
    for count in sorted({cfg.success_thresholds.hold_steps, 25}):
        candidate = copy.deepcopy(cfg)
        candidate.success_thresholds.hold_steps = count
        complete = compute_success_hold_reward(torch.arange(1, count + 1), candidate)
        interrupted = compute_success_hold_reward(torch.arange(1, count), candidate).tolist()
        interrupted += [0.0]
        hold[str(count)] = {
            "hold_seconds": count * DT,
            "one_complete_hold_component_sum": float(complete.sum()),
            "one_interrupted_hold_component_sum": sum(interrupted),
            "one_interrupted_hold_component_discounted": discounted_sum(interrupted, gamma),
            "note": "component only; no dynamics, other rewards, or terminal penalties",
        }

    # A cost decreases by one, then returns. Isolate the un-discounted progress
    # formula used by gather/flatness; this is not the total task reward.
    costs = [2.0, 1.0, 2.0]
    progress = [costs[i] - costs[i + 1] for i in range(2)]
    potential = [-cost for cost in costs]
    shaped = [gamma * potential[i + 1] - potential[i] for i in range(2)]
    telescoping = -potential[0] + gamma**2 * potential[-1]
    assert math.isclose(discounted_sum(shaped, gamma), telescoping, abs_tol=1e-12)
    terminal_sums = []
    for path in ([-2.0, -1.0, 0.0], [-2.0, -3.0, 0.0]):
        terms = [gamma * path[i + 1] - path[i] for i in range(2)]
        terminal_sums.append(discounted_sum(terms, gamma))
    assert all(math.isclose(value, 2.0, abs_tol=1e-12) for value in terminal_sums)

    threshold = 2 * 0.05 / 1.15 - 1
    latent_threshold = math.atanh(threshold)
    sigma = math.exp(-0.5)
    p_stop = 0.5 * (1 + math.erf(latent_threshold / (sigma * math.sqrt(2))))
    zero = LocalGoalRequest.from_action(np.zeros(4), np.zeros(3), 0)
    corner = LocalGoalRequest.from_action(np.ones(4), np.zeros(3), 0)

    module_root = ROOT / "source/lunar_rover_tasks/lunar_rover_tasks/tasks/multi_rover_gathering"
    sources = [
        Path(__file__),
        ROOT / "scripts/_common.py",
        ROOT / "scripts/train_local_goal_nmpc.py",
        *[
            module_root / name
            for name in (
                "reward.py",
                "termination.py",
                "observation.py",
                "state.py",
                "communication.py",
                "gathering_env.py",
                "gathering_env_cfg.py",
                "local_goal/contracts.py",
                "local_goal/learning.py",
                "local_goal/environment.py",
                "local_goal/planner.py",
            )
        ],
    ]
    return {
        "scope": "algebra and frozen legacy observation probes; zero training or solver steps",
        "libraries": {"torch": torch.__version__, "numpy": np.__version__},
        "config": str(config.resolve().relative_to(ROOT)),
        "resolved_config_sha256": hashlib.sha256(
            json.dumps(raw, sort_keys=True).encode()
        ).hexdigest(),
        "source_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sources
        },
        "discount": {
            "low_step_s": DT,
            "gamma": gamma,
            "lambda": lam,
            "macro_gamma": gamma**steps,
            "reward_half_life_s": DT * math.log(0.5) / math.log(gamma),
            "gae_trace_half_life_s": DT * math.log(0.5) / math.log(gamma * lam),
            "discount_by_seconds": {str(t): gamma ** round(t / DT) for t in (5, 30, 96, 180)},
            "note": "trace half-life is not the policy memory horizon",
        },
        "timeout": timeout,
        "observation_alias": alias,
        "hold_component": hold,
        "progress_cycle": {
            "costs": costs,
            "undiscounted_component_sum": sum(progress),
            "discounted_component_sum": discounted_sum(progress, gamma),
            "discount_consistent_potential_sum": discounted_sum(shaped, gamma),
            "potential_boundary_term": telescoping,
            "zero_terminal_potential_two_path_sums": terminal_sums,
            "note": "PBRS has a boundary term; true terminal potential must be zero",
        },
        "action": {
            "zero_action_speed_cap_mps": zero.speed_cap,
            "corner_displacement_m": float(np.linalg.norm(corner.target[:2])),
            "stop_speed_cap_threshold_mps": 0.05,
            "normalized_stop_threshold": threshold,
            "zero_mean_initial_gaussian_stop_probability": p_stop,
            "four_independent_stop_requests_probability": p_stop**4,
            "note": "analytic zero-mean prior, not actual initialized/learned policy statistics; "
            "speed cap stop alone does not suppress yaw or prove task success",
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/experiment/exp167_local_goal_nmpc.yaml"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; choose a new file to preserve prior audit results")
    result = audit(args.config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "source_sha256"}, indent=2
        )
    )


if __name__ == "__main__":
    main()
