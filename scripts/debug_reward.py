#!/usr/bin/env python
"""Print first-stage reward terms for a short random rollout."""

from __future__ import annotations

import argparse
import json

from _common import cfg_from_experiment
from lunar_rover_tasks.tasks.multi_rover_gathering.gathering_env import MultiRoverGatheringCore


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment/exp_001_minimal.yaml")
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    cfg = cfg_from_experiment(args.config)
    cfg.simulation.device = args.device
    env = MultiRoverGatheringCore(cfg)
    last = None
    for _ in range(args.steps):
        last = env.step(env.random_actions())
    terms = last.info["reward_terms"]
    summary = {
        "gather": float(terms.gather.mean().detach().cpu()),
        "oracle": float(terms.oracle.mean().detach().cpu()),
        "energy": float(terms.energy.mean().detach().cpu()),
        "safety": float(terms.safety.mean().detach().cpu()),
        "terrain": float(terms.terrain.mean().detach().cpu()),
        "flatness": float(terms.flatness.mean().detach().cpu()),
        "motion": float(terms.motion.mean().detach().cpu()),
        "consistency": float(terms.consistency.mean().detach().cpu()),
        "success_hold": float(terms.success_hold.mean().detach().cpu()),
        "terminal": float(terms.terminal.mean().detach().cpu()),
        "total": float(terms.total.mean().detach().cpu()),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
