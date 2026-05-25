#!/usr/bin/env python
"""Check observation dimensions and oracle separation."""

from __future__ import annotations

import argparse
import json

import torch

from _common import cfg_from_experiment
from lunar_rover_tasks.tasks.multi_rover_gathering.gathering_env import MultiRoverGatheringCore


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment/exp_001_minimal.yaml")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    cfg = cfg_from_experiment(args.config)
    cfg.simulation.device = args.device
    env = MultiRoverGatheringCore(cfg)
    actor_obs, critic_state = env.get_observations()
    if not torch.isfinite(actor_obs).all() or not torch.isfinite(critic_state).all():
        raise RuntimeError("observation/state contains non-finite values")

    old_obs = actor_obs.clone()
    env.oracle_point += 1000.0
    changed_actor, changed_state = env.get_observations()
    actor_delta = (changed_actor - old_obs).abs().max().item()
    state_delta = (changed_state - critic_state).abs().max().item()
    if actor_delta != 0.0:
        raise RuntimeError("oracle information leaked into actor observation")
    if state_delta <= 0.0:
        raise RuntimeError("oracle perturbation did not affect critic state")

    print(
        json.dumps(
            {
                "actor_obs_shape": list(actor_obs.shape),
                "critic_state_shape": list(critic_state.shape),
                "oracle_actor_delta": actor_delta,
                "oracle_critic_delta": state_delta,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

