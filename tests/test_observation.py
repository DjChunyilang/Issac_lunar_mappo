from __future__ import annotations

import torch

from lunar_rover_tasks.tasks.multi_rover_gathering.gathering_env import MultiRoverGatheringCore
from lunar_rover_tasks.tasks.multi_rover_gathering.gathering_env_cfg import make_debug_cfg


def test_actor_observation_does_not_change_when_oracle_changes() -> None:
    cfg = make_debug_cfg(num_envs=2, device="cpu")
    env = MultiRoverGatheringCore(cfg)
    actor_obs, critic_state = env.get_observations()
    assert actor_obs.shape == (2, 4, cfg.actor_obs_dim)
    assert critic_state.shape == (2, cfg.critic_state_dim)
    baseline = actor_obs.clone()
    env.oracle_point += 123.0
    changed_actor, _ = env.get_observations()
    assert torch.allclose(changed_actor, baseline)


def test_critic_state_changes_when_oracle_changes() -> None:
    cfg = make_debug_cfg(num_envs=2, device="cpu")
    env = MultiRoverGatheringCore(cfg)
    _, critic_state = env.get_observations()
    env.oracle_point += 123.0
    _, changed_state = env.get_observations()
    assert not torch.allclose(changed_state, critic_state)
