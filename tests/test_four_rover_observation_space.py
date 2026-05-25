from __future__ import annotations

import torch

from lunar_rover_tasks.tasks.multi_rover_gathering.gathering_env import MultiRoverGatheringCore
from lunar_rover_tasks.tasks.multi_rover_gathering.gathering_env_cfg import make_debug_cfg


def test_four_rover_observation_layout_and_oracle_boundary() -> None:
    cfg = make_debug_cfg(num_envs=1, device="cpu")
    env = MultiRoverGatheringCore(cfg)
    env.positions.copy_(
        torch.tensor(
            [[[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [-1.0, 1.0, 0.0], [1.0, 1.0, 0.0]]],
            device=env.device,
        )
    )
    env.yaws.zero_()
    env.velocities_xy.zero_()
    env.angular_velocities.zero_()
    env.oracle_point.zero_()

    actor_obs, critic_state = env.get_observations()

    assert actor_obs.shape == (1, 4, cfg.actor_obs_dim)
    assert critic_state.shape == (1, cfg.critic_state_dim)
    assert torch.isfinite(actor_obs).all()
    assert torch.isfinite(critic_state).all()

    ego_end = cfg.observation.ego_dim
    neighbor_end = ego_end + cfg.observation.max_neighbors * cfg.observation.neighbor_dim
    terrain_end = neighbor_end + cfg.observation.terrain_dim
    neighbor = actor_obs[..., ego_end:neighbor_end].reshape(
        1,
        cfg.task.n_agents,
        cfg.observation.max_neighbors,
        cfg.observation.neighbor_dim,
    )
    terrain = actor_obs[..., neighbor_end:terrain_end]

    assert torch.all(neighbor[..., 6] == 1.0)
    assert torch.allclose(terrain, torch.zeros_like(terrain))

    baseline_actor = actor_obs.clone()
    baseline_critic = critic_state.clone()
    env.oracle_point += 123.0
    changed_actor, changed_critic = env.get_observations()

    assert torch.allclose(changed_actor, baseline_actor)
    assert not torch.allclose(changed_critic, baseline_critic)

