"""First-stage multi-rover gathering proxy environment.

This module provides a torch-vectorized task core and a gymnasium wrapper. The proxy dynamics are
intentional: the design documents do not define a concrete rover USD/URDF asset or articulation
control interface yet, so this layer validates the planning, observation, reward, and training
contracts before swapping in a true Isaac Sim robot articulation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from lunar_rover_tasks.tasks.multi_rover_gathering.action_interpreter import decode_action
from lunar_rover_tasks.tasks.multi_rover_gathering.gathering_env_cfg import (
    MultiRoverGatheringEnvCfg,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.metrics import TeamMetrics, compute_team_metrics
from lunar_rover_tasks.tasks.multi_rover_gathering.observation import build_actor_observation
from lunar_rover_tasks.tasks.multi_rover_gathering.oracle import (
    compute_geometric_median,
    compute_mean_oracle_distance,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.reward import RewardTerms, compute_reward
from lunar_rover_tasks.tasks.multi_rover_gathering.simple_controller import (
    ControlCommand,
    compute_control,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.state import build_critic_state
from lunar_rover_tasks.tasks.multi_rover_gathering.terrain_features import (
    build_local_terrain_grid,
    is_flat_terrain,
    make_terrain_runtime,
    query_height,
    query_terrain_features,
    randomize_terrain_runtime,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.termination import (
    DoneFlags,
    compute_done,
    compute_success_gates,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.trajectory_generator import (
    Trajectory,
    generate_trajectory,
)
from lunar_rover_tasks.utils.math_utils import finite_or_raise, seed_torch, wrap_to_pi

try:
    import gymnasium as gym

    _GymEnvBase = gym.Env
except Exception:
    _GymEnvBase = object


@dataclass(slots=True)
class StepOutput:
    actor_obs: torch.Tensor
    critic_state: torch.Tensor
    rewards: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    info: dict[str, Any]


class MultiRoverGatheringCore:
    """Torch-vectorized first-stage task core."""

    def __init__(self, cfg: MultiRoverGatheringEnvCfg | None = None):
        self.cfg = cfg or MultiRoverGatheringEnvCfg()
        if self.cfg.simulation.device == "cuda" and not torch.cuda.is_available():
            self.cfg.simulation.device = "cpu"
        self.device = torch.device(self.cfg.simulation.device)
        self.num_envs = self.cfg.simulation.num_envs
        self.n_agents = self.cfg.task.n_agents
        self.generator = seed_torch(self.cfg.seed, str(self.device))
        self.terrain_runtime = make_terrain_runtime(
            self.num_envs,
            device=self.device,
        )
        self.positions = torch.zeros(self.num_envs, self.n_agents, 3, device=self.device)
        self.yaws = torch.zeros(self.num_envs, self.n_agents, device=self.device)
        self.velocities_xy = torch.zeros(self.num_envs, self.n_agents, 2, device=self.device)
        self.angular_velocities = torch.zeros(self.num_envs, self.n_agents, device=self.device)
        self.previous_physical_action = torch.zeros(self.num_envs, self.n_agents, 2, device=self.device)
        self.step_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.success_hold_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.oracle_point = torch.zeros(self.num_envs, 3, device=self.device)
        self.prev_metrics = compute_team_metrics(self.positions, self.velocities_xy)
        self.prev_mean_oracle_distance = torch.zeros(self.num_envs, device=self.device)
        self.metrics = self.prev_metrics
        self.last_trajectory: Trajectory | None = None
        self.last_control: ControlCommand | None = None
        self.last_terrain_features = torch.zeros(self.num_envs, self.n_agents, 5, device=self.device)
        self.last_terrain_speed_scale = torch.ones(self.num_envs, self.n_agents, device=self.device)
        self.last_height_delta = torch.zeros(self.num_envs, self.n_agents, device=self.device)
        self.reset()

    @property
    def communication_radius(self) -> float:
        return float(self.cfg.observation.communication_radius)

    @property
    def max_episode_steps(self) -> int:
        return self.cfg.simulation.max_episode_steps

    def reset(self, env_ids: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        count = int(env_ids.numel())
        self.randomize_terrain(env_ids)
        base_angles = torch.linspace(0.0, 2.0 * torch.pi, self.n_agents + 1, device=self.device)[:-1]
        base = torch.stack((torch.cos(base_angles), torch.sin(base_angles)), dim=-1)
        radius = 3.0 + 1.0 * torch.rand(count, 1, 1, generator=self.generator, device=self.device)
        jitter = 0.35 * torch.randn(count, self.n_agents, 2, generator=self.generator, device=self.device)
        centers = torch.empty(count, 1, 2, device=self.device).uniform_(
            -1.0,
            1.0,
            generator=self.generator,
        )
        xy = centers + radius * base[None, :, :] + jitter
        self.positions[env_ids, :, :2] = xy
        if self._terrain_dynamics_enabled:
            terrain_features = query_terrain_features(
                xy,
                self.cfg.terrain,
                self.terrain_runtime.subset(env_ids),
            )
            self.positions[env_ids, :, 2] = terrain_features[..., 0]
            self.last_terrain_features[env_ids] = terrain_features
        else:
            self.positions[env_ids, :, 2] = 0.0
            self.last_terrain_features[env_ids] = 0.0
        self.last_terrain_speed_scale[env_ids] = 1.0
        self.last_height_delta[env_ids] = 0.0
        self.yaws[env_ids] = torch.atan2(-xy[..., 1], -xy[..., 0])
        self.velocities_xy[env_ids] = 0.0
        self.angular_velocities[env_ids] = 0.0
        self.previous_physical_action[env_ids] = 0.0
        self.step_count[env_ids] = 0
        self.success_hold_count[env_ids] = 0
        self.oracle_point[env_ids] = compute_geometric_median(self.positions[env_ids])
        self.metrics = compute_team_metrics(self.positions, self.velocities_xy)
        self.prev_metrics = self.metrics
        self.prev_mean_oracle_distance[env_ids] = compute_mean_oracle_distance(
            self.positions[env_ids],
            self.oracle_point[env_ids],
        )
        return self.get_observations()

    def get_observations(self) -> tuple[torch.Tensor, torch.Tensor]:
        metrics = compute_team_metrics(self.positions, self.velocities_xy)
        terrain_grid = build_local_terrain_grid(
            self.positions,
            self.yaws,
            self.cfg.terrain,
            self.terrain_runtime,
        )
        actor_obs = build_actor_observation(
            self.positions,
            self.yaws,
            self.velocities_xy,
            self.angular_velocities,
            self.communication_radius,
            self.cfg,
            terrain_grid,
        )
        critic_state = build_critic_state(
            self.positions,
            self.yaws,
            self.velocities_xy,
            self.angular_velocities,
            metrics,
            self.oracle_point,
            self.success_hold_count,
            self.cfg,
            terrain_grid,
        )
        return actor_obs, critic_state

    def step(self, action: torch.Tensor) -> StepOutput:
        action = action.to(device=self.device, dtype=torch.float32)
        if action.shape != (self.num_envs, self.n_agents, 2):
            raise ValueError(
                f"Expected action shape {(self.num_envs, self.n_agents, 2)}, got {tuple(action.shape)}"
            )

        self.prev_metrics = compute_team_metrics(self.positions, self.velocities_xy)
        previous_mean_oracle = self.prev_mean_oracle_distance.clone()
        decoded = decode_action(action, self.positions, self.yaws, self.cfg.planner)
        subgoal_terrain_features = (
            query_terrain_features(
                decoded.world_subgoal[..., :2],
                self.cfg.terrain,
                self.terrain_runtime,
            )
            if self._terrain_dynamics_enabled
            else None
        )
        trajectory = generate_trajectory(
            self.positions,
            decoded.world_subgoal,
            self.cfg.trajectory_generator,
            self.cfg.simulation.planning_dt,
            current_yaws=self.yaws,
        )
        control = compute_control(
            self.positions,
            self.yaws,
            trajectory,
            self.cfg.low_level_control,
        )
        self._integrate(control)
        self.step_count += 1

        metrics = compute_team_metrics(self.positions, self.velocities_xy)
        done, self.success_hold_count = compute_done(
            self.positions,
            self.velocities_xy,
            metrics,
            self.success_hold_count,
            self.step_count,
            self.max_episode_steps,
            self.cfg.success_thresholds,
            self.cfg.safety,
        )
        success_gates = compute_success_gates(metrics, self.velocities_xy, self.cfg.success_thresholds)
        terms, mean_oracle = compute_reward(
            self.positions,
            self.oracle_point,
            self.prev_metrics,
            metrics,
            previous_mean_oracle,
            decoded.physical,
            self.previous_physical_action,
            done,
            self.success_hold_count,
            self.last_terrain_features,
            self.cfg,
            subgoal_terrain_features=subgoal_terrain_features,
            terrain_speed_scale=self.last_terrain_speed_scale,
            height_delta=self.last_height_delta,
        )
        self.prev_mean_oracle_distance = mean_oracle
        self.previous_physical_action = decoded.physical
        self.metrics = metrics
        self.last_trajectory = trajectory
        self.last_control = control
        terrain_features = self.last_terrain_features.clone()
        terrain_speed_scale = self.last_terrain_speed_scale.clone()
        height_delta = self.last_height_delta.clone()
        terrain_runtime = self.terrain_runtime.clone()
        success_hold_count = self.success_hold_count.clone()

        if done.done.any():
            env_ids = torch.nonzero(done.done, as_tuple=False).flatten()
            self.reset(env_ids)

        actor_obs, critic_state = self.get_observations()
        self._check_finite(actor_obs, critic_state, terms.total)
        rewards = terms.total[:, None].expand(-1, self.n_agents)
        return StepOutput(
            actor_obs=actor_obs,
            critic_state=critic_state,
            rewards=rewards,
            terminated=done.terminated,
            truncated=done.truncated,
            info={
                "done": done,
                "success_gates": success_gates,
                "success_hold_count": success_hold_count,
                "reward_terms": terms,
                "metrics": metrics,
                "trajectory": trajectory,
                "control": control,
                "terrain_features": terrain_features,
                "terrain_speed_scale": terrain_speed_scale,
                "height_delta": height_delta,
                "terrain_runtime": terrain_runtime,
                "oracle_point": self.oracle_point.clone(),
            },
        )

    def random_actions(self) -> torch.Tensor:
        return torch.empty(self.num_envs, self.n_agents, 2, device=self.device).uniform_(
            -1.0,
            1.0,
            generator=self.generator,
        )

    @property
    def _terrain_dynamics_enabled(self) -> bool:
        return bool(self.cfg.terrain.dynamics_enabled) and not is_flat_terrain(self.cfg.terrain)

    def randomize_terrain(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        randomize_terrain_runtime(
            self.terrain_runtime,
            env_ids,
            self.cfg.terrain,
            generator=self.generator,
        )

    def _integrate(self, control: ControlCommand) -> None:
        dt = self.cfg.simulation.planning_dt
        old_positions = self.positions.clone()
        self.yaws = wrap_to_pi(self.yaws + control.angular * dt)
        direction = torch.stack((torch.cos(self.yaws), torch.sin(self.yaws)), dim=-1)
        speed_scale = torch.ones_like(control.linear)
        if self._terrain_dynamics_enabled:
            current_features = self.last_terrain_features
            slope_xy = current_features[..., 1:3]
            directional_slope = torch.abs((slope_xy * direction).sum(dim=-1))
            traversability = current_features[..., 4]
            speed_scale = traversability * torch.exp(
                -directional_slope * float(self.cfg.terrain.slope_speed_scale)
            )
            speed_scale = speed_scale.clamp(
                min=float(self.cfg.terrain.min_speed_scale),
                max=1.0,
            )
            delta_xy = direction * control.linear.unsqueeze(-1) * speed_scale.unsqueeze(-1) * dt
            next_xy = old_positions[..., :2] + delta_xy
            next_features = query_terrain_features(
                next_xy,
                self.cfg.terrain,
                self.terrain_runtime,
            )
            self.positions[..., :2] = next_xy
            self.positions[..., 2] = next_features[..., 0]
            self.last_terrain_features = next_features
            self.last_height_delta = self.positions[..., 2] - old_positions[..., 2]
        else:
            delta_xy = direction * control.linear.unsqueeze(-1) * dt
            self.positions[..., :2] = self.positions[..., :2] + delta_xy
            self.positions[..., 2] = 0.0
            self.last_terrain_features.zero_()
            self.last_height_delta.zero_()
        self.last_terrain_speed_scale = speed_scale
        self.velocities_xy = (self.positions[..., :2] - old_positions[..., :2]) / dt
        self.angular_velocities = control.angular

    def _check_finite(
        self,
        actor_obs: torch.Tensor,
        critic_state: torch.Tensor,
        rewards: torch.Tensor,
    ) -> None:
        finite_or_raise("actor_obs", actor_obs)
        finite_or_raise("critic_state", critic_state)
        finite_or_raise("rewards", rewards)


class MultiRoverGatheringGymEnv(_GymEnvBase):
    """Gymnasium wrapper around a single proxy vector environment."""

    metadata = {"render_modes": []}

    def __init__(self, cfg: MultiRoverGatheringEnvCfg | None = None, render_mode: str | None = None):
        del render_mode
        import gymnasium as gym

        self.cfg = cfg or MultiRoverGatheringEnvCfg()
        self.cfg.simulation.num_envs = 1
        self.core = MultiRoverGatheringCore(self.cfg)
        self.observation_space = gym.spaces.Dict(
            {
                "policy": gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.cfg.task.n_agents, self.cfg.actor_obs_dim),
                    dtype=np.float32,
                ),
                "critic": gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.cfg.critic_state_dim,),
                    dtype=np.float32,
                ),
            }
        )
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.cfg.task.n_agents, 2),
            dtype=np.float32,
        )

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        del options
        if seed is not None:
            self.cfg.seed = seed
        actor_obs, critic_state = self.core.reset()
        return self._pack_obs(actor_obs, critic_state), {}

    def step(self, action):
        action_tensor = torch.as_tensor(action, dtype=torch.float32, device=self.core.device).unsqueeze(0)
        output = self.core.step(action_tensor)
        per_agent_reward = output.rewards[0].detach().cpu().numpy().astype(np.float32)
        reward = float(per_agent_reward.mean())
        terminated = bool(output.terminated[0].detach().cpu())
        truncated = bool(output.truncated[0].detach().cpu())
        output.info["per_agent_reward"] = per_agent_reward
        return self._pack_obs(output.actor_obs, output.critic_state), reward, terminated, truncated, output.info

    def _pack_obs(self, actor_obs: torch.Tensor, critic_state: torch.Tensor) -> dict[str, np.ndarray]:
        return {
            "policy": actor_obs[0].detach().cpu().numpy().astype(np.float32),
            "critic": critic_state[0].detach().cpu().numpy().astype(np.float32),
        }

    def close(self) -> None:
        return None


class MultiRoverGatheringSKRLEnv:
    """Minimal multi-agent interface consumed by SKRL's IsaacLabMultiAgentWrapper."""

    def __init__(self, cfg: MultiRoverGatheringEnvCfg | None = None):
        import gymnasium as gym

        self.cfg = cfg or MultiRoverGatheringEnvCfg()
        self.core = MultiRoverGatheringCore(self.cfg)
        self.device = self.core.device
        self.num_envs = self.core.num_envs
        self.num_agents = self.core.n_agents
        self.max_num_agents = self.core.n_agents
        self.possible_agents = [f"rover_{i}" for i in range(self.core.n_agents)]
        self.agents = list(self.possible_agents)
        obs_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.cfg.actor_obs_dim,),
            dtype=np.float32,
        )
        action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_spaces = {agent: obs_space for agent in self.possible_agents}
        self.action_spaces = {agent: action_space for agent in self.possible_agents}
        self.state_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.cfg.critic_state_dim,),
            dtype=np.float32,
        )
        self.unwrapped = self

    @property
    def state_spaces(self):
        return {agent: self.state_space for agent in self.possible_agents}

    def reset(self, seed: int | None = None):
        if seed is not None:
            self.cfg.seed = seed
        actor_obs, _ = self.core.reset()
        return self._split_agents(actor_obs), {}

    def step(self, actions: dict[str, torch.Tensor]):
        action = torch.stack([actions[agent] for agent in self.possible_agents], dim=1)
        output = self.core.step(action)
        observations = self._split_agents(output.actor_obs)
        rewards = {
            agent: output.rewards[:, index] for index, agent in enumerate(self.possible_agents)
        }
        terminated = {agent: output.terminated for agent in self.possible_agents}
        truncated = {agent: output.truncated for agent in self.possible_agents}
        return observations, rewards, terminated, truncated, output.info

    def state(self) -> torch.Tensor:
        _, critic_state = self.core.get_observations()
        return critic_state

    def render(self, *args, **kwargs) -> None:
        return None

    def close(self) -> None:
        return None

    def _split_agents(self, actor_obs: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            agent: actor_obs[:, index, :]
            for index, agent in enumerate(self.possible_agents)
        }


MultiRoverGatheringEnv = MultiRoverGatheringCore
