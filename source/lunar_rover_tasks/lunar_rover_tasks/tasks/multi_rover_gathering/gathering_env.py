"""First-stage multi-rover gathering proxy environment.

This module provides a torch-vectorized task core and a gymnasium wrapper. The proxy dynamics are
intentional: the design documents do not define a concrete rover USD/URDF asset or articulation
control interface yet, so this layer validates the planning, observation, reward, and training
contracts before swapping in a true Isaac Sim robot articulation.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from typing import Any

import numpy as np
import torch

from lunar_rover_tasks.tasks.multi_rover_gathering.action_interpreter import (
    DIFFERENTIAL_PRIMITIVE_ACTION_COUNT,
    SPATIOTEMPORAL_ACTION_COUNT,
    apply_formation_center_correction,
    apply_flat_geometry_capture,
    apply_terminal_slot_capture,
    decode_action,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.analytical_prd import (
    compute_analytical_prd_baseline,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.active_dstc_runtime import (
    ActiveDSTCRuntime,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.communication import (
    CommunicationSnapshot,
    TieredCommunicationCache,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.gathering_env_cfg import (
    MultiRoverGatheringEnvCfg,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.metrics import TeamMetrics, compute_team_metrics
from lunar_rover_tasks.tasks.multi_rover_gathering.mapf_diagnostics import (
    TrajectoryConflictTracker,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.observation import build_actor_observation
from lunar_rover_tasks.tasks.multi_rover_gathering.oracle import (
    OptimalGatherPointResult,
    compute_oracle_distances,
    compute_mean_oracle_distance,
    search_optimal_gather_point,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.reward import (
    RewardTerms,
    compute_centroid_flatness_cost,
    compute_centroid_flatness_reward,
    compute_reward,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.simple_controller import (
    ControlCommand,
    apply_control_safety_projection,
    compute_control,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.state import build_critic_state
from lunar_rover_tasks.tasks.multi_rover_gathering.subgoal_filter import apply_subgoal_filter
from lunar_rover_tasks.tasks.multi_rover_gathering.terrain_features import (
    GatherPointFlatness,
    build_local_terrain_grid,
    build_multiscale_local_terrain_observation,
    build_multiscale_site_belief_observation,
    evaluate_gather_point_flatness,
    is_flat_terrain,
    make_terrain_runtime,
    query_height,
    query_terrain_features,
    randomize_terrain_runtime,
    sample_trajectory_terrain_risk,
    search_local_flatness_center,
    summarize_local_terrain_grid_per_agent,
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
        self.committed_plan_local_xy = torch.zeros(
            self.num_envs, self.n_agents, 2, device=self.device
        )
        self.committed_plan_world_subgoal = torch.zeros(
            self.num_envs, self.n_agents, 3, device=self.device
        )
        self.committed_reference_speed = torch.zeros(
            self.num_envs, self.n_agents, device=self.device
        )
        self.committed_planned_yaw_delta = torch.zeros(
            self.num_envs, self.n_agents, device=self.device
        )
        self.coordination_token = torch.zeros(
            self.num_envs, self.n_agents, device=self.device
        )
        self.step_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.global_step_count = 0
        self.success_hold_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.oracle_point = torch.zeros(self.num_envs, 3, device=self.device)
        # For the v6 execution contract, each rover receives one assigned
        # member of a symmetric formation around ``oracle_point``.  The
        # assignment is fixed for an episode, minimizing reset-time travel and
        # preserving an exact oracle-centroid target without agent IDs.
        self.gather_slot_points = torch.zeros(
            self.num_envs,
            self.n_agents,
            3,
            device=self.device,
        )
        # The actor normally observes the fixed reset-time assignment above.
        # An opt-in terminal contract can replace it for the next action with
        # a current-centroid flat-site assignment while leaving reward and
        # success targets unchanged.
        self.execution_slot_points = torch.zeros_like(self.gather_slot_points)
        self.dynamic_terminal_slot_goal_active = torch.zeros(
            self.num_envs,
            dtype=torch.bool,
            device=self.device,
        )
        self._execution_slot_permutations = torch.tensor(
            list(permutations(range(self.n_agents))),
            dtype=torch.long,
            device=self.device,
        )
        self.oracle_search_objective = torch.zeros(self.num_envs, device=self.device)
        self.oracle_search_feasible = torch.zeros(
            self.num_envs,
            dtype=torch.bool,
            device=self.device,
        )
        self.oracle_search_mean_distance = torch.zeros(self.num_envs, device=self.device)
        self.oracle_search_max_distance = torch.zeros(self.num_envs, device=self.device)
        self.oracle_search_path_risk = torch.zeros(self.num_envs, device=self.device)
        self.oracle_search_path_height_change = torch.zeros(
            self.num_envs,
            device=self.device,
        )
        self.oracle_search_height_range = torch.zeros(self.num_envs, device=self.device)
        self.oracle_search_max_slope = torch.zeros(self.num_envs, device=self.device)
        self.prev_metrics = compute_team_metrics(self.positions, self.velocities_xy)
        self.prev_mean_oracle_distance = torch.zeros(self.num_envs, device=self.device)
        self.prev_oracle_distance_per_agent = torch.zeros(
            self.num_envs, self.n_agents, device=self.device
        )
        self.prev_centroid_flatness_cost = torch.zeros(
            self.num_envs,
            device=self.device,
        )
        self.prev_gather_point_flatness_ok = torch.ones(
            self.num_envs,
            dtype=torch.bool,
            device=self.device,
        )
        self.metrics = self.prev_metrics
        self.last_trajectory: Trajectory | None = None
        self.last_control: ControlCommand | None = None
        self.last_terrain_features = torch.zeros(self.num_envs, self.n_agents, 5, device=self.device)
        self.last_terrain_speed_scale = torch.ones(self.num_envs, self.n_agents, device=self.device)
        self.last_height_delta = torch.zeros(self.num_envs, self.n_agents, device=self.device)
        self.last_steering_angle = torch.zeros(self.num_envs, self.n_agents, device=self.device)
        self.last_actual_yaw_rate = torch.zeros(self.num_envs, self.n_agents, device=self.device)
        self.last_left_wheel_speed = torch.zeros(
            self.num_envs, self.n_agents, device=self.device
        )
        self.last_right_wheel_speed = torch.zeros_like(self.last_left_wheel_speed)
        self.last_turning_radius = torch.full(
            (self.num_envs, self.n_agents),
            float("inf"),
            device=self.device,
        )
        self.communication_cache: TieredCommunicationCache | None = None
        self.last_communication_snapshot: CommunicationSnapshot | None = None
        if self.cfg.observation.schema_version in {
            "ego_v8_decentralized_tiered",
            "ego_v9_multiscale_intent",
            "ego_v10_multiscale_diff_intent",
            "ego_v11_multiscale_site_belief",
        }:
            self.communication_cache = TieredCommunicationCache(
                num_envs=self.num_envs,
                n_agents=self.n_agents,
                max_neighbors=self.cfg.observation.max_neighbors,
                device=self.device,
                full_radius_m=self.communication_radius,
                map_max_distance_m=(
                    2.0 * float(self.cfg.safety.world_xy_limit) * (2.0**0.5)
                ),
                include_plan_intent=(
                    self.cfg.observation.schema_version
                    in {
                        "ego_v9_multiscale_intent",
                        "ego_v10_multiscale_diff_intent",
                        "ego_v11_multiscale_site_belief",
                    }
                ),
                include_plan_yaw=(
                    self.cfg.observation.schema_version
                    in {
                        "ego_v10_multiscale_diff_intent",
                        "ego_v11_multiscale_site_belief",
                    }
                ),
            )
        self.active_dstc_runtime = (
            ActiveDSTCRuntime(
                num_envs=self.num_envs,
                n_agents=self.n_agents,
                device=self.device,
                cfg=self.cfg.active_dstc,
                gather_cfg=self.cfg.gather_point,
            )
            if self.cfg.task.active_dstc_actor_enabled
            else None
        )
        self.trajectory_conflicts = TrajectoryConflictTracker(
            self.num_envs,
            self.n_agents,
            self.device,
        )
        self.reset()

    @property
    def communication_radius(self) -> float:
        return float(self.cfg.observation.communication_radius)

    @property
    def max_episode_steps(self) -> int:
        return self.cfg.simulation.max_episode_steps

    def _terrain_grid(self) -> torch.Tensor:
        return build_local_terrain_grid(
            self.positions,
            self.yaws,
            self.cfg.terrain,
            self.terrain_runtime,
        )

    def _actor_terrain_observation(self) -> torch.Tensor:
        if self.cfg.observation.schema_version == "ego_v11_multiscale_site_belief":
            if self.cfg.task.active_dstc_actor_enabled:
                if self.active_dstc_runtime is None:
                    raise RuntimeError("Active-DSTC Actor runtime is not initialized.")
                return build_multiscale_site_belief_observation(
                    self.positions,
                    self.yaws,
                    self.active_dstc_runtime.target_points,
                    self.cfg.terrain,
                    self.terrain_runtime,
                    site_radius=float(self.cfg.observation.site_belief_radius),
                    potential_sigma=float(self.cfg.observation.site_belief_sigma),
                    site_valid=self.active_dstc_runtime.target_valid,
                )
            if not self.cfg.task.diagnostic_site_belief_enabled:
                raise RuntimeError("Site-belief schema has no configured source.")
            return build_multiscale_site_belief_observation(
                self.positions,
                self.yaws,
                self.oracle_point,
                self.cfg.terrain,
                self.terrain_runtime,
                site_radius=float(self.cfg.observation.site_belief_radius),
                potential_sigma=float(self.cfg.observation.site_belief_sigma),
            )
        if self.cfg.observation.schema_version in {
            "ego_v9_multiscale_intent",
            "ego_v10_multiscale_diff_intent",
        }:
            return build_multiscale_local_terrain_observation(
                self.positions,
                self.yaws,
                self.cfg.terrain,
                self.terrain_runtime,
            )
        return self._terrain_grid()

    def _reset_communication(self, env_ids: torch.Tensor) -> None:
        if self.communication_cache is None:
            return
        terrain_grid = self._terrain_grid()
        self.communication_cache.reset(
            env_ids,
            self.positions,
            self.velocities_xy,
            self.yaws,
            summarize_local_terrain_grid_per_agent(terrain_grid),
            committed_world_subgoal=self.committed_plan_world_subgoal,
            committed_reference_speed=self.committed_reference_speed,
            coordination_token=self.coordination_token,
            committed_planned_yaw_delta=self.committed_planned_yaw_delta,
        )

    def _advance_communication(self) -> None:
        if self.communication_cache is None:
            return
        terrain_grid = self._terrain_grid()
        self.communication_cache.advance(
            dt=float(self.cfg.simulation.planning_dt),
            positions=self.positions,
            velocities_xy=self.velocities_xy,
            yaws=self.yaws,
            terrain_summary=summarize_local_terrain_grid_per_agent(terrain_grid),
            committed_world_subgoal=self.committed_plan_world_subgoal,
            committed_reference_speed=self.committed_reference_speed,
            committed_planned_yaw_delta=self.committed_planned_yaw_delta,
            coordination_token=self.coordination_token,
        )

    def _effective_initial_state_values(self) -> tuple[float, float, float, float]:
        initial_state = self.cfg.initial_state
        if (
            not bool(initial_state.curriculum_enabled)
            or int(initial_state.progress_timestep_override) < 0
        ):
            return (
                float(initial_state.spawn_radius_min),
                float(initial_state.spawn_radius_max),
                float(initial_state.center_xy_range),
                float(initial_state.jitter_std),
            )

        step = max(0, int(initial_state.progress_timestep_override))
        warmup = max(0, int(initial_state.curriculum_warmup_timesteps))
        ramp = max(1, int(initial_state.curriculum_ramp_timesteps))
        alpha = 0.0 if step < warmup else min(1.0, (step - warmup) / float(ramp))

        def lerp(start: float, end: float) -> float:
            return float(start) + alpha * (float(end) - float(start))

        return (
            lerp(initial_state.curriculum_start_spawn_radius_min, initial_state.spawn_radius_min),
            lerp(initial_state.curriculum_start_spawn_radius_max, initial_state.spawn_radius_max),
            lerp(initial_state.curriculum_start_center_xy_range, initial_state.center_xy_range),
            lerp(initial_state.curriculum_start_jitter_std, initial_state.jitter_std),
        )

    def refresh_oracle_point(
        self,
        env_ids: torch.Tensor | None = None,
    ) -> OptimalGatherPointResult:
        """Recompute the fixed per-episode terrain-aware oracle point."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        with torch.no_grad():
            result = search_optimal_gather_point(
                self.positions[env_ids],
                self.cfg.terrain,
                self.cfg.gather_point,
                self.terrain_runtime.subset(env_ids),
                world_xy_limit=float(self.cfg.safety.world_xy_limit),
            )
        self.oracle_point[env_ids] = result.point
        self.oracle_search_objective[env_ids] = result.objective
        self.oracle_search_feasible[env_ids] = result.feasible
        self.oracle_search_mean_distance[env_ids] = result.mean_distance
        self.oracle_search_max_distance[env_ids] = result.max_distance
        self.oracle_search_path_risk[env_ids] = result.path_risk
        self.oracle_search_path_height_change[env_ids] = result.path_height_change
        self.oracle_search_height_range[env_ids] = result.flatness.height_range
        self.oracle_search_max_slope[env_ids] = result.flatness.max_slope
        self._refresh_execution_slot_points(env_ids)
        self.prev_mean_oracle_distance[env_ids] = compute_mean_oracle_distance(
            self.positions[env_ids],
            self._oracle_reward_target()[env_ids],
        )
        self.prev_oracle_distance_per_agent[env_ids] = compute_oracle_distances(
            self.positions[env_ids],
            self._oracle_reward_target()[env_ids],
        )
        return result

    def _oracle_reward_target(self) -> torch.Tensor:
        """Select the progress-shaping target without changing the success site.

        Success and terrain flatness always use the actual team centroid and
        ``oracle_point`` remains the searched site for the critic.  The opt-in
        slot target only changes dense oracle-progress shaping for formation
        observation schemas that provide each rover a stable assignment.
        """
        if self.cfg.task.execution_slot_reward_target:
            return self.gather_slot_points
        return self.oracle_point

    def _refresh_execution_slot_points(self, env_ids: torch.Tensor) -> None:
        """Assign fixed symmetric execution slots around the searched point.

        The brute-force assignment is intentionally small (at most ``4!`` in
        the current task).  Slots are evenly spaced, so their arithmetic mean
        is exactly the searched terrain-aware point.  Selecting the least-cost
        permutation from initial rover positions avoids a learned identity or
        global-coordinate dependency and prevents needless crossing.
        """
        shared_point = self.oracle_point[env_ids]
        if self.cfg.observation.schema_version not in {
            "ego_v6_gather_slot_goal",
            "ego_v7_gather_site_and_slot_goal",
        }:
            self.gather_slot_points[env_ids] = shared_point[:, None, :]
            self.execution_slot_points[env_ids] = self.gather_slot_points[env_ids]
            self.dynamic_terminal_slot_goal_active[env_ids] = False
            return
        self.gather_slot_points[env_ids] = self._assign_symmetric_slots(
            shared_point,
            self.positions[env_ids],
        )
        self.execution_slot_points[env_ids] = self.gather_slot_points[env_ids]
        self.dynamic_terminal_slot_goal_active[env_ids] = False

    def _assign_symmetric_slots(
        self,
        centers: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """Assign nearest symmetric slots around per-environment centres."""
        slot_angles = torch.arange(
            self.n_agents,
            device=self.device,
            dtype=positions.dtype,
        ) * (2.0 * torch.pi / float(self.n_agents))
        slot_offsets = float(self.cfg.gather_point.execution_slot_radius) * torch.stack(
            (torch.cos(slot_angles), torch.sin(slot_angles)),
            dim=-1,
        )
        unassigned_slots = centers[:, None, :].expand(-1, self.n_agents, -1).clone()
        unassigned_slots[..., :2] += slot_offsets
        travel_cost = (
            positions[:, :, None, :2] - unassigned_slots[:, None, :, :2]
        ).square().sum(dim=-1)
        agent_ids = torch.arange(self.n_agents, device=self.device)
        permutation_costs = torch.stack(
            [travel_cost[:, agent_ids, permutation].sum(dim=-1)
             for permutation in self._execution_slot_permutations],
            dim=1,
        )
        assignment = self._execution_slot_permutations[permutation_costs.argmin(dim=1)]
        return torch.gather(
            unassigned_slots,
            dim=1,
            index=assignment[..., None].expand(-1, -1, 3),
        )

    def _refresh_dynamic_terminal_slot_goal(self, metrics: TeamMetrics) -> None:
        """Expose a local real-flat terminal target under the slot contract.

        This updates only the actor-facing target for the *next* action.  The
        fixed reset-time slots continue to define dense reward shaping, and
        success remains independently checked at the actual team centroid.
        """
        self.execution_slot_points.copy_(self.gather_slot_points)
        self.dynamic_terminal_slot_goal_active.zero_()
        task_cfg = self.cfg.task
        if not (
            task_cfg.dynamic_terminal_slot_goal_enabled
            and self.cfg.observation.schema_version
            in {"ego_v6_gather_slot_goal", "ego_v7_gather_site_and_slot_goal"}
        ):
            return

        terminal = (
            metrics.dmax
            <= float(self.cfg.success_thresholds.dmax)
            * float(task_cfg.dynamic_terminal_slot_goal_dmax_multiplier)
        ) & (
            metrics.dispersion
            <= float(self.cfg.success_thresholds.dispersion)
            * float(task_cfg.dynamic_terminal_slot_goal_dispersion_multiplier)
        )
        if task_cfg.dynamic_terminal_slot_goal_require_flatness_failure:
            flatness = self.evaluate_current_gather_point_flatness(metrics)
            terminal = terminal & ~flatness.is_flat
        env_ids = torch.nonzero(terminal, as_tuple=False).flatten()
        if env_ids.numel() == 0:
            return

        gather_cfg = self.cfg.gather_point
        search = search_local_flatness_center(
            metrics.centroid[env_ids, :2],
            self.cfg.terrain,
            self.terrain_runtime.subset(env_ids),
            search_radius=float(task_cfg.dynamic_terminal_slot_goal_search_radius),
            samples=int(task_cfg.dynamic_terminal_slot_goal_search_samples),
            flatness_radius=float(gather_cfg.flatness_radius),
            flatness_rings=int(gather_cfg.flatness_rings),
            flatness_samples_per_ring=int(gather_cfg.flatness_samples_per_ring),
            max_height_range=float(gather_cfg.max_height_range),
            max_slope=float(gather_cfg.max_slope),
        )
        found_ids = env_ids[search.found_flat]
        if found_ids.numel() == 0:
            return
        centers = self.gather_slot_points[found_ids].mean(dim=1).clone()
        centers[..., :2] = search.target_xy[search.found_flat]
        self.execution_slot_points[found_ids] = self._assign_symmetric_slots(
            centers,
            self.positions[found_ids],
        )
        self.dynamic_terminal_slot_goal_active[found_ids] = True

    def _flat_geometry_capture_slot_points(self, metrics: TeamMetrics) -> torch.Tensor:
        """Return current-centroid slots, optionally reassigned by travel cost.

        The assignment used for observations is fixed at reset.  That remains
        important for a stable actor contract, but terminal geometric capture
        should not command a rover across the formation merely because its
        reset-time slot is stale.  This method is execution-only and never
        changes the observation target or success predicate.
        """
        control_cfg = self.cfg.low_level_control
        if not control_cfg.flat_geometry_capture_dynamic_assignment:
            return self.gather_slot_points

        slot_angles = torch.arange(
            self.n_agents,
            device=self.device,
            dtype=self.positions.dtype,
        ) * (2.0 * torch.pi / float(self.n_agents))
        slot_offsets = float(self.cfg.gather_point.execution_slot_radius) * torch.stack(
            (torch.cos(slot_angles), torch.sin(slot_angles)),
            dim=-1,
        )
        unassigned_slots = metrics.centroid[:, None, :].expand(-1, self.n_agents, -1).clone()
        unassigned_slots[..., :2] += slot_offsets
        travel_cost = (
            self.positions[:, :, None, :2] - unassigned_slots[:, None, :, :2]
        ).square().sum(dim=-1)
        agent_ids = torch.arange(self.n_agents, device=self.device)
        permutation_costs = torch.stack(
            [travel_cost[:, agent_ids, permutation].sum(dim=-1)
             for permutation in self._execution_slot_permutations],
            dim=1,
        )
        assignment = self._execution_slot_permutations[permutation_costs.argmin(dim=1)]
        return torch.gather(
            unassigned_slots,
            dim=1,
            index=assignment[..., None].expand(-1, -1, 3),
        )

    def evaluate_current_gather_point_flatness(
        self,
        metrics: TeamMetrics | None = None,
        env_ids: torch.Tensor | None = None,
    ) -> GatherPointFlatness:
        """Evaluate the actual team centroid footprint on the current terrain."""
        runtime = self.terrain_runtime
        if env_ids is not None:
            env_ids = env_ids.to(device=self.device, dtype=torch.long)
            runtime = self.terrain_runtime.subset(env_ids)
        if metrics is None:
            positions = self.positions if env_ids is None else self.positions[env_ids]
            velocities = (
                self.velocities_xy
                if env_ids is None
                else self.velocities_xy[env_ids]
            )
            metrics = compute_team_metrics(positions, velocities)
        gather_cfg = self.cfg.gather_point
        return evaluate_gather_point_flatness(
            metrics.centroid[..., :2],
            self.cfg.terrain,
            runtime,
            radius=float(gather_cfg.flatness_radius),
            rings=int(gather_cfg.flatness_rings),
            samples_per_ring=int(gather_cfg.flatness_samples_per_ring),
            max_height_range=float(gather_cfg.max_height_range),
            max_slope=float(gather_cfg.max_slope),
        )

    def _select_terminal_formation_center(
        self,
        metrics: TeamMetrics,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return fixed-slot centre or a nearby truly flat terminal centre."""
        target_xy = self.gather_slot_points[..., :2].mean(dim=1)
        local_search_active = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        control_cfg = self.cfg.low_level_control
        if not (
            control_cfg.formation_center_correction_enabled
            and control_cfg.formation_center_local_flatness_search_enabled
            and control_cfg.formation_center_correction_max_offset > 0.0
            and control_cfg.formation_center_correction_gain > 0.0
        ):
            return target_xy, local_search_active

        terminal = (
            metrics.dmax
            <= float(self.cfg.success_thresholds.dmax)
            * float(control_cfg.formation_center_activation_dmax_multiplier)
        ) & (
            metrics.dispersion
            <= float(self.cfg.success_thresholds.dispersion)
            * float(control_cfg.formation_center_activation_dispersion_multiplier)
        )
        if control_cfg.formation_center_correction_require_flatness_failure:
            terminal = terminal & ~self.prev_gather_point_flatness_ok
        env_ids = torch.nonzero(terminal, as_tuple=False).flatten()
        if env_ids.numel() == 0:
            return target_xy, local_search_active

        gather_cfg = self.cfg.gather_point
        search = search_local_flatness_center(
            metrics.centroid[env_ids, :2],
            self.cfg.terrain,
            self.terrain_runtime.subset(env_ids),
            search_radius=float(control_cfg.formation_center_local_flatness_search_radius),
            samples=int(control_cfg.formation_center_local_flatness_search_samples),
            flatness_radius=float(gather_cfg.flatness_radius),
            flatness_rings=int(gather_cfg.flatness_rings),
            flatness_samples_per_ring=int(gather_cfg.flatness_samples_per_ring),
            max_height_range=float(gather_cfg.max_height_range),
            max_slope=float(gather_cfg.max_slope),
        )
        target_xy[env_ids] = torch.where(
            search.found_flat[:, None],
            search.target_xy,
            target_xy[env_ids],
        )
        local_search_active[env_ids] = search.found_flat
        return target_xy, local_search_active

    def reset(self, env_ids: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        count = int(env_ids.numel())
        self.randomize_terrain(env_ids)
        base_angles = torch.linspace(0.0, 2.0 * torch.pi, self.n_agents + 1, device=self.device)[:-1]
        base = torch.stack((torch.cos(base_angles), torch.sin(base_angles)), dim=-1)
        spawn_radius_min, spawn_radius_max, center_xy_range, jitter_std = (
            self._effective_initial_state_values()
        )
        radius_span = max(
            spawn_radius_max - spawn_radius_min,
            0.0,
        )
        radius = spawn_radius_min + radius_span * torch.rand(
            count,
            1,
            1,
            generator=self.generator,
            device=self.device,
        )
        jitter = jitter_std * torch.randn(
            count,
            self.n_agents,
            2,
            generator=self.generator,
            device=self.device,
        )
        centers = torch.empty(count, 1, 2, device=self.device).uniform_(
            -center_xy_range,
            center_xy_range,
            generator=self.generator,
        )
        if bool(self.cfg.initial_state.randomize_formation_rotation):
            formation_rotation = torch.empty(
                count,
                1,
                device=self.device,
            ).uniform_(-torch.pi, torch.pi, generator=self.generator)
            cos_rotation = torch.cos(formation_rotation)
            sin_rotation = torch.sin(formation_rotation)
            rotated_base = torch.stack(
                (
                    cos_rotation * base[None, :, 0]
                    - sin_rotation * base[None, :, 1],
                    sin_rotation * base[None, :, 0]
                    + cos_rotation * base[None, :, 1],
                ),
                dim=-1,
            )
        else:
            rotated_base = base[None, :, :].expand(count, -1, -1)
        xy = centers + radius * rotated_base + jitter
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
        self.last_steering_angle[env_ids] = 0.0
        self.last_actual_yaw_rate[env_ids] = 0.0
        self.last_turning_radius[env_ids] = float("inf")
        if bool(self.cfg.initial_state.randomize_agent_yaws):
            self.yaws[env_ids] = torch.empty(
                count,
                self.n_agents,
                device=self.device,
            ).uniform_(-torch.pi, torch.pi, generator=self.generator)
        else:
            self.yaws[env_ids] = torch.atan2(-xy[..., 1], -xy[..., 0])
        self.velocities_xy[env_ids] = 0.0
        self.angular_velocities[env_ids] = 0.0
        self.previous_physical_action[env_ids] = 0.0
        self.committed_plan_local_xy[env_ids] = 0.0
        self.committed_plan_world_subgoal[env_ids] = self.positions[env_ids]
        self.committed_reference_speed[env_ids] = 0.0
        self.committed_planned_yaw_delta[env_ids] = 0.0
        self.last_left_wheel_speed[env_ids] = 0.0
        self.last_right_wheel_speed[env_ids] = 0.0
        self.coordination_token[env_ids] = torch.empty(
            count,
            self.n_agents,
            device=self.device,
        ).uniform_(-1.0, 1.0, generator=self.generator)
        self.step_count[env_ids] = 0
        self.success_hold_count[env_ids] = 0
        self.trajectory_conflicts.reset(env_ids)
        self._reset_communication(env_ids)
        if self.active_dstc_runtime is not None:
            self.active_dstc_runtime.reset(
                env_ids,
                self.positions,
                self.yaws,
                self.cfg.terrain,
                self.terrain_runtime,
            )
            self.oracle_point[env_ids] = 0.0
            self.oracle_search_objective[env_ids] = 0.0
            self.oracle_search_feasible[env_ids] = False
            self.oracle_search_mean_distance[env_ids] = 0.0
            self.oracle_search_max_distance[env_ids] = 0.0
            self.oracle_search_path_risk[env_ids] = 0.0
            self.oracle_search_path_height_change[env_ids] = 0.0
            self.oracle_search_height_range[env_ids] = 0.0
            self.oracle_search_max_slope[env_ids] = 0.0
            self.prev_mean_oracle_distance[env_ids] = 0.0
            self.prev_oracle_distance_per_agent[env_ids] = 0.0
        else:
            self.refresh_oracle_point(env_ids)
        reset_metrics = compute_team_metrics(
            self.positions[env_ids],
            self.velocities_xy[env_ids],
        )
        reset_flatness = self.evaluate_current_gather_point_flatness(
            reset_metrics,
            env_ids=env_ids,
        )
        self.prev_centroid_flatness_cost[env_ids] = compute_centroid_flatness_cost(
            reset_flatness.height_range,
            reset_flatness.max_slope,
            self.cfg,
        )
        self.prev_gather_point_flatness_ok[env_ids] = reset_flatness.is_flat
        self.metrics = compute_team_metrics(self.positions, self.velocities_xy)
        self.prev_metrics = self.metrics
        return self.get_observations()

    def get_observations(self) -> tuple[torch.Tensor, torch.Tensor]:
        metrics = compute_team_metrics(self.positions, self.velocities_xy)
        self._refresh_dynamic_terminal_slot_goal(metrics)
        terrain_grid = self._terrain_grid()
        actor_terrain = self._actor_terrain_observation()
        critic_agent_terrain = (
            build_multiscale_local_terrain_observation(
                self.positions,
                self.yaws,
                self.cfg.terrain,
                self.terrain_runtime,
            )
            if self.cfg.state.include_multiscale_agent_terrain
            and actor_terrain.shape[-1] != 224
            else actor_terrain
        )
        communication_snapshot = (
            self.communication_cache.snapshot()
            if self.communication_cache is not None
            else None
        )
        self.last_communication_snapshot = communication_snapshot
        execution_target = None
        execution_slot_target = None
        if self.cfg.task.explicit_goal_in_execution:
            schema = self.cfg.observation.schema_version
            if schema == "ego_v6_gather_slot_goal":
                execution_target = self.execution_slot_points
            else:
                execution_target = self.oracle_point
                if schema == "ego_v7_gather_site_and_slot_goal":
                    execution_slot_target = self.execution_slot_points
        actor_obs = build_actor_observation(
            self.positions,
            self.yaws,
            self.velocities_xy,
            self.angular_velocities,
            self.communication_radius,
            self.cfg,
            actor_terrain,
            metrics,
            self.success_hold_count,
            gather_site_point=execution_target,
            gather_slot_point=execution_slot_target,
            communication_snapshot=communication_snapshot,
            committed_plan_local_xy=self.committed_plan_local_xy,
            committed_reference_speed=self.committed_reference_speed,
            committed_planned_yaw_delta=self.committed_planned_yaw_delta,
            coordination_token=self.coordination_token,
        )
        critic_site_point = (
            self.active_dstc_runtime.critic_site_points(self.positions)
            if self.active_dstc_runtime is not None
            else self.oracle_point
        )
        critic_state = build_critic_state(
            self.positions,
            self.yaws,
            self.velocities_xy,
            self.angular_velocities,
            metrics,
            critic_site_point,
            self.success_hold_count,
            self.cfg,
            terrain_grid,
            critic_agent_terrain,
        )
        return actor_obs, critic_state

    def step(self, action: torch.Tensor) -> StepOutput:
        discrete_actions = self.cfg.planner.action_type in {
            "spatiotemporal_primitives",
            "differential_trajectory_primitives",
        }
        differential_primitives = (
            self.cfg.planner.action_type == "differential_trajectory_primitives"
        )
        action = action.to(
            device=self.device,
            dtype=torch.long if discrete_actions else torch.float32,
        )
        if discrete_actions and action.shape == (self.num_envs, self.n_agents, 1):
            action = action.squeeze(-1)
        expected_shape = (
            (self.num_envs, self.n_agents)
            if discrete_actions
            else (self.num_envs, self.n_agents, 2)
        )
        if action.shape != expected_shape:
            raise ValueError(f"Expected action shape {expected_shape}, got {tuple(action.shape)}")

        self.prev_metrics = compute_team_metrics(self.positions, self.velocities_xy)
        previous_dstc_potential = (
            self.active_dstc_runtime.potential.clone()
            if self.active_dstc_runtime is not None
            else torch.zeros(self.num_envs, device=self.device)
        )
        previous_dstc_committed = (
            self.active_dstc_runtime.committed.clone()
            if self.active_dstc_runtime is not None
            else torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        )
        previous_dstc_site_distance = (
            self.active_dstc_runtime.mean_committed_distance(self.positions)
            if self.active_dstc_runtime is not None
            else torch.zeros(self.num_envs, device=self.device)
        )
        previous_mean_oracle = self.prev_mean_oracle_distance.clone()
        previous_oracle_per_agent = self.prev_oracle_distance_per_agent.clone()
        raw_decoded = decode_action(action, self.positions, self.yaws, self.cfg.planner)
        filter_cfg = self.cfg.planner.subgoal_filter
        filter_progress = (
            int(filter_cfg.progress_timestep_override)
            if int(filter_cfg.progress_timestep_override) >= 0
            else int(self.global_step_count)
        )
        filter_result = apply_subgoal_filter(
            raw_decoded,
            self.positions,
            self.yaws,
            self.cfg,
            self.terrain_runtime,
            progress_timestep=filter_progress,
            deterministic=bool(filter_cfg.deterministic_eval),
            generator=self.generator,
        )
        slot_capture = apply_terminal_slot_capture(
            filter_result.decoded,
            gather_slot_points=self.gather_slot_points,
            dmax=self.prev_metrics.dmax,
            dispersion=self.prev_metrics.dispersion,
            dmax_threshold=float(self.cfg.success_thresholds.dmax),
            dispersion_threshold=float(self.cfg.success_thresholds.dispersion),
            enabled=bool(self.cfg.low_level_control.terminal_slot_capture_enabled),
            activation_dmax_multiplier=float(
                self.cfg.low_level_control.terminal_slot_capture_dmax_multiplier
            ),
            activation_dispersion_multiplier=float(
                self.cfg.low_level_control.terminal_slot_capture_dispersion_multiplier
            ),
            blend=float(self.cfg.low_level_control.terminal_slot_capture_blend),
        )
        flat_geometry_capture = apply_flat_geometry_capture(
            slot_capture.decoded,
            gather_slot_points=self._flat_geometry_capture_slot_points(self.prev_metrics),
            centroid_xy=self.prev_metrics.centroid[..., :2],
            dmax=self.prev_metrics.dmax,
            dispersion=self.prev_metrics.dispersion,
            flatness_ok=self.prev_gather_point_flatness_ok,
            dmax_threshold=float(self.cfg.success_thresholds.dmax),
            dispersion_threshold=float(self.cfg.success_thresholds.dispersion),
            enabled=bool(self.cfg.low_level_control.flat_geometry_capture_enabled),
            activation_dmax_multiplier=float(
                self.cfg.low_level_control.flat_geometry_capture_dmax_multiplier
            ),
            activation_dispersion_multiplier=float(
                self.cfg.low_level_control.flat_geometry_capture_dispersion_multiplier
            ),
            blend=float(self.cfg.low_level_control.flat_geometry_capture_blend),
        )
        formation_center_xy, local_flatness_search_active = self._select_terminal_formation_center(
            self.prev_metrics
        )
        correction = apply_formation_center_correction(
            flat_geometry_capture.decoded,
            centroid_xy=self.prev_metrics.centroid[..., :2],
            dmax=self.prev_metrics.dmax,
            dispersion=self.prev_metrics.dispersion,
            # Fixed slots are assigned once per episode and their mean equals
            # the terrain-aware searched point exactly.
            formation_center_xy=formation_center_xy,
            dmax_threshold=float(self.cfg.success_thresholds.dmax),
            dispersion_threshold=float(self.cfg.success_thresholds.dispersion),
            enabled=bool(self.cfg.low_level_control.formation_center_correction_enabled),
            activation_dmax_multiplier=float(
                self.cfg.low_level_control.formation_center_activation_dmax_multiplier
            ),
            activation_dispersion_multiplier=float(
                self.cfg.low_level_control.formation_center_activation_dispersion_multiplier
            ),
            max_offset=float(self.cfg.low_level_control.formation_center_correction_max_offset),
            gain=float(self.cfg.low_level_control.formation_center_correction_gain),
            flatness_ok=self.prev_gather_point_flatness_ok,
            require_flatness_failure=bool(
                self.cfg.low_level_control.formation_center_correction_require_flatness_failure
            ),
        )
        decoded = correction.decoded
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
            reference_speed=decoded.reference_speed,
            motion_direction=(decoded.motion_direction if differential_primitives else None),
            planned_yaw_delta=(
                decoded.planned_yaw_delta if differential_primitives else None
            ),
            primitive_type=(decoded.primitive_type if differential_primitives else None),
        )
        self.committed_plan_local_xy = decoded.local_subgoal_xy.clone()
        self.committed_plan_world_subgoal = decoded.world_subgoal.clone()
        self.committed_reference_speed = (
            decoded.reference_speed.clone()
            if decoded.reference_speed is not None
            else torch.full(
                (self.num_envs, self.n_agents),
                float(self.cfg.trajectory_generator.reference_speed),
                device=self.device,
            )
        )
        self.committed_planned_yaw_delta = (
            decoded.planned_yaw_delta.clone()
            if decoded.planned_yaw_delta is not None
            else torch.zeros(
                self.num_envs,
                self.n_agents,
                device=self.device,
            )
        )
        path_terrain = (
            sample_trajectory_terrain_risk(
                trajectory.points,
                self.cfg.terrain,
                self.terrain_runtime,
            )
            if self._terrain_dynamics_enabled
            else None
        )
        if (
            path_terrain is not None
            and self.cfg.reward_coefficients.path_terrain_relative_cost != 0.0
        ):
            if discrete_actions:
                raise ValueError(
                    "path_terrain_relative_cost is incompatible with discrete "
                    "spatiotemporal primitives; use primitive terrain regret instead."
                )
            straight_action = action.clone()
            straight_action[..., 1] = 0.0
            straight_decoded = decode_action(
                straight_action,
                self.positions,
                self.yaws,
                self.cfg.planner,
            )
            straight_trajectory = generate_trajectory(
                self.positions,
                straight_decoded.world_subgoal,
                self.cfg.trajectory_generator,
                self.cfg.simulation.planning_dt,
                current_yaws=self.yaws,
            )
            reference_risk = sample_trajectory_terrain_risk(
                straight_trajectory.points,
                self.cfg.terrain,
                self.terrain_runtime,
            )["risk_mean"]
            path_terrain["reference_risk_mean"] = reference_risk
            path_terrain["relative_risk_mean"] = (
                path_terrain["risk_mean"] - reference_risk
            )
        conflict_safe_distance = max(
            float(self.cfg.success_thresholds.min_pairwise_distance),
            float(self.cfg.safety.collision_distance),
        )
        trajectory_conflicts = self.trajectory_conflicts.update(
            trajectory.points,
            conflict_safe_distance,
            timestamps=(
                trajectory.timestamps
                if self.cfg.trajectory_generator.time_parameterization
                == "arc_length_reference_speed"
                else None
            ),
        )
        if self.communication_cache is not None:
            pair_age = 0.5 * (
                self.communication_cache.age
                + self.communication_cache.age.transpose(1, 2)
            )
            active = trajectory_conflicts["active"]
            active_count = active.sum(dim=(1, 2)).clamp_min(1)
            trajectory_conflicts["message_age_at_conflict"] = (
                pair_age.masked_fill(~active, 0.0).sum(dim=(1, 2)) / active_count
            )
            pair_full = self.communication_cache.full & self.communication_cache.full.transpose(1, 2)
            trajectory_conflicts["full_message_conflict_ratio"] = (
                pair_full.masked_fill(~active, False).sum(dim=(1, 2)).float()
                / active_count
            )
        else:
            trajectory_conflicts["message_age_at_conflict"] = torch.zeros(
                self.num_envs,
                device=self.device,
            )
            trajectory_conflicts["full_message_conflict_ratio"] = torch.zeros(
                self.num_envs,
                device=self.device,
            )
        resolved_count = trajectory_conflicts["resolved"].sum(dim=(1, 2)).clamp_min(1)
        trajectory_conflicts["mean_conflict_resolution_steps"] = (
            trajectory_conflicts["resolved_steps"].sum(dim=(1, 2)) / resolved_count
        )
        control = compute_control(
            self.positions,
            self.yaws,
            trajectory,
            self.cfg.low_level_control,
            self.cfg.simulation.planning_dt,
        )
        raw_control = control
        control_safety = apply_control_safety_projection(
            control,
            self.positions,
            self.yaws,
            self.prev_metrics,
            self.cfg.low_level_control,
            self.cfg.success_thresholds,
            self.cfg.simulation.planning_dt,
            communication_radius=self.communication_radius,
        )
        control = control_safety.control
        self._integrate(control)
        self._advance_communication()
        self.global_step_count += 1
        self.step_count += 1
        if self.active_dstc_runtime is not None:
            self.active_dstc_runtime.update(
                self.positions,
                self.yaws,
                self.cfg.terrain,
                self.terrain_runtime,
                step_counts=self.step_count,
            )

        metrics = compute_team_metrics(self.positions, self.velocities_xy)
        gather_point_flatness = self.evaluate_current_gather_point_flatness(metrics)
        centroid_flatness_cost = compute_centroid_flatness_cost(
            gather_point_flatness.height_range,
            gather_point_flatness.max_slope,
            self.cfg,
        )
        (
            centroid_flatness_reward,
            centroid_flatness_progress,
            centroid_flatness_activation,
        ) = compute_centroid_flatness_reward(
            self.prev_centroid_flatness_cost,
            centroid_flatness_cost,
            self.prev_metrics.dmax,
            metrics.dmax,
            self.cfg,
        )
        flatness_ok = (
            gather_point_flatness.is_flat
            if self.cfg.gather_point.require_flat_for_success
            else torch.ones_like(gather_point_flatness.is_flat)
        )
        done, self.success_hold_count = compute_done(
            self.positions,
            self.velocities_xy,
            metrics,
            self.success_hold_count,
            self.step_count,
            self.max_episode_steps,
            self.cfg.success_thresholds,
            self.cfg.safety,
            flatness_ok=flatness_ok,
        )
        success_gates = compute_success_gates(
            metrics,
            self.velocities_xy,
            self.cfg.success_thresholds,
            flatness_ok=flatness_ok,
        )
        active_dstc_reward = torch.zeros(self.num_envs, device=self.device)
        if self.active_dstc_runtime is not None:
            coefficients = self.cfg.reward_coefficients
            belief_progress = (
                self.active_dstc_runtime.potential - previous_dstc_potential
            )
            new_commit = (
                self.active_dstc_runtime.committed & ~previous_dstc_committed
            ).float()
            current_site_distance = (
                self.active_dstc_runtime.mean_committed_distance(self.positions)
            )
            site_progress = torch.where(
                previous_dstc_committed,
                previous_dstc_site_distance - current_site_distance,
                torch.zeros_like(current_site_distance),
            )
            active_dstc_reward = (
                float(coefficients.dstc_belief_progress) * belief_progress
                + float(coefficients.dstc_commit_bonus) * new_commit
                + float(coefficients.dstc_site_distance_progress) * site_progress
            )
        terms, mean_oracle = compute_reward(
            self.positions,
            self._oracle_reward_target(),
            self.prev_metrics,
            metrics,
            previous_mean_oracle,
            decoded.physical,
            self.previous_physical_action,
            done,
            self.success_hold_count,
            self.last_terrain_features,
            self.cfg,
            oracle_feasible=self.oracle_search_feasible,
            subgoal_terrain_features=subgoal_terrain_features,
            terrain_speed_scale=self.last_terrain_speed_scale,
            height_delta=self.last_height_delta,
            path_terrain_risk_mean=(
                path_terrain["risk_mean"] if path_terrain is not None else None
            ),
            path_terrain_risk_max=(
                path_terrain["risk_max"] if path_terrain is not None else None
            ),
            path_terrain_reference_risk_mean=(
                path_terrain.get("reference_risk_mean")
                if path_terrain is not None
                else None
            ),
            path_height_change_mean=(
                path_terrain["height_change_mean"] if path_terrain is not None else None
            ),
            filter_raw_path_risk_mean=(
                filter_result.info["raw_path_terrain_risk_mean"]
                if isinstance(filter_result.info.get("raw_path_terrain_risk_mean"), torch.Tensor)
                else None
            ),
            filter_deviation=(
                filter_result.info["suggested_subgoal_deviation"]
                if isinstance(filter_result.info.get("suggested_subgoal_deviation"), torch.Tensor)
                else None
            ),
            centroid_flatness_reward=centroid_flatness_reward,
            active_dstc_reward=active_dstc_reward,
        )
        analytical_prd = None
        if self.cfg.task.analytical_prd_enabled:
            analytical_prd = compute_analytical_prd_baseline(
                positions=self.positions,
                oracle_target=self._oracle_reward_target(),
                previous_oracle_distances=previous_oracle_per_agent,
                physical_action=decoded.physical,
                previous_physical_action=self.previous_physical_action,
                done=done,
                terrain_features=self.last_terrain_features,
                reward_terms=terms,
                cfg=self.cfg,
                oracle_feasible=self.oracle_search_feasible,
                subgoal_terrain_features=subgoal_terrain_features,
                terrain_speed_scale=self.last_terrain_speed_scale,
                height_delta=self.last_height_delta,
                path_terrain_risk_mean=(
                    path_terrain["risk_mean"] if path_terrain is not None else None
                ),
                path_terrain_risk_max=(
                    path_terrain["risk_max"] if path_terrain is not None else None
                ),
                path_terrain_reference_risk_mean=(
                    path_terrain.get("reference_risk_mean")
                    if path_terrain is not None
                    else None
                ),
                path_height_change_mean=(
                    path_terrain["height_change_mean"]
                    if path_terrain is not None
                    else None
                ),
                filter_raw_path_risk_mean=(
                    filter_result.info["raw_path_terrain_risk_mean"]
                    if isinstance(
                        filter_result.info.get("raw_path_terrain_risk_mean"),
                        torch.Tensor,
                    )
                    else None
                ),
                filter_deviation=(
                    filter_result.info["suggested_subgoal_deviation"]
                    if isinstance(
                        filter_result.info.get("suggested_subgoal_deviation"),
                        torch.Tensor,
                    )
                    else None
                ),
            )
        self.prev_mean_oracle_distance = mean_oracle
        self.prev_oracle_distance_per_agent = compute_oracle_distances(
            self.positions,
            self._oracle_reward_target(),
        )
        self.prev_centroid_flatness_cost = centroid_flatness_cost
        # Keep next-step control state independent from this step's diagnostic
        # snapshot. ``reset()`` updates selected entries in place after a done;
        # aliasing this tensor would retroactively change success_gates for the
        # completed step.
        self.prev_gather_point_flatness_ok = gather_point_flatness.is_flat.clone()
        self.previous_physical_action = decoded.physical
        self.metrics = metrics
        self.last_trajectory = trajectory
        self.last_control = control
        terrain_features = self.last_terrain_features.clone()
        terrain_speed_scale = self.last_terrain_speed_scale.clone()
        height_delta = self.last_height_delta.clone()
        path_terrain_snapshot = (
            {key: value.clone() for key, value in path_terrain.items()}
            if path_terrain is not None
            else None
        )
        action_filter_snapshot = {
            key: value.clone() if isinstance(value, torch.Tensor) else value
            for key, value in filter_result.info.items()
        }
        formation_center_correction_snapshot = {
            "active": correction.active.clone(),
            "offset_xy": correction.offset_xy.clone(),
            "local_flatness_search_active": local_flatness_search_active.clone(),
        }
        terminal_slot_capture_snapshot = {
            "active": slot_capture.active.clone(),
        }
        flat_geometry_capture_snapshot = {
            "active": flat_geometry_capture.active.clone(),
        }
        dynamic_terminal_slot_goal_snapshot = {
            "active": self.dynamic_terminal_slot_goal_active.clone(),
        }
        control_safety_snapshot = {
            key: value.clone() if isinstance(value, torch.Tensor) else value
            for key, value in control_safety.info.items()
        }
        kinematics_snapshot = {
            "kinematic_model": self.cfg.low_level_control.kinematic_model,
            "steering_angle": self.last_steering_angle.clone(),
            "actual_yaw_rate": self.last_actual_yaw_rate.clone(),
            "turning_radius": self.last_turning_radius.clone(),
        }
        terrain_runtime = self.terrain_runtime.clone()
        success_hold_count = self.success_hold_count.clone()
        oracle_point = self.oracle_point.clone()
        oracle_search_snapshot = {
            "method": self.cfg.gather_point.search_method,
            "objective": self.oracle_search_objective.clone(),
            "feasible": self.oracle_search_feasible.clone(),
            "mean_distance": self.oracle_search_mean_distance.clone(),
            "max_distance": self.oracle_search_max_distance.clone(),
            "path_risk": self.oracle_search_path_risk.clone(),
            "path_height_change": self.oracle_search_path_height_change.clone(),
            "height_range": self.oracle_search_height_range.clone(),
            "max_slope": self.oracle_search_max_slope.clone(),
        }
        gather_point_flatness_snapshot = GatherPointFlatness(
            height_range=gather_point_flatness.height_range.clone(),
            max_slope=gather_point_flatness.max_slope.clone(),
            mean_slope=gather_point_flatness.mean_slope.clone(),
            is_flat=gather_point_flatness.is_flat.clone(),
        )
        centroid_flatness_reward_snapshot = {
            "cost": centroid_flatness_cost.clone(),
            "progress": centroid_flatness_progress.clone(),
            "activation": centroid_flatness_activation.clone(),
        }
        analytical_prd_snapshot = (
            {
                "reward_sources": {
                    "node": analytical_prd.node.clone(),
                    "team_residual": analytical_prd.team_residual.clone(),
                },
                "local_other": analytical_prd.local_other.clone(),
                "near_other": analytical_prd.near_other.clone(),
                "collision_other": analytical_prd.collision_other.clone(),
                "failure_other": analytical_prd.failure_other.clone(),
                "loo_baseline": analytical_prd.total.clone(),
                "source_reconstruction_error": (
                    analytical_prd.source_reconstruction_error.clone()
                ),
                "own_action_invariance_error": (
                    analytical_prd.own_action_invariance_error.clone()
                ),
                "actual_collision_participants": (
                    analytical_prd.actual_collision_participants.clone()
                ),
                "team_reward_preservation_error": torch.zeros_like(terms.total),
            }
            if analytical_prd is not None
            else None
        )
        # Centralized diagnostic snapshot taken before auto-reset. It is exposed
        # only through ``info`` and is never part of the Actor observation or the
        # execution chain. Offline audits need it to compute terminal-transition
        # progress without accidentally reading the next episode's reset state.
        positions_snapshot = self.positions.clone()
        active_dstc_snapshot = (
            {
                **self.active_dstc_runtime.diagnostics(),
                "target_points": self.active_dstc_runtime.target_points.clone(),
                "target_valid": self.active_dstc_runtime.target_valid.clone(),
                "committed_centers": (
                    self.active_dstc_runtime.committed_centers.clone()
                ),
                "reward": active_dstc_reward.clone(),
            }
            if self.active_dstc_runtime is not None
            else None
        )

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
                "positions": positions_snapshot,
                "trajectory": trajectory,
                "trajectory_conflicts": {
                    key: value.clone() for key, value in trajectory_conflicts.items()
                },
                "control": control,
                "raw_control": raw_control,
                "control_safety": control_safety_snapshot,
                "kinematics": kinematics_snapshot,
                "wheel_commands": {
                    "left_radps": self.last_left_wheel_speed.clone(),
                    "right_radps": self.last_right_wheel_speed.clone(),
                },
                "terrain_features": terrain_features,
                "terrain_speed_scale": terrain_speed_scale,
                "height_delta": height_delta,
                "path_terrain": path_terrain_snapshot,
                "action_filter": action_filter_snapshot,
                "formation_center_correction": formation_center_correction_snapshot,
                "terminal_slot_capture": terminal_slot_capture_snapshot,
                "flat_geometry_capture": flat_geometry_capture_snapshot,
                "dynamic_terminal_slot_goal": dynamic_terminal_slot_goal_snapshot,
                "terrain_runtime": terrain_runtime,
                "gather_point_flatness": gather_point_flatness_snapshot,
                "centroid_flatness_reward": centroid_flatness_reward_snapshot,
                "analytical_prd": analytical_prd_snapshot,
                "communication": (
                    {
                        key: value.clone()
                        for key, value in self.last_communication_snapshot.diagnostics.items()
                    }
                    if self.last_communication_snapshot is not None
                    else None
                ),
                "active_dstc": active_dstc_snapshot,
                "oracle_point": oracle_point,
                "oracle_search": oracle_search_snapshot,
            },
        )

    def random_actions(self) -> torch.Tensor:
        if self.cfg.planner.action_type in {
            "spatiotemporal_primitives",
            "differential_trajectory_primitives",
        }:
            action_count = (
                DIFFERENTIAL_PRIMITIVE_ACTION_COUNT
                if self.cfg.planner.action_type
                == "differential_trajectory_primitives"
                else SPATIOTEMPORAL_ACTION_COUNT
            )
            return torch.randint(
                0,
                action_count,
                (self.num_envs, self.n_agents),
                device=self.device,
                generator=self.generator,
            )
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
        model = str(self.cfg.low_level_control.kinematic_model)
        if model == "unicycle":
            yaw_rate = control.angular
            self.yaws = wrap_to_pi(self.yaws + yaw_rate * dt)
            direction = torch.stack((torch.cos(self.yaws), torch.sin(self.yaws)), dim=-1)
            steering_angle = torch.zeros_like(control.angular)
        elif model in {"bicycle", "differential_drive"}:
            heading_for_slope = torch.stack((torch.cos(self.yaws), torch.sin(self.yaws)), dim=-1)
            direction = heading_for_slope
            yaw_rate = torch.zeros_like(control.angular)
            steering_angle = torch.zeros_like(control.angular)
        else:
            raise ValueError(f"Unsupported kinematic_model: {model}")
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
        linear_eff = control.linear * speed_scale
        if model == "differential_drive":
            radius = float(self.cfg.low_level_control.wheel_radius_m)
            track = float(self.cfg.low_level_control.track_width_m)
            max_wheel = float(self.cfg.low_level_control.max_wheel_speed_radps)
            left = (control.linear - 0.5 * track * control.angular) / radius
            right = (control.linear + 0.5 * track * control.angular) / radius
            left = left.clamp(-max_wheel, max_wheel)
            right = right.clamp(-max_wheel, max_wheel)
            self.last_left_wheel_speed = left
            self.last_right_wheel_speed = right
            wheel_linear = 0.5 * radius * (left + right)
            wheel_yaw_rate = radius * (right - left) / track
            linear_eff = wheel_linear * speed_scale
            yaw_rate = wheel_yaw_rate * speed_scale
            midpoint_yaw = wrap_to_pi(self.yaws + 0.5 * yaw_rate * dt)
            direction = torch.stack(
                (torch.cos(midpoint_yaw), torch.sin(midpoint_yaw)),
                dim=-1,
            )
            self.yaws = wrap_to_pi(self.yaws + yaw_rate * dt)
        if model == "bicycle":
            wheelbase = float(self.cfg.low_level_control.wheelbase_m)
            max_steer = float(self.cfg.low_level_control.max_steer_angle_rad)
            eps = 1.0e-6
            steer_demand = torch.atan(
                wheelbase * control.angular / control.linear.abs().clamp_min(eps)
            )
            steering_angle = torch.clamp(steer_demand, -max_steer, max_steer)
            yaw_rate = torch.where(
                linear_eff.abs() > eps,
                linear_eff / wheelbase * torch.tan(steering_angle),
                torch.zeros_like(linear_eff),
            )
            midpoint_yaw = wrap_to_pi(self.yaws + 0.5 * yaw_rate * dt)
            direction = torch.stack((torch.cos(midpoint_yaw), torch.sin(midpoint_yaw)), dim=-1)
            self.yaws = wrap_to_pi(self.yaws + yaw_rate * dt)
            self.last_left_wheel_speed.zero_()
            self.last_right_wheel_speed.zero_()
        elif model == "unicycle":
            self.last_left_wheel_speed.zero_()
            self.last_right_wheel_speed.zero_()
        delta_xy = direction * linear_eff.unsqueeze(-1) * dt
        next_xy = old_positions[..., :2] + delta_xy
        self.positions[..., :2] = next_xy
        if self._terrain_dynamics_enabled:
            next_features = query_terrain_features(
                next_xy,
                self.cfg.terrain,
                self.terrain_runtime,
            )
            self.positions[..., 2] = next_features[..., 0]
            self.last_terrain_features = next_features
            self.last_height_delta = self.positions[..., 2] - old_positions[..., 2]
        else:
            self.positions[..., 2] = 0.0
            self.last_terrain_features.zero_()
            self.last_height_delta.zero_()
        self.last_terrain_speed_scale = speed_scale
        self.velocities_xy = (self.positions[..., :2] - old_positions[..., :2]) / dt
        self.angular_velocities = yaw_rate
        self.last_steering_angle = steering_angle
        self.last_actual_yaw_rate = yaw_rate
        self.last_turning_radius = torch.where(
            yaw_rate.abs() > 1.0e-6,
            linear_eff.abs() / yaw_rate.abs().clamp_min(1.0e-6),
            torch.full_like(linear_eff, float("inf")),
        )

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
        self.action_space = (
            gym.spaces.MultiDiscrete(
                np.full(
                    self.cfg.task.n_agents,
                    (
                        DIFFERENTIAL_PRIMITIVE_ACTION_COUNT
                        if self.cfg.planner.action_type
                        == "differential_trajectory_primitives"
                        else SPATIOTEMPORAL_ACTION_COUNT
                    ),
                    dtype=np.int64,
                )
            )
            if self.cfg.planner.action_type
            in {"spatiotemporal_primitives", "differential_trajectory_primitives"}
            else gym.spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(self.cfg.task.n_agents, 2),
                dtype=np.float32,
            )
        )

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        del options
        if seed is not None:
            self.cfg.seed = seed
        actor_obs, critic_state = self.core.reset()
        return self._pack_obs(actor_obs, critic_state), {}

    def step(self, action):
        action_tensor = torch.as_tensor(
            action,
            dtype=(
                torch.long
                if self.cfg.planner.action_type
                in {"spatiotemporal_primitives", "differential_trajectory_primitives"}
                else torch.float32
            ),
            device=self.core.device,
        ).unsqueeze(0)
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
        action_space = (
            gym.spaces.Discrete(
                DIFFERENTIAL_PRIMITIVE_ACTION_COUNT
                if self.cfg.planner.action_type
                == "differential_trajectory_primitives"
                else SPATIOTEMPORAL_ACTION_COUNT
            )
            if self.cfg.planner.action_type
            in {"spatiotemporal_primitives", "differential_trajectory_primitives"}
            else gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        )
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
