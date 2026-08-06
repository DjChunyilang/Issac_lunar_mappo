"""Configuration for the first-stage multi-rover gathering proxy task."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import pi


@dataclass(slots=True)
class SimulationCfg:
    device: str = "cuda"
    headless: bool = True
    num_envs: int = 32
    physics_dt: float = 0.05
    control_decimation: int = 4
    episode_length_s: float = 20.0

    @property
    def planning_dt(self) -> float:
        return self.physics_dt * self.control_decimation

    @property
    def max_episode_steps(self) -> int:
        return int(self.episode_length_s / self.planning_dt)


@dataclass(slots=True)
class TaskCfg:
    name: str = "Isaac-MultiRover-Gathering-Direct-v0"
    n_agents: int = 4
    scene_dim: str = "2.5D/3D"
    explicit_goal_in_execution: bool = False
    # Opt in only when the execution observation exposes a fixed per-rover
    # formation slot.  This lets oracle-progress shaping match the safe
    # formation geometry rather than pulling every rover into the same point.
    execution_slot_reward_target: bool = False
    # Optional observation-level terminal target.  The static reset-time
    # slots remain the reward target, while the actor can be shown a freshly
    # assigned symmetric formation around a nearby truly flat footprint.
    # This is deliberately an execution-goal contract change, not a success
    # proxy or a post-processing action override.
    dynamic_terminal_slot_goal_enabled: bool = False
    dynamic_terminal_slot_goal_dmax_multiplier: float = 1.25
    dynamic_terminal_slot_goal_dispersion_multiplier: float = 1.25
    dynamic_terminal_slot_goal_require_flatness_failure: bool = True
    dynamic_terminal_slot_goal_search_radius: float = 0.25
    dynamic_terminal_slot_goal_search_samples: int = 8
    oracle_optimal_gather_point_in_training: bool = True
    docking_considered: bool = False


@dataclass(slots=True)
class InitialStateCfg:
    spawn_radius_min: float = 3.0
    spawn_radius_max: float = 4.0
    center_xy_range: float = 1.0
    jitter_std: float = 0.35
    curriculum_enabled: bool = False
    curriculum_start_spawn_radius_min: float = 3.0
    curriculum_start_spawn_radius_max: float = 4.0
    curriculum_start_center_xy_range: float = 1.0
    curriculum_start_jitter_std: float = 0.35
    curriculum_warmup_timesteps: int = 0
    curriculum_ramp_timesteps: int = 1
    progress_timestep_override: int = -1


@dataclass(slots=True)
class SubgoalFilterCfg:
    enabled: bool = False
    mode: str = "terrain_safe_candidate"
    rho_scales: list[float] = field(default_factory=lambda: [0.65, 1.0])
    beta_offsets_deg: list[float] = field(
        default_factory=lambda: [-45.0, -22.5, 0.0, 22.5, 45.0]
    )
    path_samples: int = 5
    intent_deviation_weight: float = 0.35
    path_terrain_mean_weight: float = 0.70
    path_terrain_max_weight: float = 0.50
    path_height_change_weight: float = 0.20
    subgoal_terrain_weight: float = 0.30
    endpoint_near_weight: float = 2.0
    endpoint_collision_weight: float = 1000.0
    path_near_weight: float = 0.0
    path_collision_weight: float = 0.0
    mutual_path_near_weight: float = 0.0
    mutual_path_collision_weight: float = 0.0
    visible_neighbor_center_weight: float = 0.0
    center_progress_weight: float = 0.0
    center_progress_margin: float = 0.0
    hold_zone_dmax_multiplier: float = 0.0
    hold_zone_dispersion_multiplier: float = 0.0
    hold_zone_rho_weight: float = 0.0
    hold_zone_spacing_weight: float = 0.0
    hold_zone_pairwise_distance: float = 0.0
    hold_zone_override_after_warmup: bool = False
    endpoint_safe_distance: float = 0.0
    path_safe_distance: float = 0.0
    hard_endpoint_near_filter: bool = False
    hard_path_collision_filter: bool = False
    hard_center_progress_filter: bool = False
    center_progress_slack: float = 0.0
    hard_constraint_penalty: float = 10000.0
    safety_override_after_warmup: bool = False
    collision_override_after_warmup: bool = False
    warmup_timesteps: int = 0
    ramp_timesteps: int = 1
    apply_probability_end: float = 1.0
    score_scale_start: float = 1.0
    score_scale_end: float = 1.0
    deterministic_improvement_margin: float = 0.0
    progress_timestep_override: int = -1
    deterministic_eval: bool = False


@dataclass(slots=True)
class PlannerCfg:
    action_type: str = "local_subgoal_polar"
    action_dim: int = 2
    rho_max: float = 1.2
    beta_max: float = pi / 2.0
    subgoal_filter: SubgoalFilterCfg = field(default_factory=SubgoalFilterCfg)


@dataclass(slots=True)
class TrajectoryGeneratorCfg:
    n_trajectory_points: int = 8
    geometry_method: str = "line"
    reference_speed: float = 0.8
    quintic_tangent_scale: float = 0.5
    end_heading_mode: str = "subgoal_direction"
    # ``planning_step`` preserves the historical fixed-dt timestamp contract.
    # New physically timed runs explicitly opt into arc-length timing.
    time_parameterization: str = "planning_step"


@dataclass(slots=True)
class LowLevelControlCfg:
    first_stage_mode: str = "simplified_velocity_tracking"
    kinematic_model: str = "unicycle"
    wheelbase_m: float = 0.65
    max_steer_angle_rad: float = 0.610865
    max_linear_speed: float = 1.0
    max_angular_speed: float = 2.5
    k_linear: float = 1.6
    k_angular: float = 3.0
    # Historical runs select trajectory point index 1. New time-consistent
    # runs interpolate the trajectory at the next physical planning instant.
    tracking_point_mode: str = "fixed_index"
    safety_projection_enabled: bool = False
    projection_activation_distance: float = 0.0
    projection_stop_distance: float = 0.0
    projection_horizon_s: float = 0.40
    projection_strength: float = 0.75
    projection_min_linear_scale: float = 0.25
    projection_damp_nonclosing_near: bool = True
    projection_directional_agent_scale: bool = False
    projection_directional_agent_scale_mode: str = "fraction"
    success_zone_damping_enabled: bool = False
    success_zone_dmax_multiplier: float = 1.0
    success_zone_dispersion_multiplier: float = 1.0
    success_zone_linear_scale: float = 0.75
    # Optional execution-layer common translation for the fixed symmetric
    # formation. It preserves slot offsets and only biases subgoals toward the
    # searched formation centre when the team is near terminal geometry.
    formation_center_correction_enabled: bool = False
    formation_center_activation_dmax_multiplier: float = 1.75
    formation_center_activation_dispersion_multiplier: float = 1.75
    formation_center_correction_max_offset: float = 0.35
    formation_center_correction_gain: float = 0.55
    # If enabled, the common translation only runs while the prior actual
    # centroid footprint fails the same flatness gate used for success.
    formation_center_correction_require_flatness_failure: bool = False
    # Optional terminal-only local search around the actual centroid.  A
    # common target is replaced only when a nearby full success footprint is
    # truly flat, preserving all fixed-slot offsets and success semantics.
    formation_center_local_flatness_search_enabled: bool = False
    formation_center_local_flatness_search_radius: float = 0.25
    formation_center_local_flatness_search_samples: int = 8
    # Optional terminal capture blends the actor subgoal toward each rover's
    # fixed assigned slot. It never uses a shared geometric midpoint.
    terminal_slot_capture_enabled: bool = False
    terminal_slot_capture_dmax_multiplier: float = 1.75
    terminal_slot_capture_dispersion_multiplier: float = 1.75
    terminal_slot_capture_blend: float = 0.65
    # Optional in-place shape capture for an already-flat centroid that has
    # not yet met dmax/dispersion.  It reuses fixed slot offsets around the
    # actual centroid, so it cannot use an oracle point as a success proxy.
    flat_geometry_capture_enabled: bool = False
    flat_geometry_capture_dmax_multiplier: float = 1.75
    flat_geometry_capture_dispersion_multiplier: float = 1.75
    flat_geometry_capture_blend: float = 0.15
    # Reassign symmetric slots from the current rover positions before the
    # in-place capture. This avoids applying a stale reset-time assignment
    # after rover identities have crossed during the approach.
    flat_geometry_capture_dynamic_assignment: bool = False


@dataclass(slots=True)
class TerrainCfg:
    type: str = "flat_proxy"
    amplitude: float = 0.0
    wavelength: float = 4.0
    roughness_scale: float = 1.0
    traversability_slope_scale: float = 0.6
    dynamics_enabled: bool = False
    slope_speed_scale: float = 0.75
    min_speed_scale: float = 0.35
    crater_count: int = 0
    crater_min_radius: float = 0.45
    crater_max_radius: float = 1.20
    crater_depth_to_diameter: float = 0.06
    crater_rim_height_to_diameter: float = 0.015
    crater_field_size: float = 9.0
    crater_seed: int = 11
    randomize_per_reset: bool = False
    random_translation_m: float = 0.0
    random_yaw_rad: float = 0.0
    amplitude_scale_min: float = 1.0
    amplitude_scale_max: float = 1.0
    crater_radius_scale_min: float = 1.0
    crater_radius_scale_max: float = 1.0
    crater_depth_scale_min: float = 1.0
    crater_depth_scale_max: float = 1.0


@dataclass(slots=True)
class GatherPointCfg:
    """Terrain-aware oracle search and gathering-site acceptance criteria."""

    search_method: str = "terrain_aware_multiresolution"
    coarse_grid_size: int = 9
    refinement_grid_size: int = 5
    refinement_levels: int = 2
    search_margin: float = 1.5
    global_fallback_enabled: bool = True
    global_grid_size: int = 33
    global_beam_width: int = 32
    global_refinement_levels: int = 2
    global_max_envs_per_batch: int = 8
    flatness_radius: float = 0.75
    flatness_rings: int = 3
    flatness_samples_per_ring: int = 12
    max_height_range: float = 0.18
    max_slope: float = 0.25
    # Optional execution robustness envelope for the search only.  A positive
    # radius requires the complete success footprint to remain flat when its
    # center is displaced around the searched point.
    robustness_radius: float = 0.0
    robustness_samples: int = 8
    mean_distance_weight: float = 1.0
    max_distance_weight: float = 0.25
    path_risk_weight: float = 0.75
    path_height_change_weight: float = 0.25
    flatness_weight: float = 0.25
    path_samples: int = 5
    infeasible_penalty: float = 1000.0
    max_envs_per_batch: int = 64
    require_flat_for_success: bool = True
    # Execution-only formation radius around the terrain-aware search result.
    # With four evenly-spaced slots, 0.35 m yields about 0.495 m adjacent
    # separation, above the default 0.42 m success clearance.
    execution_slot_radius: float = 0.35


@dataclass(slots=True)
class RewardWeightsCfg:
    gather: float = 1.0
    oracle: float = 0.5
    energy: float = 0.02
    safety: float = 1.0
    terrain: float = 1.0
    flatness: float = 0.0
    motion: float = 0.05
    consistency: float = 0.02
    terminal: float = 1.0


@dataclass(slots=True)
class RewardCoefficientsCfg:
    dmax_progress: float = 2.0
    dispersion_progress: float = 1.0
    dmax_level: float = 0.0
    dispersion_level: float = 0.0
    oracle_mean_distance_progress: float = 1.5
    path_length: float = 0.2
    slope_cost: float = 0.0
    turn_cost: float = 0.05
    terrain_cost: float = 0.0
    subgoal_terrain_cost: float = 0.0
    terrain_speed_loss_cost: float = 0.0
    terrain_height_change_cost: float = 0.0
    path_terrain_mean_cost: float = 0.0
    path_terrain_max_cost: float = 0.0
    path_terrain_relative_cost: float = 0.0
    path_height_change_cost: float = 0.0
    filter_raw_path_risk_cost: float = 0.0
    filter_deviation_cost: float = 0.0
    centroid_flatness_progress: float = 0.0
    centroid_flatness_excess: float = 0.0
    centroid_flatness_dmax_multiplier: float = 2.0
    inter_agent_collision: float = 8.0
    near_distance: float = 0.5
    terminal_pairwise_gap: float = 0.0
    terminal_pairwise_dmax_multiplier: float = 1.0
    terminal_pairwise_dispersion_multiplier: float = 1.0
    subgoal_turn: float = 0.05
    subgoal_stagnation: float = 0.1
    action_consistency: float = 0.02
    success_hold_step: float = 0.0
    success_bonus: float = 10.0
    failure_penalty: float = 10.0
    timeout_penalty: float = 0.0


@dataclass(slots=True)
class SuccessThresholdsCfg:
    dmax: float = 1.25
    dispersion: float = 0.30
    speed: float = 0.25
    hold_steps: int = 8
    min_pairwise_distance: float = 0.0


@dataclass(slots=True)
class SafetyCfg:
    world_xy_limit: float = 12.0
    collision_distance: float = 0.28
    near_distance: float = 0.65


@dataclass(slots=True)
class ObservationCfg:
    communication_radius: float = 6.0
    schema_version: str = "ego_v3_local_terrain_grid"
    max_neighbors: int = 3
    ego_dim: int = 10
    neighbor_dim: int = 7
    terrain_dim: int = 50
    aggregation_dim: int = 5

    @property
    def effective_neighbor_dim(self) -> int:
        if self.schema_version == "ego_v8_decentralized_tiered":
            return 12
        return self.neighbor_dim

    @property
    def terminal_gate_dim(self) -> int:
        if self.schema_version == "ego_v4_terminal_gate":
            return 5
        return 0

    @property
    def gather_site_goal_dim(self) -> int:
        """Dimension of the execution-time terrain-aware gather-site target.

        ``ego_v5_gather_site_goal`` contains a rover-frame vector and distance
        to the common point selected by the terrain-aware search.
        ``ego_v6_gather_slot_goal`` has the same dimensionality, but targets a
        rover-specific, symmetry-preserving slot around that searched point.
        Neither schema exposes global coordinates or terrain-search internals.
        """
        if self.schema_version in {
            "ego_v5_gather_site_goal",
            "ego_v6_gather_slot_goal",
        }:
            return 3
        if self.schema_version == "ego_v7_gather_site_and_slot_goal":
            return 6
        return 0

    @property
    def actor_obs_dim(self) -> int:
        return (
            self.ego_dim
            + self.max_neighbors * self.effective_neighbor_dim
            + self.terrain_dim
            + self.aggregation_dim
            + self.terminal_gate_dim
            + self.gather_site_goal_dim
        )


@dataclass(slots=True)
class StateCfg:
    agent_state_dim: int = 8
    team_state_dim: int = 8
    terrain_state_dim: int = 5
    oracle_state_dim: int = 9
    include_terminal_min_pairwise: bool = False


@dataclass(slots=True)
class MultiRoverGatheringEnvCfg:
    simulation: SimulationCfg = field(default_factory=SimulationCfg)
    task: TaskCfg = field(default_factory=TaskCfg)
    initial_state: InitialStateCfg = field(default_factory=InitialStateCfg)
    planner: PlannerCfg = field(default_factory=PlannerCfg)
    trajectory_generator: TrajectoryGeneratorCfg = field(default_factory=TrajectoryGeneratorCfg)
    low_level_control: LowLevelControlCfg = field(default_factory=LowLevelControlCfg)
    terrain: TerrainCfg = field(default_factory=TerrainCfg)
    gather_point: GatherPointCfg = field(default_factory=GatherPointCfg)
    reward_weights: RewardWeightsCfg = field(default_factory=RewardWeightsCfg)
    reward_coefficients: RewardCoefficientsCfg = field(default_factory=RewardCoefficientsCfg)
    success_thresholds: SuccessThresholdsCfg = field(default_factory=SuccessThresholdsCfg)
    safety: SafetyCfg = field(default_factory=SafetyCfg)
    observation: ObservationCfg = field(default_factory=ObservationCfg)
    state: StateCfg = field(default_factory=StateCfg)
    seed: int = 7

    @property
    def actor_obs_dim(self) -> int:
        return self.observation.actor_obs_dim

    @property
    def critic_state_dim(self) -> int:
        terminal_team_dim = (
            1
            if self.state.include_terminal_min_pairwise
            or self.observation.schema_version == "ego_v4_terminal_gate"
            else 0
        )
        return (
            self.task.n_agents * self.state.agent_state_dim
            + self.state.team_state_dim
            + terminal_team_dim
            + self.state.terrain_state_dim
            + self.state.oracle_state_dim
        )


def make_debug_cfg(num_envs: int = 8, device: str = "cpu") -> MultiRoverGatheringEnvCfg:
    cfg = MultiRoverGatheringEnvCfg()
    cfg.simulation.num_envs = num_envs
    cfg.simulation.device = device
    cfg.simulation.episode_length_s = 8.0
    return cfg
