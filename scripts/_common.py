"""Shared script helpers."""

from __future__ import annotations

from copy import deepcopy
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "source" / "lunar_rover_tasks"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from lunar_rover_tasks.tasks.multi_rover_gathering.gathering_env_cfg import (  # noqa: E402
    MultiRoverGatheringEnvCfg,
)


def _deep_merge_yaml(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_yaml(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_yaml(path: str | Path, *, _ancestry: tuple[Path, ...] = ()) -> dict:
    """Load an experiment YAML, resolving an optional relative ``extends`` base."""
    config_path = Path(path).resolve()
    if config_path in _ancestry:
        chain = " -> ".join(str(item) for item in (*_ancestry, config_path))
        raise ValueError(f"Cyclic YAML extends chain: {chain}")
    with config_path.open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"YAML config must be a mapping: {config_path}")
    extends = payload.pop("extends", None)
    if extends is None:
        return payload
    if not isinstance(extends, str) or not extends.strip():
        raise ValueError("YAML 'extends' must be a non-empty relative or absolute path.")
    base_path = Path(extends)
    if not base_path.is_absolute():
        base_path = config_path.parent / base_path
    base = load_yaml(base_path, _ancestry=(*_ancestry, config_path))
    return _deep_merge_yaml(base, payload)


def _require_mapping(section: str, values) -> dict:
    if values is None:
        return {}
    if not isinstance(values, dict):
        raise ValueError(f"Config section '{section}' must be a mapping.")
    return values


def _apply_values(target, values: dict, section: str) -> None:
    values = _require_mapping(section, values)
    unknown = sorted(key for key in values if not hasattr(target, key))
    if unknown:
        unknown_keys = ", ".join(f"{section}.{key}" for key in unknown)
        raise ValueError(f"Unsupported config key(s): {unknown_keys}.")
    for key, value in values.items():
        current = getattr(target, key)
        setattr(target, key, type(current)(value))


def _validate_reward_section(reward: dict) -> None:
    supported = {"weights", "coefficients"}
    unknown = sorted(key for key in reward if key not in supported)
    if not unknown:
        return
    if "oracle_weight" in unknown:
        raise ValueError(
            "Unsupported config key 'reward.oracle_weight'; use "
            "'reward.weights.oracle' so oracle ablations are wired into cfg.reward_weights."
        )
    unknown_keys = ", ".join(f"reward.{key}" for key in unknown)
    raise ValueError(
        f"Unsupported config key(s): {unknown_keys}. Supported reward keys are "
        "'reward.weights' and 'reward.coefficients'."
    )


def _apply_observation_values(cfg: MultiRoverGatheringEnvCfg, values: dict) -> None:
    values = _require_mapping("observation", values)
    supported = {"communication_radius", "schema_version"}
    unknown = sorted(key for key in values if key not in supported)
    if unknown:
        unknown_keys = ", ".join(f"observation.{key}" for key in unknown)
        raise ValueError(
            f"Unsupported config key(s): {unknown_keys}. This loader only supports "
            "'observation.communication_radius' and 'observation.schema_version'; "
            "observation dimensions are fixed by code."
        )
    if "communication_radius" in values:
        cfg.observation.communication_radius = float(values["communication_radius"])
    if "schema_version" in values:
        schema_version = str(values["schema_version"])
        supported_schemas = {
            "ego_v3_local_terrain_grid",
            "ego_v4_terminal_gate",
            "ego_v5_gather_site_goal",
            "ego_v6_gather_slot_goal",
            "ego_v7_gather_site_and_slot_goal",
        }
        if schema_version not in supported_schemas:
            raise ValueError(
                "observation.schema_version must be one of: "
                f"{', '.join(sorted(supported_schemas))}."
            )
        cfg.observation.schema_version = schema_version


def _apply_state_values(cfg: MultiRoverGatheringEnvCfg, values: dict) -> None:
    values = _require_mapping("state", values)
    supported = {"include_terminal_min_pairwise"}
    unknown = sorted(key for key in values if key not in supported)
    if unknown:
        unknown_keys = ", ".join(f"state.{key}" for key in unknown)
        raise ValueError(
            f"Unsupported config key(s): {unknown_keys}. This loader only supports "
            "'state.include_terminal_min_pairwise'; state dimensions are fixed by code."
        )
    if "include_terminal_min_pairwise" in values:
        cfg.state.include_terminal_min_pairwise = bool(
            values["include_terminal_min_pairwise"]
        )


def _apply_planner_values(cfg: MultiRoverGatheringEnvCfg, values: dict) -> None:
    values = _require_mapping("planner", values)
    supported = {"rho_max", "beta_max", "subgoal_filter"}
    unknown = sorted(key for key in values if key not in supported)
    if unknown:
        unknown_keys = ", ".join(f"planner.{key}" for key in unknown)
        raise ValueError(f"Unsupported config key(s): {unknown_keys}.")
    cfg.planner.rho_max = float(values.get("rho_max", cfg.planner.rho_max))
    cfg.planner.beta_max = float(values.get("beta_max", cfg.planner.beta_max))
    filter_values = _require_mapping(
        "planner.subgoal_filter",
        values.get("subgoal_filter", {}),
    )
    filter_unknown = sorted(
        key for key in filter_values if not hasattr(cfg.planner.subgoal_filter, key)
    )
    if filter_unknown:
        unknown_keys = ", ".join(
            f"planner.subgoal_filter.{key}" for key in filter_unknown
        )
        raise ValueError(f"Unsupported config key(s): {unknown_keys}.")
    for key, value in filter_values.items():
        current = getattr(cfg.planner.subgoal_filter, key)
        if isinstance(current, bool):
            converted = bool(value)
        elif isinstance(current, int):
            converted = int(value)
        elif isinstance(current, float):
            converted = float(value)
        elif isinstance(current, list):
            converted = [float(item) for item in value]
        else:
            converted = type(current)(value)
        setattr(cfg.planner.subgoal_filter, key, converted)
    if not cfg.planner.subgoal_filter.rho_scales:
        raise ValueError("planner.subgoal_filter.rho_scales must not be empty.")
    if not cfg.planner.subgoal_filter.beta_offsets_deg:
        raise ValueError("planner.subgoal_filter.beta_offsets_deg must not be empty.")
    if cfg.planner.subgoal_filter.path_samples <= 0:
        raise ValueError("planner.subgoal_filter.path_samples must be positive.")
    if not (0.0 <= cfg.planner.subgoal_filter.apply_probability_end <= 1.0):
        raise ValueError("planner.subgoal_filter.apply_probability_end must be in [0, 1].")
    if cfg.planner.subgoal_filter.ramp_timesteps <= 0:
        raise ValueError("planner.subgoal_filter.ramp_timesteps must be positive.")
    if cfg.planner.subgoal_filter.warmup_timesteps < 0:
        raise ValueError("planner.subgoal_filter.warmup_timesteps must be non-negative.")


def _apply_trajectory_generator_values(
    cfg: MultiRoverGatheringEnvCfg,
    values: dict,
) -> None:
    _apply_values(cfg.trajectory_generator, values, "trajectory_generator")
    if cfg.trajectory_generator.geometry_method not in {"line", "quintic"}:
        raise ValueError(
            "trajectory_generator.geometry_method must be one of: line, quintic."
        )
    if cfg.trajectory_generator.n_trajectory_points < 2:
        raise ValueError("trajectory_generator.n_trajectory_points must be at least 2.")
    if cfg.trajectory_generator.quintic_tangent_scale < 0.0:
        raise ValueError("trajectory_generator.quintic_tangent_scale must be non-negative.")
    if cfg.trajectory_generator.end_heading_mode not in {"subgoal_direction"}:
        raise ValueError(
            "trajectory_generator.end_heading_mode currently supports only subgoal_direction."
        )


def _validate_low_level_control(cfg: MultiRoverGatheringEnvCfg) -> None:
    if cfg.low_level_control.kinematic_model not in {"unicycle", "bicycle"}:
        raise ValueError("low_level_control.kinematic_model must be one of: unicycle, bicycle.")
    if cfg.low_level_control.wheelbase_m <= 0.0:
        raise ValueError("low_level_control.wheelbase_m must be positive.")
    if cfg.low_level_control.max_steer_angle_rad <= 0.0:
        raise ValueError("low_level_control.max_steer_angle_rad must be positive.")
    if cfg.low_level_control.formation_center_activation_dmax_multiplier < 1.0:
        raise ValueError(
            "low_level_control.formation_center_activation_dmax_multiplier "
            "must be >= 1.0."
        )
    if cfg.low_level_control.formation_center_activation_dispersion_multiplier < 1.0:
        raise ValueError(
            "low_level_control.formation_center_activation_dispersion_multiplier "
            "must be >= 1.0."
        )
    if cfg.low_level_control.formation_center_correction_max_offset < 0.0:
        raise ValueError(
            "low_level_control.formation_center_correction_max_offset must be non-negative."
        )
    if not 0.0 <= cfg.low_level_control.formation_center_correction_gain <= 1.0:
        raise ValueError(
            "low_level_control.formation_center_correction_gain must be in [0, 1]."
        )
    if cfg.low_level_control.formation_center_local_flatness_search_radius < 0.0:
        raise ValueError(
            "low_level_control.formation_center_local_flatness_search_radius "
            "must be non-negative."
        )
    if cfg.low_level_control.formation_center_local_flatness_search_samples < 4:
        raise ValueError(
            "low_level_control.formation_center_local_flatness_search_samples "
            "must be at least 4."
        )
    if cfg.low_level_control.terminal_slot_capture_dmax_multiplier < 1.0:
        raise ValueError(
            "low_level_control.terminal_slot_capture_dmax_multiplier must be >= 1.0."
        )
    if cfg.low_level_control.terminal_slot_capture_dispersion_multiplier < 1.0:
        raise ValueError(
            "low_level_control.terminal_slot_capture_dispersion_multiplier must be >= 1.0."
        )
    if not 0.0 <= cfg.low_level_control.terminal_slot_capture_blend <= 1.0:
        raise ValueError(
            "low_level_control.terminal_slot_capture_blend must be in [0, 1]."
        )
    if cfg.low_level_control.flat_geometry_capture_dmax_multiplier < 1.0:
        raise ValueError(
            "low_level_control.flat_geometry_capture_dmax_multiplier must be >= 1.0."
        )
    if cfg.low_level_control.flat_geometry_capture_dispersion_multiplier < 1.0:
        raise ValueError(
            "low_level_control.flat_geometry_capture_dispersion_multiplier must be >= 1.0."
        )
    if not 0.0 <= cfg.low_level_control.flat_geometry_capture_blend <= 1.0:
        raise ValueError(
            "low_level_control.flat_geometry_capture_blend must be in [0, 1]."
        )


def _validate_dynamic_terminal_slot_goal(cfg: MultiRoverGatheringEnvCfg) -> None:
    task = cfg.task
    if task.dynamic_terminal_slot_goal_dmax_multiplier < 1.0:
        raise ValueError(
            "task.dynamic_terminal_slot_goal_dmax_multiplier must be >= 1.0."
        )
    if task.dynamic_terminal_slot_goal_dispersion_multiplier < 1.0:
        raise ValueError(
            "task.dynamic_terminal_slot_goal_dispersion_multiplier must be >= 1.0."
        )
    if task.dynamic_terminal_slot_goal_search_radius < 0.0:
        raise ValueError(
            "task.dynamic_terminal_slot_goal_search_radius must be non-negative."
        )
    if task.dynamic_terminal_slot_goal_search_samples < 4:
        raise ValueError(
            "task.dynamic_terminal_slot_goal_search_samples must be at least 4."
        )


def _validate_gather_point(cfg: MultiRoverGatheringEnvCfg) -> None:
    gather = cfg.gather_point
    supported_methods = {
        "terrain_aware_multiresolution",
        "geometric_median",
    }
    if gather.search_method not in supported_methods:
        raise ValueError(
            "gather_point.search_method must be one of: "
            f"{', '.join(sorted(supported_methods))}."
        )
    if gather.coarse_grid_size < 3 or gather.coarse_grid_size % 2 == 0:
        raise ValueError("gather_point.coarse_grid_size must be an odd integer >= 3.")
    if gather.refinement_grid_size < 3 or gather.refinement_grid_size % 2 == 0:
        raise ValueError("gather_point.refinement_grid_size must be an odd integer >= 3.")
    if gather.refinement_levels < 0:
        raise ValueError("gather_point.refinement_levels must be non-negative.")
    if gather.search_margin < 0.0:
        raise ValueError("gather_point.search_margin must be non-negative.")
    if gather.global_grid_size < 3 or gather.global_grid_size % 2 == 0:
        raise ValueError("gather_point.global_grid_size must be an odd integer >= 3.")
    if gather.global_beam_width <= 0:
        raise ValueError("gather_point.global_beam_width must be positive.")
    if gather.global_refinement_levels < 0:
        raise ValueError("gather_point.global_refinement_levels must be non-negative.")
    if gather.global_max_envs_per_batch <= 0:
        raise ValueError("gather_point.global_max_envs_per_batch must be positive.")
    if gather.flatness_radius <= 0.0:
        raise ValueError("gather_point.flatness_radius must be positive.")
    if gather.flatness_rings <= 0:
        raise ValueError("gather_point.flatness_rings must be positive.")
    if gather.flatness_samples_per_ring < 4:
        raise ValueError("gather_point.flatness_samples_per_ring must be at least 4.")
    if gather.max_height_range <= 0.0:
        raise ValueError("gather_point.max_height_range must be positive.")
    if gather.max_slope <= 0.0:
        raise ValueError("gather_point.max_slope must be positive.")
    if gather.robustness_radius < 0.0:
        raise ValueError("gather_point.robustness_radius must be non-negative.")
    if gather.robustness_radius > 0.0 and gather.robustness_samples < 4:
        raise ValueError(
            "gather_point.robustness_samples must be at least 4 when "
            "robustness_radius is positive."
        )
    weights = (
        gather.mean_distance_weight,
        gather.max_distance_weight,
        gather.path_risk_weight,
        gather.path_height_change_weight,
        gather.flatness_weight,
    )
    if any(weight < 0.0 for weight in weights):
        raise ValueError("gather_point objective weights must be non-negative.")
    if not any(weight > 0.0 for weight in weights):
        raise ValueError("At least one gather_point objective weight must be positive.")
    if gather.path_samples <= 0:
        raise ValueError("gather_point.path_samples must be positive.")
    if gather.infeasible_penalty <= 0.0:
        raise ValueError("gather_point.infeasible_penalty must be positive.")
    if gather.max_envs_per_batch <= 0:
        raise ValueError("gather_point.max_envs_per_batch must be positive.")
    if gather.execution_slot_radius <= 0.0:
        raise ValueError("gather_point.execution_slot_radius must be positive.")
    if gather.flatness_radius >= cfg.safety.world_xy_limit:
        raise ValueError(
            "gather_point.flatness_radius must be smaller than safety.world_xy_limit."
        )


def cfg_from_experiment(path: str | Path) -> MultiRoverGatheringEnvCfg:
    data = load_yaml(path)
    data = _require_mapping(str(path), data)
    if "experiment" not in data:
        raise ValueError(
            f"{path} is not an experiment config. cfg_from_experiment reads one "
            "experiment YAML and does not merge configs/agent, configs/env, "
            "configs/task, or configs/reward fragments."
        )
    cfg = MultiRoverGatheringEnvCfg()
    experiment = _require_mapping("experiment", data.get("experiment", {}))
    simulation = _require_mapping("simulation", data.get("simulation", {}))
    task = _require_mapping("task", data.get("task", {}))
    initial_state = _require_mapping("initial_state", data.get("initial_state", {}))
    planner = _require_mapping("planner", data.get("planner", {}))
    trajectory_generator = _require_mapping(
        "trajectory_generator",
        data.get("trajectory_generator", {}),
    )
    low_level_control = _require_mapping("low_level_control", data.get("low_level_control", {}))
    terrain = _require_mapping("terrain", data.get("terrain", {}))
    gather_point = _require_mapping("gather_point", data.get("gather_point", {}))
    reward = _require_mapping("reward", data.get("reward", {}))
    observation = _require_mapping("observation", data.get("observation", {}))
    state = _require_mapping("state", data.get("state", {}))
    safety = _require_mapping("safety", data.get("safety", {}))
    success_thresholds = _require_mapping("success_thresholds", data.get("success_thresholds", {}))
    algorithm = _require_mapping("algorithm", data.get("algorithm", {}))
    del algorithm
    _validate_reward_section(reward)

    cfg.seed = int(experiment.get("seed", cfg.seed))
    cfg.simulation.num_envs = int(experiment.get("num_envs", cfg.simulation.num_envs))
    cfg.simulation.device = str(experiment.get("device", cfg.simulation.device))
    cfg.simulation.episode_length_s = float(
        simulation.get("episode_length_s", cfg.simulation.episode_length_s)
    )
    cfg.simulation.physics_dt = float(simulation.get("physics_dt", cfg.simulation.physics_dt))
    cfg.simulation.control_decimation = int(
        simulation.get("control_decimation", cfg.simulation.control_decimation)
    )
    supported_task_keys = {
        "n_agents",
        "explicit_goal_in_execution",
        "execution_slot_reward_target",
        "dynamic_terminal_slot_goal_enabled",
        "dynamic_terminal_slot_goal_dmax_multiplier",
        "dynamic_terminal_slot_goal_dispersion_multiplier",
        "dynamic_terminal_slot_goal_require_flatness_failure",
        "dynamic_terminal_slot_goal_search_radius",
        "dynamic_terminal_slot_goal_search_samples",
    }
    task_unknown = sorted(key for key in task if key not in supported_task_keys)
    if task_unknown:
        unknown_keys = ", ".join(f"task.{key}" for key in task_unknown)
        raise ValueError(f"Unsupported config key(s): {unknown_keys}.")
    cfg.task.n_agents = int(task.get("n_agents", cfg.task.n_agents))
    cfg.task.explicit_goal_in_execution = bool(
        task.get("explicit_goal_in_execution", cfg.task.explicit_goal_in_execution)
    )
    cfg.task.execution_slot_reward_target = bool(
        task.get(
            "execution_slot_reward_target",
            cfg.task.execution_slot_reward_target,
        )
    )
    cfg.task.dynamic_terminal_slot_goal_enabled = bool(
        task.get(
            "dynamic_terminal_slot_goal_enabled",
            cfg.task.dynamic_terminal_slot_goal_enabled,
        )
    )
    cfg.task.dynamic_terminal_slot_goal_dmax_multiplier = float(
        task.get(
            "dynamic_terminal_slot_goal_dmax_multiplier",
            cfg.task.dynamic_terminal_slot_goal_dmax_multiplier,
        )
    )
    cfg.task.dynamic_terminal_slot_goal_dispersion_multiplier = float(
        task.get(
            "dynamic_terminal_slot_goal_dispersion_multiplier",
            cfg.task.dynamic_terminal_slot_goal_dispersion_multiplier,
        )
    )
    cfg.task.dynamic_terminal_slot_goal_require_flatness_failure = bool(
        task.get(
            "dynamic_terminal_slot_goal_require_flatness_failure",
            cfg.task.dynamic_terminal_slot_goal_require_flatness_failure,
        )
    )
    cfg.task.dynamic_terminal_slot_goal_search_radius = float(
        task.get(
            "dynamic_terminal_slot_goal_search_radius",
            cfg.task.dynamic_terminal_slot_goal_search_radius,
        )
    )
    cfg.task.dynamic_terminal_slot_goal_search_samples = int(
        task.get(
            "dynamic_terminal_slot_goal_search_samples",
            cfg.task.dynamic_terminal_slot_goal_search_samples,
        )
    )
    _validate_dynamic_terminal_slot_goal(cfg)
    _apply_values(cfg.initial_state, initial_state, "initial_state")
    if cfg.initial_state.spawn_radius_min <= 0.0:
        raise ValueError("initial_state.spawn_radius_min must be positive.")
    if cfg.initial_state.spawn_radius_max < cfg.initial_state.spawn_radius_min:
        raise ValueError(
            "initial_state.spawn_radius_max must be >= initial_state.spawn_radius_min."
        )
    if cfg.initial_state.center_xy_range < 0.0:
        raise ValueError("initial_state.center_xy_range must be non-negative.")
    if cfg.initial_state.jitter_std < 0.0:
        raise ValueError("initial_state.jitter_std must be non-negative.")
    if cfg.initial_state.curriculum_start_spawn_radius_min <= 0.0:
        raise ValueError(
            "initial_state.curriculum_start_spawn_radius_min must be positive."
        )
    if (
        cfg.initial_state.curriculum_start_spawn_radius_max
        < cfg.initial_state.curriculum_start_spawn_radius_min
    ):
        raise ValueError(
            "initial_state.curriculum_start_spawn_radius_max must be >= "
            "initial_state.curriculum_start_spawn_radius_min."
        )
    if cfg.initial_state.curriculum_start_center_xy_range < 0.0:
        raise ValueError(
            "initial_state.curriculum_start_center_xy_range must be non-negative."
        )
    if cfg.initial_state.curriculum_start_jitter_std < 0.0:
        raise ValueError("initial_state.curriculum_start_jitter_std must be non-negative.")
    if cfg.initial_state.curriculum_warmup_timesteps < 0:
        raise ValueError("initial_state.curriculum_warmup_timesteps must be non-negative.")
    if cfg.initial_state.curriculum_ramp_timesteps <= 0:
        raise ValueError("initial_state.curriculum_ramp_timesteps must be positive.")
    if cfg.initial_state.progress_timestep_override < -1:
        raise ValueError("initial_state.progress_timestep_override must be >= -1.")
    _apply_planner_values(cfg, planner)
    _apply_trajectory_generator_values(cfg, trajectory_generator)
    _apply_values(cfg.low_level_control, low_level_control, "low_level_control")
    _validate_low_level_control(cfg)
    cfg.terrain.type = str(terrain.get("type", cfg.terrain.type))
    cfg.terrain.amplitude = float(terrain.get("amplitude", cfg.terrain.amplitude))
    cfg.terrain.wavelength = float(terrain.get("wavelength", cfg.terrain.wavelength))
    cfg.terrain.roughness_scale = float(
        terrain.get("roughness_scale", cfg.terrain.roughness_scale)
    )
    cfg.terrain.traversability_slope_scale = float(
        terrain.get("traversability_slope_scale", cfg.terrain.traversability_slope_scale)
    )
    cfg.terrain.dynamics_enabled = bool(
        terrain.get("dynamics_enabled", cfg.terrain.dynamics_enabled)
    )
    cfg.terrain.slope_speed_scale = float(
        terrain.get("slope_speed_scale", cfg.terrain.slope_speed_scale)
    )
    cfg.terrain.min_speed_scale = float(
        terrain.get("min_speed_scale", cfg.terrain.min_speed_scale)
    )
    cfg.terrain.crater_count = int(terrain.get("crater_count", cfg.terrain.crater_count))
    cfg.terrain.crater_min_radius = float(
        terrain.get("crater_min_radius", cfg.terrain.crater_min_radius)
    )
    cfg.terrain.crater_max_radius = float(
        terrain.get("crater_max_radius", cfg.terrain.crater_max_radius)
    )
    cfg.terrain.crater_depth_to_diameter = float(
        terrain.get("crater_depth_to_diameter", cfg.terrain.crater_depth_to_diameter)
    )
    cfg.terrain.crater_rim_height_to_diameter = float(
        terrain.get("crater_rim_height_to_diameter", cfg.terrain.crater_rim_height_to_diameter)
    )
    cfg.terrain.crater_field_size = float(
        terrain.get("crater_field_size", cfg.terrain.crater_field_size)
    )
    cfg.terrain.crater_seed = int(terrain.get("crater_seed", cfg.terrain.crater_seed))
    cfg.terrain.randomize_per_reset = bool(
        terrain.get("randomize_per_reset", cfg.terrain.randomize_per_reset)
    )
    cfg.terrain.random_translation_m = float(
        terrain.get("random_translation_m", cfg.terrain.random_translation_m)
    )
    cfg.terrain.random_yaw_rad = float(
        terrain.get("random_yaw_rad", cfg.terrain.random_yaw_rad)
    )
    cfg.terrain.amplitude_scale_min = float(
        terrain.get("amplitude_scale_min", cfg.terrain.amplitude_scale_min)
    )
    cfg.terrain.amplitude_scale_max = float(
        terrain.get("amplitude_scale_max", cfg.terrain.amplitude_scale_max)
    )
    cfg.terrain.crater_radius_scale_min = float(
        terrain.get("crater_radius_scale_min", cfg.terrain.crater_radius_scale_min)
    )
    cfg.terrain.crater_radius_scale_max = float(
        terrain.get("crater_radius_scale_max", cfg.terrain.crater_radius_scale_max)
    )
    cfg.terrain.crater_depth_scale_min = float(
        terrain.get("crater_depth_scale_min", cfg.terrain.crater_depth_scale_min)
    )
    cfg.terrain.crater_depth_scale_max = float(
        terrain.get("crater_depth_scale_max", cfg.terrain.crater_depth_scale_max)
    )
    _apply_values(cfg.reward_weights, reward.get("weights", {}), "reward.weights")
    _apply_values(cfg.reward_coefficients, reward.get("coefficients", {}), "reward.coefficients")
    if cfg.reward_coefficients.centroid_flatness_progress < 0.0:
        raise ValueError("reward.coefficients.centroid_flatness_progress must be non-negative.")
    if cfg.reward_coefficients.centroid_flatness_excess < 0.0:
        raise ValueError("reward.coefficients.centroid_flatness_excess must be non-negative.")
    if cfg.reward_coefficients.centroid_flatness_dmax_multiplier <= 1.0:
        raise ValueError(
            "reward.coefficients.centroid_flatness_dmax_multiplier must be greater than 1."
        )
    _apply_observation_values(cfg, observation)
    has_gather_site_goal = cfg.observation.schema_version in {
        "ego_v5_gather_site_goal",
        "ego_v6_gather_slot_goal",
        "ego_v7_gather_site_and_slot_goal",
    }
    if has_gather_site_goal != cfg.task.explicit_goal_in_execution:
        raise ValueError(
            "task.explicit_goal_in_execution must be true exactly when "
            "observation.schema_version is an execution gather-site goal schema."
        )
    if cfg.task.execution_slot_reward_target and cfg.observation.schema_version not in {
        "ego_v6_gather_slot_goal",
        "ego_v7_gather_site_and_slot_goal",
    }:
        raise ValueError(
            "task.execution_slot_reward_target requires an execution-slot "
            "observation schema."
        )
    if cfg.task.dynamic_terminal_slot_goal_enabled and cfg.observation.schema_version not in {
        "ego_v6_gather_slot_goal",
        "ego_v7_gather_site_and_slot_goal",
    }:
        raise ValueError(
            "task.dynamic_terminal_slot_goal_enabled requires an execution-slot "
            "observation schema."
        )
    _apply_state_values(cfg, state)
    _apply_values(cfg.safety, safety, "safety")
    _apply_values(cfg.gather_point, gather_point, "gather_point")
    _validate_gather_point(cfg)
    _apply_values(cfg.success_thresholds, success_thresholds, "success_thresholds")
    if cfg.success_thresholds.min_pairwise_distance > 0.0:
        if not (
            cfg.safety.collision_distance
            < cfg.success_thresholds.min_pairwise_distance
            < cfg.success_thresholds.dmax
        ):
            raise ValueError(
                "success_thresholds.min_pairwise_distance must satisfy "
                "safety.collision_distance < min_pairwise_distance < success_thresholds.dmax."
            )
    return cfg


def ensure_output_dir(path: str | Path) -> Path:
    output = ROOT / Path(path)
    output.mkdir(parents=True, exist_ok=True)
    return output
