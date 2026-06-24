"""Shared script helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "source" / "lunar_rover_tasks"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from lunar_rover_tasks.tasks.multi_rover_gathering.gathering_env_cfg import (  # noqa: E402
    MultiRoverGatheringEnvCfg,
)


def load_yaml(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream) or {}


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
    supported = {"communication_radius"}
    unknown = sorted(key for key in values if key not in supported)
    if unknown:
        unknown_keys = ", ".join(f"observation.{key}" for key in unknown)
        raise ValueError(
            f"Unsupported config key(s): {unknown_keys}. This loader only supports "
            "'observation.communication_radius'; observation dimensions and schema "
            "version are fixed by code."
        )
    if "communication_radius" in values:
        cfg.observation.communication_radius = float(values["communication_radius"])


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
    planner = _require_mapping("planner", data.get("planner", {}))
    low_level_control = _require_mapping("low_level_control", data.get("low_level_control", {}))
    terrain = _require_mapping("terrain", data.get("terrain", {}))
    reward = _require_mapping("reward", data.get("reward", {}))
    observation = _require_mapping("observation", data.get("observation", {}))
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
    cfg.task.n_agents = int(task.get("n_agents", cfg.task.n_agents))
    cfg.planner.rho_max = float(planner.get("rho_max", cfg.planner.rho_max))
    cfg.planner.beta_max = float(planner.get("beta_max", cfg.planner.beta_max))
    _apply_values(cfg.low_level_control, low_level_control, "low_level_control")
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
    _apply_observation_values(cfg, observation)
    _apply_values(cfg.safety, safety, "safety")
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
