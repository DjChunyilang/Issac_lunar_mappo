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


def cfg_from_experiment(path: str | Path) -> MultiRoverGatheringEnvCfg:
    data = load_yaml(path)
    cfg = MultiRoverGatheringEnvCfg()
    experiment = data.get("experiment", {})
    simulation = data.get("simulation", {})
    task = data.get("task", {})
    planner = data.get("planner", {})
    terrain = data.get("terrain", {})
    algorithm = data.get("algorithm", {})
    del algorithm

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
    cfg.terrain.type = str(terrain.get("type", cfg.terrain.type))
    cfg.terrain.amplitude = float(terrain.get("amplitude", cfg.terrain.amplitude))
    cfg.terrain.wavelength = float(terrain.get("wavelength", cfg.terrain.wavelength))
    cfg.terrain.roughness_scale = float(
        terrain.get("roughness_scale", cfg.terrain.roughness_scale)
    )
    cfg.terrain.traversability_slope_scale = float(
        terrain.get("traversability_slope_scale", cfg.terrain.traversability_slope_scale)
    )
    return cfg


def ensure_output_dir(path: str | Path) -> Path:
    output = ROOT / Path(path)
    output.mkdir(parents=True, exist_ok=True)
    return output
