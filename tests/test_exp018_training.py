from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from _common import cfg_from_experiment, load_yaml  # noqa: E402
from lunar_rover_tasks.tasks.multi_rover_gathering.gathering_env import (  # noqa: E402
    MultiRoverGatheringCore,
)


CONFIG = ROOT / "configs/experiment/exp018_randomized_terrain_pure_rl.yaml"


def test_exp018_randomized_terrain_and_reward_contract() -> None:
    raw = load_yaml(CONFIG)
    cfg = cfg_from_experiment(CONFIG)
    terrain = raw["terrain"]
    reward = raw["reward"]
    algorithm = raw["algorithm"]

    assert terrain["randomize_per_reset"] is True
    assert terrain["amplitude"] == pytest.approx(0.10)
    assert terrain["crater_count"] == 9
    assert terrain["crater_depth_to_diameter"] == pytest.approx(0.12)
    assert terrain["min_speed_scale"] == pytest.approx(0.22)
    assert terrain["amplitude_scale_min"] < terrain["amplitude_scale_max"]
    assert terrain["crater_radius_scale_min"] < terrain["crater_radius_scale_max"]
    assert terrain["crater_depth_scale_min"] < terrain["crater_depth_scale_max"]
    assert reward["weights"]["terrain"] == pytest.approx(0.30)
    assert cfg.reward_coefficients.subgoal_terrain_cost == pytest.approx(0.45)
    assert cfg.reward_coefficients.terrain_speed_loss_cost == pytest.approx(0.30)
    assert cfg.reward_coefficients.terrain_height_change_cost == pytest.approx(0.25)
    assert algorithm["mode"] == "pure_rl"
    assert algorithm["update_mode"] == "shared_joint"
    assert algorithm["bc_updates"] == 0


def test_exp018_parallel_environments_start_with_distinct_maps() -> None:
    cfg = cfg_from_experiment(CONFIG)
    cfg.simulation.device = "cpu"
    cfg.simulation.num_envs = 32
    env = MultiRoverGatheringCore(cfg)

    assert torch.unique(env.terrain_runtime.phase).numel() == 32
    assert torch.unique(env.terrain_runtime.translation_xy, dim=0).shape[0] == 32
    actor_obs, critic_state = env.get_observations()
    assert actor_obs.shape == (32, 4, 86)
    assert critic_state.shape == (32, 54)
    assert torch.isfinite(actor_obs).all()
    assert torch.isfinite(critic_state).all()
