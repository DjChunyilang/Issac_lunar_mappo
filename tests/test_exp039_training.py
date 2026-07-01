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
from lunar_rover_tasks.tasks.multi_rover_gathering.action_interpreter import (  # noqa: E402
    decode_action,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.subgoal_filter import (  # noqa: E402
    apply_subgoal_filter,
)


CONFIG = ROOT / "configs/experiment/exp039_randomized_terrain_hard_near_stabilizer.yaml"


def test_exp039_config_enables_hard_near_override() -> None:
    raw = load_yaml(CONFIG)
    cfg = cfg_from_experiment(CONFIG)
    exp038 = cfg_from_experiment(
        ROOT / "configs/experiment/exp038_randomized_terrain_success_zone_stabilizer.yaml"
    )
    filter_cfg = cfg.planner.subgoal_filter

    assert raw["experiment"]["name"] == "exp039_randomized_terrain_hard_near_stabilizer"
    assert cfg.simulation.max_episode_steps == exp038.simulation.max_episode_steps == 320
    assert raw["algorithm"]["mode"] == "pure_rl"
    assert raw["algorithm"]["update_mode"] == "shared_joint"
    assert raw["algorithm"]["bc_updates"] == 0
    assert filter_cfg.hard_endpoint_near_filter
    assert filter_cfg.safety_override_after_warmup
    assert filter_cfg.collision_override_after_warmup
    assert filter_cfg.endpoint_safe_distance > exp038.planner.subgoal_filter.endpoint_safe_distance
    assert filter_cfg.hold_zone_pairwise_distance > exp038.planner.subgoal_filter.hold_zone_pairwise_distance
    assert filter_cfg.apply_probability_end > exp038.planner.subgoal_filter.apply_probability_end


def test_exp039_near_unsafe_raw_action_is_overridden_after_warmup() -> None:
    cfg = cfg_from_experiment(CONFIG)
    cfg.simulation.device = "cpu"
    cfg.terrain.type = "flat_proxy"
    positions = torch.tensor([[[0.0, 0.0, 0.0], [0.40, 0.0, 0.0]]])
    yaws = torch.tensor([[1.5708, -1.5708]])
    raw_action = torch.tensor([[[-0.5, -1.0], [-0.5, -1.0]]])
    decoded = decode_action(raw_action, positions, yaws, cfg.planner)

    result = apply_subgoal_filter(
        decoded,
        positions,
        yaws,
        cfg,
        progress_timestep=8192,
        deterministic=True,
    )

    assert result.info["safety_override"].any()
    assert result.info["applied"].any()
    assert result.info["endpoint_near_violation"].max() <= 1.0e-6


def test_exp039_warmup_only_scores_near_unsafe_raw_action() -> None:
    cfg = cfg_from_experiment(CONFIG)
    cfg.simulation.device = "cpu"
    cfg.terrain.type = "flat_proxy"
    positions = torch.tensor([[[0.0, 0.0, 0.0], [0.40, 0.0, 0.0]]])
    yaws = torch.tensor([[1.5708, -1.5708]])
    raw_action = torch.tensor([[[-0.5, -1.0], [-0.5, -1.0]]])
    decoded = decode_action(raw_action, positions, yaws, cfg.planner)

    result = apply_subgoal_filter(
        decoded,
        positions,
        yaws,
        cfg,
        progress_timestep=0,
        deterministic=True,
    )

    assert not result.info["safety_override"].any()
    assert not result.info["applied"].any()
