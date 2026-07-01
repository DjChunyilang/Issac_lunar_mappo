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


CONFIG = ROOT / "configs/experiment/exp041_randomized_terrain_hold_zone_override.yaml"


def test_exp041_config_enables_only_hold_zone_override() -> None:
    raw = load_yaml(CONFIG)
    cfg = cfg_from_experiment(CONFIG)
    exp038 = cfg_from_experiment(
        ROOT / "configs/experiment/exp038_randomized_terrain_success_zone_stabilizer.yaml"
    )
    filter_cfg = cfg.planner.subgoal_filter

    assert raw["experiment"]["name"] == "exp041_randomized_terrain_hold_zone_override"
    assert cfg.simulation.max_episode_steps == 320
    assert raw["algorithm"]["mode"] == "pure_rl"
    assert raw["algorithm"]["bc_updates"] == 0
    assert filter_cfg.hold_zone_override_after_warmup
    assert not filter_cfg.hard_endpoint_near_filter
    assert not filter_cfg.safety_override_after_warmup
    assert filter_cfg.hold_zone_spacing_weight > exp038.planner.subgoal_filter.hold_zone_spacing_weight
    assert filter_cfg.hold_zone_pairwise_distance > exp038.planner.subgoal_filter.hold_zone_pairwise_distance


def test_exp041_hold_zone_override_forces_spacing_improvement_after_warmup() -> None:
    cfg = cfg_from_experiment(CONFIG)
    cfg.simulation.device = "cpu"
    cfg.terrain.type = "flat_proxy"
    positions = torch.tensor([[[0.0, 0.0, 0.0], [0.43, 0.0, 0.0]]])
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

    assert result.info["hold_zone_activation"].float().mean() > 0.0
    assert result.info["hold_zone_override"].any()
    assert result.info["applied"].any()
    assert (
        result.info["hold_zone_spacing_violation"]
        <= result.info["raw_hold_zone_spacing_violation"] + 1.0e-6
    ).all()


def test_exp041_hold_zone_override_inactive_during_warmup() -> None:
    cfg = cfg_from_experiment(CONFIG)
    cfg.simulation.device = "cpu"
    cfg.terrain.type = "flat_proxy"
    positions = torch.tensor([[[0.0, 0.0, 0.0], [0.43, 0.0, 0.0]]])
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

    assert not result.info["hold_zone_override"].any()
