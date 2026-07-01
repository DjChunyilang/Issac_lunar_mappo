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


CONFIG = ROOT / "configs/experiment/exp040_randomized_terrain_soft_hold_stabilizer.yaml"


def test_exp040_config_strengthens_soft_hold_without_hard_near_filter() -> None:
    raw = load_yaml(CONFIG)
    cfg = cfg_from_experiment(CONFIG)
    exp038 = cfg_from_experiment(
        ROOT / "configs/experiment/exp038_randomized_terrain_success_zone_stabilizer.yaml"
    )
    filter_cfg = cfg.planner.subgoal_filter

    assert raw["experiment"]["name"] == "exp040_randomized_terrain_soft_hold_stabilizer"
    assert cfg.simulation.max_episode_steps == 320
    assert raw["algorithm"]["mode"] == "pure_rl"
    assert raw["algorithm"]["bc_updates"] == 0
    assert not filter_cfg.hard_endpoint_near_filter
    assert not filter_cfg.safety_override_after_warmup
    assert filter_cfg.hold_zone_spacing_weight > exp038.planner.subgoal_filter.hold_zone_spacing_weight
    assert filter_cfg.hold_zone_pairwise_distance > exp038.planner.subgoal_filter.hold_zone_pairwise_distance
    assert filter_cfg.apply_probability_end > exp038.planner.subgoal_filter.apply_probability_end
    assert filter_cfg.deterministic_improvement_margin < exp038.planner.subgoal_filter.deterministic_improvement_margin


def test_exp040_compact_near_success_team_reports_stronger_hold_zone_costs() -> None:
    cfg = cfg_from_experiment(CONFIG)
    cfg.simulation.device = "cpu"
    cfg.simulation.num_envs = 1
    cfg.terrain.type = "flat_proxy"
    cfg.planner.subgoal_filter.progress_timestep_override = 8192
    cfg.planner.subgoal_filter.deterministic_eval = True
    env = MultiRoverGatheringCore(cfg)
    env.positions[:] = torch.tensor(
        [[[0.0, 0.0, 0.0], [0.43, 0.0, 0.0], [0.0, 0.43, 0.0], [0.43, 0.43, 0.0]]]
    )

    out = env.step(torch.zeros(cfg.simulation.num_envs, cfg.task.n_agents, 2))
    action_filter = out.info["action_filter"]

    assert action_filter["hold_zone_activation"].float().mean() > 0.0
    assert action_filter["raw_hold_zone_spacing_violation"].float().mean() > 0.0
    assert torch.isfinite(action_filter["hold_zone_spacing_violation"]).all()
