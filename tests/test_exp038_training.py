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


CONFIG = ROOT / "configs/experiment/exp038_randomized_terrain_success_zone_stabilizer.yaml"


def test_exp038_config_targets_success_zone_stability() -> None:
    raw = load_yaml(CONFIG)
    cfg = cfg_from_experiment(CONFIG)
    exp037 = cfg_from_experiment(
        ROOT / "configs/experiment/exp037_randomized_terrain_directional_mask_timeout260.yaml"
    )
    filter_cfg = cfg.planner.subgoal_filter
    control_cfg = cfg.low_level_control

    assert raw["experiment"]["name"] == "exp038_randomized_terrain_success_zone_stabilizer"
    assert exp037.simulation.max_episode_steps == 260
    assert cfg.simulation.max_episode_steps == 320
    assert raw["algorithm"]["mode"] == "pure_rl"
    assert raw["algorithm"]["update_mode"] == "shared_joint"
    assert raw["algorithm"]["bc_updates"] == 0
    assert filter_cfg.mode == "terrain_safe_candidate_hold_progress_curriculum"
    assert filter_cfg.hold_zone_dmax_multiplier == pytest.approx(1.0)
    assert filter_cfg.hold_zone_dispersion_multiplier == pytest.approx(1.0)
    assert filter_cfg.hold_zone_rho_weight > 0.0
    assert filter_cfg.hold_zone_spacing_weight > exp037.planner.subgoal_filter.hold_zone_spacing_weight
    assert filter_cfg.hold_zone_pairwise_distance > cfg.success_thresholds.min_pairwise_distance
    assert control_cfg.success_zone_damping_enabled
    assert control_cfg.success_zone_linear_scale < 1.0
    assert control_cfg.projection_activation_distance > exp037.low_level_control.projection_activation_distance
    assert control_cfg.projection_stop_distance >= cfg.success_thresholds.min_pairwise_distance


def test_exp038_compact_safe_team_activates_hold_and_damping() -> None:
    cfg = cfg_from_experiment(CONFIG)
    cfg.simulation.device = "cpu"
    cfg.simulation.num_envs = 1
    cfg.terrain.type = "flat_proxy"
    cfg.planner.subgoal_filter.progress_timestep_override = 8192
    cfg.planner.subgoal_filter.deterministic_eval = True
    env = MultiRoverGatheringCore(cfg)
    env.positions[:] = torch.tensor(
        [[[-0.30, -0.30, 0.0], [0.30, -0.30, 0.0], [-0.30, 0.30, 0.0], [0.30, 0.30, 0.0]]]
    )

    out = env.step(torch.zeros(cfg.simulation.num_envs, cfg.task.n_agents, 2))
    action_filter = out.info["action_filter"]
    control_safety = out.info["control_safety"]

    assert action_filter["hold_zone_activation"].float().mean() > 0.0
    assert torch.isfinite(action_filter["hold_zone_rho_cost"]).all()
    assert torch.isfinite(action_filter["hold_zone_spacing_violation"]).all()
    assert control_safety["success_zone_active"].float().mean() == pytest.approx(1.0)
    assert control_safety["linear_scale"].max() <= cfg.low_level_control.success_zone_linear_scale + 1.0e-6


def test_exp038_noncompact_team_keeps_hold_zone_inactive() -> None:
    cfg = cfg_from_experiment(CONFIG)
    cfg.simulation.device = "cpu"
    cfg.simulation.num_envs = 1
    cfg.terrain.type = "flat_proxy"
    cfg.planner.subgoal_filter.progress_timestep_override = 8192
    cfg.planner.subgoal_filter.deterministic_eval = True
    env = MultiRoverGatheringCore(cfg)
    env.positions[:] = torch.tensor(
        [[[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [3.0, 3.0, 0.0]]]
    )

    out = env.step(torch.zeros(cfg.simulation.num_envs, cfg.task.n_agents, 2))

    assert out.info["action_filter"]["hold_zone_activation"].float().mean() == pytest.approx(0.0)
    assert out.info["control_safety"]["success_zone_active"].float().mean() == pytest.approx(0.0)
