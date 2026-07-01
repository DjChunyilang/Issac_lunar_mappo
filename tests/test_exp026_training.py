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


CONFIG = ROOT / "configs/experiment/exp026_randomized_terrain_hold_stable_filter.yaml"


def test_exp026_config_enables_hold_stable_filter_contract() -> None:
    raw = load_yaml(CONFIG)
    cfg = cfg_from_experiment(CONFIG)
    filter_cfg = cfg.planner.subgoal_filter

    assert raw["experiment"]["name"] == "exp026_randomized_terrain_hold_stable_filter"
    assert raw["algorithm"]["mode"] == "pure_rl"
    assert raw["algorithm"]["update_mode"] == "shared_joint"
    assert raw["algorithm"]["bc_updates"] == 0
    assert filter_cfg.enabled
    assert filter_cfg.mode == "terrain_safe_candidate_hold_progress_curriculum"
    assert filter_cfg.path_samples == 9
    assert len(filter_cfg.rho_scales) * len(filter_cfg.beta_offsets_deg) == 28
    assert filter_cfg.rho_scales == pytest.approx([0.45, 0.65, 0.85, 1.0])
    assert filter_cfg.hold_zone_dmax_multiplier == pytest.approx(1.20)
    assert filter_cfg.hold_zone_dispersion_multiplier == pytest.approx(1.60)
    assert filter_cfg.hold_zone_rho_weight == pytest.approx(0.65)
    assert filter_cfg.hold_zone_spacing_weight == pytest.approx(2.20)
    assert filter_cfg.hold_zone_pairwise_distance == pytest.approx(0.48)
    assert filter_cfg.endpoint_safe_distance > cfg.success_thresholds.min_pairwise_distance
    assert filter_cfg.path_safe_distance > cfg.safety.collision_distance
    assert not filter_cfg.hard_endpoint_near_filter
    assert not filter_cfg.hard_path_collision_filter
    assert not filter_cfg.hard_center_progress_filter
    assert filter_cfg.collision_override_after_warmup


def test_exp026_step_reports_hold_zone_metrics() -> None:
    cfg = cfg_from_experiment(CONFIG)
    cfg.simulation.device = "cpu"
    cfg.simulation.num_envs = 4
    cfg.planner.subgoal_filter.progress_timestep_override = 6144
    cfg.planner.subgoal_filter.deterministic_eval = True
    env = MultiRoverGatheringCore(cfg)

    action = torch.zeros(cfg.simulation.num_envs, cfg.task.n_agents, 2)
    out = env.step(action)
    action_filter = out.info["action_filter"]

    assert action_filter["enabled"]
    assert action_filter["candidate_count"] == 28
    for key in [
        "hold_zone_activation",
        "hold_zone_rho_cost",
        "hold_zone_spacing_violation",
        "raw_hold_zone_rho_cost",
        "raw_hold_zone_spacing_violation",
    ]:
        assert action_filter[key].shape == (cfg.simulation.num_envs, cfg.task.n_agents)
        assert torch.isfinite(action_filter[key]).all(), key


def test_exp026_hold_zone_activation_for_compact_team() -> None:
    cfg = cfg_from_experiment(CONFIG)
    cfg.simulation.device = "cpu"
    cfg.simulation.num_envs = 1
    cfg.terrain.type = "flat_proxy"
    cfg.planner.subgoal_filter.progress_timestep_override = 6144
    cfg.planner.subgoal_filter.deterministic_eval = True
    env = MultiRoverGatheringCore(cfg)
    env.positions[:] = torch.tensor(
        [[[0.0, 0.0, 0.0], [0.7, 0.0, 0.0], [0.0, 0.7, 0.0], [0.7, 0.7, 0.0]]]
    )

    action = torch.zeros(cfg.simulation.num_envs, cfg.task.n_agents, 2)
    out = env.step(action)
    action_filter = out.info["action_filter"]

    assert action_filter["hold_zone_activation"].float().mean() > 0.0
    assert (
        action_filter["hold_zone_rho_cost"]
        <= action_filter["raw_hold_zone_rho_cost"] + 1.0e-6
    ).all()


def test_exp025_config_keeps_hold_zone_disabled_for_compatibility() -> None:
    cfg = cfg_from_experiment(
        ROOT / "configs/experiment/exp025_randomized_terrain_dense_mutual_filter.yaml"
    )
    filter_cfg = cfg.planner.subgoal_filter

    assert filter_cfg.mode == "terrain_safe_candidate_mutual_progress_curriculum"
    assert filter_cfg.hold_zone_dmax_multiplier == pytest.approx(0.0)
    assert filter_cfg.hold_zone_rho_weight == pytest.approx(0.0)
    assert filter_cfg.hold_zone_spacing_weight == pytest.approx(0.0)
