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


CONFIG = ROOT / "configs/experiment/exp027_randomized_terrain_strict_hold_filter.yaml"


def test_exp027_config_uses_strict_hold_zone_contract() -> None:
    raw = load_yaml(CONFIG)
    cfg = cfg_from_experiment(CONFIG)
    filter_cfg = cfg.planner.subgoal_filter

    assert raw["experiment"]["name"] == "exp027_randomized_terrain_strict_hold_filter"
    assert raw["algorithm"]["mode"] == "pure_rl"
    assert raw["algorithm"]["update_mode"] == "shared_joint"
    assert raw["algorithm"]["bc_updates"] == 0
    assert filter_cfg.enabled
    assert filter_cfg.mode == "terrain_safe_candidate_hold_progress_curriculum"
    assert filter_cfg.path_samples == 9
    assert len(filter_cfg.rho_scales) * len(filter_cfg.beta_offsets_deg) == 28
    assert filter_cfg.rho_scales == pytest.approx([0.60, 0.80, 1.0, 1.08])
    assert filter_cfg.hold_zone_dmax_multiplier == pytest.approx(1.00)
    assert filter_cfg.hold_zone_dispersion_multiplier == pytest.approx(1.00)
    assert filter_cfg.hold_zone_rho_weight == pytest.approx(0.35)
    assert filter_cfg.hold_zone_spacing_weight == pytest.approx(1.40)
    assert filter_cfg.hold_zone_pairwise_distance == pytest.approx(0.44)
    assert filter_cfg.endpoint_safe_distance == pytest.approx(cfg.success_thresholds.min_pairwise_distance)
    assert filter_cfg.path_safe_distance > cfg.safety.collision_distance
    assert not filter_cfg.hard_endpoint_near_filter
    assert not filter_cfg.hard_path_collision_filter
    assert not filter_cfg.hard_center_progress_filter
    assert filter_cfg.collision_override_after_warmup


def test_exp027_hold_zone_does_not_activate_before_success_zone() -> None:
    cfg = cfg_from_experiment(CONFIG)
    cfg.simulation.device = "cpu"
    cfg.simulation.num_envs = 1
    cfg.terrain.type = "flat_proxy"
    cfg.planner.subgoal_filter.progress_timestep_override = 6144
    cfg.planner.subgoal_filter.deterministic_eval = True
    env = MultiRoverGatheringCore(cfg)
    # dmax is inside exp026's 1.2x relaxed hold zone but outside the true success dmax.
    env.positions[:] = torch.tensor(
        [[[0.0, 0.0, 0.0], [1.35, 0.0, 0.0], [0.0, 0.35, 0.0], [1.35, 0.35, 0.0]]]
    )

    out = env.step(torch.zeros(cfg.simulation.num_envs, cfg.task.n_agents, 2))
    action_filter = out.info["action_filter"]

    assert action_filter["hold_zone_activation"].float().mean() == pytest.approx(0.0)
    assert action_filter["hold_zone_rho_cost"].float().mean() == pytest.approx(0.0)


def test_exp027_hold_zone_activates_only_in_compact_success_zone() -> None:
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

    out = env.step(torch.zeros(cfg.simulation.num_envs, cfg.task.n_agents, 2))
    action_filter = out.info["action_filter"]

    assert action_filter["hold_zone_activation"].float().mean() > 0.0
    assert torch.isfinite(action_filter["hold_zone_rho_cost"]).all()
    assert torch.isfinite(action_filter["hold_zone_spacing_violation"]).all()


def test_exp027_step_reports_finite_filter_metrics() -> None:
    cfg = cfg_from_experiment(CONFIG)
    cfg.simulation.device = "cpu"
    cfg.simulation.num_envs = 4
    cfg.planner.subgoal_filter.progress_timestep_override = 6144
    cfg.planner.subgoal_filter.deterministic_eval = True
    env = MultiRoverGatheringCore(cfg)

    out = env.step(torch.zeros(cfg.simulation.num_envs, cfg.task.n_agents, 2))
    action_filter = out.info["action_filter"]

    assert action_filter["enabled"]
    assert action_filter["candidate_count"] == 28
    for key, value in action_filter.items():
        if isinstance(value, torch.Tensor) and value.is_floating_point():
            assert torch.isfinite(value).all(), key
