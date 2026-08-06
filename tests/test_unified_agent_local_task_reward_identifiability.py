from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

pytest.importorskip("skrl")

from analyze_unified_agent_local_task_reward_identifiability import (  # noqa: E402
    RAW_TARGET_NAMES,
    build_targets,
    fit_target_normalization,
    unified_local_task_gate,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.gathering_env import (  # noqa: E402
    MultiRoverGatheringCore,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.gathering_env_cfg import (  # noqa: E402
    MultiRoverGatheringEnvCfg,
)


def test_unified_target_uses_training_statistics_and_equal_weight_sum() -> None:
    training = torch.tensor(
        [[-1.0, 0.0, 1.0], [0.0, 2.0, 2.0], [1.0, 4.0, 3.0]]
    )
    mean, std = fit_target_normalization(training)
    targets = build_targets(training, mean, std)

    assert targets.shape == (3, 4)
    assert torch.equal(targets[:, :3], training)
    assert torch.allclose(targets[:, 3], ((training - mean) / std).sum(dim=-1))
    assert targets[:, 3].mean().item() == pytest.approx(0.0, abs=1.0e-6)


def test_step_info_position_snapshot_matches_pre_reset_metrics() -> None:
    cfg = MultiRoverGatheringEnvCfg()
    cfg.simulation.device = "cpu"
    cfg.simulation.num_envs = 2
    cfg.simulation.episode_length_s = cfg.simulation.planning_dt
    env = MultiRoverGatheringCore(cfg)

    output = env.step(torch.zeros(2, cfg.task.n_agents, 2))

    assert output.info["done"].done.all()
    assert torch.allclose(
        output.info["positions"].mean(dim=1),
        output.info["metrics"].centroid,
        atol=1.0e-6,
    )


def _aggregate(raw_gain: float = 0.18, unified_gain: float = 0.24) -> dict:
    result = {
        name: {
            "mean_mse_improvement_fraction": raw_gain,
            "minimum_seed_mse_improvement_fraction": raw_gain,
        }
        for name in RAW_TARGET_NAMES
    }
    result["unified_local_task"] = {
        "mean_mse_improvement_fraction": unified_gain,
        "minimum_seed_mse_improvement_fraction": unified_gain,
    }
    return result


def _validation() -> dict:
    distributions = {
        "local_gather_progress": {
            "std": 0.01,
            "active_rate": 0.9,
            "positive_rate": 0.45,
            "negative_rate": 0.45,
        },
        "local_terrain_result": {
            "std": 0.02,
            "active_rate": 0.8,
            "positive_rate": 0.35,
            "negative_rate": 0.30,
        },
        "local_safety_progress": {
            "std": 0.005,
            "active_rate": 0.12,
            "positive_rate": 0.06,
            "negative_rate": 0.06,
        },
    }
    return {
        "40023": {"target_distribution": distributions},
        "41023": {
            "target_distribution": {
                name: dict(values) for name, values in distributions.items()
            }
        },
    }


def test_unified_gate_passes_only_as_plan_authorization() -> None:
    gate = unified_local_task_gate(
        aggregate=_aggregate(),
        validation=_validation(),
        actor_parameters_unchanged=True,
        actor_output_change=0.0,
        actor_observation_dim=101,
        own_action_input_dim=103,
    )
    assert gate["passed"]


def test_unified_gate_stops_on_one_weak_component_or_coverage_failure() -> None:
    aggregate = _aggregate()
    aggregate["local_terrain_result"]["minimum_seed_mse_improvement_fraction"] = 0.14
    validation = _validation()
    validation["41023"]["target_distribution"]["local_safety_progress"][
        "active_rate"
    ] = 0.079
    gate = unified_local_task_gate(
        aggregate=aggregate,
        validation=validation,
        actor_parameters_unchanged=True,
        actor_output_change=0.0,
        actor_observation_dim=101,
        own_action_input_dim=103,
    )
    assert not gate["passed"]
    assert not gate["checks"]["local_terrain_result_every_seed_action_gain_ge_0_15"]
    assert not gate["checks"]["safety_every_seed_active_rate_ge_0_08"]
