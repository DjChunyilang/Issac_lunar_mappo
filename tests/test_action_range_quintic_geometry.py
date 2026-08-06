from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from _common import cfg_from_experiment  # noqa: E402
from analyze_action_range_quintic_geometry import (  # noqa: E402
    _checks,
    action_range_geometry_outcomes,
    geometry_decision,
    summarize_geometry_timeline,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.gathering_env import (  # noqa: E402
    MultiRoverGatheringCore,
)


def _timeline() -> dict[str, torch.Tensor]:
    time_steps, num_envs, n_agents = 20, 1, 4
    shape = (time_steps, num_envs, n_agents, n_agents, n_agents)
    local = torch.zeros(shape)
    axis = torch.zeros(shape)
    grid = torch.zeros(shape)
    line = torch.zeros(shape)
    endpoint = torch.zeros(shape)
    grid[11, 0, 0, 0, 1] = 0.04
    line[11, 0, 0, 0, 1] = 0.05
    endpoint[11, 0, 0, 0, 1] = 0.06
    axis[11, 0, 0, 0, 1] = 0.01
    joint = torch.zeros(shape)
    joint[11, 0, 0, 0, 1] = 1.0
    positions = torch.zeros((time_steps, num_envs, n_agents, 3))
    positions[..., 0, 0] = 0.0
    positions[..., 1, 0] = 1.0
    positions[..., 2, 0] = 4.0
    positions[..., 3, 0] = 7.0
    positions[19, 0, 1, 0] = 0.1
    done = torch.zeros((time_steps, num_envs), dtype=torch.bool)
    collision = torch.zeros_like(done)
    done[19, 0] = True
    collision[19, 0] = True
    return {
        "best_local_quintic_gain": local,
        "best_axis_quintic_gain": axis,
        "best_grid_quintic_gain": grid,
        "best_grid_line_gain": line,
        "best_grid_endpoint_gain": endpoint,
        "best_grid_joint_dimension": joint,
        "positions_after": positions,
        "done": done,
        "collision_done": collision,
        "sampled_actions": torch.zeros((time_steps, num_envs, n_agents, 2)),
    }


def test_summary_detects_range_and_joint_dimension_recovery() -> None:
    summary = summarize_geometry_timeline(_timeline(), collision_distance=0.28)
    h8 = summary["horizons"]["8"]
    assert h8["local_quintic_actionable"]["mean"] == pytest.approx(0.0)
    assert h8["grid_quintic_actionable"]["mean"] == pytest.approx(1.0)
    assert h8["range_recovery"]["mean"] == pytest.approx(1.0)
    assert h8["joint_dimension_recovery"]["mean"] == pytest.approx(1.0)
    assert h8["quintic_geometry_loss"]["mean"] == pytest.approx(0.0)
    assert summary["nonparticipant_pair_gain_abs_max"] == 0.0


def test_existing_grid_contains_axis_candidates_and_is_non_intervening() -> None:
    cfg = cfg_from_experiment(
        ROOT / "configs/experiment/exp150_collision_participant_actor_credit.yaml"
    )
    cfg.simulation.device = "cpu"
    cfg.simulation.num_envs = 2
    env = MultiRoverGatheringCore(cfg)
    actions = torch.zeros((2, env.n_agents, 2))
    positions_before = env.positions.clone()
    yaws_before = env.yaws.clone()
    result = action_range_geometry_outcomes(env, actions, action_delta=0.15)
    assert torch.equal(env.positions, positions_before)
    assert torch.equal(env.yaws, yaws_before)
    assert torch.all(
        result["best_axis_quintic_gain"]
        <= result["best_grid_quintic_gain"] + 1.0e-7
    )
    assert result["best_grid_quintic_gain"][:, 2, 0, 1].abs().amax() == 0.0


def test_decision_identifies_local_coverage() -> None:
    combinations = {}
    for index in range(4):
        combinations[str(index)] = {
            "passed": True,
            "summary": {
                "horizons": {
                    "8": {
                        "local_quintic_actionable": {"mean": 0.65},
                        "grid_quintic_actionable": {"mean": 0.85},
                        "grid_line_actionable": {"mean": 0.9},
                        "grid_endpoint_actionable": {"mean": 0.95},
                        "range_recovery": {"mean": 0.2},
                        "quintic_geometry_loss": {"mean": 0.05},
                        "path_crossing_loss": {"mean": 0.05},
                    }
                }
            },
        }
    decision = geometry_decision(combinations)
    assert decision["status"] == "local_coverage_bottleneck"
    assert decision["training_authorized"] is False


def test_exp152_reconstruction_is_required() -> None:
    summary = summarize_geometry_timeline(_timeline(), collision_distance=0.28)
    expected = {
        "summary": {
            "horizons": {
                str(h): {
                    "unconstrained_actionable": {
                        "mean": summary["horizons"][str(h)][
                            "local_quintic_actionable"
                        ]["mean"]
                    }
                }
                for h in (1, 2, 4, 8, 16)
            }
        }
    }
    checks, error = _checks(
        summary,
        expected_exp152=expected,
        actor_digest_unchanged=True,
        actor_probe_action_max_abs_change=0.0,
        executed_action_max_abs_error=0.0,
    )
    assert error == 0.0
    assert checks["exp152_local_reconstruction_error_le_1e_6"] is True
    assert checks["collision_episodes_ge_100"] is False
