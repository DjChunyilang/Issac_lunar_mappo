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
from analyze_action_planning_controllability_decomposition import (  # noqa: E402
    _combination_checks,
    controllability_decision,
    layered_counterfactual_outcomes,
    summarize_layered_timeline,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.gathering_env import (  # noqa: E402
    MultiRoverGatheringCore,
)


def _synthetic_timeline() -> dict[str, torch.Tensor]:
    time_steps, num_envs, n_agents = 20, 1, 4
    shape = (time_steps, num_envs, n_agents, n_agents, n_agents)
    layers = {
        name: torch.zeros(shape)
        for name in ("unconstrained", "risk", "dmax", "joint")
    }
    # Eight steps before collision, only rover 0 can improve the pair. The
    # candidate passes terrain but violates the dmax condition.
    layers["unconstrained"][11, 0, 0, 0, 1] = 0.04
    layers["risk"][11, 0, 0, 0, 1] = 0.035
    layers["dmax"][11, 0, 0, 0, 1] = 0.01
    layers["joint"][11, 0, 0, 0, 1] = 0.005
    # Sixteen steps before collision all layers are actionable.
    for value in layers.values():
        value[3, 0, 0, 0, 1] = 0.03
    radius = torch.zeros(shape)
    bearing = torch.zeros(shape)
    radius[11, 0, 0, 0, 1] = 0.02
    bearing[11, 0, 0, 0, 1] = 0.04
    control_response = torch.zeros(shape)
    control_response[11, 0, 0, 0, 1] = 0.1
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
    zeros = torch.zeros(shape)
    return {
        **{f"best_{name}_pair_gain": value for name, value in layers.items()},
        "best_radius_pair_gain": radius,
        "best_bearing_pair_gain": bearing,
        "best_unconstrained_control_response": control_response,
        "best_unconstrained_control_linear_delta": zeros.clone(),
        "best_unconstrained_control_angular_delta": zeros.clone(),
        "baseline_linear_saturated": zeros.clone(),
        "baseline_angular_saturated": zeros.clone(),
        "best_candidate_linear_saturated": zeros.clone(),
        "best_candidate_angular_saturated": zeros.clone(),
        "positions_after": positions,
        "done": done,
        "collision_done": collision,
        "sampled_actions": torch.zeros((time_steps, num_envs, n_agents, 2)),
    }


def test_summary_attributes_dmax_block_and_control_transmission() -> None:
    summary = summarize_layered_timeline(
        _synthetic_timeline(),
        collision_distance=0.28,
    )
    h8 = summary["horizons"]["8"]
    assert h8["unconstrained_actionable"]["mean"] == pytest.approx(1.0)
    assert h8["risk_actionable"]["mean"] == pytest.approx(1.0)
    assert h8["dmax_actionable"]["mean"] == pytest.approx(0.0)
    assert h8["joint_actionable"]["mean"] == pytest.approx(0.0)
    assert h8["terrain_blocked"]["mean"] == pytest.approx(0.0)
    assert h8["dmax_blocked"]["mean"] == pytest.approx(1.0)
    assert h8["control_transmitted"]["mean"] == pytest.approx(1.0)
    assert h8["bearing_best"]["mean"] == pytest.approx(1.0)
    assert summary["nonparticipant_pair_gain_abs_max"] == 0.0


def test_layered_counterfactual_is_non_intervening_and_pair_local() -> None:
    cfg = cfg_from_experiment(
        ROOT / "configs/experiment/exp150_collision_participant_actor_credit.yaml"
    )
    cfg.simulation.device = "cpu"
    cfg.simulation.num_envs = 2
    env = MultiRoverGatheringCore(cfg)
    actions = torch.zeros((2, env.n_agents, 2))
    positions_before = env.positions.clone()
    yaws_before = env.yaws.clone()
    result = layered_counterfactual_outcomes(
        env,
        actions,
        action_delta=0.15,
        terrain_risk_tolerance=0.01,
        endpoint_dmax_tolerance=0.02,
    )
    assert torch.equal(env.positions, positions_before)
    assert torch.equal(env.yaws, yaws_before)
    assert result["best_unconstrained_pair_gain"][:, 2, 0, 1].abs().amax() == 0.0
    unconstrained = result["best_unconstrained_pair_gain"]
    assert torch.all(result["best_risk_pair_gain"] <= unconstrained + 1.0e-7)
    assert torch.all(result["best_dmax_pair_gain"] <= unconstrained + 1.0e-7)
    assert torch.all(
        result["best_joint_pair_gain"]
        <= torch.minimum(
            result["best_risk_pair_gain"], result["best_dmax_pair_gain"]
        )
        + 1.0e-7
    )
    assert torch.all(result["best_radius_pair_gain"] <= unconstrained + 1.0e-7)
    assert torch.all(result["best_bearing_pair_gain"] <= unconstrained + 1.0e-7)


def test_decision_identifies_gather_constraint_dominance() -> None:
    combinations = {}
    for index in range(4):
        horizons = {
            "8": {
                "unconstrained_actionable": {"mean": 0.8},
                "control_transmitted": {"mean": 0.9},
                "joint_actionable": {"mean": 0.5},
                "terrain_blocked": {"mean": 0.1},
                "dmax_blocked": {"mean": 0.4},
            },
            "16": {"unconstrained_actionable": {"mean": 0.7}},
        }
        combinations[str(index)] = {"passed": True, "summary": {"horizons": horizons}}
    decision = controllability_decision(combinations)
    assert decision["status"] == "objective_tradeoff_bottleneck"
    assert decision["attribution"] == "gather_constraint_dominant"
    assert decision["training_authorized"] is False


def test_exp151_reconstruction_is_required() -> None:
    summary = summarize_layered_timeline(
        _synthetic_timeline(),
        collision_distance=0.28,
    )
    expected = {
        "summary": {
            "horizons": {
                str(h): {
                    "any_actionable": {
                        "mean": summary["horizons"][str(h)]["joint_actionable"]["mean"]
                    }
                }
                for h in (1, 2, 4, 8, 16)
            }
        }
    }
    checks, error = _combination_checks(
        summary,
        expected_exp151=expected,
        actor_digest_unchanged=True,
        actor_probe_action_max_abs_change=0.0,
        executed_action_max_abs_error=0.0,
    )
    assert error == 0.0
    assert checks["exp151_joint_reconstruction_error_le_1e_6"] is True
    assert checks["collision_episodes_ge_100"] is False
