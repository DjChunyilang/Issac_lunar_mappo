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
from analyze_collision_credit_causal_validity import (  # noqa: E402
    causal_validity_decision,
    local_counterfactual_outcomes,
    summarize_causal_timeline,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.gathering_env import (  # noqa: E402
    MultiRoverGatheringCore,
)


def _synthetic_timeline() -> dict[str, torch.Tensor]:
    time_steps, num_envs, n_agents = 20, 1, 4
    pair_shape = (time_steps, num_envs, n_agents, n_agents)
    modified_pair_shape = (time_steps, num_envs, n_agents, n_agents, n_agents)
    baseline = torch.full(pair_shape, 0.4)
    best = torch.zeros(modified_pair_shape)
    feasible = torch.zeros_like(best)
    # Eight steps before collision: agent 0 has a strong alternative, agent 1
    # has only a weak alternative. Nonparticipants 2/3 remain exactly zero.
    best[11, 0, 0, 0, 1] = 0.04
    best[11, 0, 1, 0, 1] = 0.015
    feasible[11, 0, 0, 0, 1] = 0.03
    feasible[11, 0, 1, 0, 1] = 0.01
    # Sixteen steps before collision both participants have useful alternatives.
    best[3, 0, 0, 0, 1] = 0.035
    best[3, 0, 1, 0, 1] = 0.035
    feasible[3, 0, 0, 0, 1] = 0.03
    feasible[3, 0, 1, 0, 1] = 0.03
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
    zeros = torch.zeros_like(best)
    return {
        "baseline_pair_distance": baseline,
        "best_pair_gain": best,
        "best_pair_gain_risk_delta": zeros.clone(),
        "best_pair_gain_endpoint_dmax_delta": zeros.clone(),
        "best_feasible_pair_gain": feasible,
        "best_feasible_risk_delta": zeros.clone(),
        "best_feasible_endpoint_dmax_delta": zeros.clone(),
        "positions_after": positions,
        "done": done,
        "collision_done": collision,
        "sampled_actions": torch.zeros((time_steps, num_envs, n_agents, 2)),
    }


def test_causal_summary_reports_actionability_and_asymmetry() -> None:
    summary = summarize_causal_timeline(
        _synthetic_timeline(),
        collision_distance=0.28,
        gamma=0.99,
        trace_lambda=0.95,
    )
    assert summary["collision_episodes"] == 1
    assert summary["collision_pairs"] == 1
    h8 = summary["horizons"]["8"]
    assert h8["any_actionable"]["mean"] == pytest.approx(1.0)
    assert h8["both_actionable"]["mean"] == pytest.approx(0.0)
    assert h8["participant_actionable"]["mean"] == pytest.approx(0.5)
    assert h8["responsibility_asymmetry"]["median"] == pytest.approx(0.5)
    assert h8["equal_credit_supported"]["mean"] == pytest.approx(0.0)
    assert h8["trace_weight"] == pytest.approx((0.99 * 0.95) ** 8)
    h16 = summary["horizons"]["16"]
    assert h16["both_actionable"]["mean"] == pytest.approx(1.0)
    assert h16["responsibility_asymmetry"]["median"] == pytest.approx(0.0)
    assert summary["nonparticipant_pair_gain_abs_max"] == pytest.approx(0.0)


def test_local_counterfactual_does_not_mutate_environment_and_nonparticipant_is_zero() -> None:
    cfg = cfg_from_experiment(
        ROOT / "configs/experiment/exp150_collision_participant_actor_credit.yaml"
    )
    cfg.simulation.device = "cpu"
    cfg.simulation.num_envs = 2
    env = MultiRoverGatheringCore(cfg)
    actions = torch.zeros((2, env.n_agents, 2))
    positions_before = env.positions.clone()
    yaws_before = env.yaws.clone()
    result = local_counterfactual_outcomes(
        env,
        actions,
        action_delta=0.15,
        terrain_risk_tolerance=0.01,
        endpoint_dmax_tolerance=0.02,
    )
    assert torch.equal(env.positions, positions_before)
    assert torch.equal(env.yaws, yaws_before)
    # Changing rover 2 cannot alter the trajectory distance of pair (0, 1).
    assert result["best_pair_gain"][:, 2, 0, 1].abs().amax().item() == 0.0


def test_decision_only_authorizes_another_frozen_audit() -> None:
    combinations = {}
    for index in range(4):
        combinations[str(index)] = {
            "passed": True,
            "summary": {
                "horizons": {
                    "8": {
                        "any_actionable": {"mean": 0.8},
                        "equal_credit_supported": {"mean": 0.3},
                        "responsibility_asymmetry": {"median": 0.6},
                    },
                    "16": {"any_actionable": {"mean": 0.7}},
                }
            },
        }
    decision = causal_validity_decision(combinations)
    assert decision["status"] == "equal_participant_credit_causally_invalid"
    assert decision["next_stage"] == (
        "frozen_counterfactual_difference_advantage_audit_only"
    )
    assert decision["training_authorized"] is False


def test_missing_timeline_field_is_rejected() -> None:
    timeline = _synthetic_timeline()
    del timeline["best_feasible_pair_gain"]
    with pytest.raises(ValueError, match="best_feasible_pair_gain"):
        summarize_causal_timeline(
            timeline,
            collision_distance=0.28,
            gamma=0.99,
            trace_lambda=0.95,
        )
