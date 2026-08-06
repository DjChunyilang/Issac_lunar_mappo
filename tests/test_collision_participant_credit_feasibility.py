from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from analyze_collision_participant_credit_feasibility import (  # noqa: E402
    _combination_checks,
    summarize_collision_timeline,
)


def _synthetic_timeline() -> dict[str, torch.Tensor]:
    time_steps, num_envs, n_agents = 6, 1, 4
    actions = torch.zeros((time_steps, num_envs, n_agents, 2))
    actions[..., 0] = 0.5
    physical = torch.zeros_like(actions)
    physical[..., 0] = 0.9
    near_before = torch.full((time_steps, num_envs, n_agents), 0.8)
    near_after = near_before - 0.02
    dmax_before = torch.linspace(2.0, 1.5, time_steps).view(time_steps, 1)
    dmax_after = dmax_before - 0.05
    repeated = torch.zeros((time_steps, num_envs, n_agents, n_agents), dtype=torch.bool)
    repeated[1:5, 0, 0, 1] = True
    repeated[1:5, 0, 1, 0] = True
    active = repeated.clone()
    positions = torch.zeros((time_steps, num_envs, n_agents, 3))
    positions[..., 0, 0] = 0.0
    positions[..., 1, 0] = 1.0
    positions[..., 2, 0] = 4.0
    positions[..., 3, 0] = 7.0
    positions[5, 0, 1, 0] = 0.1
    done = torch.zeros((time_steps, num_envs), dtype=torch.bool)
    collision = torch.zeros_like(done)
    done[5, 0] = True
    collision[5, 0] = True
    zeros = torch.zeros((time_steps, num_envs))
    gather = torch.full_like(zeros, 0.1)
    safety = torch.full_like(zeros, -0.2)
    terminal = zeros.clone()
    terminal[5, 0] = -55.0
    total = gather + safety + terminal
    return {
        "actions": actions,
        "physical_actions": physical,
        "near_before": near_before,
        "near_after": near_after,
        "dmax_before": dmax_before,
        "dmax_after": dmax_after,
        "repeated_pairs": repeated,
        "active_pairs": active,
        "positions_after": positions,
        "done": done,
        "collision_done": collision,
        "success_done": torch.zeros_like(done),
        "gather_reward": gather,
        "safety_reward": safety,
        "terrain_reward": zeros,
        "terminal_reward": terminal,
        "total_reward": total,
    }


def test_collision_summary_identifies_pair_without_symmetric_double_count() -> None:
    summary = summarize_collision_timeline(
        _synthetic_timeline(),
        collision_distance=0.28,
    )
    assert summary["collision_episodes"] == 1
    assert summary["collision_pair_count"]["median"] == pytest.approx(1.0)
    assert summary["participant_count"]["median"] == pytest.approx(2.0)
    assert summary["nonparticipant_fraction"]["mean"] == pytest.approx(0.5)
    assert summary["collision_pair_repeated_recall"]["4"]["mean"] == pytest.approx(1.0)
    assert summary["collision_pair_repeated_precision_h8"]["mean"] == pytest.approx(1.0)
    assert summary["first_repeated_lead_steps"]["median"] == pytest.approx(4.0)
    assert summary["precollision_reward_sum"]["gather"]["mean"] == pytest.approx(0.5)
    assert summary["terminal_reward"]["terminal"]["mean"] == pytest.approx(-55.0)


def test_combination_gate_requires_sample_size_and_all_preregistered_metrics() -> None:
    summary = summarize_collision_timeline(
        _synthetic_timeline(),
        collision_distance=0.28,
    )
    checks = _combination_checks(summary, actor_unchanged=True)
    assert checks["collision_episodes_ge_100"] is False
    assert checks["nonparticipant_fraction_mean_ge_0_25"] is True
    assert checks["participant_count_median_le_2"] is True
    assert checks["repeated_recall_h8_ge_0_80"] is True
    assert checks["repeated_recall_h16_ge_0_90"] is True
    assert checks["first_repeated_lead_median_ge_4"] is True
    assert checks["actor_checkpoint_unchanged"] is True


def test_missing_timeline_field_is_rejected() -> None:
    timeline = _synthetic_timeline()
    del timeline["positions_after"]
    with pytest.raises(ValueError, match="positions_after"):
        summarize_collision_timeline(timeline, collision_distance=0.28)
