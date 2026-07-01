from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from _common import cfg_from_experiment, load_yaml  # noqa: E402
from run_exp017_pure_rl_long import (  # noqa: E402
    CHECKPOINT_INTERVAL,
    MILESTONES,
    ROLLOUT_STEPS,
    engineering_acceptance,
    milestone_records,
)
from shared_policy_mappo import scheduled_entropy_scale  # noqa: E402
from train_skrl_mappo import (  # noqa: E402
    checkpoint_rank,
    checkpoint_teacher_metadata,
    pure_rl_long_acceptance,
    pure_rl_long_checkpoint_rank,
)


CONFIG = ROOT / "configs/experiment/exp017_shared_mappo_pure_rl_comm12.yaml"


def _evaluation(timestep: int, *, timeout: float = 1.0) -> dict:
    return {
        "candidate_timestep": timestep,
        "checkpoint": f"/tmp/ppo_timestep_{timestep:06d}.pt",
        "dmax_reduction_ratio": 0.40,
        "success_rate": 0.10,
        "collision_rate": 0.02,
        "timeout_rate": timeout,
    }


def test_exp017_config_contract() -> None:
    raw = load_yaml(CONFIG)
    cfg = cfg_from_experiment(CONFIG)
    exp = raw["experiment"]
    algo = raw["algorithm"]

    assert exp["num_envs"] == 2048
    assert exp["rollout_steps"] == 32
    assert exp["checkpoint_interval"] == 1024
    assert cfg.simulation.max_episode_steps == 220
    assert cfg.observation.communication_radius == pytest.approx(12.0)
    assert algo["mode"] == "pure_rl"
    assert algo["update_mode"] == "shared_joint"
    assert algo["bc_updates"] == 0
    assert algo["teacher_mode"] is None
    assert algo["teacher_stop_radius"] is None
    assert algo["teacher_max_rho"] is None
    assert algo["entropy_schedule_timesteps"] == 4096


def test_exp017_entropy_schedule_reaches_floor_and_holds() -> None:
    assert scheduled_entropy_scale(0.002, 0.0005, 0, 10240, 4096) == pytest.approx(
        0.002
    )
    assert scheduled_entropy_scale(
        0.002, 0.0005, 4095, 10240, 4096
    ) == pytest.approx(0.0005)
    assert scheduled_entropy_scale(
        0.002, 0.0005, 8191, 10240, 4096
    ) == pytest.approx(0.0005)


def test_exp017_timeout_is_recorded_but_not_ranked() -> None:
    low_timeout = _evaluation(1024, timeout=0.0)
    high_timeout = _evaluation(1024, timeout=1.0)

    assert pure_rl_long_acceptance(low_timeout)["passed"]
    assert pure_rl_long_acceptance(high_timeout)["passed"]
    assert pure_rl_long_checkpoint_rank(low_timeout) == pure_rl_long_checkpoint_rank(
        high_timeout
    )


def test_exp017_rank_prefers_later_checkpoint_on_exact_tie() -> None:
    early = _evaluation(1024)
    late = _evaluation(4096)

    assert pure_rl_long_checkpoint_rank(late) < pure_rl_long_checkpoint_rank(early)


def test_strict_rank_does_not_let_zero_timeout_threshold_hide_collision() -> None:
    collision_safe = _evaluation(9216, timeout=0.0107)
    collision_safe["success_rate"] = 0.9756
    collision_safe["collision_rate"] = 0.0137
    collision_over = _evaluation(10240, timeout=0.0088)
    collision_over["success_rate"] = 0.9697
    collision_over["collision_rate"] = 0.0215

    assert checkpoint_rank(collision_safe) < checkpoint_rank(collision_over)


def test_exp017_milestones_map_to_expected_env_steps() -> None:
    records = milestone_records(
        [_evaluation(step) for step in MILESTONES],
        num_envs=2048,
        timesteps=10240,
    )

    assert [(item["label"], item["timestep"], item["env_steps"]) for item in records] == [
        ("2m", 1024, 2_097_152),
        ("8m", 4096, 8_388_608),
        ("20m", 10240, 20_971_520),
    ]


def test_exp017_pure_rl_checkpoint_teacher_metadata_is_null() -> None:
    metadata = checkpoint_teacher_metadata(
        bc_updates=0,
        teacher_mode="visible_local_centroid",
        teacher_stop_radius=0.54,
        teacher_max_rho=0.8,
    )

    assert metadata == {
        "teacher_mode": None,
        "teacher_stop_radius": None,
        "teacher_max_rho": None,
    }


def test_exp017_engineering_acceptance_contract(tmp_path: Path) -> None:
    train_metrics = tmp_path / "train_metrics.jsonl"
    train_metrics.write_text(json.dumps({"nan_flag": False}) + "\n", encoding="utf-8")
    summary = {
        "status": "ok",
        "update_mode": "shared_joint",
        "communication_radius": 12.0,
        "candidate_count": 10,
        "observation_schema_version": "ego_v3_local_terrain_grid",
        "actor_obs_dim": 86,
        "critic_state_dim": 54,
        "bc": {"updates": 0},
        "training_diagnostics": {
            "optimizer_count": 1,
            "joint_update_count": 320,
            "critic_update_count": 320,
            "policy_parameter_delta_l2": 0.1,
            "terrain_input_weight_delta_l2": 0.01,
            "post_training_action_std": 0.2,
        },
    }
    milestones = milestone_records(
        [_evaluation(step) for step in MILESTONES],
        num_envs=2048,
        timesteps=10240,
    )

    acceptance = engineering_acceptance(
        summary,
        milestones,
        timesteps=10240,
        train_metrics_path=train_metrics,
    )

    assert ROLLOUT_STEPS == 32
    assert CHECKPOINT_INTERVAL == 1024
    assert acceptance["passed"]
