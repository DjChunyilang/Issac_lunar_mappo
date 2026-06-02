from __future__ import annotations

import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from _common import cfg_from_experiment  # noqa: E402
from evaluate_proxy_policy import evaluate_checkpoint  # noqa: E402
from lunar_rover_tasks.tasks.multi_rover_gathering.gathering_env import MultiRoverGatheringCore  # noqa: E402
from lunar_rover_tasks.tasks.multi_rover_gathering.gathering_env_cfg import make_debug_cfg  # noqa: E402
from run_proxy_convergence_suite import build_strict_acceptance  # noqa: E402
from train import Actor, Critic  # noqa: E402
from train_proxy_convergence import run_behavior_cloning, scripted_gather_action, strict_acceptance  # noqa: E402


def test_cfg_from_experiment_parses_reward_control_and_success_thresholds(tmp_path: Path) -> None:
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "experiment": {"num_envs": 3, "device": "cpu"},
                "low_level_control": {"max_linear_speed": 0.7},
                "reward": {
                    "weights": {"energy": 0.123},
                    "coefficients": {"dmax_level": 0.25, "success_bonus": 12.0},
                },
                "safety": {"near_distance": 0.95},
                "success_thresholds": {"dmax": 0.9, "hold_steps": 4},
            }
        ),
        encoding="utf-8",
    )
    cfg = cfg_from_experiment(config_path)
    assert cfg.simulation.num_envs == 3
    assert cfg.low_level_control.max_linear_speed == 0.7
    assert cfg.reward_weights.energy == 0.123
    assert cfg.reward_coefficients.dmax_level == 0.25
    assert cfg.reward_coefficients.success_bonus == 12.0
    assert cfg.safety.near_distance == 0.95
    assert cfg.success_thresholds.dmax == 0.9
    assert cfg.success_thresholds.hold_steps == 4


def test_behavior_cloning_reduces_scripted_action_mse() -> None:
    cfg = make_debug_cfg(num_envs=4, device="cpu")
    env = MultiRoverGatheringCore(cfg)
    actor = Actor(cfg.actor_obs_dim)
    actor_obs, _ = env.get_observations()
    target = scripted_gather_action(env).reshape(-1, 2)
    obs = actor_obs.reshape(-1, actor_obs.shape[-1])
    with torch.no_grad():
        before = torch.nn.functional.mse_loss(actor(obs).mean, target)
    run_behavior_cloning(actor, cfg, steps=20, batch_size=64, learning_rate=5.0e-3)
    with torch.no_grad():
        after = torch.nn.functional.mse_loss(actor(obs).mean, target)
    assert after < before


def test_safety_aware_teacher_reduces_rho_near_centroid() -> None:
    cfg = make_debug_cfg(num_envs=1, device="cpu")
    env = MultiRoverGatheringCore(cfg)
    env.positions[0, :, :2] = torch.tensor(
        [[0.20, 0.0], [0.0, 0.20], [-0.20, 0.0], [0.0, -0.20]],
        dtype=torch.float32,
    )
    env.yaws.zero_()
    direct = scripted_gather_action(env, safety_aware=False)
    safe = scripted_gather_action(env, stop_radius=0.45, slow_distance=0.4, safety_aware=True)
    direct_rho = 0.5 * (direct[..., 0] + 1.0) * cfg.planner.rho_max
    safe_rho = 0.5 * (safe[..., 0] + 1.0) * cfg.planner.rho_max
    assert safe_rho.max() < direct_rho.max()


def test_evaluate_proxy_policy_outputs_finite_ratio(tmp_path: Path) -> None:
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(
        yaml.safe_dump({"experiment": {"num_envs": 2, "device": "cpu"}, "simulation": {"episode_length_s": 2.0}}),
        encoding="utf-8",
    )
    cfg = cfg_from_experiment(config_path)
    actor = Actor(cfg.actor_obs_dim)
    critic = Critic(cfg.critic_state_dim)
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save({"actor": actor.state_dict(), "critic": critic.state_dict(), "cfg": {}}, checkpoint_path)
    result = evaluate_checkpoint(
        config_path,
        checkpoint_path,
        device="cpu",
        num_envs=2,
        steps=3,
        output=tmp_path / "eval.json",
    )
    assert result["status"] == "ok"
    assert result["initial_dmax"] > 0.0
    assert torch.isfinite(torch.tensor(result["dmax_reduction_ratio"]))
    assert "min_nearest_distance" in result
    assert "near_violation_rate" in result
    assert "collision_episode_ids" in result
    assert Path(result["artifact"]).exists()


def test_strict_acceptance_and_suite_summary() -> None:
    passing = {
        "dmax_reduction_ratio": 0.1,
        "success_rate": 0.95,
        "collision_rate": 0.0,
        "timeout_rate": 0.0,
        "phase": "ppo",
    }
    failing = {
        "dmax_reduction_ratio": 0.3,
        "success_rate": 0.95,
        "collision_rate": 0.0,
        "timeout_rate": 0.0,
        "phase": "bc",
    }
    assert strict_acceptance(passing)["passed"]
    assert not strict_acceptance(failing)["passed"]
    assert strict_acceptance(passing, required_phase="ppo")["passed"]
    phase_fail = {**passing, "phase": "bc"}
    assert not strict_acceptance(phase_fail, required_phase="ppo")["passed"]
    suite = build_strict_acceptance(
        [
            {
                "mode": "bc_ppo",
                "seed": 23,
                "run_name": "bc_ppo_seed_23",
                "checkpoint_path": "a.pt",
                "best_metrics": passing,
                "strict_acceptance": strict_acceptance(passing),
            },
            {
                "mode": "bc_ppo",
                "seed": 31,
                "run_name": "bc_ppo_seed_31",
                "checkpoint_path": "b.pt",
                "best_metrics": passing,
                "strict_acceptance": strict_acceptance(passing),
            },
        ]
    )
    assert suite["passed"]
    assert len(suite["seeds"]) == 2
