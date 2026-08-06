from __future__ import annotations

import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("skrl")

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from _common import cfg_from_experiment, load_yaml  # noqa: E402
from shared_policy_mappo import (  # noqa: E402
    SharedPolicyMAPPO,
    collision_termination_cost,
    episode_equivalent_collision_rate,
    lagrangian_multiplier_update,
)
from skrl.multi_agents.torch.mappo.mappo import compute_gae  # noqa: E402
from train_skrl_mappo import (  # noqa: E402
    SKRLValue,
    build_mappo_config,
    build_skrl_mappo_memories,
    build_skrl_mappo_models,
    skrl_mappo_checkpoint_payload,
)
from run_exp142_collision_lagrangian_screen import component_gate  # noqa: E402
from lunar_rover_tasks.tasks.multi_rover_gathering.gathering_env import (  # noqa: E402
    MultiRoverGatheringSKRLEnv,
)


CONFIG = ROOT / "configs/experiment/exp142_collision_lagrangian_component.yaml"


def _make_env(num_envs: int = 2) -> MultiRoverGatheringSKRLEnv:
    cfg = cfg_from_experiment(CONFIG)
    cfg.simulation.device = "cpu"
    cfg.simulation.num_envs = num_envs
    return MultiRoverGatheringSKRLEnv(cfg)


def test_exp142_is_strict_b0_plus_one_training_constraint() -> None:
    raw = load_yaml(CONFIG)
    cfg = cfg_from_experiment(CONFIG)
    algorithm = raw["algorithm"]
    assert raw["experiment"]["name"] == "exp142_collision_lagrangian_component"
    assert algorithm["collision_constraint_enabled"] is True
    assert algorithm["collision_cost_discount_factor"] == pytest.approx(0.99)
    assert algorithm["collision_cost_gae_lambda"] == pytest.approx(0.95)
    assert algorithm["collision_cost_limit"] == pytest.approx(0.02)
    assert algorithm["collision_episode_steps"] == 480
    assert algorithm["lagrangian_init"] == 0.0
    assert algorithm["lagrangian_learning_rate"] == pytest.approx(0.1)
    assert algorithm["lagrangian_max"] == pytest.approx(2.0)
    assert algorithm["actor_credit_assignment"] == "none"
    assert algorithm["actor_credit_scale"] == 0.0
    assert algorithm["bc_updates"] == 0
    assert algorithm["init_checkpoint"] is None
    assert cfg.actor_obs_dim == 101
    assert cfg.critic_state_dim == 54
    assert not cfg.low_level_control.safety_projection_enabled
    assert not cfg.planner.subgoal_filter.enabled


def test_episode_equivalent_rate_and_projected_dual_update() -> None:
    costs = torch.zeros((64, 10, 1))
    costs[0, 0] = 1.0
    rate = episode_equivalent_collision_rate(costs, episode_steps=480)
    assert rate == pytest.approx(0.75)
    assert lagrangian_multiplier_update(
        0.0,
        rate,
        collision_limit=0.02,
        learning_rate=0.1,
        maximum=2.0,
    ) == pytest.approx(0.073)
    assert lagrangian_multiplier_update(
        0.0,
        0.0,
        collision_limit=0.02,
        learning_rate=0.1,
        maximum=2.0,
    ) == 0.0
    assert lagrangian_multiplier_update(
        1.99,
        1.0,
        collision_limit=0.02,
        learning_rate=0.1,
        maximum=2.0,
    ) == 2.0


def test_cost_is_exact_collision_flag_and_cost_gae_stops_at_done() -> None:
    done = SimpleNamespace(collision=torch.tensor([False, True, False]))
    cost = collision_termination_cost(done, device=torch.device("cpu"))
    assert torch.equal(cost, torch.tensor([[0.0], [1.0], [0.0]]))

    rewards = torch.tensor([[[0.0]], [[1.0]], [[9.0]]])
    terminated = torch.tensor([[[False]], [[True]], [[False]]])
    truncated = torch.zeros_like(terminated)
    values = torch.zeros_like(rewards)
    returns, _ = compute_gae(
        rewards=rewards,
        terminated=terminated,
        truncated=truncated,
        values=values,
        last_values=torch.zeros((1, 1)),
        discount_factor=0.99,
        lambda_coefficient=0.95,
        time_limit_bootstrap=True,
    )
    # The post-reset reward 9 must not leak backward through the terminal at t=1.
    assert returns[:, 0, 0].tolist() == pytest.approx([0.99 * 0.95, 1.0, 9.0])

    truncated[1] = True
    terminated.zero_()
    returns_at_timeout, _ = compute_gae(
        rewards=rewards,
        terminated=terminated,
        truncated=truncated,
        values=values,
        last_values=torch.zeros((1, 1)),
        discount_factor=0.99,
        lambda_coefficient=0.95,
        time_limit_bootstrap=True,
    )
    assert returns_at_timeout[:, 0, 0].tolist() == pytest.approx(
        [0.99 * 0.95, 1.0, 9.0]
    )


def test_collision_constraint_uses_second_optimizer_and_cost_memory() -> None:
    env = _make_env()
    models = build_skrl_mappo_models(env, shared_actor=True, centralized_critic=True)
    memories = build_skrl_mappo_memories(env, rollout_steps=4)
    first = env.possible_agents[0]
    cost_value = SKRLValue(
        env.observation_spaces[first],
        env.state_space,
        env.action_spaces[first],
        env.device,
        architecture="mlp_v1",
    )
    empty_kwargs = {agent: {} for agent in env.possible_agents}
    agent = SharedPolicyMAPPO(
        possible_agents=env.possible_agents,
        models=models,
        memories=memories,
        observation_spaces=env.observation_spaces,
        state_spaces=env.state_spaces,
        action_spaces=env.action_spaces,
        device=env.device,
        cfg=build_mappo_config(
            {"ppo_epochs": 1, "mini_batches": 1},
            {"rollout_steps": 4},
            empty_kwargs,
        ),
        collision_constraint_enabled=True,
        collision_cost_value=cost_value,
    )
    assert agent.optimizer_count == 2
    assert agent.collision_cost_optimizer is not agent.shared_optimizer
    assert "collision_cost" in memories[first].tensors
    assert "collision_cost_values" in memories[first].tensors
    for other in env.possible_agents[1:]:
        assert "collision_cost" not in memories[other].tensors


def test_checkpoint_records_constraint_but_actor_output_is_invariant() -> None:
    env = _make_env()
    models = build_skrl_mappo_models(env, shared_actor=True, centralized_critic=True)
    first = env.possible_agents[0]
    cost_value = SKRLValue(
        env.observation_spaces[first],
        env.state_space,
        env.action_spaces[first],
        env.device,
        architecture="mlp_v1",
    )
    raw = load_yaml(CONFIG)
    observations, _ = env.reset()
    policy = models[first]["policy"]
    with torch.no_grad():
        before, _ = policy.compute(
            {"observations": observations[first]}, role="policy"
        )
    payload = skrl_mappo_checkpoint_payload(
        models,
        env.possible_agents,
        raw_cfg=raw,
        shared_actor=True,
        centralized_critic=True,
        shared_value=True,
        timesteps=64,
        actor_obs_dim=101,
        critic_state_dim=54,
        collision_cost_value=cost_value,
        lagrangian_multiplier=0.37,
    )
    mutated = copy.deepcopy(payload)
    mutated["collision_constraint"]["lagrangian_multiplier"] = 1.91
    for value in mutated["collision_constraint"]["cost_value"].values():
        value.add_(100.0)
    with torch.no_grad():
        after, _ = policy.compute(
            {"observations": observations[first]}, role="policy"
        )
    assert payload["metadata"]["collision_constraint_enabled"] is True
    assert payload["collision_constraint"]["lagrangian_multiplier"] == pytest.approx(0.37)
    assert torch.equal(before, after)


def test_collision_constraint_rejects_actor_credit_stacking() -> None:
    env = _make_env()
    models = build_skrl_mappo_models(env, shared_actor=True, centralized_critic=True)
    memories = build_skrl_mappo_memories(env, rollout_steps=4)
    first = env.possible_agents[0]
    cost_value = SKRLValue(
        env.observation_spaces[first],
        env.state_space,
        env.action_spaces[first],
        env.device,
        architecture="mlp_v1",
    )
    empty_kwargs = {agent: {} for agent in env.possible_agents}
    with pytest.raises(ValueError, match="cannot be combined"):
        SharedPolicyMAPPO(
            possible_agents=env.possible_agents,
            models=models,
            memories=memories,
            observation_spaces=env.observation_spaces,
            state_spaces=env.state_spaces,
            action_spaces=env.action_spaces,
            device=env.device,
            cfg=build_mappo_config(
                {"ppo_epochs": 1, "mini_batches": 1},
                {"rollout_steps": 4},
                empty_kwargs,
            ),
            actor_credit_scale=0.25,
            collision_constraint_enabled=True,
            collision_cost_value=cost_value,
        )


def _passing_component_inputs() -> tuple[dict, dict, dict]:
    history = []
    for index, rate in enumerate([0.20, 0.20, 0.18, 0.16, 0.12, 0.10, 0.08, 0.06]):
        history.append(
            {
                "update": float(index + 1),
                "episode_equivalent_collision_rate": rate,
                "lagrangian_multiplier_applied": 0.1,
                "lagrangian_multiplier": 0.2,
                "cost_value_loss": 0.01,
            }
        )
    summary = {
        "status": "ok",
        "training_diagnostics": {
            "policy_parameters_finite": True,
            "collision_cost_value_parameters_finite": True,
            "policy_parameter_delta_l2": 1.0,
            "neighbor_encoder_parameter_delta_l2": 1.0,
            "terrain_encoder_parameter_delta_l2": 1.0,
            "reward_critic_parameter_delta_l2": 1.0,
            "collision_cost_value_parameter_delta_l2": 1.0,
            "collision_cost_critic_update_count": 8,
            "post_training_action_std": 0.1,
            "lagrangian_multiplier": 0.2,
            "bc_updates": 0,
            "bc_parameter_delta_l2": 0.0,
            "actor_credit_assignment": "none",
            "collision_constraint_history": history,
        },
        "final_eval": {
            "collision_rate": 0.06,
            "success_rate": 0.04,
            "dmax_reduction_ratio": 0.24,
            "timeout_rate": 0.70,
        },
    }
    conflicts = {
        "per_seed": {
            "28023": {"failed_repeated_event_count": {"median": 13.0}},
            "29023": {"failed_repeated_event_count": {"median": 14.0}},
        }
    }
    baseline = {
        "per_seed": {
            "28023": {"failed_repeated_event_count": {"median": 17.0}},
            "29023": {"failed_repeated_event_count": {"median": 18.0}},
        }
    }
    return summary, conflicts, baseline


def test_exp142_component_gate_requires_all_registered_effects() -> None:
    result = component_gate(*_passing_component_inputs())
    assert result["passed"]
    assert result["checks"]["training_collision_rate_reduced_30pct"]
    assert result["checks"]["every_seed_repeated_conflicts_reduced_20pct"]


def test_exp142_component_gate_stops_when_lambda_is_pinned() -> None:
    summary, conflicts, baseline = _passing_component_inputs()
    history = summary["training_diagnostics"]["collision_constraint_history"]
    for row in history[-2:]:
        row["lagrangian_multiplier"] = 2.0
    summary["training_diagnostics"]["lagrangian_multiplier"] = 2.0
    result = component_gate(summary, conflicts, baseline)
    assert not result["passed"]
    assert not result["checks"]["final_lambda_in_open_interval"]
    assert not result["checks"]["lambda_not_long_pinned_at_upper_bound"]
