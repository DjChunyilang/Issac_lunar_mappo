from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

pytest.importorskip("skrl")

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from lunar_rover_tasks.tasks.multi_rover_gathering.gathering_env import MultiRoverGatheringSKRLEnv  # noqa: E402
from lunar_rover_tasks.tasks.multi_rover_gathering.gathering_env_cfg import make_debug_cfg  # noqa: E402
from _common import cfg_from_experiment, load_yaml  # noqa: E402
from _skrl_metadata import (  # noqa: E402
    CheckpointCompatibilityError,
    DEFAULT_TRAINING_SEMANTICS,
    observation_interface_metadata,
    resolve_checkpoint_name,
    resolve_training_semantics,
    sanitize_checkpoint_name,
    validate_checkpoint_compatibility,
)
from train_skrl_mappo import (  # noqa: E402
    _collect_on_policy_tail_bc_samples,
    _collect_teacher_rollout_tail_bc_samples,
    _action_telemetry,
    _nearest_distances,
    _randomize_bc_state,
    _reward_breakdown,
    append_metrics_jsonl,
    build_mappo_config,
    build_skrl_mappo_memories,
    build_skrl_mappo_models,
    environment_geometry_metadata,
    critic_state_slices_metadata,
    initialize_skrl_mappo_models_from_checkpoint,
    observation_slices_metadata,
    run_skrl_behavior_cloning,
    scripted_gather_action,
    skrl_mappo_checkpoint_payload,
    terrain_sanity_metrics,
    terrain_input_weight_delta_l2,
    terrain_input_weight_snapshot,
)
from shared_policy_mappo import (  # noqa: E402
    SharedPolicyMAPPO,
    linear_schedule,
    primary_preserving_gradient_merge,
)
from diagnose_cuda_training_signal import diagnose  # noqa: E402


def _make_env() -> MultiRoverGatheringSKRLEnv:
    cfg = make_debug_cfg(num_envs=2, device="cpu")
    return MultiRoverGatheringSKRLEnv(cfg)


def test_skrl_shared_actor_parameters_are_shared() -> None:
    env = _make_env()
    models = build_skrl_mappo_models(env, shared_actor=True, centralized_critic=True)

    first_policy = models[env.possible_agents[0]]["policy"]
    first_param_ptrs = tuple(parameter.data_ptr() for parameter in first_policy.parameters())
    for agent_id in env.possible_agents:
        policy = models[agent_id]["policy"]
        assert policy is first_policy
        assert tuple(parameter.data_ptr() for parameter in policy.parameters()) == first_param_ptrs


def test_skrl_mappo_receives_shared_actor_instance() -> None:
    from skrl.multi_agents.torch.mappo import MAPPO

    env = _make_env()
    models = build_skrl_mappo_models(env, shared_actor=True, centralized_critic=True)
    memories = build_skrl_mappo_memories(env, rollout_steps=4)
    empty_kwargs = {agent_id: {} for agent_id in env.possible_agents}

    agent = MAPPO(
        possible_agents=env.possible_agents,
        models=models,
        memories=memories,
        observation_spaces=env.observation_spaces,
        state_spaces=env.state_spaces,
        action_spaces=env.action_spaces,
        device=env.device,
        cfg={
            "rollouts": 4,
            "learning_epochs": 1,
            "mini_batches": 1,
            "learning_rate_scheduler_kwargs": empty_kwargs,
            "observation_preprocessor_kwargs": empty_kwargs,
            "state_preprocessor_kwargs": empty_kwargs,
            "value_preprocessor_kwargs": empty_kwargs,
        },
    )

    first_policy = agent.policies[env.possible_agents[0]]
    for agent_id in env.possible_agents:
        assert agent.policies[agent_id] is first_policy


def test_shared_policy_mappo_uses_one_optimizer() -> None:
    env = _make_env()
    models = build_skrl_mappo_models(env, shared_actor=True, centralized_critic=True)
    memories = build_skrl_mappo_memories(env, rollout_steps=4)
    empty_kwargs = {agent_id: {} for agent_id in env.possible_agents}
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
    )

    assert agent.optimizer_count == 1
    assert len({id(optimizer) for optimizer in agent.optimizers.values()}) == 1


def test_primary_preserving_gradient_merge_projects_and_caps_auxiliary() -> None:
    parameter = torch.nn.Parameter(torch.zeros(2))
    merged, metrics = primary_preserving_gradient_merge(
        (torch.tensor([1.0, 0.0]),),
        (torch.tensor([-2.0, 4.0]),),
        (parameter,),
        auxiliary_scale=0.25,
    )
    assert metrics["conflict"] == 1.0
    assert 0.0 <= metrics["projected_primary_dot"] <= 1.0e-6
    assert metrics["auxiliary_norm_cap_scale"] == pytest.approx(0.25)
    assert merged[0].tolist() == pytest.approx([1.0, 0.25])
    assert metrics["combined_primary_cosine"] >= 0.970


def test_exp131_enables_only_primary_projected_credit_combination() -> None:
    raw = load_yaml(
        ROOT
        / "configs/experiment/exp131_decentralized_b0_primary_projected_terrain_credit.yaml"
    )
    assert raw["algorithm"]["actor_credit_assignment"] == "terrain_relative_centered"
    assert raw["algorithm"]["actor_credit_scale"] == pytest.approx(0.25)
    assert raw["algorithm"]["actor_credit_gradient_mode"] == "primary_projected_norm_cap"
    assert raw["algorithm"]["mixed_precision"] is False


def test_shared_policy_mappo_joint_update_merges_actor_samples_once() -> None:
    from skrl.envs.wrappers.torch import wrap_env
    from skrl.trainers.torch import SequentialTrainer

    env = _make_env()
    models = build_skrl_mappo_models(env, shared_actor=True, centralized_critic=True)
    memories = build_skrl_mappo_memories(env, rollout_steps=4)
    empty_kwargs = {agent_id: {} for agent_id in env.possible_agents}
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
    )
    trainer = SequentialTrainer(
        env=wrap_env(env, wrapper="isaaclab-multi-agent", verbose=False),
        agents=agent,
        cfg={
            "timesteps": 8,
            "headless": True,
            "disable_progressbar": True,
            "close_environment_at_exit": False,
        },
    )

    trainer.train()

    assert agent.joint_update_count == 2
    assert agent.critic_update_count == 2
    assert agent.last_actor_sample_count == 4 * 4 * env.num_envs
    assert agent.last_critic_sample_count == 4 * env.num_envs


def test_exp015_mappo_parameters_are_mapped_to_skrl() -> None:
    raw = load_yaml(ROOT / "configs/experiment/exp015_skrl_weak_warmup_medium_soft.yaml")
    config = build_mappo_config(
        raw["algorithm"],
        raw["experiment"],
        {"rover_0": {}},
    )

    assert config["rollouts"] == 128
    assert config["learning_epochs"] == 4
    assert config["mini_batches"] == 16
    assert config["discount_factor"] == pytest.approx(0.99)
    assert config["gae_lambda"] == pytest.approx(0.95)
    assert config["learning_rate"] == pytest.approx(1.2e-4)
    assert config["ratio_clip"] == pytest.approx(0.2)
    assert config["value_clip"] == pytest.approx(0.2)
    assert config["entropy_loss_scale"] == pytest.approx(0.006)
    assert config["value_loss_scale"] == pytest.approx(0.5)
    assert config["grad_norm_clip"] == pytest.approx(0.5)


def test_skrl_behavior_cloning_updates_shared_policy() -> None:
    env = _make_env()
    models = build_skrl_mappo_models(env, shared_actor=True, centralized_critic=True)
    policy = models[env.possible_agents[0]]["policy"]
    before = [parameter.detach().clone() for parameter in policy.parameters()]

    records = run_skrl_behavior_cloning(
        policy,
        env.cfg,
        updates=2,
        batch_size=32,
        learning_rate=1.0e-3,
    )

    assert len(records) == 2
    assert all(torch.isfinite(torch.tensor(record["bc_loss"])) for record in records)
    assert any(
        not torch.allclose(initial, current)
        for initial, current in zip(before, policy.parameters(), strict=True)
    )


def test_on_policy_tail_bc_can_anchor_nonterminal_policy_actions() -> None:
    cfg = make_debug_cfg(num_envs=2, device="cpu")
    cfg.task.explicit_goal_in_execution = True
    cfg.observation.schema_version = "ego_v6_gather_slot_goal"
    env = MultiRoverGatheringSKRLEnv(cfg)
    models = build_skrl_mappo_models(env, shared_actor=True, centralized_critic=True)
    policy = models[env.possible_agents[0]]["policy"]

    records = run_skrl_behavior_cloning(
        policy,
        env.cfg,
        updates=1,
        batch_size=32,
        learning_rate=1.0e-4,
        teacher_mode="oracle_slots",
        teacher_slow_distance=0.0,
        bc_on_policy_rollout_steps=2,
        bc_on_policy_tail_fraction=0.5,
        bc_on_policy_dmax_multiplier=20.0,
        bc_on_policy_dispersion_multiplier=100.0,
        bc_on_policy_min_teacher_disagreement=0.0,
        bc_on_policy_anchor_base_policy=True,
    )

    assert records[0]["bc_on_policy_tail_samples"] > 0
    assert records[0]["bc_on_policy_anchor_base_policy"] == 1


def test_exp015_terrain_sanity_matches_medium_soft_target() -> None:
    cfg = cfg_from_experiment(
        ROOT / "configs/experiment/exp015_skrl_weak_warmup_medium_soft.yaml"
    )
    metrics = terrain_sanity_metrics(cfg, "cpu")

    assert metrics["height_range"] == pytest.approx(0.397, abs=0.015)
    assert metrics["min_traversability"] == pytest.approx(0.341, abs=0.02)
    assert metrics["mean_speed_scale"] == pytest.approx(0.663, abs=0.02)


def test_visible_local_teacher_ignores_invisible_rover() -> None:
    env = _make_env().core
    env.cfg.observation.communication_radius = 2.5
    env.positions[..., :2] = torch.tensor(
        [
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [8.0, 8.0]],
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [8.0, 8.0]],
        ]
    )
    env.yaws.zero_()
    before = scripted_gather_action(env, visible_local=True)
    env.positions[:, 3, :2] = torch.tensor([[20.0, -20.0], [-20.0, 20.0]])
    after = scripted_gather_action(env, visible_local=True)

    assert torch.allclose(before[:, 0], after[:, 0])


def test_exp016_bc_states_are_safe_and_teacher_labels_not_saturated() -> None:
    cfg = cfg_from_experiment(ROOT / "configs/experiment/exp016_shared_mappo_comm12.yaml")
    cfg.simulation.device = "cpu"
    cfg.simulation.num_envs = 512
    env = MultiRoverGatheringSKRLEnv(cfg).core
    _randomize_bc_state(
        env,
        visible_local=True,
        yaw_noise_degrees=15.0,
        min_nearest_distance=0.32,
    )
    action = scripted_gather_action(
        env,
        stop_radius=0.54,
        slow_distance=0.47,
        max_rho=0.8,
        visible_local=True,
        terrain_scale=True,
    )

    assert torch.all(_nearest_distances(env.positions) >= 0.32)
    assert float((action[..., 0] >= 0.95).float().mean()) == 0.0
    assert float((action[..., 1].abs() >= 0.95).float().mean()) < 0.05


def test_oracle_ring_teacher_targets_spaced_slots_at_the_execution_goal() -> None:
    cfg = make_debug_cfg(num_envs=1, device="cpu")
    cfg.task.explicit_goal_in_execution = True
    cfg.observation.schema_version = "ego_v5_gather_site_goal"
    env = MultiRoverGatheringSKRLEnv(cfg).core
    env.oracle_point.zero_()
    env.positions[0, :, :2] = torch.tensor(
        [[2.0, 0.0], [0.0, 2.0], [-2.0, 0.0], [0.0, -2.0]]
    )
    env.yaws.copy_(torch.tensor([[torch.pi, -torch.pi / 2.0, 0.0, torch.pi / 2.0]]))

    action = scripted_gather_action(
        env,
        stop_radius=0.45,
        slow_distance=0.0,
        teacher_mode="oracle_ring",
    )
    rho = 0.5 * (action[..., 0] + 1.0) * cfg.planner.rho_max

    assert torch.allclose(action[..., 1], torch.zeros_like(action[..., 1]), atol=1.0e-5)
    assert torch.allclose(rho, torch.full_like(rho, cfg.planner.rho_max))

    env.oracle_point[:, 0] = 1.0
    changed = scripted_gather_action(
        env,
        stop_radius=0.45,
        slow_distance=0.0,
        teacher_mode="oracle_ring",
    )
    assert not torch.allclose(changed, action)


def test_oracle_translating_ring_teacher_moves_formation_center_toward_goal() -> None:
    cfg = make_debug_cfg(num_envs=1, device="cpu")
    cfg.task.explicit_goal_in_execution = True
    cfg.observation.schema_version = "ego_v5_gather_site_goal"
    env = MultiRoverGatheringSKRLEnv(cfg).core
    env.positions[0, :, :2] = torch.tensor(
        [[-2.0, -0.4], [-2.0, 0.4], [-1.2, -0.4], [-1.2, 0.4]]
    )
    env.yaws.zero_()
    env.oracle_point.copy_(torch.tensor([[2.0, 0.0, 0.0]]))

    action = scripted_gather_action(
        env,
        stop_radius=0.45,
        slow_distance=0.0,
        max_rho=1.0,
        teacher_mode="oracle_translating_ring",
        teacher_center_step=0.65,
    )

    assert torch.all(action[..., 0] > -1.0)
    assert torch.all(action[..., 1].abs() < 0.95)


def test_oracle_slots_teacher_follows_fixed_symmetric_execution_targets() -> None:
    cfg = make_debug_cfg(num_envs=1, device="cpu")
    cfg.task.explicit_goal_in_execution = True
    cfg.observation.schema_version = "ego_v6_gather_slot_goal"
    env = MultiRoverGatheringSKRLEnv(cfg).core
    env.positions.zero_()
    env.yaws.copy_(torch.tensor([[0.0, torch.pi / 2.0, torch.pi, -torch.pi / 2.0]]))
    env.gather_slot_points.copy_(
        torch.tensor([[[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [-2.0, 0.0, 0.0], [0.0, -2.0, 0.0]]])
    )

    action = scripted_gather_action(
        env,
        slow_distance=0.0,
        teacher_mode="oracle_slots",
    )

    assert torch.allclose(action[..., 1], torch.zeros_like(action[..., 1]), atol=1.0e-5)
    assert torch.allclose(action[..., 0], torch.ones_like(action[..., 0]))


def test_terminal_flat_slots_teacher_follows_actor_visible_dynamic_targets() -> None:
    cfg = make_debug_cfg(num_envs=1, device="cpu")
    cfg.task.explicit_goal_in_execution = True
    cfg.task.dynamic_terminal_slot_goal_enabled = True
    cfg.observation.schema_version = "ego_v6_gather_slot_goal"
    cfg.planner.rho_max = 4.0
    env = MultiRoverGatheringSKRLEnv(cfg).core
    env.positions.zero_()
    env.yaws.copy_(torch.tensor([[0.0, torch.pi / 2.0, torch.pi, -torch.pi / 2.0]]))
    env.gather_slot_points.copy_(
        torch.tensor([[[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [-3.0, 0.0, 0.0], [0.0, -3.0, 0.0]]])
    )
    env.execution_slot_points.copy_(
        torch.tensor([[[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [-2.0, 0.0, 0.0], [0.0, -2.0, 0.0]]])
    )

    action = scripted_gather_action(
        env,
        slow_distance=0.0,
        teacher_mode="terminal_flat_slots",
    )

    assert torch.allclose(action[..., 1], torch.zeros_like(action[..., 1]), atol=1.0e-5)
    assert torch.allclose(action[..., 0], torch.zeros_like(action[..., 0]), atol=1.0e-5)


def test_on_policy_tail_bc_samples_match_actor_and_teacher_contract() -> None:
    cfg = make_debug_cfg(num_envs=4, device="cpu")
    cfg.task.explicit_goal_in_execution = True
    cfg.observation.schema_version = "ego_v6_gather_slot_goal"
    cfg.gather_point.execution_slot_radius = 0.35
    env = MultiRoverGatheringSKRLEnv(cfg).core
    models = build_skrl_mappo_models(
        MultiRoverGatheringSKRLEnv(cfg),
        shared_actor=True,
        centralized_critic=True,
    )
    policy = models["rover_0"]["policy"]

    observations, targets = _collect_on_policy_tail_bc_samples(
        policy,
        env,
        rollout_steps=2,
        teacher_stop_radius=0.45,
        teacher_slow_distance=0.0,
        teacher_max_rho=None,
        teacher_mode="oracle_slots",
        teacher_terrain_scale=False,
        teacher_center_step=0.65,
        dmax_multiplier=20.0,
        dispersion_multiplier=100.0,
        min_teacher_disagreement=0.0,
    )

    assert observations.ndim == 2
    assert observations.shape[1] == cfg.actor_obs_dim
    assert targets.shape == (cfg.simulation.num_envs * cfg.task.n_agents * 2, 2)
    assert torch.isfinite(observations).all()
    assert torch.isfinite(targets).all()


def test_teacher_rollout_tail_samples_match_actor_and_teacher_contract() -> None:
    cfg = make_debug_cfg(num_envs=4, device="cpu")
    cfg.task.explicit_goal_in_execution = True
    cfg.observation.schema_version = "ego_v6_gather_slot_goal"
    cfg.gather_point.execution_slot_radius = 0.35
    env = MultiRoverGatheringSKRLEnv(cfg).core

    observations, targets = _collect_teacher_rollout_tail_bc_samples(
        env,
        rollout_steps=2,
        teacher_stop_radius=0.45,
        teacher_slow_distance=0.0,
        teacher_max_rho=None,
        teacher_mode="oracle_slots",
        teacher_terrain_scale=False,
        teacher_center_step=0.65,
        dmax_multiplier=20.0,
        dispersion_multiplier=100.0,
    )

    assert observations.ndim == 2
    assert observations.shape[1] == cfg.actor_obs_dim
    assert targets.shape == (cfg.simulation.num_envs * cfg.task.n_agents * 2, 2)
    assert torch.isfinite(observations).all()
    assert torch.isfinite(targets).all()


def test_exp016_initial_log_std_and_checkpoint_metadata() -> None:
    raw = load_yaml(ROOT / "configs/experiment/exp016_shared_mappo_comm12.yaml")
    env = _make_env()
    models = build_skrl_mappo_models(
        env,
        initial_log_std=float(raw["algorithm"]["initial_log_std"]),
    )
    policy = models[env.possible_agents[0]]["policy"]
    payload = skrl_mappo_checkpoint_payload(
        models,
        env.possible_agents,
        raw_cfg=raw,
        shared_actor=True,
        centralized_critic=True,
        shared_value=True,
        timesteps=64,
        observation_schema_version=env.cfg.observation.schema_version,
        actor_obs_dim=env.cfg.actor_obs_dim,
        critic_state_dim=env.cfg.critic_state_dim,
        extra_metadata={
            "update_mode": "shared_joint",
            "communication_radius": 12.0,
            "teacher_mode": "visible_local_centroid",
        },
    )

    assert torch.allclose(policy.log_std_parameter, torch.full((2,), -1.0))
    assert payload["metadata"]["update_mode"] == "shared_joint"
    assert payload["metadata"]["communication_radius"] == pytest.approx(12.0)
    assert payload["metadata"]["teacher_mode"] == "visible_local_centroid"


def test_checkpoint_initialization_loads_weights_and_resets_training_state(
    tmp_path: Path,
) -> None:
    env = _make_env()
    raw = {"experiment": {"name": "warm_start"}, "algorithm": {}}
    source_models = build_skrl_mappo_models(
        env,
        shared_actor=True,
        centralized_critic=True,
        shared_value=True,
    )
    with torch.no_grad():
        next(source_models[env.possible_agents[0]]["policy"].parameters()).fill_(0.125)
        next(source_models[env.possible_agents[0]]["value"].parameters()).fill_(0.25)
    checkpoint_path = tmp_path / "source.pt"
    torch.save(
        skrl_mappo_checkpoint_payload(
            source_models,
            env.possible_agents,
            raw_cfg=raw,
            shared_actor=True,
            centralized_critic=True,
            shared_value=True,
            timesteps=123,
            observation_schema_version=env.cfg.observation.schema_version,
            actor_obs_dim=env.cfg.actor_obs_dim,
            critic_state_dim=env.cfg.critic_state_dim,
        ),
        checkpoint_path,
    )

    target_models = build_skrl_mappo_models(
        env,
        shared_actor=True,
        centralized_critic=True,
        shared_value=True,
    )
    metadata = initialize_skrl_mappo_models_from_checkpoint(
        checkpoint_path,
        target_models,
        env.possible_agents,
        env.cfg,
        actor_architecture="mlp_v1",
        critic_architecture="mlp_v1",
        device=torch.device("cpu"),
    )
    target_policy = target_models[env.possible_agents[0]]["policy"]
    target_value = target_models[env.possible_agents[0]]["value"]
    assert torch.allclose(next(target_policy.parameters()), torch.full_like(next(target_policy.parameters()), 0.125))
    assert torch.allclose(next(target_value.parameters()), torch.full_like(next(target_value.parameters()), 0.25))
    assert metadata["init_checkpoint_source_timestep"] == 123


def test_exp016_communication_radius_covers_standard_initial_team() -> None:
    cfg = cfg_from_experiment(ROOT / "configs/experiment/exp016_shared_mappo_comm12.yaml")
    cfg.simulation.device = "cpu"
    cfg.simulation.num_envs = 512
    env = MultiRoverGatheringSKRLEnv(cfg).core
    pairwise = torch.cdist(env.positions[..., :2], env.positions[..., :2])

    assert cfg.observation.communication_radius == pytest.approx(12.0)
    assert torch.all(pairwise <= cfg.observation.communication_radius)


def test_entropy_linear_schedule_hits_endpoints() -> None:
    assert linear_schedule(0.002, 0.0005, 0, 1024) == pytest.approx(0.002)
    assert linear_schedule(0.002, 0.0005, 1023, 1024) == pytest.approx(0.0005)


def test_skrl_value_uses_centralized_state() -> None:
    env = _make_env()
    models = build_skrl_mappo_models(env, shared_actor=True, centralized_critic=True)
    value = models[env.possible_agents[0]]["value"]
    actor_obs, critic_state = env.core.get_observations()

    assert value.num_observations == env.cfg.actor_obs_dim
    assert value.num_states == env.cfg.critic_state_dim
    assert value.net[0].in_features == env.cfg.critic_state_dim

    values, _ = value.compute(
        {
            "observations": actor_obs[:, 0, :],
            "states": critic_state,
        },
        role="value",
    )
    assert values.shape == (env.num_envs, 1)

    with pytest.raises(RuntimeError):
        value.compute(
            {
                "observations": actor_obs[:, 0, :],
                "states": actor_obs[:, 0, :],
            },
            role="value",
        )


def test_actor_does_not_receive_state_or_oracle() -> None:
    env = _make_env()
    models = build_skrl_mappo_models(env, shared_actor=True, centralized_critic=True)
    policy = models[env.possible_agents[0]]["policy"]
    actor_obs, critic_state = env.core.get_observations()

    assert policy.num_observations == env.cfg.actor_obs_dim
    assert policy.net[0].in_features == env.cfg.actor_obs_dim
    assert env.cfg.actor_obs_dim != env.cfg.critic_state_dim

    agent_obs = actor_obs[:, 0, :]
    mean_a, _ = policy.compute(
        {
            "observations": agent_obs,
            "states": critic_state,
        },
        role="policy",
    )
    mean_b, _ = policy.compute(
        {
            "observations": agent_obs,
            "states": critic_state + 123.0,
        },
        role="policy",
    )
    assert torch.allclose(mean_a, mean_b)

    env.core.oracle_point += 123.0
    changed_actor_obs, _ = env.core.get_observations()
    assert torch.allclose(changed_actor_obs[:, 0, :], agent_obs)
    mean_c, _ = policy.compute(
        {
            "observations": changed_actor_obs[:, 0, :],
            "states": critic_state,
        },
        role="policy",
    )
    assert torch.allclose(mean_a, mean_c)


def test_skrl_checkpoint_metadata_marks_smoke_semantics() -> None:
    env = _make_env()
    models = build_skrl_mappo_models(env, shared_actor=True, centralized_critic=True)
    payload = skrl_mappo_checkpoint_payload(
        models,
        env.possible_agents,
        raw_cfg={"experiment": {"name": "metadata_test"}},
        shared_actor=True,
        centralized_critic=True,
        shared_value=True,
        timesteps=4,
        observation_schema_version=env.cfg.observation.schema_version,
        actor_obs_dim=env.cfg.actor_obs_dim,
        critic_state_dim=env.cfg.critic_state_dim,
    )

    assert payload["metadata"]["training_semantics"] == DEFAULT_TRAINING_SEMANTICS
    assert payload["metadata"]["experiment_name"] == "metadata_test"
    assert payload["metadata"]["algorithm_mode"] is None
    assert payload["metadata"]["observation_schema_version"] == "ego_v3_local_terrain_grid"
    assert payload["metadata"]["actor_obs_dim"] == 86
    assert payload["metadata"]["critic_state_dim"] == 54
    assert payload["metadata"]["shared_actor"] is True
    assert payload["metadata"]["centralized_critic"] is True
    for agent_id in env.possible_agents:
        assert "policy" in payload[agent_id]
        assert "value" in payload[agent_id]


def test_skrl_metadata_resolves_semantics_without_training_imports() -> None:
    minimal = load_yaml(ROOT / "configs/experiment/exp_001_minimal.yaml")
    pure_rl = load_yaml(ROOT / "configs/experiment/exp_006_ppo_selected_pure_rl.yaml")
    explicit = {"algorithm": {"training_semantics": "Research Smoke"}}

    assert resolve_training_semantics(minimal) == "skrl_mappo_smoke"
    assert resolve_training_semantics(pure_rl) == "skrl_mappo_pure_rl"
    assert resolve_training_semantics(explicit) == "research_smoke"


def test_skrl_checkpoint_name_is_sanitized_and_uses_pt_suffix() -> None:
    assert sanitize_checkpoint_name("run 01") == "run_01.pt"
    assert sanitize_checkpoint_name("already.pt") == "already.pt"
    assert resolve_checkpoint_name(
        {"experiment": {"name": "exp 01"}},
        ROOT / "configs/experiment/exp_001_minimal.yaml",
    ) == "exp_01_skrl_mappo.pt"
    assert resolve_checkpoint_name(
        {"experiment": {"checkpoint_name": "custom-name"}},
        ROOT / "configs/experiment/exp_001_minimal.yaml",
    ) == "custom-name.pt"

    for bad_name in ("../escape.pt", "nested/file.pt", r"nested\\file.pt", "/tmp/file.pt"):
        with pytest.raises(ValueError):
            sanitize_checkpoint_name(bad_name)


def test_skrl_spaces_and_policy_input_track_actor_observation_schema() -> None:
    env = _make_env()
    models = build_skrl_mappo_models(env, shared_actor=True, centralized_critic=True)
    policy = models[env.possible_agents[0]]["policy"]

    assert env.cfg.observation.schema_version == "ego_v3_local_terrain_grid"
    assert env.cfg.actor_obs_dim == 86
    assert env.observation_spaces[env.possible_agents[0]].shape == (env.cfg.actor_obs_dim,)
    assert policy.net[0].in_features == env.cfg.actor_obs_dim


def test_branched_actor_and_structured_critic_use_fixed_slices() -> None:
    env = _make_env()
    models = build_skrl_mappo_models(
        env,
        shared_actor=True,
        centralized_critic=True,
        actor_architecture="branched_v1",
        critic_architecture="structured_v1",
        initial_log_std=-1.0,
    )
    policy = models[env.possible_agents[0]]["policy"]
    value = models[env.possible_agents[0]]["value"]
    actor_obs, critic_state = env.core.get_observations()

    assert policy.architecture == "branched_v1"
    assert value.architecture == "structured_v1"
    assert policy.ego_encoder[0].in_features == 10
    assert policy.neighbor_encoder[0].in_features == 21
    assert policy.terrain_encoder[0].in_features == 50
    assert policy.aggregation_encoder[0].in_features == 5
    assert policy.trunk[0].in_features == 160
    assert value.agent_encoder[0].in_features == 8
    assert value.team_encoder[0].in_features == 8
    assert value.terrain_encoder[0].in_features == 5
    assert value.oracle_encoder[0].in_features == 9
    assert observation_slices_metadata()["terrain"] == {"start": 31, "end": 81, "dim": 50}

    means, _ = policy.compute(
        {"observations": actor_obs.reshape(-1, env.cfg.actor_obs_dim)},
        role="policy",
    )
    values, _ = value.compute({"states": critic_state}, role="value")

    assert means.shape == (env.num_envs * env.cfg.task.n_agents, 2)
    assert values.shape == (env.num_envs, 1)
    assert torch.allclose(policy.log_std_parameter, torch.full((2,), -1.0))


def test_terminal_gate_actor_and_critic_use_v2_slices() -> None:
    cfg = make_debug_cfg(num_envs=2, device="cpu")
    cfg.observation.schema_version = "ego_v4_terminal_gate"
    env = MultiRoverGatheringSKRLEnv(cfg)
    models = build_skrl_mappo_models(
        env,
        shared_actor=True,
        centralized_critic=True,
        actor_architecture="branched_v2",
        critic_architecture="structured_v2",
        initial_log_std=-1.0,
    )
    policy = models[env.possible_agents[0]]["policy"]
    value = models[env.possible_agents[0]]["value"]
    actor_obs, critic_state = env.core.get_observations()

    assert policy.architecture == "branched_v2"
    assert value.architecture == "structured_v2"
    assert env.cfg.actor_obs_dim == 91
    assert env.cfg.critic_state_dim == 55
    assert policy.terminal_gate_encoder[0].in_features == 5
    assert policy.trunk[0].in_features == 176
    assert value.team_encoder[0].in_features == 9
    assert observation_slices_metadata(91)["terminal_gate"] == {
        "start": 86,
        "end": 91,
        "dim": 5,
    }
    assert critic_state_slices_metadata(55)["team"] == {"start": 32, "end": 41, "dim": 9}

    means, _ = policy.compute(
        {"observations": actor_obs.reshape(-1, env.cfg.actor_obs_dim)},
        role="policy",
    )
    values, _ = value.compute({"states": critic_state}, role="value")

    assert means.shape == (env.num_envs * env.cfg.task.n_agents, 2)
    assert values.shape == (env.num_envs, 1)


def test_gather_site_goal_actor_uses_v3_slices() -> None:
    cfg = make_debug_cfg(num_envs=2, device="cpu")
    cfg.task.explicit_goal_in_execution = True
    cfg.observation.schema_version = "ego_v5_gather_site_goal"
    env = MultiRoverGatheringSKRLEnv(cfg)
    models = build_skrl_mappo_models(
        env,
        shared_actor=True,
        centralized_critic=True,
        actor_architecture="branched_v3",
        critic_architecture="structured_v1",
        initial_log_std=-1.0,
    )
    policy = models[env.possible_agents[0]]["policy"]
    actor_obs, critic_state = env.core.get_observations()

    assert policy.architecture == "branched_v3"
    assert env.cfg.actor_obs_dim == 89
    assert env.cfg.critic_state_dim == 54
    assert policy.gather_site_goal_encoder[0].in_features == 3
    assert policy.trunk[0].in_features == 176
    assert observation_slices_metadata(89)["gather_site_goal"] == {
        "start": 86,
        "end": 89,
        "dim": 3,
    }

    means, _ = policy.compute(
        {"observations": actor_obs.reshape(-1, env.cfg.actor_obs_dim)},
        role="policy",
    )
    assert means.shape == (env.num_envs * env.cfg.task.n_agents, 2)
    assert critic_state.shape == (env.num_envs, env.cfg.critic_state_dim)


def test_site_and_slot_goal_actor_uses_v4_slices() -> None:
    cfg = make_debug_cfg(num_envs=2, device="cpu")
    cfg.task.explicit_goal_in_execution = True
    cfg.observation.schema_version = "ego_v7_gather_site_and_slot_goal"
    env = MultiRoverGatheringSKRLEnv(cfg)
    models = build_skrl_mappo_models(
        env,
        shared_actor=True,
        centralized_critic=True,
        actor_architecture="branched_v4",
        critic_architecture="structured_v1",
        initial_log_std=-1.0,
    )
    policy = models[env.possible_agents[0]]["policy"]
    actor_obs, critic_state = env.core.get_observations()

    assert policy.architecture == "branched_v4"
    assert env.cfg.actor_obs_dim == 92
    assert env.cfg.critic_state_dim == 54
    assert policy.gather_site_and_slot_goal_encoder[0].in_features == 6
    assert policy.trunk[0].in_features == 176
    assert observation_slices_metadata(92)["gather_site_and_slot_goal"] == {
        "start": 86,
        "end": 92,
        "dim": 6,
    }

    means, _ = policy.compute(
        {"observations": actor_obs.reshape(-1, env.cfg.actor_obs_dim)},
        role="policy",
    )
    assert means.shape == (env.num_envs * env.cfg.task.n_agents, 2)
    assert critic_state.shape == (env.num_envs, env.cfg.critic_state_dim)


def test_exp051_actor_can_pair_with_terminal_min_pairwise_critic_state() -> None:
    cfg = make_debug_cfg(num_envs=2, device="cpu")
    cfg.state.include_terminal_min_pairwise = True
    env = MultiRoverGatheringSKRLEnv(cfg)
    models = build_skrl_mappo_models(
        env,
        shared_actor=True,
        centralized_critic=True,
        actor_architecture="branched_v1",
        critic_architecture="structured_v2",
    )
    policy = models[env.possible_agents[0]]["policy"]
    value = models[env.possible_agents[0]]["value"]
    actor_obs, critic_state = env.core.get_observations()

    assert env.cfg.observation.schema_version == "ego_v3_local_terrain_grid"
    assert env.cfg.actor_obs_dim == 86
    assert env.cfg.critic_state_dim == 55
    assert policy.architecture == "branched_v1"
    assert value.architecture == "structured_v2"
    assert policy.aggregation_encoder[0].in_features == 5
    assert not hasattr(policy, "terminal_gate_encoder")
    assert value.team_encoder[0].in_features == 9

    means, _ = policy.compute(
        {"observations": actor_obs.reshape(-1, env.cfg.actor_obs_dim)},
        role="policy",
    )
    values, _ = value.compute({"states": critic_state}, role="value")

    assert means.shape == (env.num_envs * env.cfg.task.n_agents, 2)
    assert values.shape == (env.num_envs, 1)


def test_branched_actor_terrain_branch_weight_updates() -> None:
    env = _make_env()
    models = build_skrl_mappo_models(
        env,
        shared_actor=True,
        centralized_critic=True,
        actor_architecture="branched_v1",
        critic_architecture="structured_v1",
    )
    policy = models[env.possible_agents[0]]["policy"]
    actor_obs, _ = env.core.get_observations()
    observations = actor_obs[:, 0, :].detach().clone()
    observations[:, 31:81] = observations[:, 31:81] + torch.linspace(
        0.0,
        1.0,
        observations[:, 31:81].numel(),
    ).reshape_as(observations[:, 31:81])
    optimizer = torch.optim.Adam(policy.parameters(), lr=1.0e-2)
    snapshot = terrain_input_weight_snapshot(policy)

    mean, _ = policy.compute({"observations": observations}, role="policy")
    loss = (mean[:, 0] - 0.25).square().mean() + mean[:, 1].square().mean()
    loss.backward()
    optimizer.step()

    assert terrain_input_weight_delta_l2(policy, snapshot) > 0.0


def test_checkpoint_architecture_metadata_and_mismatch_rejection() -> None:
    env = _make_env()
    raw = {
        "experiment": {"name": "architecture_test"},
        "algorithm": {
            "actor_architecture": "branched_v1",
            "critic_architecture": "structured_v1",
        },
        "low_level_control": {"kinematic_model": "bicycle"},
        "trajectory_generator": {"geometry_method": "quintic"},
    }
    models = build_skrl_mappo_models(
        env,
        shared_actor=True,
        centralized_critic=True,
        actor_architecture="branched_v1",
        critic_architecture="structured_v1",
    )
    payload = skrl_mappo_checkpoint_payload(
        models,
        env.possible_agents,
        raw_cfg=raw,
        shared_actor=True,
        centralized_critic=True,
        shared_value=True,
        timesteps=32,
        observation_schema_version=env.cfg.observation.schema_version,
        actor_obs_dim=env.cfg.actor_obs_dim,
        critic_state_dim=env.cfg.critic_state_dim,
    )

    metadata = validate_checkpoint_compatibility(
        payload,
        env.cfg,
        expected_actor_architecture="branched_v1",
        expected_critic_architecture="structured_v1",
    )
    assert metadata["actor_architecture"] == "branched_v1"
    assert metadata["critic_architecture"] == "structured_v1"
    assert metadata["kinematic_model"] == "bicycle"
    assert metadata["trajectory_geometry_method"] == "quintic"
    assert metadata["observation_slices"]["terrain"]["dim"] == 50

    with pytest.raises(CheckpointCompatibilityError):
        validate_checkpoint_compatibility(
            payload,
            env.cfg,
            expected_actor_architecture="mlp_v1",
            expected_critic_architecture="structured_v1",
        )

    old_style_current_schema = {"metadata": observation_interface_metadata(env.cfg)}
    validate_checkpoint_compatibility(
        old_style_current_schema,
        env.cfg,
        expected_actor_architecture="mlp_v1",
        expected_critic_architecture="mlp_v1",
    )
    with pytest.raises(CheckpointCompatibilityError):
        validate_checkpoint_compatibility(
            old_style_current_schema,
            env.cfg,
            expected_actor_architecture="branched_v1",
        )


def test_exp042_config_selects_structured_bicycle_quintic_stack() -> None:
    config = ROOT / "configs/experiment/exp042_structured_actor_bicycle_quintic_probe.yaml"
    raw = load_yaml(config)
    cfg = cfg_from_experiment(config)

    assert raw["algorithm"]["actor_architecture"] == "branched_v1"
    assert raw["algorithm"]["critic_architecture"] == "structured_v1"
    assert cfg.low_level_control.kinematic_model == "bicycle"
    assert cfg.low_level_control.wheelbase_m == pytest.approx(0.65)
    assert cfg.trajectory_generator.geometry_method == "quintic"
    assert cfg.observation.communication_radius == pytest.approx(0.0)
    assert cfg.safety.world_xy_limit == pytest.approx(12.5)
    assert cfg.terrain.crater_field_size == pytest.approx(25.0)
    assert cfg.actor_obs_dim == 86
    assert cfg.critic_state_dim == 54


def test_exp043_config_is_direct_long_structured_map25_contract() -> None:
    config = ROOT / "configs/experiment/exp043_structured_bicycle_quintic_map25_long.yaml"
    raw = load_yaml(config)
    cfg = cfg_from_experiment(config)

    assert raw["experiment"]["name"] == "exp043_structured_bicycle_quintic_map25_long"
    assert raw["experiment"]["rollout_steps"] == 64
    assert raw["experiment"]["checkpoint_interval"] == 1024
    assert raw["algorithm"]["actor_architecture"] == "branched_v1"
    assert raw["algorithm"]["critic_architecture"] == "structured_v1"
    assert raw["algorithm"]["bc_updates"] == 0
    assert raw["algorithm"]["entropy_schedule_timesteps"] == 8192
    assert cfg.low_level_control.kinematic_model == "bicycle"
    assert cfg.low_level_control.max_steer_angle_rad == pytest.approx(0.698132)
    assert cfg.trajectory_generator.geometry_method == "quintic"
    assert cfg.trajectory_generator.n_trajectory_points == 12
    assert cfg.observation.communication_radius == pytest.approx(0.0)
    assert cfg.safety.world_xy_limit == pytest.approx(12.5)
    assert cfg.terrain.crater_field_size == pytest.approx(25.0)
    assert cfg.terrain.crater_count == 48
    assert cfg.initial_state.spawn_radius_min == pytest.approx(4.5)
    assert cfg.initial_state.spawn_radius_max == pytest.approx(6.5)
    assert cfg.initial_state.center_xy_range == pytest.approx(3.0)
    assert cfg.planner.subgoal_filter.hold_zone_override_after_warmup is True
    assert cfg.planner.subgoal_filter.hold_zone_spacing_weight == pytest.approx(8.0)
    assert cfg.actor_obs_dim == 86
    assert cfg.critic_state_dim == 54
    assert environment_geometry_metadata(cfg) == {
        "world_xy_limit": 12.5,
        "map_size_m": 25.0,
        "terrain_crater_field_size": 25.0,
    }


def test_exp044_config_enables_initial_state_curriculum_after_exp043_failure() -> None:
    config = ROOT / "configs/experiment/exp044_structured_bicycle_quintic_map25_curriculum.yaml"
    raw = load_yaml(config)
    cfg = cfg_from_experiment(config)

    assert raw["experiment"]["name"] == "exp044_structured_bicycle_quintic_map25_curriculum"
    assert raw["experiment"]["rollout_steps"] == 64
    assert raw["algorithm"]["actor_architecture"] == "branched_v1"
    assert raw["algorithm"]["critic_architecture"] == "structured_v1"
    assert raw["algorithm"]["initial_log_std"] == pytest.approx(-1.1)
    assert raw["algorithm"]["entropy_loss_scale"] == pytest.approx(0.0015)
    assert raw["algorithm"]["bc_updates"] == 0
    assert cfg.initial_state.curriculum_enabled is True
    assert cfg.initial_state.curriculum_start_spawn_radius_min == pytest.approx(3.0)
    assert cfg.initial_state.curriculum_start_spawn_radius_max == pytest.approx(4.0)
    assert cfg.initial_state.spawn_radius_min == pytest.approx(3.8)
    assert cfg.initial_state.spawn_radius_max == pytest.approx(5.2)
    assert cfg.initial_state.curriculum_warmup_timesteps == 4096
    assert cfg.initial_state.curriculum_ramp_timesteps == 8192
    assert cfg.terrain.crater_count == 36
    assert cfg.terrain.crater_field_size == pytest.approx(25.0)
    assert cfg.observation.communication_radius == pytest.approx(0.0)
    assert cfg.low_level_control.kinematic_model == "bicycle"
    assert cfg.trajectory_generator.geometry_method == "quintic"


def test_exp045_config_bootstraps_local_success_after_exp044_timeout() -> None:
    config = (
        ROOT
        / "configs/experiment/exp045_structured_bicycle_quintic_map25_local_success_bootstrap.yaml"
    )
    raw = load_yaml(config)
    cfg = cfg_from_experiment(config)

    assert (
        raw["experiment"]["name"]
        == "exp045_structured_bicycle_quintic_map25_local_success_bootstrap"
    )
    assert raw["experiment"]["rollout_steps"] == 64
    assert raw["algorithm"]["actor_architecture"] == "branched_v1"
    assert raw["algorithm"]["critic_architecture"] == "structured_v1"
    assert raw["algorithm"]["bc_updates"] == 0
    assert cfg.initial_state.curriculum_enabled is True
    assert cfg.initial_state.curriculum_start_spawn_radius_min == pytest.approx(1.6)
    assert cfg.initial_state.curriculum_start_spawn_radius_max == pytest.approx(2.4)
    assert cfg.initial_state.spawn_radius_min == pytest.approx(2.4)
    assert cfg.initial_state.spawn_radius_max == pytest.approx(3.4)
    assert cfg.initial_state.center_xy_range == pytest.approx(1.0)
    assert cfg.planner.rho_max == pytest.approx(1.6)
    assert cfg.planner.beta_max == pytest.approx(1.0471975512)
    assert cfg.planner.subgoal_filter.apply_probability_end == pytest.approx(0.35)
    assert cfg.reward_weights.terrain == pytest.approx(0.20)
    assert cfg.reward_coefficients.dmax_progress == pytest.approx(5.5)
    assert cfg.terrain.crater_count == 30
    assert cfg.terrain.crater_field_size == pytest.approx(25.0)
    assert cfg.observation.communication_radius == pytest.approx(0.0)
    assert cfg.low_level_control.kinematic_model == "bicycle"
    assert cfg.low_level_control.max_steer_angle_rad == pytest.approx(0.785398)
    assert cfg.trajectory_generator.geometry_method == "quintic"


def test_exp046_config_releases_terminal_hold_after_exp045_partial_success() -> None:
    config = (
        ROOT
        / "configs/experiment/exp046_structured_bicycle_quintic_map25_local_hold_release.yaml"
    )
    raw = load_yaml(config)
    cfg = cfg_from_experiment(config)

    assert raw["experiment"]["name"] == "exp046_structured_bicycle_quintic_map25_local_hold_release"
    assert raw["algorithm"]["actor_architecture"] == "branched_v1"
    assert raw["algorithm"]["critic_architecture"] == "structured_v1"
    assert raw["algorithm"]["bc_updates"] == 0
    assert cfg.initial_state.curriculum_enabled is True
    assert cfg.initial_state.curriculum_start_spawn_radius_min == pytest.approx(1.6)
    assert cfg.initial_state.curriculum_start_spawn_radius_max == pytest.approx(2.4)
    assert cfg.initial_state.spawn_radius_min == pytest.approx(2.4)
    assert cfg.initial_state.spawn_radius_max == pytest.approx(3.4)
    assert cfg.planner.rho_max == pytest.approx(1.6)
    assert cfg.planner.beta_max == pytest.approx(1.0471975512)
    assert cfg.planner.subgoal_filter.apply_probability_end == pytest.approx(0.22)
    assert cfg.planner.subgoal_filter.score_scale_end == pytest.approx(0.35)
    assert cfg.low_level_control.projection_activation_distance == pytest.approx(0.68)
    assert cfg.low_level_control.projection_strength == pytest.approx(0.70)
    assert cfg.low_level_control.projection_min_linear_scale == pytest.approx(0.40)
    assert cfg.low_level_control.success_zone_linear_scale == pytest.approx(0.65)
    assert cfg.reward_weights.terrain == pytest.approx(0.15)
    assert cfg.reward_coefficients.dmax_progress == pytest.approx(7.0)
    assert cfg.reward_coefficients.dispersion_progress == pytest.approx(3.2)
    assert cfg.reward_coefficients.success_bonus == pytest.approx(85.0)
    assert cfg.reward_coefficients.timeout_penalty == pytest.approx(45.0)
    assert cfg.terrain.crater_count == 30
    assert cfg.terrain.crater_field_size == pytest.approx(25.0)
    assert cfg.observation.communication_radius == pytest.approx(0.0)
    assert cfg.low_level_control.kinematic_model == "bicycle"
    assert cfg.trajectory_generator.geometry_method == "quintic"


def test_exp047_config_targets_terminal_convergence_after_exp046_timeout() -> None:
    config = (
        ROOT
        / "configs/experiment/exp047_structured_bicycle_quintic_map25_terminal_convergence.yaml"
    )
    raw = load_yaml(config)
    cfg = cfg_from_experiment(config)

    assert (
        raw["experiment"]["name"]
        == "exp047_structured_bicycle_quintic_map25_terminal_convergence"
    )
    assert raw["algorithm"]["actor_architecture"] == "branched_v1"
    assert raw["algorithm"]["critic_architecture"] == "structured_v1"
    assert raw["algorithm"]["bc_updates"] == 0
    assert cfg.initial_state.curriculum_enabled is True
    assert cfg.initial_state.curriculum_start_spawn_radius_min == pytest.approx(1.6)
    assert cfg.initial_state.curriculum_start_spawn_radius_max == pytest.approx(2.4)
    assert cfg.initial_state.spawn_radius_min == pytest.approx(2.4)
    assert cfg.initial_state.spawn_radius_max == pytest.approx(3.4)
    assert cfg.planner.rho_max == pytest.approx(1.6)
    assert cfg.planner.subgoal_filter.apply_probability_end < 0.22
    assert cfg.planner.subgoal_filter.score_scale_end < 0.35
    assert cfg.planner.subgoal_filter.hold_zone_pairwise_distance == pytest.approx(0.48)
    assert cfg.planner.subgoal_filter.hold_zone_spacing_weight == pytest.approx(3.20)
    assert cfg.planner.subgoal_filter.endpoint_safe_distance == pytest.approx(0.42)
    assert cfg.low_level_control.projection_activation_distance == pytest.approx(0.62)
    assert cfg.low_level_control.projection_strength == pytest.approx(0.50)
    assert cfg.low_level_control.projection_min_linear_scale == pytest.approx(0.55)
    assert cfg.low_level_control.success_zone_linear_scale == pytest.approx(0.80)
    assert cfg.safety.collision_distance < cfg.success_thresholds.min_pairwise_distance
    assert cfg.success_thresholds.min_pairwise_distance < cfg.success_thresholds.dmax
    assert cfg.safety.near_distance == pytest.approx(0.75)
    assert cfg.reward_weights.terrain == pytest.approx(0.12)
    assert cfg.reward_coefficients.dmax_progress == pytest.approx(9.0)
    assert cfg.reward_coefficients.dispersion_progress == pytest.approx(4.5)
    assert cfg.reward_coefficients.near_distance == pytest.approx(3.0)
    assert cfg.reward_coefficients.success_bonus == pytest.approx(115.0)
    assert cfg.reward_coefficients.timeout_penalty == pytest.approx(65.0)
    assert cfg.terrain.crater_count == 30
    assert cfg.terrain.crater_field_size == pytest.approx(25.0)
    assert cfg.observation.communication_radius == pytest.approx(0.0)
    assert cfg.low_level_control.kinematic_model == "bicycle"
    assert cfg.trajectory_generator.geometry_method == "quintic"


def test_exp048_config_increases_terminal_drive_after_exp047_timeout() -> None:
    config = (
        ROOT
        / "configs/experiment/exp048_structured_bicycle_quintic_map25_terminal_drive.yaml"
    )
    raw = load_yaml(config)
    cfg = cfg_from_experiment(config)

    assert raw["experiment"]["name"] == "exp048_structured_bicycle_quintic_map25_terminal_drive"
    assert raw["algorithm"]["actor_architecture"] == "branched_v1"
    assert raw["algorithm"]["critic_architecture"] == "structured_v1"
    assert raw["algorithm"]["bc_updates"] == 0
    assert cfg.initial_state.curriculum_enabled is True
    assert cfg.initial_state.curriculum_start_spawn_radius_min == pytest.approx(1.6)
    assert cfg.initial_state.spawn_radius_max == pytest.approx(3.4)
    assert cfg.planner.subgoal_filter.apply_probability_end == pytest.approx(0.18)
    assert cfg.planner.subgoal_filter.score_scale_end == pytest.approx(0.30)
    assert cfg.planner.subgoal_filter.hold_zone_pairwise_distance == pytest.approx(0.46)
    assert cfg.planner.subgoal_filter.visible_neighbor_center_weight == pytest.approx(1.90)
    assert cfg.planner.subgoal_filter.center_progress_weight == pytest.approx(3.20)
    assert cfg.low_level_control.max_linear_speed == pytest.approx(1.35)
    assert cfg.low_level_control.projection_strength == pytest.approx(0.45)
    assert cfg.low_level_control.projection_min_linear_scale == pytest.approx(0.65)
    assert cfg.low_level_control.success_zone_linear_scale == pytest.approx(0.95)
    assert cfg.safety.collision_distance < cfg.success_thresholds.min_pairwise_distance
    assert cfg.success_thresholds.min_pairwise_distance < cfg.success_thresholds.dmax
    assert cfg.safety.near_distance == pytest.approx(0.72)
    assert cfg.reward_weights.terrain == pytest.approx(0.10)
    assert cfg.reward_coefficients.dispersion_progress == pytest.approx(6.0)
    assert cfg.reward_coefficients.timeout_penalty == pytest.approx(80.0)
    assert cfg.reward_coefficients.success_bonus == pytest.approx(130.0)
    assert cfg.terrain.crater_field_size == pytest.approx(25.0)
    assert cfg.observation.communication_radius == pytest.approx(0.0)
    assert cfg.low_level_control.kinematic_model == "bicycle"
    assert cfg.trajectory_generator.geometry_method == "quintic"


def test_exp049_config_targets_terminal_spacing_timeout_from_exp048() -> None:
    config = (
        ROOT
        / "configs/experiment/exp049_structured_bicycle_quintic_map25_terminal_spacing.yaml"
    )
    raw = load_yaml(config)
    cfg = cfg_from_experiment(config)

    assert raw["experiment"]["name"] == "exp049_structured_bicycle_quintic_map25_terminal_spacing"
    assert raw["algorithm"]["actor_architecture"] == "branched_v1"
    assert raw["algorithm"]["critic_architecture"] == "structured_v1"
    assert raw["algorithm"]["bc_updates"] == 0
    assert cfg.initial_state.curriculum_enabled is True
    assert cfg.initial_state.curriculum_start_spawn_radius_min == pytest.approx(1.6)
    assert cfg.initial_state.spawn_radius_max == pytest.approx(3.4)
    assert cfg.planner.subgoal_filter.hold_zone_pairwise_distance == pytest.approx(0.52)
    assert cfg.planner.subgoal_filter.hold_zone_spacing_weight == pytest.approx(4.60)
    assert cfg.planner.subgoal_filter.endpoint_safe_distance == pytest.approx(0.44)
    assert cfg.planner.subgoal_filter.path_safe_distance == pytest.approx(0.32)
    assert cfg.planner.subgoal_filter.apply_probability_end == pytest.approx(0.20)
    assert cfg.planner.subgoal_filter.score_scale_end == pytest.approx(0.32)
    assert cfg.low_level_control.max_linear_speed == pytest.approx(1.32)
    assert cfg.low_level_control.projection_activation_distance == pytest.approx(0.64)
    assert cfg.low_level_control.projection_strength == pytest.approx(0.55)
    assert cfg.low_level_control.projection_min_linear_scale == pytest.approx(0.58)
    assert cfg.low_level_control.success_zone_linear_scale == pytest.approx(0.88)
    assert cfg.safety.collision_distance < cfg.success_thresholds.min_pairwise_distance
    assert cfg.success_thresholds.min_pairwise_distance < cfg.success_thresholds.dmax
    assert cfg.reward_coefficients.near_distance == pytest.approx(3.4)
    assert cfg.reward_coefficients.dispersion_progress == pytest.approx(6.2)
    assert cfg.reward_coefficients.timeout_penalty == pytest.approx(90.0)
    assert cfg.reward_coefficients.success_bonus == pytest.approx(135.0)
    assert cfg.terrain.crater_field_size == pytest.approx(25.0)
    assert cfg.observation.communication_radius == pytest.approx(0.0)
    assert cfg.low_level_control.kinematic_model == "bicycle"
    assert cfg.trajectory_generator.geometry_method == "quintic"


def test_exp058_config_only_extends_discount_horizon_from_exp051() -> None:
    baseline_path = ROOT / "configs/experiment/exp051_structured_bicycle_quintic_map25_ppo_stability.yaml"
    config = ROOT / "configs/experiment/exp058_structured_bicycle_quintic_map25_gamma995.yaml"
    baseline = load_yaml(baseline_path)
    raw = load_yaml(config)
    cfg = cfg_from_experiment(config)

    assert raw["experiment"]["name"] == "exp058_structured_bicycle_quintic_map25_gamma995"
    assert raw["algorithm"]["training_semantics"] == "exp058_structured_bicycle_quintic_map25_gamma995"
    assert raw["algorithm"]["gamma"] == pytest.approx(0.995)
    assert baseline["algorithm"]["gamma"] == pytest.approx(0.99)
    assert raw["algorithm"]["gae_lambda"] == baseline["algorithm"]["gae_lambda"]
    assert raw["algorithm"]["clip_epsilon"] == baseline["algorithm"]["clip_epsilon"]
    assert raw["algorithm"]["learning_rate"] == baseline["algorithm"]["learning_rate"]
    assert raw["algorithm"]["entropy_schedule_timesteps"] == baseline["algorithm"]["entropy_schedule_timesteps"]
    assert raw["planner"] == baseline["planner"]
    assert raw["low_level_control"] == baseline["low_level_control"]
    assert raw["reward"] == baseline["reward"]
    assert raw["success_thresholds"] == baseline["success_thresholds"]
    assert cfg.low_level_control.kinematic_model == "bicycle"
    assert cfg.trajectory_generator.geometry_method == "quintic"
    assert cfg.planner.subgoal_filter.apply_probability_end == pytest.approx(0.18)


def test_exp059_config_only_shortens_gae_trace_from_exp051() -> None:
    baseline_path = ROOT / "configs/experiment/exp051_structured_bicycle_quintic_map25_ppo_stability.yaml"
    config = ROOT / "configs/experiment/exp059_structured_bicycle_quintic_map25_gae090.yaml"
    baseline = load_yaml(baseline_path)
    raw = load_yaml(config)
    cfg = cfg_from_experiment(config)

    assert raw["experiment"]["name"] == "exp059_structured_bicycle_quintic_map25_gae090"
    assert raw["algorithm"]["training_semantics"] == "exp059_structured_bicycle_quintic_map25_gae090"
    assert raw["algorithm"]["gae_lambda"] == pytest.approx(0.90)
    assert baseline["algorithm"]["gae_lambda"] == pytest.approx(0.95)
    assert raw["algorithm"]["gamma"] == baseline["algorithm"]["gamma"]
    assert raw["algorithm"]["clip_epsilon"] == baseline["algorithm"]["clip_epsilon"]
    assert raw["algorithm"]["learning_rate"] == baseline["algorithm"]["learning_rate"]
    assert raw["algorithm"]["entropy_schedule_timesteps"] == baseline["algorithm"]["entropy_schedule_timesteps"]
    assert raw["planner"] == baseline["planner"]
    assert raw["low_level_control"] == baseline["low_level_control"]
    assert raw["reward"] == baseline["reward"]
    assert raw["success_thresholds"] == baseline["success_thresholds"]
    assert cfg.low_level_control.kinematic_model == "bicycle"
    assert cfg.trajectory_generator.geometry_method == "quintic"
    assert cfg.planner.subgoal_filter.apply_probability_end == pytest.approx(0.18)


def test_exp060_config_only_raises_value_loss_from_exp051() -> None:
    baseline_path = ROOT / "configs/experiment/exp051_structured_bicycle_quintic_map25_ppo_stability.yaml"
    config = ROOT / "configs/experiment/exp060_structured_bicycle_quintic_map25_value075.yaml"
    baseline = load_yaml(baseline_path)
    raw = load_yaml(config)
    cfg = cfg_from_experiment(config)

    assert raw["experiment"]["name"] == "exp060_structured_bicycle_quintic_map25_value075"
    assert raw["algorithm"]["training_semantics"] == "exp060_structured_bicycle_quintic_map25_value075"
    assert raw["algorithm"]["value_loss_coef"] == pytest.approx(0.75)
    assert baseline["algorithm"]["value_loss_coef"] == pytest.approx(0.5)
    assert raw["algorithm"]["gamma"] == baseline["algorithm"]["gamma"]
    assert raw["algorithm"]["gae_lambda"] == baseline["algorithm"]["gae_lambda"]
    assert raw["algorithm"]["clip_epsilon"] == baseline["algorithm"]["clip_epsilon"]
    assert raw["algorithm"]["learning_rate"] == baseline["algorithm"]["learning_rate"]
    assert raw["algorithm"]["entropy_schedule_timesteps"] == baseline["algorithm"]["entropy_schedule_timesteps"]
    assert raw["planner"] == baseline["planner"]
    assert raw["low_level_control"] == baseline["low_level_control"]
    assert raw["reward"] == baseline["reward"]
    assert raw["success_thresholds"] == baseline["success_thresholds"]
    assert cfg.low_level_control.kinematic_model == "bicycle"
    assert cfg.trajectory_generator.geometry_method == "quintic"
    assert cfg.planner.subgoal_filter.apply_probability_end == pytest.approx(0.18)


def test_exp061_config_only_adds_terminal_gate_observation_from_exp051() -> None:
    baseline_path = ROOT / "configs/experiment/exp051_structured_bicycle_quintic_map25_ppo_stability.yaml"
    config = ROOT / "configs/experiment/exp061_structured_bicycle_quintic_map25_terminal_gate_obs.yaml"
    baseline = load_yaml(baseline_path)
    raw = load_yaml(config)
    cfg = cfg_from_experiment(config)

    assert raw["experiment"]["name"] == "exp061_structured_bicycle_quintic_map25_terminal_gate_obs"
    assert raw["algorithm"]["training_semantics"] == "exp061_structured_bicycle_quintic_map25_terminal_gate_obs"
    assert raw["observation"]["schema_version"] == "ego_v4_terminal_gate"
    assert raw["observation"]["communication_radius"] == baseline["observation"]["communication_radius"]
    assert raw["algorithm"]["actor_architecture"] == "branched_v2"
    assert raw["algorithm"]["critic_architecture"] == "structured_v2"
    assert raw["algorithm"]["gamma"] == baseline["algorithm"]["gamma"]
    assert raw["algorithm"]["gae_lambda"] == baseline["algorithm"]["gae_lambda"]
    assert raw["algorithm"]["clip_epsilon"] == baseline["algorithm"]["clip_epsilon"]
    assert raw["algorithm"]["learning_rate"] == baseline["algorithm"]["learning_rate"]
    assert raw["algorithm"]["value_loss_coef"] == baseline["algorithm"]["value_loss_coef"]
    assert raw["planner"] == baseline["planner"]
    assert raw["low_level_control"] == baseline["low_level_control"]
    assert raw["reward"] == baseline["reward"]
    assert raw["success_thresholds"] == baseline["success_thresholds"]
    assert cfg.actor_obs_dim == 91
    assert cfg.critic_state_dim == 55
    assert cfg.low_level_control.kinematic_model == "bicycle"
    assert cfg.trajectory_generator.geometry_method == "quintic"


def test_exp062_config_only_adds_critic_minpair_state_from_exp051() -> None:
    baseline_path = ROOT / "configs/experiment/exp051_structured_bicycle_quintic_map25_ppo_stability.yaml"
    config = ROOT / "configs/experiment/exp062_structured_bicycle_quintic_map25_critic_minpair.yaml"
    baseline = load_yaml(baseline_path)
    raw = load_yaml(config)
    cfg = cfg_from_experiment(config)

    assert raw["experiment"]["name"] == "exp062_structured_bicycle_quintic_map25_critic_minpair"
    assert raw["algorithm"]["training_semantics"] == "exp062_structured_bicycle_quintic_map25_critic_minpair"
    assert raw["observation"] == baseline["observation"]
    assert raw["state"]["include_terminal_min_pairwise"] is True
    assert raw["algorithm"]["actor_architecture"] == baseline["algorithm"]["actor_architecture"]
    assert raw["algorithm"]["critic_architecture"] == "structured_v2"
    assert raw["algorithm"]["gamma"] == baseline["algorithm"]["gamma"]
    assert raw["algorithm"]["gae_lambda"] == baseline["algorithm"]["gae_lambda"]
    assert raw["algorithm"]["clip_epsilon"] == baseline["algorithm"]["clip_epsilon"]
    assert raw["algorithm"]["learning_rate"] == baseline["algorithm"]["learning_rate"]
    assert raw["algorithm"]["value_loss_coef"] == baseline["algorithm"]["value_loss_coef"]
    assert raw["planner"] == baseline["planner"]
    assert raw["low_level_control"] == baseline["low_level_control"]
    assert raw["reward"] == baseline["reward"]
    assert raw["success_thresholds"] == baseline["success_thresholds"]
    assert cfg.observation.schema_version == "ego_v3_local_terrain_grid"
    assert cfg.actor_obs_dim == 86
    assert cfg.critic_state_dim == 55
    assert cfg.low_level_control.kinematic_model == "bicycle"
    assert cfg.trajectory_generator.geometry_method == "quintic"


def test_skrl_telemetry_jsonl_is_written(tmp_path: Path) -> None:
    metrics = {
        "timesteps": 32,
        "wall_time_s": 1.25,
        "device": "cuda",
        "cuda_available": True,
        "mean_reward": None,
        "episode_length": None,
        "mean_pairwise_distance": None,
        "mean_oracle_distance": None,
        "success_rate": None,
        "nan_flag": False,
        "checkpoint_path": str(tmp_path / "checkpoint.pt"),
        "training_semantics": "skrl_mappo_pure_rl",
        "observation_schema_version": "ego_v3_local_terrain_grid",
        "actor_obs_dim": 86,
        "critic_state_dim": 54,
    }

    metrics_path = append_metrics_jsonl(tmp_path, metrics)

    assert metrics_path.exists()
    lines = metrics_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    for key in (
        "device",
        "training_semantics",
        "observation_schema_version",
        "actor_obs_dim",
        "critic_state_dim",
        "nan_flag",
        "checkpoint_path",
    ):
        assert key in parsed
    assert parsed["nan_flag"] is False


def test_checkpoint_compatibility_requires_current_schema_and_dimensions() -> None:
    env = _make_env()
    current = {"metadata": observation_interface_metadata(env.cfg)}

    metadata = validate_checkpoint_compatibility(current, env.cfg)

    assert metadata["observation_schema_version"] == "ego_v3_local_terrain_grid"
    assert metadata["actor_obs_dim"] == 86
    assert metadata["critic_state_dim"] == 54


@pytest.mark.parametrize(
    "metadata",
    [
        {},
        {
            "observation_schema_version": "ego_v2_speed_angular",
            "actor_obs_dim": 41,
            "critic_state_dim": 54,
        },
        {
            "observation_schema_version": "ego_v3_local_terrain_grid",
            "actor_obs_dim": 41,
            "critic_state_dim": 54,
        },
    ],
)
def test_old_missing_or_wrong_checkpoint_interface_is_rejected(metadata: dict) -> None:
    env = _make_env()

    with pytest.raises(CheckpointCompatibilityError):
        validate_checkpoint_compatibility({"metadata": metadata}, env.cfg)


def test_skrl_action_telemetry_reports_normalized_and_physical_scale() -> None:
    env = _make_env()
    action = torch.tensor(
        [
            [[-1.0, -1.0], [1.0, 1.0], [0.0, 0.0], [0.5, -0.5]],
            [[-0.25, 0.25], [0.75, -0.75], [0.95, 0.95], [-0.95, -0.95]],
        ],
        dtype=torch.float32,
    )

    telemetry = _action_telemetry(action, env.cfg)

    assert telemetry["physical_rho_min"] == pytest.approx(0.0)
    assert telemetry["physical_rho_max"] == pytest.approx(env.cfg.planner.rho_max)
    assert telemetry["physical_beta_max"] == pytest.approx(env.cfg.planner.beta_max)
    assert telemetry["physical_beta_min"] == pytest.approx(-env.cfg.planner.beta_max)
    assert telemetry["action_forward_high_saturation_fraction"] > 0.0
    assert telemetry["action_turn_abs_saturation_fraction"] > 0.0
    assert telemetry["physical_rho_max_config"] == pytest.approx(env.cfg.planner.rho_max)


def test_skrl_reward_breakdown_reports_weighted_contributions() -> None:
    env = _make_env()
    action = torch.zeros(env.num_envs, env.cfg.task.n_agents, env.cfg.planner.action_dim)
    output = env.core.step(action)

    breakdown = _reward_breakdown(output.info, env.cfg)

    for component in (
        "gather",
        "oracle",
        "energy",
        "safety",
        "terrain",
        "flatness",
        "motion",
        "consistency",
        "success_hold",
        "terminal",
    ):
        assert f"reward_raw_{component}" in breakdown
        assert f"reward_weight_{component}" in breakdown
        assert f"reward_contribution_{component}" in breakdown
        assert f"reward_abs_share_{component}" in breakdown
    assert breakdown["cohesion_pairwise_reward"] is None
    assert breakdown["reward_weighted_total"] == pytest.approx(breakdown["reward_contribution_sum"])


def test_cuda_training_diagnosis_summarizes_action_and_reward_components(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.jsonl"
    rows = [
        {
            "run_id": "diagnostic",
            "mean_pairwise_distance": 5.0,
            "mean_oracle_distance": 4.0,
            "success_rate": 0.0,
            "mean_reward": -0.1,
            "action_saturation_fraction": 0.05,
            "action_near_zero_fraction": 0.01,
            "action_forward_high_saturation_fraction": 0.0,
            "action_forward_low_saturation_fraction": 0.0,
            "action_turn_abs_saturation_fraction": 0.0,
            "physical_rho_high_fraction": 0.0,
            "physical_rho_low_fraction": 0.0,
            "physical_beta_abs_high_fraction": 0.0,
            "reward_contribution_gather": 0.1,
            "reward_abs_share_gather": 0.4,
            "reward_raw_gather": 0.1,
        },
        {
            "run_id": "diagnostic",
            "mean_pairwise_distance": 4.8,
            "mean_oracle_distance": 3.8,
            "success_rate": 0.0,
            "mean_reward": 0.2,
            "action_saturation_fraction": 0.35,
            "action_near_zero_fraction": 0.02,
            "action_forward_high_saturation_fraction": 0.30,
            "action_forward_low_saturation_fraction": 0.0,
            "action_turn_abs_saturation_fraction": 0.25,
            "physical_rho_high_fraction": 0.30,
            "physical_rho_low_fraction": 0.0,
            "physical_beta_abs_high_fraction": 0.25,
            "reward_contribution_gather": 0.3,
            "reward_abs_share_gather": 0.5,
            "reward_raw_gather": 0.3,
            "reward_weighted_total": 0.2,
            "reward_contribution_sum": 0.2,
            "reward_positive_contribution_sum": 0.3,
            "reward_negative_contribution_sum": -0.1,
            "reward_abs_contribution_sum": 0.4,
            "reward_dominant_positive_component": "gather",
            "reward_dominant_negative_component": "motion",
            "success_done": 0,
            "timeout_done": 2,
            "collision_done": 0,
            "safety_done": 0,
            "other_done": 0,
        },
    ]
    metrics_path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    summary = diagnose(metrics_path)

    assert summary["action_scale_summary"]["flags"] == [
        "normalized_action_saturation",
        "forward_high_saturation",
        "turn_saturation",
        "physical_rho_high_saturation",
        "physical_beta_saturation",
    ]
    assert summary["reward_component_summary"]["dominant_positive_component"] == "gather"
    assert summary["reward_component_trends"]["gather"]["contribution"]["last"] == pytest.approx(0.3)
    assert "action_scale_ablation" in summary["next_experiment_focus"]


def test_cuda_training_diagnosis_preserves_flatness_reward_signal(tmp_path: Path) -> None:
    metrics_path = tmp_path / "flatness_metrics.jsonl"
    row = {
        "run_id": "flatness_diagnostic",
        "mean_pairwise_distance": 1.0,
        "mean_oracle_distance": 1.0,
        "success_rate": 0.1,
        "mean_reward": 0.1,
        "reward_raw_flatness": 0.2,
        "reward_contribution_flatness": 0.2,
        "reward_abs_share_flatness": 0.6,
        "reward_abs_share_gather": 0.2,
        "reward_abs_share_oracle": 0.1,
        "reward_abs_share_safety": 0.1,
    }
    metrics_path.write_text(json.dumps(row), encoding="utf-8")

    summary = diagnose(metrics_path)

    assert summary["reward_component_trends"]["flatness"]["raw"]["last"] == pytest.approx(0.2)
    assert summary["reward_component_summary"]["contribution"]["flatness"] == pytest.approx(0.2)
    assert summary["reward_component_summary"]["abs_share"]["flatness"] == pytest.approx(0.6)
    assert "reward_signal_balance" not in summary["next_experiment_focus"]
