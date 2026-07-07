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
    _action_telemetry,
    _nearest_distances,
    _randomize_bc_state,
    _reward_breakdown,
    append_metrics_jsonl,
    build_mappo_config,
    build_skrl_mappo_memories,
    build_skrl_mappo_models,
    environment_geometry_metadata,
    observation_slices_metadata,
    run_skrl_behavior_cloning,
    scripted_gather_action,
    skrl_mappo_checkpoint_payload,
    terrain_sanity_metrics,
    terrain_input_weight_delta_l2,
    terrain_input_weight_snapshot,
)
from shared_policy_mappo import SharedPolicyMAPPO, linear_schedule  # noqa: E402
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
