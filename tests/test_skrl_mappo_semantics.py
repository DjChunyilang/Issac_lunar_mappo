from __future__ import annotations

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
from _common import load_yaml  # noqa: E402
from _skrl_metadata import (  # noqa: E402
    DEFAULT_TRAINING_SEMANTICS,
    resolve_checkpoint_name,
    resolve_training_semantics,
    sanitize_checkpoint_name,
)
from train_skrl_mappo import (  # noqa: E402
    build_skrl_mappo_memories,
    build_skrl_mappo_models,
    skrl_mappo_checkpoint_payload,
)


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
    )

    assert payload["metadata"]["training_semantics"] == DEFAULT_TRAINING_SEMANTICS
    assert payload["metadata"]["experiment_name"] == "metadata_test"
    assert payload["metadata"]["algorithm_mode"] is None
    assert payload["metadata"]["observation_schema_version"] == "ego_v2_speed_angular"
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

    assert env.cfg.observation.schema_version == "ego_v2_speed_angular"
    assert env.observation_spaces[env.possible_agents[0]].shape == (env.cfg.actor_obs_dim,)
    assert policy.net[0].in_features == env.cfg.actor_obs_dim
