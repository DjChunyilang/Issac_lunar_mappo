from __future__ import annotations

import copy

import torch
import yaml

from _common import cfg_from_experiment, load_yaml
from lunar_rover_tasks.tasks.multi_rover_gathering.gathering_env import (
    MultiRoverGatheringCore,
    MultiRoverGatheringSKRLEnv,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.terrain_features import (
    build_multiscale_site_belief_observation,
)
from train_skrl_mappo import SKRLCategoricalPolicy, build_skrl_mappo_models


CONFIG = "configs/experiment/exp157_h1_site_belief_n1.yaml"


def _cfg(num_envs: int = 2):
    cfg = cfg_from_experiment(CONFIG)
    cfg.simulation.device = "cpu"
    cfg.simulation.num_envs = num_envs
    return cfg


def test_h1_contract_is_explicitly_diagnostic_and_pure_rl() -> None:
    cfg = _cfg()
    raw = load_yaml(CONFIG)
    assert cfg.actor_obs_dim == 407
    assert cfg.critic_state_dim == 950
    assert cfg.task.diagnostic_site_belief_enabled
    assert not cfg.task.explicit_goal_in_execution
    assert cfg.reward_weights.oracle == 0.5
    assert raw["algorithm"]["actor_architecture"] == "multiscale_n1_cnn"
    assert raw["algorithm"]["bc_updates"] == 0
    assert raw["algorithm"]["init_checkpoint"] is None
    assert not cfg.low_level_control.safety_projection_enabled
    assert not cfg.planner.subgoal_filter.enabled


def test_v11_actor_and_critic_shapes_are_decoupled() -> None:
    env = MultiRoverGatheringSKRLEnv(_cfg())
    actor_obs, critic_state = env.core.get_observations()
    assert actor_obs.shape == (2, 4, 407)
    assert critic_state.shape == (2, 950)
    terrain = actor_obs[..., 66:402].reshape(2, 4, 112, 3)
    assert torch.all((terrain[..., 2] >= 0.0) & (terrain[..., 2] <= 1.0))
    models = build_skrl_mappo_models(
        env,
        actor_architecture="multiscale_n1_cnn",
        critic_architecture="structured_multiscale_v3",
    )
    policy = models["rover_0"]["policy"]
    assert isinstance(policy, SKRLCategoricalPolicy)
    logits, _ = policy.compute({"observations": actor_obs.reshape(-1, 407)})
    values, _ = models["rover_0"]["value"].compute(
        {"states": critic_state}, role="value"
    )
    assert logits.shape == (8, 47)
    assert values.shape == (2, 1)
    assert torch.isfinite(logits).all() and torch.isfinite(values).all()
    assert sum(parameter.numel() for parameter in policy.parameters()) <= 120_000


def test_site_belief_channel_is_se2_invariant() -> None:
    positions = torch.tensor(
        [[[1.0, -0.5, 0.0], [-0.2, 0.8, 0.0], [0.5, 1.1, 0.0], [-1.0, -0.7, 0.0]]]
    )
    yaws = torch.tensor([[0.2, -1.0, 1.4, 2.2]])
    site = torch.tensor([[2.1, 1.7, 0.0]])
    original = build_multiscale_site_belief_observation(
        positions, yaws, site, site_radius=0.75, potential_sigma=2.0
    ).reshape(1, 4, 112, 3)[..., 2]
    angle = torch.tensor(0.83)
    rotation = torch.tensor(
        [[torch.cos(angle), -torch.sin(angle)], [torch.sin(angle), torch.cos(angle)]]
    )
    translation = torch.tensor([1.4, -2.0])
    transformed_positions = positions.clone()
    transformed_positions[..., :2] = positions[..., :2] @ rotation.T + translation
    transformed_site = site.clone()
    transformed_site[..., :2] = site[..., :2] @ rotation.T + translation
    transformed = build_multiscale_site_belief_observation(
        transformed_positions,
        yaws + angle,
        transformed_site,
        site_radius=0.75,
        potential_sigma=2.0,
    ).reshape(1, 4, 112, 3)[..., 2]
    assert torch.allclose(original, transformed, atol=1.0e-6, rtol=0.0)


def test_v11_schema_cannot_be_enabled_without_diagnostic_gate(tmp_path) -> None:
    raw = copy.deepcopy(load_yaml(CONFIG))
    raw["task"]["diagnostic_site_belief_enabled"] = False
    config = tmp_path / "invalid.yaml"
    config.write_text(yaml.safe_dump(raw), encoding="utf-8")
    try:
        cfg_from_experiment(config)
    except ValueError as error:
        assert "diagnostic_site_belief_enabled" in str(error)
    else:
        raise AssertionError("v11 must require the explicit H1 diagnostic gate")


def test_exp156_v10_contract_remains_295_dimensional() -> None:
    cfg = cfg_from_experiment(
        "configs/experiment/exp156_differential_multiscale_ablation.yaml"
    )
    assert cfg.actor_obs_dim == 295
    assert not cfg.task.diagnostic_site_belief_enabled
