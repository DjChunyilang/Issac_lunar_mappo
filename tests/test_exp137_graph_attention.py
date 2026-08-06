from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from _common import cfg_from_experiment, load_yaml  # noqa: E402
from _skrl_metadata import (  # noqa: E402
    CheckpointCompatibilityError,
    validate_checkpoint_compatibility,
)
from train_skrl_mappo import build_skrl_mappo_models  # noqa: E402
from lunar_rover_tasks.tasks.multi_rover_gathering.gathering_env import (  # noqa: E402
    MultiRoverGatheringSKRLEnv,
)


B0_CONFIG = (
    ROOT
    / "configs"
    / "experiment"
    / "exp125_decentralized_tiered_b0_pure_rl_relative_quintic.yaml"
)
B2_CONFIG = (
    ROOT
    / "configs"
    / "experiment"
    / "exp137_decentralized_b2_graph_attention.yaml"
)


def _build_policy():
    cfg = cfg_from_experiment(B2_CONFIG)
    cfg.simulation.device = "cpu"
    cfg.simulation.num_envs = 2
    env = MultiRoverGatheringSKRLEnv(cfg)
    models = build_skrl_mappo_models(
        env,
        shared_actor=True,
        centralized_critic=True,
        actor_architecture="branched_v6_graph_attention",
        critic_architecture="structured_v1",
    )
    return cfg, env, models[env.possible_agents[0]]["policy"]


def _policy_mean(policy, observations: torch.Tensor) -> torch.Tensor:
    return policy.compute({"observations": observations}, role="policy")[0]


def _permuted_neighbor_slots(observations: torch.Tensor, order: list[int]) -> torch.Tensor:
    result = observations.clone()
    neighbors = observations[..., 10:46].reshape(*observations.shape[:-1], 3, 12)
    result[..., 10:46] = neighbors[..., order, :].reshape(*observations.shape[:-1], 36)
    return result


def test_exp137_changes_only_neighbor_architecture_from_selected_b0() -> None:
    b0 = load_yaml(B0_CONFIG)
    b2 = load_yaml(B2_CONFIG)
    assert b2["experiment"]["name"] == "exp137_decentralized_b2_graph_attention"
    assert b2["algorithm"]["actor_architecture"] == "branched_v6_graph_attention"
    assert b2["algorithm"]["bc_updates"] == 0
    assert b2["algorithm"]["init_checkpoint"] is None

    normalized_b0 = copy.deepcopy(b0)
    normalized_b2 = copy.deepcopy(b2)
    normalized_b2["experiment"] = normalized_b0["experiment"]
    normalized_b2["algorithm"]["training_semantics"] = normalized_b0["algorithm"][
        "training_semantics"
    ]
    normalized_b2["algorithm"]["actor_architecture"] = normalized_b0["algorithm"][
        "actor_architecture"
    ]
    assert normalized_b2 == normalized_b0


def test_graph_attention_is_neighbor_permutation_invariant() -> None:
    cfg, env, policy = _build_policy()
    observations, _ = env.core.get_observations()
    flat = observations.reshape(-1, cfg.actor_obs_dim)
    reference = _policy_mean(policy, flat)
    for order in ([2, 0, 1], [1, 2, 0], [2, 1, 0]):
        permuted = _permuted_neighbor_slots(flat, list(order))
        actual = _policy_mean(policy, permuted)
        torch.testing.assert_close(actual, reference, rtol=0.0, atol=1.0e-6)


def test_invalid_neighbor_content_cannot_leak_through_attention() -> None:
    cfg, env, policy = _build_policy()
    observations, _ = env.core.get_observations()
    flat = observations.reshape(-1, cfg.actor_obs_dim)
    neighbors = flat[..., 10:46].reshape(-1, 3, 12)
    neighbors[:, 1, 11] = 0.0
    reference = _policy_mean(policy, flat)

    perturbed = flat.clone()
    perturbed_neighbors = perturbed[..., 10:46].reshape(-1, 3, 12)
    perturbed_neighbors[:, 1, :11] = torch.linspace(
        -100.0,
        100.0,
        11,
        dtype=perturbed.dtype,
    )
    perturbed_neighbors[:, 1, 11] = 0.0
    actual = _policy_mean(policy, perturbed)
    torch.testing.assert_close(actual, reference, rtol=0.0, atol=0.0)


def test_no_neighbor_aggregation_is_zero_and_policy_is_finite() -> None:
    cfg, env, policy = _build_policy()
    observations, _ = env.core.get_observations()
    flat = observations.reshape(-1, cfg.actor_obs_dim)
    flat[..., 10:46] = 0.0
    ego_embedding = policy.ego_encoder(flat[..., :10])
    encoded = policy.neighbor_encoder(flat[..., 10:46], ego_embedding)
    assert encoded.shape == (flat.shape[0], 48)
    assert torch.equal(encoded, torch.zeros_like(encoded))
    assert torch.isfinite(_policy_mean(policy, flat)).all()


def test_graph_attention_neighbor_encoder_receives_gradients() -> None:
    cfg, env, policy = _build_policy()
    observations, _ = env.core.get_observations()
    flat = observations.reshape(-1, cfg.actor_obs_dim)
    mean = _policy_mean(policy, flat)
    mean.square().mean().backward()
    gradients = [
        parameter.grad
        for parameter in policy.neighbor_encoder.parameters()
        if parameter.requires_grad
    ]
    assert gradients
    assert all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients if gradient is not None)
    assert sum(float(gradient.abs().sum()) for gradient in gradients if gradient is not None) > 0.0


def test_b0_checkpoint_architecture_is_rejected_by_b2() -> None:
    cfg = cfg_from_experiment(B2_CONFIG)
    checkpoint = {
        "metadata": {
            "observation_schema_version": cfg.observation.schema_version,
            "actor_obs_dim": cfg.actor_obs_dim,
            "critic_state_dim": cfg.critic_state_dim,
            "actor_architecture": "branched_v5",
            "critic_architecture": "structured_v1",
        }
    }
    with pytest.raises(CheckpointCompatibilityError, match="actor architecture"):
        validate_checkpoint_compatibility(
            checkpoint,
            cfg,
            expected_actor_architecture="branched_v6_graph_attention",
            expected_critic_architecture="structured_v1",
        )
