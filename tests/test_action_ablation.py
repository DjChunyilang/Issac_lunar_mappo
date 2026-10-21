"""T11 preparation contracts; synthetic replay never starts environment training."""

from copy import deepcopy
from dataclasses import replace
import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
from torch.distributions import Bernoulli, Categorical, Normal, TransformedDistribution
from torch.distributions.transforms import AffineTransform, TanhTransform

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from _common import cfg_from_experiment
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.action_ablation import (
    REPRESENTATIONS,
    ActionContract,
    ActionPayload,
    AblationPolicy,
    AblationSample,
    BoundedMoveDistribution,
    build_ablation_policies,
    clipped_actor_objective,
    decode_payload,
    replay_action_sequence,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.actions import PolarHoldDistribution
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.baselines import (
    PRIMITIVE_VERSION,
    primitive_fingerprint,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.learning import ppo_update


@pytest.fixture(scope="module")
def spec():
    return cfg_from_experiment(ROOT / "configs/experiment/upstream_scale_s01_review_v2.yaml").spec


@pytest.fixture
def policies(spec):
    torch.set_num_threads(1)
    torch.manual_seed(23)
    return build_ablation_policies(991, spec)


def payload(contract, continuous, *, hold=None, ids=None):
    continuous = torch.as_tensor(continuous, dtype=torch.float64)
    shape = continuous.shape[:-1]
    return ActionPayload(
        continuous,
        torch.zeros(shape, dtype=torch.bool) if hold is None else torch.as_tensor(hold),
        torch.full(shape, -1, dtype=torch.int64) if ids is None else torch.as_tensor(ids),
        contract.version,
        PRIMITIVE_VERSION if contract.representation == "A3" else None,
        primitive_fingerprint() if contract.representation == "A3" else None,
    )


def synthetic_sequence(policy, *, length=4, valid=None, ended=None):
    generator = torch.Generator().manual_seed(47)
    observations = torch.randn(length, 2, 4, 411, generator=generator) * .03
    initial = torch.randn(2, 4, 128, generator=generator) * .1
    valid = torch.ones(length, 2, dtype=torch.bool) if valid is None else valid
    ended = torch.zeros_like(valid) if ended is None else ended
    hidden, samples = initial, []
    with torch.no_grad():
        for index, observation in enumerate(observations):
            sample = policy.sample(observation, hidden)
            samples.append(sample)
            hidden = torch.where(
                (ended[index] | ~valid[index])[:, None, None], 0., sample.hidden
            )
    return observations, initial, samples, valid, ended, hidden


@pytest.mark.parametrize("representation", ["A0", "A2"])
def test_continuous_log_probability_matches_independent_change_of_measure(spec, representation):
    contract = ActionContract.from_spec(representation, spec)
    dimension = contract.continuous_dim
    mean = torch.linspace(-.4, .3, 3 * dimension, dtype=torch.float64).reshape(3, dimension)
    std = torch.linspace(.35, .9, dimension, dtype=torch.float64)
    latent = torch.linspace(-2., 2., 3 * dimension, dtype=torch.float64).reshape(3, dimension)
    distribution = BoundedMoveDistribution(mean, std, contract)
    transforms = [TanhTransform(cache_size=1)]
    if representation == "A0":
        transforms.append(AffineTransform(
            torch.tensor([contract.rho_max / 2, 0.], dtype=torch.float64),
            torch.tensor([contract.rho_max / 2, contract.beta_max], dtype=torch.float64),
        ))
    reference = TransformedDistribution(Normal(mean, std), transforms)
    physical_or_normalized = distribution.transform(latent)
    torch.testing.assert_close(
        distribution.log_prob(latent), reference.log_prob(physical_or_normalized).sum(-1),
        atol=1e-10, rtol=1e-8,
    )
    expected_measure = "physical_rho_beta" if representation == "A0" else "normalized_cartesian_yaw_speed"
    assert contract.metadata["probability_measure"] == expected_measure
    if representation == "A2":
        # A physical request density has a different additive constant; it is not
        # the original tanh action density used for sampling and replay.
        physical_log = distribution.log_prob(latent) - math.log(
            contract.rho_max ** 2 * math.pi * contract.speed_cap_max / 2
        )
        assert not torch.allclose(physical_log, distribution.log_prob(latent))


def test_a1_wrapper_preserves_conditional_distribution_and_seeded_samples(policies):
    policy = policies["A1"]
    hidden = torch.linspace(-.2, .2, 8 * 128).reshape(2, 4, 128)
    actual = policy.action_distribution(hidden)
    reference = PolarHoldDistribution(
        policy.mean(hidden), policy.log_std.clamp(-5, 1).exp(),
        policy.hold_logit(hidden).squeeze(-1), policy.contract.rho_max, policy.contract.beta_max,
    )
    with torch.random.fork_rng():
        torch.manual_seed(31)
        expected_action, expected_latent, expected_log = reference.sample()
        torch.manual_seed(31)
        action, latent, log = actual.sample()
    torch.testing.assert_close(action.physical, expected_action.physical, atol=0, rtol=0)
    assert torch.equal(action.hold, expected_action.hold)
    torch.testing.assert_close(latent, expected_latent, atol=0, rtol=0)
    torch.testing.assert_close(log, expected_log, atol=0, rtol=0)


def test_a1_hold_log_probability_ignores_continuous_density_but_entropy_is_conditional(spec):
    contract = ActionContract.from_spec("A1", spec)
    mean = torch.zeros(3, 2, dtype=torch.float64, requires_grad=True)
    std = torch.full((3, 2), .7, dtype=torch.float64, requires_grad=True)
    logits = torch.tensor([-2., 0., 2.], dtype=torch.float64, requires_grad=True)
    distribution = PolarHoldDistribution(mean, std, logits, contract.rho_max, contract.beta_max)
    log = distribution.log_prob(torch.full_like(mean, float("nan")), torch.ones(3, dtype=torch.bool))
    torch.testing.assert_close(log, Bernoulli(logits=logits).log_prob(torch.ones(3, dtype=torch.float64)))
    log.sum().backward()
    assert mean.grad is None and std.grad is None
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    distribution.entropy().sum().backward()
    assert std.grad is not None and torch.isfinite(std.grad).all() and std.grad.abs().sum() > 0


def test_a3_all_native_ids_have_categorical_probability_and_frozen_library_identity(policies):
    policy = policies["A3"].double()
    hidden = torch.zeros(47, 128, dtype=torch.float64)
    with torch.no_grad():
        policy.primitive_logits.weight.zero_()
        policy.primitive_logits.bias.copy_(torch.linspace(-2., 2., 47, dtype=torch.float64))
    ids = torch.arange(47, dtype=torch.int64)
    action = payload(policy.contract, torch.empty(47, 0), ids=ids)
    sample = AblationSample(action, torch.empty(47, 0, dtype=torch.float64), torch.zeros(47), hidden)
    logs, entropy = policy.evaluate_sample(hidden, sample)
    reference = Categorical(logits=policy.primitive_logits.bias)
    torch.testing.assert_close(logs, reference.log_prob(ids), atol=1e-10, rtol=1e-8)
    torch.testing.assert_close(logs.exp().sum(), torch.tensor(1., dtype=torch.float64))
    torch.testing.assert_close(entropy, reference.entropy().expand(47))
    library_record = json.loads((ROOT / "configs/experiment/primitive47_library_v1.json").read_text())
    assert action.primitive_fingerprint == library_record["sha256"]
    assert action.primitive_version == library_record["library"]["version"]
    poses = np.zeros((1, 47, 3))
    requests = decode_payload(replace(action, continuous=action.continuous[None], hold=action.hold[None], primitive_id=ids[None]), policy.contract, poses, [0.])
    assert [request.primitive_id for request in requests] == list(range(47))
    assert all(request.goal_xy is None for request in requests)
    assert requests[0].primitive_id == 0 and not requests[0].hold


@pytest.mark.parametrize("representation", ["A0", "A2"])
def test_zero_motion_parameters_remain_move(spec, representation):
    contract = ActionContract.from_spec(representation, spec)
    continuous = [[[0., 0.]]] if representation == "A0" else [[[0., 0., 0., -1.]]]
    pose = np.array([[[1., 2., .7]]])
    request = decode_payload(payload(contract, continuous), contract, pose, [4.])[0]
    assert not request.hold and request.primitive_id is None
    np.testing.assert_array_equal(request.goal_xy, pose[0, 0, :2])
    assert request.timestamp == 4. and request.expires_at == 4. + contract.request_seconds
    if representation == "A2":
        assert request.speed_cap == 0.
        assert request.terminal_yaw == pytest.approx(.7)


@pytest.mark.parametrize("representation", ["A0", "A1", "A2"])
def test_decoding_is_se2_equivariant_and_requests_freeze_original_state(spec, representation):
    contract = ActionContract.from_spec(representation, spec)
    physical = [[[1.2, -.4], [0., 0.]]] if representation != "A2" else [[[.3, -.2, .25, .2], [0., 0., 0., -1.]]]
    action = payload(contract, physical, hold=[[False, representation == "A1"]])
    pose = np.array([[[.2, -.1, .3], [-.3, .4, -.6]]])
    angle, translation = .8, np.array([4., -2.])
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    transformed = pose.copy()
    transformed[..., :2] = pose[..., :2] @ rotation.T + translation
    transformed[..., 2] += angle
    original = decode_payload(action, contract, pose, [3.])
    moved = decode_payload(action, contract, transformed, [3.])
    frozen = deepcopy(original)
    for left, right in zip(original, moved):
        np.testing.assert_allclose(right.goal_xy, rotation @ left.goal_xy + translation, atol=1e-12)
        assert left.hold == right.hold
        if left.terminal_yaw is not None:
            difference = (right.terminal_yaw - left.terminal_yaw - angle + math.pi) % (2 * math.pi) - math.pi
            assert difference == pytest.approx(0., abs=1e-12)
        assert left.speed_cap == right.speed_cap
    pose[:] = 123.
    action.continuous.fill_(0.)
    assert original == frozen


def test_four_policies_share_identical_backbone_gru_and_critic_but_only_required_heads(policies):
    reference = policies["A1"].state_dict()
    head_names = ("mean.", "log_std", "hold_logit.", "primitive_logits.")
    common = {name for name in reference if not name.startswith(head_names)}
    assert any(name.startswith("gru.") for name in common)
    assert any(name.startswith("critic.") for name in common)
    obs, hidden, states = torch.zeros(2, 4, 411), torch.zeros(2, 4, 128), torch.zeros(2, 991)
    expected_encoding, expected_values = policies["A1"].encode(obs), policies["A1"].value(states)
    for representation, policy in policies.items():
        for name in common:
            torch.testing.assert_close(policy.state_dict()[name], reference[name], atol=0, rtol=0)
        torch.testing.assert_close(policy.encode(obs), expected_encoding, atol=0, rtol=0)
        torch.testing.assert_close(policy.value(states), expected_values, atol=0, rtol=0)
        assert policy.diagnostic_only and not policy.diagnostic_metadata["training_eligible"]
        if representation == "A3":
            assert not hasattr(policy, "mean") and not hasattr(policy, "log_std")
            assert policy.primitive_logits.out_features == 47
        else:
            assert policy.mean.out_features == policy.contract.continuous_dim
            assert hasattr(policy, "hold_logit") == (representation == "A1")
        policy.sample(obs, hidden).action.validate(policy.contract, (2, 4))


@pytest.mark.parametrize("representation", list(REPRESENTATIONS))
def test_original_sample_replay_ratio_is_one_without_rewriting_actions(policies, representation):
    policy = policies[representation]
    obs, initial, samples, valid, ended, tail = synthetic_sequence(policy)
    result = replay_action_sequence(policy, obs, initial, samples, valid, ended)
    torch.testing.assert_close(result["ratio"], torch.ones_like(result["ratio"]), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(result["log_prob"], torch.stack([sample.log_prob for sample in samples]), atol=1e-6, rtol=1e-6)
    # Removing a row changes CPU GEMM batch size; numerical equivalence uses the
    # float32 replay tolerance, rather than requiring identical accumulation.
    torch.testing.assert_close(result["tail_hidden"], tail, atol=1e-7, rtol=1e-6)
    assert torch.isfinite(result["entropy"]).all()
    objective = clipped_actor_objective(result["ratio"], torch.ones_like(valid, dtype=obs.dtype), valid)
    torch.testing.assert_close(objective, torch.tensor(-1.))


@pytest.mark.parametrize("representation", list(REPRESENTATIONS))
@pytest.mark.parametrize("deterministic", [False, True])
def test_replay_gradients_match_frozen_rollout_even_when_sampling_has_a_graph(
    policies, representation, deterministic,
):
    policy = policies[representation]
    if deterministic and representation == "A1":
        # Exercise a differentiable MOVE latent instead of an all-HOLD fixture.
        with torch.no_grad():
            policy.hold_logit.bias.fill_(-1.)
    observations = torch.randn(3, 2, 4, 411) * .03
    _, initial = policy.distribution(torch.randn(2, 4, 411) * .03, torch.zeros(2, 4, 128))
    valid = torch.ones(3, 2, dtype=torch.bool)
    ended = torch.tensor([[False, False], [True, False], [False, True]])
    hidden, samples = initial, []
    for observation, done in zip(observations, ended):
        sample = policy.sample(observation, hidden, deterministic=deterministic)
        samples.append(sample)
        hidden = torch.where(done[:, None, None], 0., sample.hidden)
    # Value-derived advantages deliberately retain their critic computation graph.
    advantages = 1. + .1 * policy.value(torch.randn(3, 2, 991) * .03)
    assert initial.requires_grad and advantages.requires_grad
    assert all(sample.log_prob.requires_grad for sample in samples)
    if deterministic and representation != "A3":
        assert any(sample.latent.requires_grad for sample in samples)
    frozen = [replace(
        sample,
        action=replace(sample.action, continuous=sample.action.continuous.detach()),
        latent=sample.latent.detach(), log_prob=sample.log_prob.detach(), hidden=sample.hidden.detach(),
    ) for sample in samples]
    parameters = tuple(policy.parameters())
    reference = replay_action_sequence(policy, observations, initial.detach(), frozen, valid, ended)
    reference_loss = clipped_actor_objective(reference["ratio"], advantages.detach(), valid)
    reference_gradients = torch.autograd.grad(reference_loss, parameters, allow_unused=True)

    actual = replay_action_sequence(policy, observations, initial, samples, valid, ended)
    actual_loss = clipped_actor_objective(actual["ratio"], advantages, valid)
    stored_tensors = (initial, advantages, *(sample.log_prob for sample in samples), *(
        tensor for sample in samples for tensor in (sample.latent, sample.action.continuous)
        if tensor.requires_grad
    ))
    gradients = torch.autograd.grad(actual_loss, parameters + stored_tensors, allow_unused=True)
    actual_gradients, stored_gradients = gradients[:len(parameters)], gradients[len(parameters):]
    assert all(gradient is None for gradient in stored_gradients)
    torch.testing.assert_close(actual["ratio"], reference["ratio"], atol=0, rtol=0)
    torch.testing.assert_close(actual_loss, reference_loss, atol=0, rtol=0)
    squared_norm = 0.
    for actual_gradient, reference_gradient in zip(actual_gradients, reference_gradients):
        if reference_gradient is None:
            assert actual_gradient is None
            continue
        assert actual_gradient is not None and torch.isfinite(actual_gradient).all()
        torch.testing.assert_close(actual_gradient, reference_gradient, atol=1e-8, rtol=1e-6)
        squared_norm += float(actual_gradient.square().sum())
    assert squared_norm > 1e-8


@pytest.mark.parametrize("representation", ["A0", "A2"])
@pytest.mark.parametrize("magnitude", [30., 100.])
def test_extreme_continuous_latents_have_finite_density_and_gradient(spec, representation, magnitude):
    contract = ActionContract.from_spec(representation, spec)
    mean = torch.zeros(2, contract.continuous_dim, dtype=torch.float64, requires_grad=True)
    std = torch.full_like(mean, .5)
    latent = torch.full_like(mean, magnitude)
    latent[0] = -magnitude
    distribution = BoundedMoveDistribution(mean, std, contract)
    logs = distribution.log_prob(latent)
    assert torch.isfinite(logs).all() and torch.isfinite(distribution.entropy()).all()
    logs.sum().backward()
    assert torch.isfinite(mean.grad).all()


@pytest.mark.parametrize("representation", ["A1", "A3"])
@pytest.mark.parametrize("magnitude", [-100., 100.])
def test_extreme_discrete_logits_have_finite_entropy_and_replayed_probability(policies, representation, magnitude):
    policy = policies[representation]
    with torch.no_grad():
        if representation == "A1":
            policy.hold_logit.weight.zero_()
            policy.hold_logit.bias.fill_(magnitude)
        else:
            policy.primitive_logits.weight.zero_()
            policy.primitive_logits.bias.copy_(torch.linspace(-abs(magnitude), abs(magnitude), 47))
    sample = policy.sample(torch.zeros(2, 4, 411), torch.zeros(2, 4, 128))
    log, entropy = policy.evaluate_sample(sample.hidden, sample)
    assert torch.isfinite(log).all() and torch.isfinite(entropy).all()
    torch.testing.assert_close(log, sample.log_prob)


@pytest.mark.parametrize("defect", ["negative_rho", "large_rho", "large_beta", "nan", "hold_dtype", "hold_on_a0", "unexpected_id", "version", "shape", "primitive_identity"])
def test_invalid_polar_payload_rejected_before_decode(spec, defect):
    contract = ActionContract.from_spec("A0", spec)
    action = payload(contract, [[[1., 0.]]])
    if defect == "negative_rho": action.continuous[..., 0] = -.1
    if defect == "large_rho": action.continuous[..., 0] = contract.rho_max + .1
    if defect == "large_beta": action.continuous[..., 1] = math.pi + .1
    if defect == "nan": action.continuous[..., 0] = float("nan")
    if defect == "hold_dtype": action = replace(action, hold=action.hold.long())
    if defect == "hold_on_a0": action = replace(action, hold=torch.ones_like(action.hold))
    if defect == "unexpected_id": action = replace(action, primitive_id=torch.zeros_like(action.primitive_id))
    if defect == "version": action = replace(action, action_version="polar_hold_v2")
    if defect == "shape": action = replace(action, continuous=action.continuous.squeeze(0))
    if defect == "primitive_identity": action = replace(action, primitive_version=PRIMITIVE_VERSION)
    pose = np.zeros((1, 1, 3))
    with pytest.raises(ValueError):
        decode_payload(action, contract, pose, [0.])
    np.testing.assert_array_equal(pose, np.zeros((1, 1, 3)))


@pytest.mark.parametrize("defect", ["negative", "too_large", "float_id", "wrong_library", "wrong_fingerprint", "hold_flag", "continuous"])
def test_invalid_native_primitive_payload_rejected(spec, defect):
    contract = ActionContract.from_spec("A3", spec)
    action = payload(contract, torch.empty(1, 1, 0), ids=[[0]])
    if defect == "negative": action = replace(action, primitive_id=torch.tensor([[-1]]))
    if defect == "too_large": action = replace(action, primitive_id=torch.tensor([[47]]))
    if defect == "float_id": action = replace(action, primitive_id=action.primitive_id.float())
    if defect == "wrong_library": action = replace(action, primitive_version="other_library")
    if defect == "wrong_fingerprint": action = replace(action, primitive_fingerprint="0" * 64)
    if defect == "hold_flag": action = replace(action, hold=torch.ones_like(action.hold))
    if defect == "continuous": action = replace(action, continuous=torch.zeros(1, 1, 2))
    with pytest.raises(ValueError):
        decode_payload(action, contract, np.zeros((1, 1, 3)), [0.])


@pytest.mark.parametrize("representation", ["A0", "A1", "A2"])
def test_action_rewrite_is_rejected_even_when_still_within_bounds(policies, representation):
    policy = policies[representation]
    sample = policy.sample(torch.zeros(2, 4, 411), torch.zeros(2, 4, 128), deterministic=True)
    changed = sample.action.continuous.clone()
    changed[..., 0] += .01
    with pytest.raises(ValueError, match="changed|placeholders"):
        policy.evaluate_sample(sample.hidden, replace(sample, action=replace(sample.action, continuous=changed)))


@pytest.mark.parametrize("representation", list(REPRESENTATIONS))
def test_nan_invalid_environment_is_removed_before_encoder_and_density(policies, representation, monkeypatch, record_property):
    policy = policies[representation]
    valid = torch.ones(4, 2, dtype=torch.bool)
    valid[1, 0] = False
    obs, initial, samples, valid, ended, tail = synthetic_sequence(policy, valid=valid)
    obs[1, 0] = float("nan")
    bad = samples[1]
    continuous, latent, logs, following, ids = (bad.action.continuous.clone(), bad.latent.clone(), bad.log_prob.clone(), bad.hidden.clone(), bad.action.primitive_id.clone())
    continuous[0] = latent[0] = logs[0] = following[0] = float("nan")
    ids[0] = 100
    samples[1] = replace(bad, action=replace(bad.action, continuous=continuous, primitive_id=ids), latent=latent, log_prob=logs, hidden=following)
    original_encode = policy.encode
    seen = []
    def finite_encoder(inputs):
        assert torch.isfinite(inputs).all()
        seen.append(inputs.shape[0])
        return original_encode(inputs)
    monkeypatch.setattr(policy, "encode", finite_encoder)
    result = replay_action_sequence(policy, obs, initial, samples, valid, ended)
    assert seen == [2, 1, 2, 2]
    assert torch.isfinite(result["log_prob"]).all() and torch.isfinite(result["entropy"]).all()
    torch.testing.assert_close(result["ratio"][valid], torch.ones_like(result["ratio"][valid]), atol=1e-6, rtol=1e-6)
    # Removing invalid environments changes GEMM batch size; float32 need not be bitwise equal.
    record_property("max_cross_batch_hidden_error", float((result["tail_hidden"] - tail).abs().max().detach()))
    torch.testing.assert_close(result["tail_hidden"], tail, atol=1e-7, rtol=1e-6)


def test_all_invalid_sequence_and_empty_sequence_do_not_evaluate_network(policies, monkeypatch):
    policy = policies["A1"]
    obs, initial, samples, valid, ended, _ = synthetic_sequence(policy)
    obs.fill_(float("nan")); initial.fill_(float("nan")); valid[:] = False
    def forbidden(*args, **kwargs):
        raise AssertionError("Invalid rows must not enter network")
    monkeypatch.setattr(policy, "distribution", forbidden)
    result = replay_action_sequence(policy, obs, initial, samples, valid, ended)
    assert torch.equal(result["tail_hidden"], torch.zeros_like(initial))
    assert torch.equal(result["ratio"], torch.ones_like(result["ratio"]))
    objective = clipped_actor_objective(result["ratio"], torch.full_like(valid, float("nan"), dtype=torch.float32), valid)
    assert objective == 0.
    empty = replay_action_sequence(policy, obs[:0], torch.zeros_like(initial), [], valid[:0], ended[:0])
    assert empty["log_prob"].shape == (0, 2, 4)


@pytest.mark.parametrize("defect", ["ratio_rank", "advantage_rank", "valid_rank", "negative_ratio", "nonfinite_ratio", "nonfinite_advantage", "valid_dtype"])
def test_surrogate_rejects_invalid_shapes_and_valid_nonfinite_values(defect):
    ratios = torch.ones(2, 3, 4)
    advantages = torch.ones(2, 3)
    valid = torch.ones(2, 3, dtype=torch.bool)
    if defect == "ratio_rank": ratios = ratios[..., 0]
    if defect == "advantage_rank": advantages = advantages.unsqueeze(-1)
    if defect == "valid_rank": valid = valid.unsqueeze(-1)
    if defect == "negative_ratio": ratios[0, 0, 0] = -.1
    if defect == "nonfinite_ratio": ratios[0, 0, 0] = float("nan")
    if defect == "nonfinite_advantage": advantages[0, 0] = float("nan")
    if defect == "valid_dtype": valid = valid.long()
    with pytest.raises(ValueError):
        clipped_actor_objective(ratios, advantages, valid)


def test_surrogate_clipping_matches_independent_scalar_reference_and_excludes_invalid():
    ratios = torch.tensor([[[1.5, .5], [float("nan"), float("nan")]], [[1.5, .5], [1., 1.]]])
    advantages = torch.tensor([[2., float("nan")], [-2., 1.]])
    valid = torch.tensor([[True, False], [True, True]])
    scalar_terms = []
    for time in range(2):
        for environment in range(2):
            if not valid[time, environment]:
                continue
            for agent in range(2):
                ratio, advantage = float(ratios[time, environment, agent]), float(advantages[time, environment])
                clipped = max(.85, min(1.15, ratio))
                scalar_terms.append(min(ratio * advantage, clipped * advantage))
    actual = clipped_actor_objective(ratios, advantages, valid)
    assert float(actual) == pytest.approx(-sum(scalar_terms) / len(scalar_terms), abs=1e-7)


@pytest.mark.parametrize("representation", list(REPRESENTATIONS))
def test_async_episode_reset_and_buffer_cut_have_distinct_recurrent_behavior(policies, representation):
    policy = policies[representation]
    ended = torch.tensor([[True, False], [False, False], [False, False], [False, True]])
    obs, initial, samples, valid, ended, tail = synthetic_sequence(policy, ended=ended)
    whole = replay_action_sequence(policy, obs, initial, samples, valid, ended)
    first = replay_action_sequence(policy, obs[:2], initial, samples[:2], valid[:2], ended[:2])
    second = replay_action_sequence(policy, obs[2:], first["tail_hidden"], samples[2:], valid[2:], ended[2:])
    torch.testing.assert_close(torch.cat([first["log_prob"], second["log_prob"]]), whole["log_prob"], atol=0, rtol=0)
    torch.testing.assert_close(second["tail_hidden"], tail, atol=0, rtol=0)
    assert torch.equal(tail[1], torch.zeros_like(tail[1]))
    assert tail[0].abs().sum() > 0
    reset_at_cut = replay_action_sequence(policy, obs[2:], torch.zeros_like(first["tail_hidden"]), samples[2:], valid[2:], ended[2:])
    assert not torch.allclose(reset_at_cut["log_prob"], second["log_prob"], atol=1e-8, rtol=1e-8)


@pytest.mark.parametrize("defect", ["version", "representation", "profile", "critic_dim", "shape", "dtype", "nan", "extra_key", "missing_key"])
def test_snapshot_rejection_is_atomic(policies, defect):
    policy = policies["A1"]
    before = {key: value.clone() for key, value in policy.state_dict().items()}
    state = policy.export_diagnostic_state()
    first, last = next(iter(state["weights"])), list(state["weights"])[-1]
    state["weights"][first] += 1.  # A preceding valid update must not be applied on rejection.
    if defect == "version": state["metadata"]["state_version"] = "unrecognized"
    if defect == "representation": state = policies["A0"].export_diagnostic_state()
    if defect == "profile": state["metadata"]["profile_fingerprint"] = "0" * 64
    if defect == "critic_dim": state["metadata"]["critic_dim"] = 123
    if defect == "shape": state["weights"][last] = state["weights"][last].reshape(1, -1)
    if defect == "dtype": state["weights"][last] = state["weights"][last].double()
    if defect == "nan": state["weights"][last].fill_(float("nan"))
    if defect == "extra_key": state["weights"]["unknown"] = torch.zeros(1)
    if defect == "missing_key": del state["weights"][last]
    with pytest.raises(ValueError):
        policy.load_diagnostic_state(state)
    for key, expected in before.items():
        torch.testing.assert_close(policy.state_dict()[key], expected, atol=0, rtol=0)


@pytest.mark.parametrize("representation", list(REPRESENTATIONS))
def test_same_contract_snapshot_restore_preserves_sample_and_does_not_alias(policies, representation):
    policy = policies[representation]
    saved = policy.export_diagnostic_state()
    restored = AblationPolicy(991, policy.contract)
    restored.load_diagnostic_state(saved)
    obs, hidden = torch.zeros(2, 4, 411), torch.randn(2, 4, 128)
    expected, actual = policy.sample(obs, hidden, deterministic=True), restored.sample(obs, hidden, deterministic=True)
    torch.testing.assert_close(actual.log_prob, expected.log_prob, atol=0, rtol=0)
    torch.testing.assert_close(actual.hidden, expected.hidden, atol=0, rtol=0)
    assert torch.equal(actual.action.continuous, expected.action.continuous)
    assert torch.equal(actual.action.primitive_id, expected.action.primitive_id)
    saved["weights"][next(iter(saved["weights"]))].fill_(123.)
    for key, value in policy.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value, atol=0, rtol=0)


@pytest.mark.parametrize("representation", list(REPRESENTATIONS))
def test_legacy_ppo_rejects_preparation_policy_before_any_update(policies, representation):
    policy = policies[representation]
    before = {key: value.clone() for key, value in policy.state_dict().items()}
    with pytest.raises(ValueError, match="preparation|diagnostic|legacy PPO"):
        ppo_update(policy, None, {})
    for key, value in before.items():
        torch.testing.assert_close(policy.state_dict()[key], value, atol=0, rtol=0)
