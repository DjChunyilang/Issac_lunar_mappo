"""Local parking exploration legality, likelihood replay and persistence."""
from copy import deepcopy
from dataclasses import replace
import itertools
import math
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from _common import cfg_from_experiment
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.action_ablation import (
    ActionContract, AblationPolicy, replay_action_sequence,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_parking_exploration import (
    ParkingExploration, exploration_strength,
)


SETTINGS = ParkingExploration(300., 12., 30.)


def local_observations(side=7.2, yaw=0.):
    xy = torch.tensor([[-.5, -.5], [-.5, .5], [.5, -.5], [.5, .5]]) * side
    rotation = torch.tensor([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
    xy = xy @ rotation.T
    obs = torch.zeros(1, 4, 411)
    for agent in range(4):
        relative = xy[torch.arange(4) != agent] - xy[agent]
        row = obs[0, agent, 15:66].reshape(3, 17)
        row[:, :2] = relative.sign() * torch.log1p(relative.abs() / 12.) / math.log1p(300. / 12.)
        row[:, 11] = 1.
        obs[0, agent, [381, 391, 401]] = 1.
    return obs


@pytest.fixture
def policy():
    torch.set_num_threads(1)
    torch.manual_seed(23)
    spec = cfg_from_experiment(ROOT / 'configs/experiment/upstream_1km_curriculum_v1.yaml').spec
    p = AblationPolicy(991, ActionContract.from_spec('A1', spec))
    with torch.no_grad():
        p.hold_logit.weight.zero_()
        p.hold_logit.bias.fill_(math.log(.35 / .65))
    return p


def test_geometry_is_local_rotation_invariant_and_requires_all_full_fresh_neighbors():
    for angle in (0., .7, math.pi / 2, math.pi):
        near = local_observations(yaw=angle)
        assert SETTINGS.compact_geometry(near).all()
        assert not SETTINGS.compact_geometry(local_observations(12., angle)).any()
        # No dependence on another rover's observation or an environment ID.
        changed = near.clone()
        changed[:, 0] = local_observations(20.)[:, 0]
        assert SETTINGS.compact_geometry(changed).tolist() == [[False, True, True, True]]
        neighbors = near[..., 15:66].reshape(1, 4, 3, 17)
        neighbors[:] = neighbors[..., [2, 0, 1], :].clone()
        assert SETTINGS.compact_geometry(near).all()
    for field in (381, 391, 401, 15 + 11, 32 + 11, 49 + 11):
        obs = local_observations()
        obs[..., field] = 0.
        assert not SETTINGS.compact_geometry(obs).any()
    obs = local_observations()
    obs[..., 381] = 0.
    obs[..., 15:17] = 1e20  # Must mask before inverse logarithm.
    assert torch.isfinite(SETTINGS.logit_bias(obs)).all()
    assert not SETTINGS.compact_geometry(obs).any()
    assert SETTINGS.compact_geometry(local_observations(7.7)).all()
    assert not SETTINGS.compact_geometry(local_observations(7.8)).any()  # Dispersion > 30.
    # Dispersion alone can pass while the maximum pair distance fails.
    obs = local_observations()
    relative = torch.tensor([[12.1, 0.], [6., .1], [6., -.1]])
    obs[..., 15:66].reshape(1, 4, 3, 17)[..., :2] = (
        relative.sign() * torch.log1p(relative.abs() / 12.) / math.log1p(300. / 12.))
    assert not SETTINGS.compact_geometry(obs).any()


def test_prior_changes_only_hold_probability_and_has_independent_joint_likelihood(policy):
    obs = local_observations()
    h = torch.zeros(1, 4, 128)
    original, _ = policy.distribution(obs, h)
    policy.set_parking_exploration(SETTINGS)
    start, following = policy.distribution(obs, h)
    continued = obs.clone(); continued[..., 14] = 1.
    continuing = policy.action_distribution(following, observation=continued)
    torch.testing.assert_close(start.normal.loc, original.normal.loc)
    torch.testing.assert_close(start.normal.scale, original.normal.scale)
    torch.testing.assert_close(start.logits, original.logits + 2.4)
    torch.testing.assert_close(continuing.logits, original.logits + 4.)
    assert .85 < start.branch.probs.mean() < .86
    assert .96 < continuing.branch.probs.mean() < .98
    probability_sum = 0.
    for bits in itertools.product((0., 1.), repeat=4):
        probability_sum += start.branch.log_prob(torch.tensor([bits])).sum().exp()
    torch.testing.assert_close(probability_sum, torch.tensor(1.))
    all_hold = start.branch.log_prob(torch.ones(1, 4)).sum().exp()
    torch.testing.assert_close(all_hold, start.branch.probs.prod())
    assert .53 < all_hold < .55
    # Positive HOLD credit must retain a gradient into the learned head.
    (-start.log_prob(torch.zeros(1, 4, 2), torch.ones(1, 4, dtype=torch.bool)).mean()).backward()
    assert policy.hold_logit.bias.grad.item() < 0


def test_zero_strength_is_exact_legacy_including_rng_and_gru(policy):
    original = deepcopy(policy)
    policy.set_parking_exploration(replace(SETTINGS, strength=0.))
    obs = local_observations(); h = torch.zeros(1, 4, 128)
    torch.manual_seed(71)
    expected = original.sample(obs, h); expected_rng = torch.get_rng_state()
    torch.manual_seed(71)
    actual = policy.sample(obs, h)
    for name in ('latent', 'log_prob', 'hidden'):
        torch.testing.assert_close(getattr(actual, name), getattr(expected, name), atol=0, rtol=0)
    assert torch.equal(actual.action.hold, expected.action.hold)
    assert torch.equal(actual.action.continuous, expected.action.continuous)
    assert torch.equal(torch.get_rng_state(), expected_rng)


def test_recurrent_replay_uses_same_observation_bias_and_resets(policy):
    policy.set_parking_exploration(SETTINGS)
    obs = torch.stack([local_observations().repeat(2, 1, 1) for _ in range(4)])
    obs[1:, ..., 14] = 1.
    obs[2, 1] = local_observations(12.)[0]
    ended = torch.tensor([[False, False], [True, False], [False, True], [False, False]])
    valid = torch.ones_like(ended)
    initial = torch.zeros(2, 4, 128); hidden = initial.clone(); samples = []
    with torch.no_grad():
        for step in range(4):
            sample = policy.sample(obs[step], hidden)
            samples.append(sample)
            hidden = torch.where(ended[step, :, None, None], 0., sample.hidden)
    replay = replay_action_sequence(policy, obs, initial, samples, valid, ended)
    torch.testing.assert_close(replay['ratio'], torch.ones(4, 2, 4), atol=0, rtol=0)
    torch.testing.assert_close(replay['tail_hidden'], hidden, atol=0, rtol=0)
    with pytest.raises(ValueError, match='original observation'):
        policy.evaluate_sample(samples[0].hidden, samples[0])
    saved = policy.export_diagnostic_state()
    policy.load_diagnostic_state(saved)
    policy.set_parking_exploration(replace(SETTINGS, strength=0.))
    with pytest.raises(ValueError, match='contract/version'):
        policy.load_diagnostic_state(saved)
    wrong = replay_action_sequence(policy, obs, initial, samples, valid, ended)
    assert (wrong['ratio'] - 1.).abs().max() > .1


def test_schedule_uses_actual_physical_steps_and_is_finite():
    assert [exploration_strength(x) for x in (0, 3000, 5500, 8000, 10000)] == [1., 1., .5, 0., 0.]
    with pytest.raises(ValueError):
        exploration_strength(-1)
