"""Original reward units, one moment update per rollout, and exact restoration."""
from copy import deepcopy
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from run_upstream_t11 import make_collector
from upstream_process_rollout import ProcessUpstreamRollout
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.action_ablation import AblationPolicy, ActionContract
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.learning import bootstrap_values, duration_gae
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_decision import LearningTimeConfig
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_skrl import SkrlUpstreamTrainer
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_training import TrainingOptions, recurrent_mappo_update
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_value_normalization import RunningValueNormalization


def test_running_moments_partition_roundtrip_and_policy_identity():
    values = torch.tensor([-24., -3., 0., 12., 130.])
    n = RunningValueNormalization()
    torch.testing.assert_close(n.denormalize(values), values, rtol=0, atol=0)
    n.update(values[:2]); n.update(values[2:])
    assert int(n.count) == 5
    torch.testing.assert_close(n.mean, values.double().mean())
    torch.testing.assert_close(n.std, values.double().std(unbiased=False))
    torch.testing.assert_close(n.denormalize(n.normalize(values)), values, atol=1e-5, rtol=1e-6)
    policy = AblationPolicy(991, ActionContract('A1', 'a'*64, 4., 2.))
    old = policy.export_diagnostic_state()
    policy.enable_value_normalization()
    with pytest.raises(ValueError, match='contract'):
        policy.load_diagnostic_state(old)
    policy.value_normalizer.load_state_dict(n.state_dict())
    state = torch.randn(2, 991)
    raw = policy.critic(state).squeeze(-1)*n.std.float()+n.mean.float()
    torch.testing.assert_close(policy.value(state), raw)
    bootstrap = bootstrap_values(policy, state, torch.tensor([True, False]))
    torch.testing.assert_close(bootstrap[0], raw[0])
    assert bootstrap[1] == 0.
    saved = policy.export_diagnostic_state()
    policy.value_normalizer.update(torch.tensor([-1000.]))
    policy.load_diagnostic_state(saved)
    torch.testing.assert_close(policy.value(state), raw, rtol=0, atol=0)
    assert 'value_normalization' not in TrainingOptions().metadata


def test_gae_is_raw_actor_unchanged_and_invalid_targets_excluded():
    torch.set_num_threads(1); torch.manual_seed(23)
    original = AblationPolicy(991, ActionContract('A1', 'a'*64, 4., 2.))
    normalized = deepcopy(original); normalized.enable_value_normalization()
    obs = torch.randn(2, 2, 4, original.actor_dim)
    states = torch.randn(2, 2, 991)
    hidden = torch.zeros(2, 4, 128); hidden_rows = []; samples = []
    valid = torch.tensor([[True, True], [True, False]])
    terminated = torch.tensor([[False, True], [True, True]])
    with torch.no_grad():
        for index in range(2):
            hidden_rows.append(hidden.clone())
            sample = original.sample(obs[index], hidden); samples.append(sample)
            hidden = torch.where(terminated[index, :, None, None], 0., sample.hidden)
        values = original.value(states)
    batch = dict(obs=obs, state=states, hidden=torch.stack(hidden_rows), samples=samples,
        learning_valid=valid, action_effective=valid, reward=torch.tensor([[-5., 130.], [2., 1e9]]),
        value=values, next_value=torch.zeros_like(values), duration=torch.tensor([[25, 1], [25, 0]]),
        terminated=terminated, truncated=torch.zeros_like(valid), bootstrap_allowed=~terminated)
    args = dict(epochs=3, gradient_clipping='actor_critic', terminal_actor_credit='effective',
                gamma=.99**.2, lam=.95**.2)
    _, returns = duration_gae(batch['reward'], values, batch['next_value'], batch['duration'],
        terminated, batch['truncated'], gamma=args['gamma'], lam=args['lam'],
        bootstrap_allowed=batch['bootstrap_allowed'], learning_valid=valid)
    rng = torch.get_rng_state().clone()  # Entropy estimation draws latent samples.
    for policy, setting in ((original, 'none'), (normalized, 'running')):
        torch.set_rng_state(rng)
        result = recurrent_mappo_update(policy, torch.optim.Adam(policy.parameters(), lr=1e-4),
                                       batch, TrainingOptions(**args, value_normalization=setting))
    assert int(normalized.value_normalizer.count) == 3  # Not 3 epochs x 4 rovers.
    torch.testing.assert_close(normalized.value_normalizer.mean, returns[valid].double().mean())
    assert result['diagnostics']['return_mean'] == float(returns[valid].mean())
    assert result['epochs'][0]['ratio_min'] == result['epochs'][0]['ratio_max'] == 1.
    for name, parameter in original.named_parameters():
        if not name.startswith('critic.'):
            torch.testing.assert_close(parameter, dict(normalized.named_parameters())[name], rtol=0, atol=0)


@pytest.mark.parametrize('owners', [0, 4])
def test_actual_skrl_collectors_restore_value_units(owners):
    # 1 + 16 actual physical steps across both variants; no repeated episodes.
    torch.set_num_threads(1)
    base = 'configs/experiment/upstream_scale_s05_benchmark_cpu_review_v2.yaml'
    timing = LearningTimeConfig(5., 1.)
    c = (ProcessUpstreamRollout(base, 'A1', 23, owners=4, envs_per_owner=4, samples=8,
            episode_seconds=.2, cpu_workers=1, learning_time=timing) if owners else
         make_collector(base, 'A1', 23, num_envs=1, samples=8, episode_seconds=.2, learning_time=timing))
    try:
        options = TrainingOptions(rollout_macro_steps=1, epochs=2, gamma=c.gamma_step, lam=c.lambda_step,
            gradient_clipping='actor_critic', terminal_actor_credit='effective', value_normalization='running')
        trainer = SkrlUpstreamTrainer(c, options)
        batch, _, _ = trainer.collect()
        reward = batch['reward'].clone()
        assert c.base_steps == (16 if owners else 1)
        assert batch['terminated'].all() and torch.count_nonzero(batch['next_value']) == 0
        trainer.diagnostic_update(batch)
        assert int(c.policy.value_normalizer.count) == c.env.e
        torch.testing.assert_close(batch['reward'], reward, rtol=0, atol=0)
        with torch.no_grad():
            expected = c.policy.value(batch['state'])
        saved = trainer.state_dict()
        c.policy.value_normalizer.update(torch.tensor([10000.]))
        trainer.load_state_dict(saved)
        torch.testing.assert_close(c.policy.value(batch['state']), expected, rtol=0, atol=0)
        assert trainer.updates == 2
        incompatible = deepcopy(saved)
        incompatible['metadata']['options'].pop('value_normalization')
        with pytest.raises(ValueError, match='identity'):
            trainer.load_state_dict(incompatible)
    finally:
        if owners:
            c.close()
