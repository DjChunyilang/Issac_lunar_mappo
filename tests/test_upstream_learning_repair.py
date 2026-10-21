"""The independent value network must not throttle policy clipping."""
from copy import deepcopy
from types import SimpleNamespace
from pathlib import Path
import sys

import pytest
import torch

from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_training import (
    TrainingOptions, PreparedUpstreamTrainer, clip_learning_gradients,
)

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from upstream_decision_experiment import initialize_exploration
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.action_ablation import AblationPolicy, ActionContract
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_decision import action_can_affect_transition
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.learning import duration_gae
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal import upstream_training


def gradients():
    policy = torch.nn.Module()
    policy.actor = torch.nn.Linear(2, 1, bias=False)
    policy.critic = torch.nn.Linear(2, 1, bias=False)
    policy.actor.weight.grad = torch.tensor([[.12, .16]])
    policy.critic.weight.grad = torch.tensor([[150., 200.]])
    return policy


def test_independent_clipping_preserves_small_actor_gradient():
    policy = gradients()
    actor = policy.actor.weight.grad.clone()
    metrics = clip_learning_gradients(policy, TrainingOptions(gradient_clipping='actor_critic'))
    torch.testing.assert_close(policy.actor.weight.grad, actor, rtol=0, atol=0)
    assert policy.critic.weight.grad.norm() <= 1.
    assert metrics['actor_gradient_scale'] == 1.
    assert metrics['critic_gradient_scale'] == pytest.approx(.004)
    assert metrics['actor_gradient_norm'] == pytest.approx(.2)


def test_legacy_global_clipping_is_unchanged():
    policy = gradients()
    reference = deepcopy(policy)
    for p, q in zip(policy.parameters(), reference.parameters()):
        q.grad = p.grad.clone()
    expected = torch.nn.utils.clip_grad_norm_(reference.parameters(), 1.)
    metrics = clip_learning_gradients(policy, TrainingOptions())
    for p, q in zip(policy.parameters(), reference.parameters()):
        torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)
    assert metrics['gradient_norm'] == float(expected)
    assert metrics['actor_gradient_scale'] < .005


def test_resume_rejects_cross_clipping_semantics_before_mutation():
    trainer = object.__new__(PreparedUpstreamTrainer)
    trainer.collector = SimpleNamespace(metadata={'test': 'fixed'})
    trainer.options = TrainingOptions()
    legacy = deepcopy(trainer.metadata)
    assert 'gradient_clipping' not in legacy['options']
    trainer.options = TrainingOptions(gradient_clipping='actor_critic')
    assert trainer.metadata['options']['gradient_clipping'] == 'actor_critic'
    with pytest.raises(ValueError, match='identity'):
        trainer.load_state_dict({'metadata': legacy, 'updates': 1})
    with pytest.raises(ValueError, match='clipping'):
        TrainingOptions(gradient_clipping='unknown')
    trainer.options = TrainingOptions(terminal_actor_credit='effective')
    with pytest.raises(ValueError, match='identity'):
        trainer.load_state_dict({'metadata': legacy, 'updates': 1})


def test_smaller_exploration_uses_the_actual_probability_distribution():
    torch.manual_seed(17)
    policy = AblationPolicy(991, ActionContract('A1', 'a'*64, 4., 2.))
    before = deepcopy(policy.state_dict())
    initialize_exploration(policy, 1.)
    assert all(torch.equal(v, before[k]) for k, v in policy.state_dict().items())
    initialize_exploration(policy, .25)
    torch.testing.assert_close(policy.log_std.exp(), before['log_std'].exp()*.25)
    assert policy.log_std.requires_grad
    assert all(torch.equal(v, before[k]) for k, v in policy.state_dict().items() if k!='log_std')
    with torch.no_grad():
        sample = policy.sample(torch.zeros(2,4,policy.actor_dim),torch.zeros(2,4,128))
        replay, _ = policy.evaluate_sample(sample.hidden,sample)
    torch.testing.assert_close(replay,sample.log_prob,rtol=0,atol=0)
    assert torch.isfinite(replay).all()


def test_delay_boundary_mask_keeps_terminal_reward_in_gae():
    ended = torch.tensor([[True,True,True,False]])
    duration = torch.tensor([[1,3,4,1]])
    assert action_can_affect_transition(ended,duration,.5,.2).tolist()==[[False,False,True,True]]
    reward=torch.tensor([[11.5],[130.5]])
    value=torch.zeros_like(reward)
    duration=torch.tensor([[25],[1]])
    ended=torch.tensor([[False],[True]])
    adv,targets=duration_gae(reward,value,value,duration,ended,torch.zeros_like(ended),
                           gamma=.99**.2,lam=.95**.2)
    assert adv[0]>100 and targets[1]==130.5
    assert action_can_affect_transition(ended,duration,.5,.2).tolist()==[[True],[False]]


def test_value_only_terminal_batch_does_not_move_actor_with_adam_momentum(monkeypatch):
    policy=gradients()
    policy.value=lambda state:policy.critic(state).squeeze(-1)
    optimizer=torch.optim.Adam(policy.parameters(),lr=.01)
    optimizer.step()  # Populate both Adam momentum slots.
    before=policy.actor.weight.detach().clone()
    critic_before=policy.critic.weight.detach().clone()
    actor_step=optimizer.state[policy.actor.weight]['step'].clone()
    def replay(*args):
        score=policy.actor.weight.sum().expand(1,1,4)
        return {'ratio':score.exp(),'log_prob':score,'entropy':score}
    monkeypatch.setattr(upstream_training,'replay_action_sequence',replay)
    batch={'learning_valid':torch.ones(1,1,dtype=torch.bool),
           'action_effective':torch.zeros(1,1,dtype=torch.bool),
           'reward':torch.full((1,1),130.),'value':torch.zeros(1,1),'next_value':torch.zeros(1,1),
           'duration':torch.ones(1,1,dtype=torch.int64),'terminated':torch.ones(1,1,dtype=torch.bool),
           'truncated':torch.zeros(1,1,dtype=torch.bool),'bootstrap_allowed':torch.zeros(1,1,dtype=torch.bool),
           'obs':torch.zeros(1,1,4,1),'state':torch.ones(1,1,2),'hidden':torch.zeros(1,1,4,128),
           'samples':[SimpleNamespace(log_prob=torch.zeros(1,4),action=SimpleNamespace(hold=torch.zeros(1,4,dtype=torch.bool)))]}
    result=upstream_training.recurrent_mappo_update(policy,optimizer,batch,
        TrainingOptions(epochs=1,terminal_actor_credit='effective',gradient_clipping='actor_critic'))
    torch.testing.assert_close(policy.actor.weight,before,rtol=0,atol=0)
    assert torch.equal(optimizer.state[policy.actor.weight]['step'],actor_step)
    assert not torch.equal(policy.critic.weight,critic_before)
    assert result['diagnostics']['actor_transitions']==0
    assert result['diagnostics']['return_mean']==130.


@pytest.mark.parametrize('owners',[0,2])
def test_actual_collectors_and_skrl_preserve_value_only_terminal(owners):
    from run_upstream_t11 import make_collector
    from upstream_process_rollout import ProcessUpstreamRollout
    from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_decision import LearningTimeConfig
    from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_skrl import SkrlUpstreamTrainer
    torch.set_num_threads(1)
    base='configs/experiment/upstream_scale_s05_benchmark_cpu_review_v2.yaml'
    timing=LearningTimeConfig(5.,1.)
    c=(ProcessUpstreamRollout(base,'A1',23,owners=owners,envs_per_owner=1,samples=8,
                            episode_seconds=.2,cpu_workers=1,learning_time=timing) if owners else
       make_collector(base,'A1',23,num_envs=1,samples=8,episode_seconds=.2,learning_time=timing))
    try:
        batch,_,_=c.collect(1)
        assert c.base_steps==(owners or 1)
        assert batch['learning_valid'].all() and batch['terminated'].all()
        assert not batch['action_effective'].any()
        trainer=SkrlUpstreamTrainer(c,TrainingOptions(rollout_macro_steps=1,epochs=1,
            gamma=c.gamma_step,lam=c.lambda_step,terminal_actor_credit='effective',gradient_clipping='actor_critic'))
        before={k:v.clone() for k,v in c.policy.state_dict().items() if not k.startswith('critic.')}
        result=trainer.diagnostic_update(batch)
        assert result['diagnostics']['actor_transitions']==0
        assert all(torch.equal(c.policy.state_dict()[k],v) for k,v in before.items())
    finally:
        if owners:c.close()
