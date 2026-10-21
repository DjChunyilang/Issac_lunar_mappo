"""Decision-boundary, discount and budget contracts, using actual transitions."""
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from run_upstream_t11 import make_collector
from upstream_process_rollout import ProcessUpstreamRollout
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_decision import LearningTimeConfig, remaining_decisions
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_ablation_learning import UpstreamAblationRollout
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_environment import UpstreamEnvironment
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_reward import UpstreamRewardSettings
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.action_ablation import build_ablation_policies, replay_action_sequence
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.learning import duration_gae
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_training import TrainingOptions
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_skrl import SkrlUpstreamTrainer

BASE = 'configs/experiment/upstream_scale_s05_benchmark_cpu_review_v2.yaml'


def test_five_second_target_probability_and_discounted_accumulation():
    torch.set_num_threads(1)
    timing = LearningTimeConfig(5., 1.)
    c = make_collector(BASE, 'A1', 23, samples=8, num_envs=1, learning_time=timing)
    initial = c.state_dict()
    requests = []
    enqueue = c.env.enqueue
    def record(items, **kwargs):
        requests.append(items)
        return enqueue(items, **kwargs)
    c.env.enqueue = record
    batch, _, records = c.collect(1)
    assert batch['duration'].tolist() == [[25]] and len(batch['samples']) == 1
    assert c.base_steps == 25 and c.decisions == c.transitions == 1
    assert len(requests) == 5
    for second, rows in enumerate(requests):
        for agent, intent in enumerate(rows[0]):
            np.testing.assert_array_equal(intent.goal_xy, requests[0][0][agent].goal_xy)
            assert intent.timestamp == pytest.approx(second)
            assert intent.expires_at - intent.timestamp == pytest.approx(1.)
    raw = [r['reward'][0] for r in records if 'execution' in r]
    assert len(raw) == 25
    assert batch['reward'][0, 0].item() == pytest.approx(sum(c.gamma_step**k*r for k,r in enumerate(raw)), abs=2e-5)
    replay = replay_action_sequence(c.policy, batch['obs'], batch['hidden'][0], batch['samples'],
                                   batch['learning_valid'], batch['terminated'])
    torch.testing.assert_close(replay['ratio'], torch.ones_like(replay['ratio']), atol=1e-5, rtol=1e-5)
    saved = c.state_dict()
    next_batch, _, _ = c.collect(1)
    c.load_state_dict(saved)
    assert c.base_steps == 25
    again, _, _ = c.collect(1)
    torch.testing.assert_close(again['reward'], next_batch['reward'], atol=0, rtol=0)
    c.load_state_dict(initial)
    assert c.base_steps == c.decisions == 0


def test_asynchronous_terminal_retains_duration_and_bootstrap():
    from test_upstream_ablation_learning import scene
    torch.set_num_threads(1)
    cfg, poses = scene('A1')
    cfg = replace(cfg, simulation=replace(cfg.simulation, episode_length_s=6.))
    timing = LearningTimeConfig(5., 1.)
    g, l = timing.discounts(.2)
    env = UpstreamEnvironment(cfg, poses, reward_settings=UpstreamRewardSettings(gamma=g, lam=l))
    c = UpstreamAblationRollout(env, build_ablation_policies(991, cfg.spec)['A1'], timing)
    batch, episodes, _ = c.collect(1)
    assert batch['duration'].tolist() == [[1,25]]
    assert batch['terminal_reason'].tolist() == [[2,0]]
    assert batch['bootstrap_allowed'].tolist() == [[False,True]]
    assert c.base_steps == 26 and len(episodes) == 1
    assert not c.hidden[0].any() and c.hidden[1].any()
    tail = c.hidden[1].clone()
    following, _, _ = c.collect(1)
    assert following['duration'].tolist() == [[1,5]]
    assert following['terminal_reason'].tolist() == [[2,4]]
    assert torch.equal(following['hidden'][0,1], tail)
    assert not following['bootstrap_allowed'].any()


def test_discount_gae_terminal_cut_and_potential_identity():
    g, l = LearningTimeConfig(5., 1.).discounts(.2)
    reward = torch.tensor([[2.], [3.]], dtype=torch.float64)
    value = torch.tensor([[1.], [4.]], dtype=torch.float64)
    following = torch.tensor([[4.], [float('nan')]], dtype=torch.float64)
    duration = torch.tensor([[25], [3]])
    terminal = torch.tensor([[False], [True]])
    cut = torch.zeros_like(terminal)
    advantage, _ = duration_gae(reward, value, following, duration, terminal, cut, gamma=g, lam=l)
    assert advantage[1,0] == -1.
    assert advantage[0,0] == pytest.approx(2+g**25*4-1-(g*l)**25)
    cut[0] = True
    truncated, _ = duration_gae(reward, value, following, duration, terminal, cut, gamma=g, lam=l)
    assert truncated[0,0] == pytest.approx(2+g**25*4-1)
    from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_reward import UpstreamReward
    from types import SimpleNamespace
    spec = SimpleNamespace(task=SimpleNamespace(dmax=12.,dispersion=30.,hold_duration=5.),
        grid=SimpleNamespace(dt=.2),vehicle=SimpleNamespace(effective_radius=3.274))
    tracker=UpstreamReward(spec,1,UpstreamRewardSettings(gamma=g,lam=l,distance_level=0.,dispersion_level=0.))
    facts=[{'dmax':np.array([30.-k*.1]),'dispersion':np.array([70.-k*.2]),'hold_count':np.array([0.]),
            'flatness_cost':np.array([1.]),'minimum_distance':np.array([10.])} for k in range(26)]
    total=0.
    for k in range(25):
        task={'learning_valid':np.array([True]),'ended_this_step':np.array([False]),'terminal_reason':np.array(['none'])}
        _,terms=tracker.advance(facts[k],facts[k+1],task,np.zeros((1,4,2)),np.zeros((1,4,5)),np.ones((1,4)),np.zeros((1,4)))
        total+=g**k*terms['gather'][0]
    expected=-tracker.potentials(facts[0])['gather'][0]+g**25*tracker.potentials(facts[-1])['gather'][0]
    assert total == pytest.approx(expected, abs=1e-10)


def test_budget_reserves_whole_decisions():
    assert remaining_decisions(200000, 199600, 16, 25, 25) == 1
    assert remaining_decisions(200000, 199601, 16, 25, 25) == 0
    assert remaining_decisions(200000, 0, 16, 5, 125) == 125


def test_multi_owner_matches_single_and_rejects_cross_semantic_restore():
    torch.set_num_threads(1)
    timing = LearningTimeConfig(5.,1.)
    # One owner owns the same two-environment cohort as the serial collector.
    process = ProcessUpstreamRollout(BASE,'A1',23,owners=1,envs_per_owner=2,samples=8,
        episode_seconds=6.,cpu_workers=1,learning_time=timing)
    try:
        single=make_collector(BASE,'A1',23,samples=8,num_envs=2,episode_seconds=6.,learning_time=timing)
        state=process.state_dict(); rng=torch.get_rng_state()
        a,_,_=single.collect(2)
        torch.set_rng_state(rng)
        b,_,_=process.collect(2)
        for name in ('obs','hidden','duration','terminated','bootstrap_allowed','reward','next_value'):
            torch.testing.assert_close(a[name],b[name],atol=1e-5,rtol=1e-5)
        assert process.base_steps == single.base_steps == 60
        process.load_state_dict(state)
        assert process.base_steps == 0
        batch,_,_=process.collect(1)
        trainer=SkrlUpstreamTrainer(process,TrainingOptions(rollout_macro_steps=2,epochs=1,
            gamma=process.gamma_step,lam=process.lambda_step))
        result=trainer.diagnostic_update(batch)
        assert result['updates']==1 and all(np.isfinite(v) for v in result['epochs'][0].values())
        state['metadata']['learning_time']['policy_decision_period_s']=1.
        with pytest.raises(ValueError,match='protocol mismatch'):process.load_state_dict(state)
    finally:
        process.close()


def test_explicit_one_second_matches_legacy():
    torch.set_num_threads(1)
    old=make_collector(BASE,'A1',23,samples=8,num_envs=1)
    new=make_collector(BASE,'A1',23,samples=8,num_envs=1,learning_time=LearningTimeConfig())
    rng=torch.get_rng_state();a,_,_=old.collect(1)
    torch.set_rng_state(rng);b,_,_=new.collect(1)
    for k,v in a.items():
        if torch.is_tensor(v):torch.testing.assert_close(v,b[k],atol=0,rtol=0)
    with pytest.raises(ValueError,match='restore contract'):new.load_state_dict(old.state_dict())
