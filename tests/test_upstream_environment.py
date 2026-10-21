"""End-to-end actual sensing, queue, execution, finite reward and macro cuts."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from _common import cfg_from_experiment
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_environment import UpstreamEnvironment
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_reward import UpstreamReward


def configuration(deadline=360.):
    cfg=cfg_from_experiment('configs/experiment/upstream_scale_s03_environment_v1.yaml')
    return replace(cfg,simulation=replace(cfg.simulation,episode_length_s=deadline),mppi_samples=32)


def square(cfg):
    p=np.zeros((2,4,3));p[...,:2]=cfg.spec.task.slots;p[1,:,0]+=20.
    return p


def holds(env):
    return env.polar_requests(np.zeros((2,4)),np.zeros((2,4)),np.ones((2,4),bool))


def test_actual_queue_cut_deadline_and_once_only_returns():
    cfg=configuration(.8);env=UpstreamEnvironment(cfg,square(cfg))
    first=env.step_intents(holds(env),max_steps=2)
    assert first.duration.tolist()==[2,2] and first.bootstrap_allowed.all()
    assert len(env.pending)==8 and not any(env.intents[0])
    np.testing.assert_allclose(env.core.times,.4)
    last=env.step_intents(None)
    assert last.duration.tolist()==[2,2] and last.terminated.all()
    assert not last.bootstrap_allowed.any() and last.learning_valid.all()
    assert [r['terminal_reason'] for r in last.episodes]==['task_deadline']*2
    assert len(env.pending)==0 and len(last.episodes)==2
    for e in range(2):
        rewards=[r['reward'][e] for r in first.records+last.records if 'reward' in r]
        assert last.episodes[e]['raw_return']==pytest.approx(sum(rewards))
        assert last.episodes[e]['discounted_return']==pytest.approx(sum(.99**i*r for i,r in enumerate(rewards)))
    stopped=env.step_intents(None)
    assert not stopped.duration.any() and not stopped.reward.any() and not stopped.episodes


def test_mid_queue_restore_actual_feedback_and_async_reset_isolation():
    cfg=configuration();p=square(cfg);original=UpstreamEnvironment(cfg,p)
    original.step_intents(holds(original),max_steps=1);saved=original.state_dict()
    restored=UpstreamEnvironment(cfg,p);restored.load_state_dict(saved)
    assert len(restored.pending)==8
    a=original.step_intents(None,max_steps=2);b=restored.step_intents(None,max_steps=2)
    assert torch.equal(a.actor,b.actor) and torch.equal(a.reward,b.reward)
    np.testing.assert_array_equal(original.core.previous_control,restored.core.previous_control)
    # Exactly the selected environment loses its queue, episode and warm state.
    peer_time=restored.core.times[1];peer_return=restored.reward_tracker.raw_return[1]
    restored.reset([0])
    assert restored.core.times[0]==0 and restored.core.times[1]==peer_time
    assert restored.reward_tracker.raw_return[0]==0 and restored.reward_tracker.raw_return[1]==peer_return
    assert restored.core.episodes.tolist()==[1,0]
    bad=deepcopy(saved);bad['ended'][0]=True
    with pytest.raises(ValueError,match='disagree'):restored.load_state_dict(bad)
    assert restored.core.episodes.tolist()==[1,0]


def test_invalid_terrain_tail_does_not_bootstrap_or_enter_learning():
    cfg=configuration();env=UpstreamEnvironment(cfg,square(cfg))
    for m in env.memory[0]:m.reset()
    env._refresh=lambda active:None
    transition=env.step_intents(None,max_steps=1)
    assert transition.terminal_reason.tolist()==[5,0]
    assert transition.learning_valid.tolist()==[False,True]
    assert transition.bootstrap_allowed.tolist()==[False,True]
    assert transition.reward[0]==0 and not transition.terminal_actor[0].any()
    assert transition.episodes[0]['terminal_reason']=='environment_invalid'


def test_raw_request_expiry_is_not_extended_by_injected_delay():
    cfg=configuration();env=UpstreamEnvironment(cfg,square(cfg))
    env.step_intents(holds(env),max_steps=3)
    assert not env.pending and env.intents[0][0].timestamp==0.
    assert env.intents[0][0].expires_at==1.
    env.step_intents(None,max_steps=3)
    assert env._nominal(0,0)[1]=='expired_or_missing_intent'
    assert env.core.times[0]==pytest.approx(1.2)


def test_discounted_hold_cycle_telescopes_without_repeated_entry_gain():
    cfg=configuration();tracker=UpstreamReward(cfg.spec,2)
    base={'dmax':np.full(2,12.),'dispersion':np.full(2,30.),'flatness_cost':np.ones(2),
          'minimum_distance':np.full(2,8.),'hold_count':np.zeros(2)}
    sequence=[0,7,0,12,0,25]
    sums=np.zeros(2);previous=deepcopy(base)
    task={'learning_valid':np.ones(2,bool),'ended_this_step':np.zeros(2,bool),'terminal_reason':np.array(['none']*2,dtype=object)}
    f=np.zeros((2,4,5));f[...,4]=1.
    for t,count in enumerate(sequence[1:]):
        following=deepcopy(base);following['hold_count'][:]=count
        _,terms=tracker.advance(previous,following,task,np.zeros((2,4,2)),f,np.ones((2,4)),np.zeros((2,4)))
        sums+=.99**t*terms['success_hold'];previous=following
    expected=.99**(len(sequence)-1)*tracker.potentials(previous)['success_hold']-tracker.potentials(base)['success_hold']
    np.testing.assert_allclose(sums,expected,atol=1e-12)
    task['ended_this_step'][:]=True;task['terminal_reason'][:]='task_deadline'
    _,terms=tracker.advance(previous,previous,task,np.zeros((2,4,2)),f,np.ones((2,4)),np.zeros((2,4)))
    np.testing.assert_array_equal(terms['terminal'],[-80.,-80.])


def test_v2_acceptance_creates_distinct_lease_and_preserves_original_ticket():
    cfg=replace(configuration(),environment_execution_version='actual_loop_upstream_v2')
    env=UpstreamEnvironment(cfg,square(cfg));requests=holds(env)
    result=env.step_intents(requests,max_steps=3)
    assert env.intents[0][0].timestamp==pytest.approx(.6)
    assert env.intents[0][0].expires_at==pytest.approx(1.6)
    assert requests[0][0].timestamp==0. and requests[0][0].expires_at==1.
    event=next(r for r in result.records if r.get('event')=='request_activation')
    assert event['accepted'] and event['original_intent'].expires_at==1.
    assert event['active_intent'].expires_at==pytest.approx(1.6)


def test_actual_recurrent_cut_restore_and_one_diagnostic_ppo_update():
    from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.learning import RecurrentPolicy,ppo_update
    from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_rollout import UpstreamRollout
    torch.set_num_threads(1);torch.manual_seed(23)
    cfg=configuration();env=UpstreamEnvironment(cfg,square(cfg))
    policy=RecurrentPolicy(991,observation_version='local_intent_clock_v3',action_version='polar_hold_v2',rho_max=4.)
    with torch.no_grad():policy.hold_logit.bias.fill_(40.)
    collector=UpstreamRollout(env,policy);batch,_,_=collector.collect(1)
    assert batch['duration'].tolist()==[[5,5]] and batch['bootstrap_allowed'].all()
    assert collector.hidden.abs().sum()>0 and batch['hold'].all()
    saved=collector.state_dict();tail=collector.hidden.clone()
    next_batch,_,_=collector.collect(1)
    assert torch.equal(next_batch['hidden'][0],tail) # A buffer boundary keeps GRU state.
    new_policy=RecurrentPolicy(991,observation_version='local_intent_clock_v3',action_version='polar_hold_v2',rho_max=4.)
    restored=UpstreamRollout(UpstreamEnvironment(cfg,square(cfg)),new_policy);restored.load_state_dict(saved)
    replay,_,_=restored.collect(1)
    assert torch.equal(next_batch['action'],replay['action'])
    assert torch.allclose(next_batch['log_prob'],replay['log_prob'],atol=1e-6,rtol=1e-6)
    assert torch.equal(next_batch['reward'],replay['reward'])
    before=new_policy.critic[0].weight.detach().clone()
    optimizer=torch.optim.Adam(new_policy.parameters(),lr=1e-4)
    loss=ppo_update(new_policy,optimizer,replay,epochs=1)
    assert all(np.isfinite(value) for value in loss.values())
    assert not torch.equal(before,new_policy.critic[0].weight)


def test_safety_actual_control_broadcast_and_asynchronous_restore():
    cfg=replace(configuration(),environment_execution_version='actual_loop_upstream_v2',
                execution_safety_version='footprint_discrete_cbf_upstream_v1')
    env=UpstreamEnvironment(cfg,square(cfg))
    for e in range(2):
        for a in range(4):env.pledge_speed(e,a,0.,12.)
    env.step_intents(None,max_steps=2)
    snapshot=env.radio.snapshot(0,0,float(env.core.times[0]),env.local_pose(0,0))
    assert len(snapshot.messages)==3
    assert all(m.trajectory is not None and m.trajectory.controls.shape==(40,2) for m in snapshot.messages)
    for m in snapshot.messages:
        np.testing.assert_array_equal(m.applied_control,env.core.previous_control[0,m.sender_id])
        assert m.motion_bound.speed==0.
    saved=env.state_dict();fresh=UpstreamEnvironment(cfg,square(cfg));fresh.load_state_dict(saved)
    np.testing.assert_array_equal(env.step_intents(None,max_steps=1).reward,fresh.step_intents(None,max_steps=1).reward)
    fresh.reset([0]);assert all(p.last_control is None for p in fresh.safety[0])
    assert all(p.last_control is not None for p in fresh.safety[1])


def test_repeated_dense_frame_fast_path_matches_full_hazard_merge():
    from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_sensing import DenseSensor,DenseTerrainMemory
    cfg=configuration();p=square(cfg);frames=DenseSensor(cfg.spec,None).sample(p,p,[0.,0.])
    frame=frames[0][0];a=DenseTerrainMemory(cfg.spec);b=DenseTerrainMemory(cfg.spec)
    a.observe(frame);b.observe(frame);later=replace(frame,timestamp=.2)
    b._view=None # Force the complete reference merge, retaining identical rows.
    a.observe(later);b.observe(later)
    for key in ('xy','features','observed_at'):np.testing.assert_array_equal(a.state_dict()[key],b.state_dict()[key])
    changed=later.features.copy();changed[10,4]=0.
    hazard=replace(later,features=changed,timestamp=.4)
    a.observe(hazard);b.observe(hazard)
    for key in ('xy','features','observed_at'):np.testing.assert_array_equal(a.state_dict()[key],b.state_dict()[key])
