"""Real dense memory -> N1 shapes; sender, time and information boundaries."""
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from _common import cfg_from_experiment
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.contracts import PlannerFeedback
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_communication import (
    MESSAGE_V2,DistanceCommunicationCache,RoverBroadcast,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_observation import (
    ACTOR_DIM,CRITIC_DIM,ObservationInput,UpstreamObservationEncoder,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_sensing import DenseSensor,DenseTerrainMemory,world_to_odometry
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_planning import UpstreamIntent


@pytest.fixture(scope='module')
def cfg():return cfg_from_experiment('configs/experiment/upstream_scale_s03_observation_v1.yaml')


def scene(cfg,*,now=0.,poses=None,anchors=None):
    if poses is None:
        poses=np.zeros((2,4,3));poses[...,:2]=cfg.spec.task.slots;poses[1,:,0]+=20.
    if anchors is None:anchors=poses.copy()
    frames=DenseSensor(cfg.spec,None,map_identities=('flat_proxy_actual_sensor',)).sample(poses,anchors,[now]*2)
    memories=[[DenseTerrainMemory(cfg.spec) for _ in range(4)] for _ in range(2)]
    for e in range(2):
        for a in range(4):memories[e][a].observe(frames[e][a])
    radio=DistanceCommunicationCache(cfg.spec,anchors,message_version=MESSAGE_V2);packets=[]
    for e in range(2):
        row=[]
        for a in range(4):
            p=frames[e][a].sensor_pose;feature,_=memories[e][a].view().lookup(p[:2])
            row.append(RoverBroadcast(a,0,now,p,[0.,0.],[0.,0.],goal_xy=p[:2]+[1.,0.],
                                     version=MESSAGE_V2,terrain_features=feature[0]))
        packets.append(row)
    radio.transmit(packets,poses[...,:2],[now]*2)
    contexts=[[ObservationInput(cfg.spec,e,a,0,now,360.,frames[e][a].sensor_pose,[0.,0.],[0.,0.],
        memories[e][a].view(),radio.snapshot(e,a,now,frames[e][a].sensor_pose),
        UpstreamIntent(frames[e][a].sensor_pose[:2]+[1.,0.],False,now,now+1.)) for a in range(4)] for e in range(2)]
    return contexts,radio,memories,poses,anchors,packets


def test_real_sensor_frames_are_independent_from_actor_112_and_have_stable_dimensions(cfg):
    contexts,_,memories,poses,_,_=scene(cfg);encoder=UpstreamObservationEncoder(cfg.spec)
    observations=[[encoder.actor(c) for c in row] for row in contexts]
    for row in observations:
        for observation in row:
            assert observation.values.shape==(ACTOR_DIM,) and observation.terrain_known.all()
            assert not observation.values.flags.writeable
            assert observation.values[-1]==1.
    assert all(len(m.view().snapshot().xy)==12231 for row in memories for m in row)
    critic=encoder.critic(poses,np.zeros((2,4,2)),observations,np.zeros(2,np.int64),np.zeros(2),360.)
    assert critic.shape==(2,CRITIC_DIM) and np.isfinite(critic).all() and not critic.flags.writeable
    assert not critic[:,45:54].any() # Reserved Oracle slots.
    np.testing.assert_allclose(critic[:,32],np.sqrt(2)*7.2/12)
    np.testing.assert_allclose(critic[:,33],25.92/30)
    assert encoder.actor_tensor([]).shape==(0,411)


def test_real_n1_gru_and_critic_forward_accept_new_observations_without_network_changes(cfg):
    from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.learning import RecurrentPolicy
    contexts,_,_,poses,_,_=scene(cfg);encoder=UpstreamObservationEncoder(cfg.spec)
    observations=[[encoder.actor(c) for c in row] for row in contexts]
    actor=encoder.actor_tensor([c for row in contexts for c in row]).reshape(2,4,411)
    critic=torch.tensor(encoder.critic(poses,np.zeros((2,4,2)),observations,np.zeros(2,np.int64),[0.,0.],360.))
    policy=RecurrentPolicy(991,observation_version='local_intent_clock_v3',action_version='polar_hold_v2',rho_max=4.)
    with torch.no_grad():
        encoded=policy.encode(actor);distribution,hidden=policy.distribution(actor,torch.zeros((2,4,128)))
        values=policy.critic(critic)
    assert encoded.shape==hidden.shape==(2,4,128) and values.shape==(2,1)
    assert torch.isfinite(encoded).all() and torch.isfinite(values).all()
    # Only architecture compatibility is demonstrated; no old checkpoint is loaded.
    assert encoder.metadata['observation_version']=='dense_actor_clock_upstream_v1'


def test_velocity_and_intent_normalization_use_physical_limits_and_task_clock(cfg):
    contexts,*_=scene(cfg,now=180.);item=contexts[0][0];encoder=UpstreamObservationEncoder(cfg.spec)
    moving=replace(item,actual_velocity=[25/9,0.],previous_control=[25/9,0.],
        intent=UpstreamIntent(item.ego_pose[:2]+[4.,0.],True,180.,181.,speed_cap=25/9))
    observation=encoder.actor(moving).values
    assert observation[5]==observation[8]==observation[10]==observation[12]==1.
    assert observation[14]==1. and observation[-1]==.5
    extra=observation[295:410]
    assert extra[0]==extra[2]==extra[3]==extra[4]==extra[7]==1.
    assert extra[8]==0.


def test_full_sparse_missing_and_expired_payloads_keep_their_sender_slots(cfg):
    contexts,radio,memories,poses,anchors,packets=scene(cfg);encoder=UpstreamObservationEncoder(cfg.spec)
    poses[0,1,:2]=[65.,0.];poses[0,2,:2]=[130.,0.]
    for a in (1,2):
        p=np.r_[world_to_odometry(poses[0,a,:2][None],anchors[0,a])[0],0.]
        packets[0][a]=replace(packets[0][a],pose=p,goal_xy=p[:2]+[1.,0.],timestamp=1.2)
    packets[0][0]=replace(packets[0][0],timestamp=1.2);packets[0][3]=replace(packets[0][3],timestamp=1.2)
    radio.transmit(packets,poses[...,:2],[1.2,0.])
    item=contexts[0][0];snapshot=radio.snapshot(0,0,1.2,item.ego_pose)
    observation=encoder.actor(replace(item,now=1.2,snapshot=snapshot)).values
    for slot,sender in enumerate(snapshot.sender_indices):
        base=observation[15+17*slot:32+17*slot];extra=observation[380+10*slot:390+10*slot]
        if sender==1:assert not base[2:4].any() and not base[6:].any() and not extra[4:].any()
        if sender==2:assert not base.any() and not extra[4:].any() # Last legal stale packet, no truth fill.
    stale=radio.snapshot(0,0,3.,item.ego_pose)
    expired=encoder.actor(replace(item,now=3.,snapshot=stale)).values
    assert not expired[15:66].any() and not expired[290:295].any()


def test_sender_slot_crossing_keeps_base_and_goal_payload_from_same_message(cfg):
    contexts,radio,_,_,_,_=scene(cfg);encoder=UpstreamObservationEncoder(cfg.spec);item=contexts[0][0]
    pose=item.ego_pose.copy();pose[:2]+=[7.,0.]
    snapshot=radio.snapshot(0,0,0.,pose)
    observation=encoder.actor(replace(item,ego_pose=pose,snapshot=snapshot)).values
    for slot,message in enumerate(snapshot.messages):
        expected=(message.goal_xy-pose[:2])/cfg.spec.communication.full_radius
        np.testing.assert_allclose(observation[27+17*slot:29+17*slot],expected,atol=1e-7)
        np.testing.assert_allclose(observation[384+10*slot:386+10*slot],expected,atol=1e-7)


def test_v2_sender_terrain_gradients_transform_to_receiver_and_sparse_strips_them(cfg):
    anchors=np.zeros((1,4,3));anchors[0,0,2]=np.pi/2
    radio=DistanceCommunicationCache(cfg.spec,anchors,message_version=MESSAGE_V2)
    poses=np.array([[[0.,0.],[10.,0.],[70.,0.],[130.,0.]]])
    packets=[[RoverBroadcast(a,0,0.,np.r_[poses[0,a],0.],[0.,0.],[0.,0.],version=MESSAGE_V2,
              terrain_features=[20.,.1,0.,.01,.7]) for a in range(4)]]
    radio.transmit(packets,poses,[0.]);snapshot=radio.snapshot(0,0,0.,[0.,0.,0.])
    np.testing.assert_allclose(snapshot.messages[0].terrain_features,[20.,0.,-.1,.01,.7],atol=1e-12)
    assert not snapshot.messages[0].terrain_features.flags.writeable
    assert snapshot.messages[1].terrain_features is None
    old=DistanceCommunicationCache(cfg.spec,anchors)
    with pytest.raises(ValueError):old.transmit(packets,poses,[0.])
    with pytest.raises(ValueError,match='requires explicit'):RoverBroadcast(0,0,0.,np.zeros(3),[0.,0.],[0.,0.],terrain_features=[0.,0.,0.,0.,1.])
    restored=DistanceCommunicationCache(cfg.spec,anchors,message_version=MESSAGE_V2);restored.load_state_dict(radio.state_dict())
    with pytest.raises(ValueError,match='mismatch'):old.load_state_dict(radio.state_dict())
    np.testing.assert_array_equal(restored.snapshot(0,0,0.,[0.,0.,0.]).messages[0].terrain_features,snapshot.messages[0].terrain_features)


@pytest.mark.parametrize('damage',['receiver','time','version','sender','validity','future_terrain','velocity','feedback'])
def test_invalid_information_is_rejected_before_encoding(cfg,damage):
    contexts,*_=scene(cfg);item=contexts[0][0];fields={}
    if damage=='receiver':fields['snapshot']=replace(item.snapshot,receiver_id=1)
    if damage=='time':fields['now']=.2
    if damage=='version':
        messages=list(item.snapshot.messages);messages[0]=replace(messages[0],version='sender_message_upstream_v1',terrain_features=None)
        fields['snapshot']=replace(item.snapshot,messages=tuple(messages))
    if damage=='sender':fields['snapshot']=replace(item.snapshot,sender_indices=np.array([3,2,1]))
    if damage=='validity':fields['snapshot']=replace(item.snapshot,state_valid=np.zeros(3,bool))
    if damage=='future_terrain':
        from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_footprint import FootprintTerrain
        snapshot=item.terrain.snapshot();fields['terrain']=FootprintTerrain(replace(snapshot,observed_at=np.ones(len(snapshot.xy))),cfg.spec)
    if damage=='velocity':fields['actual_velocity']=[.1,0.]
    if damage=='feedback':fields['feedback']=PlannerFeedback(status='not_a_status')
    with pytest.raises(ValueError):replace(item,**fields)


def test_actor_same_legal_inputs_ignore_uncommunicated_truth_and_other_environment(cfg):
    contexts,radio,_,poses,_,_=scene(cfg);encoder=UpstreamObservationEncoder(cfg.spec);item=contexts[0][0]
    before=encoder.actor(item).values
    poses[1,...,:2]=500.;poses[0,3,:2]=900.
    radio.transmit([[None]*4,[None]*4],poses[...,:2],[0.,0.])
    after=encoder.actor(replace(item,snapshot=radio.snapshot(0,0,0.,item.ego_pose))).values
    np.testing.assert_array_equal(before,after)


def test_actor_is_se2_invariant_under_joint_world_and_odometry_frame_transform(cfg):
    contexts,_,_,poses,anchors,_=scene(cfg);encoder=UpstreamObservationEncoder(cfg.spec)
    old=np.stack([encoder.actor(c).values for row in contexts for c in row])
    angle=.6;c,s=np.cos(angle),np.sin(angle);rotation=np.array([[c,s],[-s,c]])
    transformed=poses.copy();transformed[...,:2]=poses[...,:2]@rotation+[30.,-20.];transformed[...,2]+=angle
    frames=anchors.copy();frames[...,:2]=anchors[...,:2]@rotation+[30.,-20.];frames[...,2]+=angle
    shifted,*_=scene(cfg,poses=transformed,anchors=frames)
    new=np.stack([encoder.actor(c).values for row in shifted for c in row])
    np.testing.assert_allclose(new,old,atol=1e-6,rtol=1e-6)


def test_actual_hold_count_is_critic_only_and_invalid_critic_time_rejected(cfg):
    contexts,_,_,poses,_,_=scene(cfg,now=6.);encoder=UpstreamObservationEncoder(cfg.spec)
    observations=[[encoder.actor(c) for c in row] for row in contexts];actor=np.stack([o.values for row in observations for o in row])
    a=encoder.critic(poses,np.zeros((2,4,2)),observations,np.array([0,0]),[6.,6.],360.)
    b=encoder.critic(poses,np.zeros((2,4,2)),observations,np.array([25,25]),[6.,6.],360.)
    changed=np.flatnonzero(a[0]!=b[0]);assert changed.tolist()==[39]
    np.testing.assert_array_equal(actor,np.stack([encoder.actor(c).values for row in contexts for c in row]))
    with pytest.raises(ValueError,match='time mismatch'):encoder.critic(poses,np.zeros((2,4,2)),observations,np.array([0,0]),[6.,6.2],360.)


def test_empty_private_memory_is_unknown_risk_without_global_fill(cfg):
    contexts,*_=scene(cfg);encoder=UpstreamObservationEncoder(cfg.spec);item=contexts[0][0]
    empty=DenseTerrainMemory(cfg.spec).view()
    observation=encoder.actor(replace(item,terrain=empty))
    grids=observation.values[66:290].reshape(112,2)
    assert not observation.terrain_known.any() and not grids[:,0].any() and (grids[:,1]==1.).all()
    assert observation.values[304]==0. # Goal patch known fraction.


def test_all_native_feedback_indices_preserve_network_layout(cfg):
    contexts,*_=scene(cfg);encoder=UpstreamObservationEncoder(cfg.spec)
    for index in range(47):
        observation=encoder.actor(replace(contexts[0][0],feedback=PlannerFeedback(status=f'primitive:{index}')))
        assert observation.values.shape==(411,)
    with pytest.raises(ValueError):replace(contexts[0][0],feedback=PlannerFeedback(status='primitive:47'))


@pytest.mark.parametrize('channel,value',[(1,.2),(3,.06),(4,0.)])
def test_hard_local_hazards_cannot_be_hidden_by_a_finite_traversability_proxy(cfg,channel,value):
    from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_footprint import FootprintTerrain
    contexts,*_=scene(cfg);item=contexts[0][0];snapshot=item.terrain.snapshot()
    features=snapshot.features.copy();features[:,channel]=value
    unsafe=FootprintTerrain(replace(snapshot,features=features),cfg.spec,query_version='batch_ties_v2')
    observation=UpstreamObservationEncoder(cfg.spec).actor(replace(item,terrain=unsafe))
    assert observation.terrain_known.all()
    assert (observation.values[66:290].reshape(112,2)[:,1]==1.).all()
