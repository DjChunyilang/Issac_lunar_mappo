"""Legal square layout, stopping capture, latched braking and information waits."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from _common import cfg_from_experiment
from diagnose_upstream_execution import synthetic_local_frame
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_communication import DistanceCommunicationCache,RoverBroadcast
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_gathering import LeaderSquareScript
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_footprint import FootprintTerrain


@pytest.fixture(scope='module')
def cfg():return cfg_from_experiment('configs/experiment/upstream_scale_s03.yaml')


@pytest.fixture(scope='module')
def terrain(cfg):return synthetic_local_frame(np.zeros(3),cfg.spec,0.,query_version='batch_ties_v2')


def messages(cfg,agent,pose,now=0.,*,leader_payload=True,leader_full=True,peer_positions=None):
    positions=np.array([[[0.,0.],[0.,7.2],[7.2,0.],[7.2,7.2]]]);positions[0,agent]=pose[:2]
    if peer_positions is not None:
        for sender,value in peer_positions.items():positions[0,sender]=value
    cache=DistanceCommunicationCache(cfg.spec,np.zeros((1,4,3)))
    packets=[[RoverBroadcast(i,0,now,np.r_[positions[0,i],0.],[0.,0.],[0.,0.],True,
        site_center=[3.6,3.6] if i==0 and leader_payload else None,
        layout_yaw=0. if i==0 and leader_payload else None) for i in range(4)]]
    if not leader_full:positions[0,0]=[0.,70.]
    cache.transmit(packets,positions,[now])
    return cache.snapshot(0,agent,now,pose)


def decide(script,cfg,terrain,pose,control=(0.,0.),now=0.,**kwargs):
    return script.decide(pose,control,control,terrain,messages(cfg,script.agent_id,pose,now,**kwargs),now,0)


def test_leader_is_corner_and_publishes_distinct_center(cfg,terrain):
    script=LeaderSquareScript(cfg.spec,0,0);pose=np.array([1.,2.,np.pi/2])
    result=decide(script,cfg,terrain,pose)
    assert result.intent.hold and result.phase=='HOLD'
    np.testing.assert_allclose(result.site_center,pose[:2]+[-3.6,3.6],atol=1e-12)
    np.testing.assert_allclose(result.slot_xy,pose[:2],atol=1e-12)
    later=decide(script,cfg,terrain,pose+[.1,0.,0.],now=1.)
    np.testing.assert_array_equal(later.intent.goal_xy,pose[:2])
    np.testing.assert_array_equal(later.site_center,result.site_center)


def test_rolling_polar_decode_preserves_four_meter_range(cfg,terrain):
    script=LeaderSquareScript(cfg.spec,0,1);pose=[0.,-5.,np.pi]
    result=decide(script,cfg,terrain,pose)
    assert result.phase=='APPROACH' and not result.intent.hold and not result.intent.stop_at_goal
    assert np.linalg.norm(result.intent.goal_xy-np.asarray(pose[:2]))==pytest.approx(4.)
    np.testing.assert_allclose(result.intent.goal_xy,[0.,-1.],atol=1e-12)
    np.testing.assert_allclose(result.slot_xy,[0.,7.2],atol=1e-12)
    assert result.intent.expires_at==1.


def test_stopping_endpoint_capture_latches_until_actual_stop(cfg,terrain):
    script=LeaderSquareScript(cfg.spec,0,2)
    first=decide(script,cfg,terrain,[7.13,0.,0.],(.2,0.))
    assert first.phase=='BRAKING' and first.intent.hold and first.stop_prediction_verified
    assert np.linalg.norm(first.stopping_endpoint[:2]-[7.2,0.])<=.2
    noisy=decide(script,cfg,terrain,[7.0,.25,0.],(.12,0.),now=1.)
    assert noisy.phase=='BRAKING' and noisy.intent.hold and noisy.reason=='latched_braking'
    np.testing.assert_array_equal(noisy.intent.goal_xy,first.intent.goal_xy)
    stopped=decide(script,cfg,terrain,[7.16,.25,0.],now=2.)
    assert stopped.phase=='HOLD' and stopped.intent.hold
    outside=decide(script,cfg,terrain,[7.2,.41,0.],now=3.)
    assert outside.phase=='APPROACH' and not outside.intent.hold


def test_higher_angular_velocity_is_not_actual_stopping(cfg,terrain):
    script=LeaderSquareScript(cfg.spec,0,2)
    result=decide(script,cfg,terrain,[7.2,0.,0.],(.0,.02))
    assert result.phase=='BRAKING' and result.intent.hold
    result=decide(script,cfg,terrain,[7.2,0.,.004],(.0,.004),now=1.)
    assert result.phase=='BRAKING' and result.reason=='latched_braking'


@pytest.mark.parametrize('delivery',['missing_layout','sparse','expired'])
def test_illegal_or_expired_leader_information_waits_without_capture(cfg,terrain,delivery):
    script=LeaderSquareScript(cfg.spec,0,1);pose=[0.,-5.,0.]
    snapshot=messages(cfg,1,pose,leader_payload=delivery!='missing_layout',leader_full=delivery!='sparse')
    now=1.2 if delivery=='expired' else 0.
    result=script.decide(pose,[0.,0.],[0.,0.],terrain,snapshot,now,0)
    assert result.intent.hold and result.temporary_wait and result.phase=='APPROACH'
    assert result.site_center is None
    recovered=decide(script,cfg,terrain,pose,now=2.)
    assert not recovered.intent.hold and recovered.phase=='APPROACH'


def test_priority_distinguishes_parked_slot_from_stationary_wait_far_away(cfg,terrain):
    script=LeaderSquareScript(cfg.spec,0,2);pose=[-5.,0.,0.]
    waiting=decide(script,cfg,terrain,pose,peer_positions={1:[0.,-12.]})
    assert waiting.intent.hold and waiting.reason=='lower_sender_priority'
    released=decide(script,cfg,terrain,pose,now=1.)
    assert not released.intent.hold and released.reason=='approach_slot'


def test_seven_degree_driving_surface_is_not_parking_capture(cfg,terrain):
    snapshot=terrain.snapshot();features=snapshot.features.copy();slope=np.tan(np.deg2rad(7.))
    features[:,0]=snapshot.xy[:,0]*slope;features[:,1]=slope
    rough=FootprintTerrain(replace(snapshot,features=features),cfg.spec,query_version='batch_ties_v2')
    script=LeaderSquareScript(cfg.spec,0,2)
    result=decide(script,cfg,rough,[7.2,0.,0.])
    assert result.phase=='APPROACH' and not result.stop_prediction_verified
    assert not result.intent.hold


def test_frozen_site_cannot_be_silently_replaced(cfg,terrain):
    script=LeaderSquareScript(cfg.spec,0,1);pose=[0.,-5.,0.]
    first=decide(script,cfg,terrain,pose)
    snapshot=messages(cfg,1,pose,1.)
    changed=list(snapshot.messages);leader=next(i for i,m in enumerate(changed) if m.sender_id==0)
    changed[leader]=replace(changed[leader],site_center=[4.,3.6])
    result=script.decide(pose,[0.,0.],[0.,0.],terrain,replace(snapshot,messages=tuple(changed)),1.,0)
    assert result.intent.hold and result.reason=='changed_frozen_layout'
    np.testing.assert_array_equal(result.site_center,first.site_center)


def test_public_restore_and_episode_guard_are_atomic(cfg,terrain):
    script=LeaderSquareScript(cfg.spec,0,2);pose=[7.13,0.,0.]
    original=decide(script,cfg,terrain,pose,(.2,0.));saved=script.state_dict()
    restored=LeaderSquareScript(cfg.spec,0,2);restored.load_state_dict(saved)
    replay=decide(restored,cfg,terrain,pose,(.2,0.),now=1.)
    assert replay.phase==original.phase
    np.testing.assert_array_equal(replay.intent.goal_xy,original.intent.goal_xy)
    bad=deepcopy(saved);bad['hold_xy']=None
    with pytest.raises(ValueError):restored.load_state_dict(bad)
    assert restored.state_dict()['phase']=='BRAKING'
    with pytest.raises(ValueError,match='without reset'):
        restored.decide(pose,[0.,0.],[0.,0.],terrain,messages(cfg,2,pose,2.),2.,1)
    restored.reset();assert restored.state_dict()['site_center'] is None


def test_same_legal_inputs_do_not_depend_on_other_environments(cfg,terrain):
    pose=[7.13,0.,0.];snapshot=messages(cfg,2,pose)
    a=LeaderSquareScript(cfg.spec,0,2);b=LeaderSquareScript(cfg.spec,9,2)
    result=a.decide(pose,[.2,0.],[.2,0.],terrain,snapshot,0.,0)
    replay=b.decide(pose,[.2,0.],[.2,0.],terrain,snapshot,0.,0)
    np.testing.assert_array_equal(result.intent.goal_xy,replay.intent.goal_xy)
    np.testing.assert_array_equal(result.stopping_endpoint,replay.stopping_endpoint)
    assert result.phase==replay.phase
