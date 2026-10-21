"""Distance/TTL, frame/sender alignment, honest forecasts and public restore."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from _common import cfg_from_experiment
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_communication import (
    DistanceCommunicationCache,MotionBoundLedger,MotionBound,RoverBroadcast,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_execution import rollout_numpy,control_error


@pytest.fixture(scope='module')
def spec():return cfg_from_experiment('configs/experiment/upstream_scale_s02.yaml').spec


def packets(positions,now=0.,epoch=0,*,hold=False):
    return [[RoverBroadcast(a,epoch,now,np.r_[positions[e,a],0.],np.zeros(2),np.zeros(2),hold,
                            goal_xy=positions[e,a]+[1.,0.]) for a in range(4)] for e in range(len(positions))]


def position():return np.array([[[0.,0.],[10.,0.],[70.,0.],[130.,0.]]])


def test_full_sparse_and_hard_disconnect_never_fill_truth_slots(spec):
    cache=DistanceCommunicationCache(spec,np.zeros((1,4,3)));pos=position()
    cache.transmit(packets(pos),pos,[0.]);snap=cache.snapshot(0,0,0.,[0.,0.,0.])
    assert snap.sender_indices.tolist()==[1,2,-1]
    assert snap.state_valid.tolist()==[True,True,False]
    assert snap.full_messages.tolist()==[True,False,False]
    assert snap.goal_valid.tolist()==[True,False,False]
    assert snap.messages[0].sender_id==1 and snap.messages[1].goal_xy is None
    assert len(snap.predictions)==2


@pytest.mark.parametrize('distance,full,received',[(60.,True,True),(60.0001,False,True),(120.,False,True),(120.0001,False,False)])
def test_radio_boundaries_are_explicit(spec,distance,full,received):
    pos=position();pos[0,1,0]=distance;pos[0,2:,0]=200.
    cache=DistanceCommunicationCache(spec,np.zeros((1,4,3)));cache.transmit(packets(pos),pos,[0.])
    snap=cache.snapshot(0,0,0.,[0.,0.,0.])
    assert (len(snap.messages)>0)==received
    if received:assert snap.messages[0].full==full


def test_distinct_full_and_sparse_update_periods_and_pure_reads(spec):
    cache=DistanceCommunicationCache(spec,np.zeros((1,4,3)));pos=position()
    cache.transmit(packets(pos),pos,[0.]);before=cache.state_dict()
    for _ in range(2):cache.snapshot(0,0,.1,[0.,0.,0.])
    np.testing.assert_array_equal(cache.state_dict()['last_sent'],before['last_sent'])
    cache.transmit(packets(pos,.2),pos,[.2]);snap=cache.snapshot(0,0,.2,[0.,0.,0.])
    assert [m.timestamp for m in snap.messages]==[.2,0.]
    cache.transmit(packets(pos,1.),pos,[1.]);snap=cache.snapshot(0,0,1.,[0.,0.,0.])
    assert [m.timestamp for m in snap.messages]==[1.,1.]


def test_expired_state_retains_only_conservative_identity_and_reachable_prediction(spec):
    cache=DistanceCommunicationCache(spec,np.zeros((1,4,3)));pos=position()
    cache.transmit(packets(pos,hold=True),pos,[0.])
    inclusive=cache.snapshot(0,0,1.,[0.,0.,0.]);assert inclusive.state_valid[:2].all()
    stale=cache.snapshot(0,0,1.2,[0.,0.,0.]);assert not stale.state_valid.any() and not stale.goal_valid.any()
    assert stale.sender_indices[:2].tolist()==[1,2]
    assert stale.predictions[0].radii[0]==pytest.approx(2*spec.vehicle.effective_radius+1.2*25/9)
    assert stale.prediction_metadata[0]['bound_source']=='global_hard_limit'
    # A stationary HOLD packet does not imply any future motion guarantee.
    fresh=cache.snapshot(0,0,0.,[0.,0.,0.])
    assert fresh.prediction_metadata[0]['stop_intent']
    assert fresh.predictions[0].radii[-1]>20.


def test_receiver_frame_conversion_and_sender_slot_crossing_keep_all_payloads_aligned(spec):
    frames=np.zeros((1,4,3));frames[0,0]=[10.,20.,np.pi/2];frames[0,1]=[12.,20.,0.]
    cache=DistanceCommunicationCache(spec,frames)
    pos=np.array([[[10.,20.],[12.,20.],[14.,20.],[200.,200.]]])
    row=packets(pos)
    row[0][0]=replace(row[0][0],pose=np.zeros(3))
    row[0][1]=RoverBroadcast(1,0,0.,np.zeros(3),np.zeros(2),np.zeros(2),goal_xy=[1.,0.],site_center=[2.,3.],layout_yaw=.1)
    row[0][2]=RoverBroadcast(2,0,0.,[14.,20.,0.],np.zeros(2),np.zeros(2),goal_xy=[14.,21.])
    cache.transmit(row,pos,[0.]);snap=cache.snapshot(0,0,0.,[0.,0.,0.])
    assert snap.sender_indices[:2].tolist()==[1,2]
    np.testing.assert_allclose(snap.messages[0].pose[:2],[0.,-2.],atol=1e-12)
    np.testing.assert_allclose(snap.messages[0].goal_xy,[0.,-3.],atol=1e-12)
    np.testing.assert_allclose(snap.messages[0].site_center,[3.,-4.],atol=1e-12)
    assert snap.messages[0].layout_yaw==pytest.approx(.1-np.pi/2)
    swapped=cache.snapshot(0,0,0.,[0.,-5.,0.])
    assert swapped.sender_indices[:2].tolist()==[2,1]
    for slot,msg in enumerate(swapped.messages):
        assert swapped.sender_indices[slot]==msg.sender_id==swapped.predictions[slot].sender_id


def test_duplicate_delivery_cannot_refresh_age_or_trajectory(spec):
    cache=DistanceCommunicationCache(spec,np.zeros((1,4,3)));pos=position();row=packets(pos)
    cache.transmit(row,pos,[0.]);cache.transmit(row,pos,[.4])
    snap=cache.snapshot(0,0,.4,[0.,0.,0.])
    assert snap.ages[0]==.4 and snap.messages[0].received_at==0.


def test_future_delivery_is_rejected_even_when_publication_was_earlier(spec):
    cache=DistanceCommunicationCache(spec,np.zeros((1,4,3)));pos=position()
    cache.transmit(packets(pos),pos,[.4])
    with pytest.raises(ValueError,match='future'):cache.snapshot(0,0,.2,[0.,0.,0.])


def test_fresh_execution_trajectory_expires_from_its_own_timestamp(spec):
    cache=DistanceCommunicationCache(spec,np.zeros((1,4,3)));pos=position()
    trajectory=rollout_numpy([10.,0.,0.],np.zeros((40,2)),spec,0.)
    row=packets(pos,.2);row[0][1]=replace(row[0][1],trajectory=trajectory)
    cache.transmit(row,pos,[.2])
    assert cache.snapshot(0,0,.6,[0.,0.,0.]).trajectory_valid[0]
    assert not cache.snapshot(0,0,.8,[0.,0.,0.]).trajectory_valid[0]
    row=packets(pos,.8);row[0][1]=replace(row[0][1],trajectory=trajectory)
    cache.transmit(row,pos,[.8])
    assert not cache.snapshot(0,0,.8,[0.,0.,0.]).trajectory_valid[0]


def test_verified_long_motion_promise_changes_bound_but_untrusted_or_short_does_not(spec):
    ledger=MotionBoundLedger(spec,1)
    bound=ledger.pledge(0,1,.05,0.,10.,[0.,0.])
    pos=position();row=packets(pos);row[0][1]=replace(row[0][1],motion_bound=bound)
    cache=DistanceCommunicationCache(spec,np.zeros((1,4,3)),ledger=ledger);cache.transmit(row,pos,[0.])
    snap=cache.snapshot(0,0,0.,[0.,0.,0.])
    assert snap.prediction_metadata[0]['speed_bound']==.05
    assert snap.predictions[0].radii[-1]==pytest.approx(2*spec.vehicle.effective_radius+.4)
    untrusted=DistanceCommunicationCache(spec,np.zeros((1,4,3)));untrusted.transmit(row,pos,[0.])
    assert untrusted.snapshot(0,0,0.,[0.,0.,0.]).prediction_metadata[0]['speed_bound']==25/9
    short=ledger.pledge(0,1,.01,0.,1.,[0.,0.]);row[0][1]=replace(row[0][1],timestamp=.2,motion_bound=short)
    row[0][0]=replace(row[0][0],timestamp=.2);row[0][2]=replace(row[0][2],timestamp=.2);row[0][3]=replace(row[0][3],timestamp=.2)
    cache.transmit(row,pos,[.2])
    assert cache.snapshot(0,0,.2,[0.,0.,0.]).prediction_metadata[0]['speed_bound']==25/9


def test_ledger_enforces_old_lower_promise_despite_new_higher_request(spec):
    ledger=MotionBoundLedger(spec,1);lower=ledger.pledge(0,0,.05,0.,10.,[0.,0.])
    ledger.pledge(0,0,2.,.2,12.,[0.,0.])
    previous=np.zeros(2)
    for i in range(40):
        applied=ledger.limit(0,0,[2.,.35],previous,i*.2)
        assert abs(applied[0])<=.05+1e-10 and not control_error(applied,previous,spec)
        previous=applied
    assert ledger.verified(lower)
    with pytest.raises(ValueError,match='lower bound'):ledger.pledge(0,0,.01,8.,12.,[.2,0.])


def test_nominal_center_shift_gets_matching_reachable_radius_once(spec):
    ledger=MotionBoundLedger(spec,1);bound=ledger.pledge(0,1,.1,0.,10.,[.1,0.])
    controls=np.zeros((40,2));controls[:4,0]=.1;controls[4,0]=.02
    trajectory=rollout_numpy([10.,0.,0.],controls,spec,0.)
    pos=position();pos[0,1]=trajectory.states[1,:2]
    row=packets(pos,.2);row[0][1]=RoverBroadcast(1,0,.2,trajectory.states[1],trajectory.velocities[0],controls[0],trajectory=trajectory,motion_bound=bound)
    cache=DistanceCommunicationCache(spec,np.zeros((1,4,3)),ledger=ledger);cache.transmit(row,pos,[.2])
    snap=cache.snapshot(0,0,.2,[0.,0.,0.]);prediction=snap.predictions[0]
    expected=2*spec.vehicle.effective_radius+.1*(prediction.times-.2)+np.linalg.norm(prediction.positions-pos[0,1],axis=-1)
    np.testing.assert_allclose(prediction.radii,expected,atol=1e-10,rtol=1e-8)


def test_uncommunicated_truth_and_other_environment_changes_do_not_change_snapshot(spec):
    frames=np.zeros((2,4,3));pos=np.repeat(position(),2,axis=0)
    cache=DistanceCommunicationCache(spec,frames);cache.transmit(packets(pos),pos,[0.,0.])
    old=cache.snapshot(0,0,.2,[0.,0.,0.])
    pos[0,3]=[500.,800.];pos[1]=[900.,600.]
    cache.transmit([[None]*4,[None]*4],pos,[.2,.2])
    new=cache.snapshot(0,0,.2,[0.,0.,0.])
    np.testing.assert_array_equal(old.sender_indices,new.sender_indices)
    for a,b in zip(old.predictions,new.predictions):np.testing.assert_array_equal(a.positions,b.positions);np.testing.assert_array_equal(a.radii,b.radii)


def test_public_restore_and_async_epoch_reset_are_atomic(spec):
    frames=np.zeros((2,4,3));pos=np.repeat(position(),2,axis=0)
    cache=DistanceCommunicationCache(spec,frames);cache.transmit(packets(pos),pos,[0.,0.]);saved=cache.state_dict()
    restored=DistanceCommunicationCache(spec,frames);restored.load_state_dict(saved)
    snap=restored.snapshot(0,0,0.,[0.,0.,0.]);assert not snap.messages[0].pose.flags.writeable
    cache.reset(0);snap=cache.snapshot(1,0,0.,[0.,0.,0.]);assert snap.sender_indices.tolist()==[1,2,-1]
    assert not cache.snapshot(0,0,0.,[0.,0.,0.]).messages
    cache.transmit(packets(pos),pos,[0.,.2]);assert not cache.snapshot(0,0,0.,[0.,0.,0.]).messages
    bad=deepcopy(saved);bad['messages'][0,0,1]=replace(bad['messages'][0,0,1],received_at=10.)
    with pytest.raises(ValueError):restored.load_state_dict(bad)
    np.testing.assert_array_equal(restored.state_dict()['last_sent'],saved['last_sent'])


def test_bound_ledger_restore_and_other_epoch_reset_keep_promises(spec):
    ledger=MotionBoundLedger(spec,2);a=ledger.pledge(0,0,.1,0.,10.,[0.,0.]);b=ledger.pledge(1,0,.2,0.,10.,[0.,0.])
    restored=MotionBoundLedger(spec,2);restored.load_state_dict(ledger.state_dict())
    assert restored.verified(a) and restored.verified(b)
    restored.reset(0);assert not restored.verified(a) and restored.verified(b)
    old=restored.state_dict();bad=deepcopy(old);bad['bounds'].append(MotionBound(0,0,0,.1,0.,10.,spec.fingerprint))
    with pytest.raises(ValueError):restored.load_state_dict(bad)
    np.testing.assert_array_equal(restored.state_dict()['episodes'],old['episodes'])


@pytest.mark.parametrize('damage', ['velocity', 'foreign_bound'])
def test_restore_rejects_inconsistent_actuator_or_certificate_atomically(spec, damage):
    cache=DistanceCommunicationCache(spec,np.zeros((1,4,3)));pos=position()
    cache.transmit(packets(pos),pos,[0.]);original=cache.state_dict();bad=deepcopy(original)
    message=bad['messages'][0,0,1]
    if damage == 'velocity':message=replace(message,actual_velocity=[.1,0.])
    else:message=replace(message,motion_bound=MotionBound(7,1,0,.1,0.,10.,spec.fingerprint))
    bad['messages'][0,0,1]=message
    with pytest.raises(ValueError):cache.load_state_dict(bad)
    np.testing.assert_array_equal(cache.snapshot(0,0,0.,[0.,0.,0.]).messages[0].actual_velocity,[0.,0.])


def test_foreign_bound_and_malformed_ledger_cannot_be_verified(spec):
    ledger=MotionBoundLedger(spec,1)
    assert not ledger.verified(MotionBound(99,1,0,.1,0.,10.,spec.fingerprint))
    with pytest.raises(ValueError):ledger.load_state_dict([])
    with pytest.raises(ValueError):MotionBoundLedger(spec,True)
    with pytest.raises(ValueError):ledger.pledge(True,0,.1,0.,10.,[0.,0.])
