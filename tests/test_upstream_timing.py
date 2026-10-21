"""Queue feedback, expiring requests, reserves and independent legacy timing."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from _common import cfg_from_experiment
from diagnose_upstream_execution import synthetic_local_frame
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_execution import rollout_numpy
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_planning import (
    PlanningAdapter,UpstreamIntent,UpstreamPlannerInput,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_timing import ActualControlHistory


@pytest.fixture(scope='module')
def cfg():return cfg_from_experiment('configs/experiment/upstream_scale_s03.yaml')


def item(cfg, *, controls=None, gain=1., previous=(1.,0.)):
    if controls is None:controls=np.array([[1.,0.],[.92,0.],[.84,0.]])
    controls=np.asarray(controls,float)
    path=rollout_numpy(np.zeros(3),controls,cfg.spec,0.,gain=gain)
    history=ActualControlHistory(0,0,cfg.spec.fingerprint,path.times,path.states,controls,path.velocities,previous)
    terrain=synthetic_local_frame(np.zeros(3),cfg.spec,0.,query_version='batch_ties_v2')
    return UpstreamPlannerInput(0,0,path.states[-1],path.velocities[-1],
        UpstreamIntent([0.,0.],True,0.,1.),terrain,(),controls[-1],path.times[-1],
        path.times[-1]+np.arange(41)*.2,(-100.,-100.,100.,100.),cfg.spec,
        information_pose=np.zeros(3),information_time=0.,actual_history=history)


class NoSolver(PlanningAdapter):
    adapter_id='history_test_no_solver_v1'
    def candidate_controls(self, item):raise AssertionError('HOLD/expiry cannot call the solver')


def test_actual_expiration_braking_during_queue_is_used_once(cfg):
    context=item(cfg);output=NoSolver(cfg).plan(context)
    assert output.valid and output.trajectory.timestamp==pytest.approx(.6)
    assert output.trajectory.states[0,0]==pytest.approx((1.+.92+.84)*.2)
    assert output.trajectory.controls[0,0]==pytest.approx(.76)
    assert output.diagnostics['delay_steps']==3 and output.diagnostics['delay_control_changes']==2
    assert output.diagnostics['information_reserve_verified']
    assert output.diagnostics['history_checked_poses']==27
    np.testing.assert_allclose(output.trajectory.states[1,0],context.ego_pose[0]+.76*.2)


def test_observed_gain_not_applied_twice_or_replaced_by_older_snapshot(cfg):
    context=item(cfg,gain=.5);NoSolver(cfg).validate_input(context)
    assert context.ego_pose[0]==pytest.approx(.276)
    output=NoSolver(cfg).plan(context)
    assert output.valid and output.trajectory.states[0,0]==pytest.approx(.276)
    # Future motion is recomputed using the frozen local terrain, not the
    # observed gain of an already-completed step.
    assert output.trajectory.velocities[0,0]==pytest.approx(.76)


def test_legacy_path_rejects_history_and_preserves_old_held_model(cfg):
    legacy=replace(cfg,timing_version='held_previous_control_v1')
    context=item(cfg)
    with pytest.raises(ValueError,match='legacy'):NoSolver(legacy).plan(context)
    with pytest.raises(ValueError,match='actual activation'):
        NoSolver(legacy).plan(replace(context,actual_history=None))
    with pytest.raises(ValueError,match='explicit queue'):
        NoSolver(cfg).plan(replace(context,actual_history=None))
    assert 'timing_version' not in cfg_from_experiment('configs/experiment/upstream_scale_s02.yaml').metadata
    assert cfg.metadata['timing_version']=='actual_control_history_v1'


@pytest.mark.parametrize('defect', ['agent','profile','grid','nonfinite','slew','gain','odometry','initial_control','version'])
def test_bad_history_is_rejected_before_solver(cfg,defect):
    history=item(cfg).actual_history;fields=deepcopy(vars(history))
    if defect=='agent':fields['agent_id']=1
    if defect=='profile':fields['profile_fingerprint']='different'
    if defect=='grid':fields['times'][1]=.1
    if defect=='nonfinite':fields['poses'][0,0]=np.nan
    if defect=='slew':fields['controls'][1,0]=.5
    if defect=='gain':fields['actual_velocities'][1,0]=.01
    if defect=='odometry':fields['poses'][1,0]+=.001
    if defect=='initial_control':fields['initial_previous_control']=[3.,0.]
    if defect=='version':fields['version']='different'
    with pytest.raises(ValueError):
        invalid=ActualControlHistory(**fields)
        replace(item(cfg),actual_history=invalid)


@pytest.mark.parametrize('defect', ['information_pose','activation_pose','activation_control','activation_velocity','time'])
def test_history_endpoints_must_join_actual_feedback(cfg,defect):
    context=item(cfg);fields={}
    if defect=='information_pose':fields['information_pose']=[.1,0.,0.]
    if defect=='activation_pose':fields['ego_pose']=context.ego_pose+[.1,0.,0.]
    if defect=='activation_control':fields['previous_control']=[.83,0.]
    if defect=='activation_velocity':fields['ego_velocity']=[.83,0.]
    if defect=='time':fields['information_time']=.1
    with pytest.raises(ValueError,match='join'):replace(context,**fields)


def test_expired_new_request_brakes_and_never_uses_old_target(cfg):
    context=item(cfg)
    expired=replace(context,intent=UpstreamIntent([100.,0.],False,0.,.4))
    output=NoSolver(cfg).plan(expired)
    assert not output.valid and output.fallback_reason=='expired_intent'
    assert output.trajectory.controls[0,0]==pytest.approx(.76)
    assert output.trajectory.states[-1,0]<2.


def test_wrong_actual_delay_count_is_rejected(cfg):
    context=item(replace(cfg,simulated_latency=.2))
    with pytest.raises(ValueError,match='rounded'):NoSolver(replace(cfg,simulated_latency=.2)).plan(context)


def test_full_actual_history_plus_stop_must_fit_original_n40_horizon(cfg):
    delayed=replace(cfg,simulated_latency=1.)
    top=cfg.spec.vehicle.forward_limit
    context=item(delayed,controls=np.tile([top,0.],(5,1)),previous=(top,0.))
    reason,detail=NoSolver(delayed)._reserve(context,[top,0.])
    assert reason=='stop_reserve_exceeds_horizon' and detail['reserve_steps']>40


def test_history_restore_cannot_cross_into_old_timing_identity(cfg):
    adapter=NoSolver(cfg);adapter.plan(item(cfg));saved=adapter.state_dict()
    with pytest.raises(ValueError,match='mismatch'):
        NoSolver(replace(cfg,timing_version='held_previous_control_v1')).load_state_dict(saved)


def test_history_owns_readonly_input_and_zero_delay_is_explicit(cfg):
    context=item(cfg);history=context.actual_history
    assert not history.controls.flags.writeable and not history.poses.flags.writeable
    instantaneous=replace(cfg,simulated_latency=0.)
    empty=ActualControlHistory(0,0,cfg.spec.fingerprint,[0.],np.zeros((1,3)),np.zeros((0,2)),np.zeros((0,2)),[0.,0.])
    now=replace(context,ego_pose=np.zeros(3),ego_velocity=np.zeros(2),previous_control=np.zeros(2),
                now=0.,times=np.arange(41)*.2,actual_history=empty)
    result=NoSolver(instantaneous).plan(now)
    assert result.valid and result.diagnostics['delay_steps']==0
