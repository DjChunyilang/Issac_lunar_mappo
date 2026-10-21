"""Actual OSQP, analytical midpoint derivatives and nonlinear full-foot review."""
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from _common import cfg_from_experiment
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_execution import midpoint_numpy,control_error
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_footprint import FootprintNeighbor
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_safety import SafetyContext,UpstreamSafetyProjector,midpoint_jacobian
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_sensing import DenseSensor,DenseTerrainMemory


@pytest.fixture(scope='module')
def scene():
    cfg=cfg_from_experiment('configs/experiment/upstream_scale_s03_environment_safety_v2.yaml')
    p=np.zeros((2,4,3));p[:,:,0]=np.arange(4)*10
    frames=DenseSensor(cfg.spec,None).sample(p,p,np.zeros(2));memory=DenseTerrainMemory(cfg.spec)
    memory.observe(frames[0][0]);return cfg,memory


@pytest.mark.parametrize('control',[[.4,.1],[-.3,-.15],[0.,.1]])
def test_exact_midpoint_jacobian_includes_yaw_and_terrain_gain(scene,control):
    cfg,_=scene;pose=np.array([1.,2.,.7]);control=np.asarray(control);gain=.6;eps=1e-6
    numerical=np.column_stack([(midpoint_numpy(pose,control+eps*np.eye(2)[j],cfg.spec,gain=gain)[0]
        -midpoint_numpy(pose,control-eps*np.eye(2)[j],cfg.spec,gain=gain)[0])/(2*eps) for j in range(2)])
    np.testing.assert_allclose(midpoint_jacobian(pose,control,gain,.2),numerical,atol=1e-10,rtol=1e-8)


def test_real_qp_joint_actuator_projection_and_public_restore(scene):
    cfg,memory=scene;safety=UpstreamSafetyProjector(cfg.spec)
    context=SafetyContext(np.zeros(3),np.zeros(2),memory.view(),(),0.,np.array([[-100.,-100.],[100.,100.]]))
    decision=safety.project(context,[.2,.1])
    assert decision.verified and decision.qp_status=='solved' and decision.status=='projected'
    assert not control_error(decision.control,[0.,0.],cfg.spec)
    assert decision.prediction.states.shape==(41,3) and 'terrain_support' in decision.constraint_sources
    fresh=UpstreamSafetyProjector(cfg.spec);fresh.load_state_dict(safety.state_dict())
    replay=fresh.project(context,[.2,.1]);np.testing.assert_allclose(decision.control,replay.control,atol=1e-7,rtol=1e-6)


def test_unavoidable_collision_brake_never_reports_safety(scene):
    cfg,memory=scene;safety=UpstreamSafetyProjector(cfg.spec)
    times=cfg.spec.grid.times(0.);radius=2*cfg.spec.vehicle.effective_radius
    peer=FootprintNeighbor.for_spec(1,times,np.tile([radius+.02,0.],(41,1)),np.full(41,radius),cfg.spec)
    context=SafetyContext(np.zeros(3),np.array([.3,0.]),memory.view(),(peer,),0.,hold=True)
    decision=safety.project(context,[1.,0.])
    assert decision.status=='unverified_brake' and not decision.verified and decision.prediction is None
    assert decision.control[0]<.3 and not control_error(decision.control,context.previous_control,cfg.spec)


def test_unknown_and_hold_use_braking_without_new_motion_goal(scene):
    cfg,memory=scene;safety=UpstreamSafetyProjector(cfg.spec)
    hold=SafetyContext(np.zeros(3),np.array([.12,.03]),memory.view(),(),0.,hold=True)
    stopped=safety.project(hold,[2.,.3])
    assert stopped.verified and stopped.iterations==0 and stopped.control[0]<.12
    empty=DenseTerrainMemory(cfg.spec)
    rejected=safety.project(replace(hold,terrain=empty.view(),hold=False),[2.,.3])
    assert not rejected.verified and rejected.status=='unverified_brake'
