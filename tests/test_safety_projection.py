"""Real OSQP and nonlinear proxy checks, including honest failure feedback."""

from dataclasses import replace
from copy import deepcopy
import numpy as np
import pytest

from test_planner_baselines import low_input
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.safety import (
    SafetyInput, SafetyProjector, position_jacobian,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.execution import dynamics
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.protocol import (
    LocalIntent, TerrainView, IdentifiedNeighbor,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.contracts import NeighborPrediction
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.memory import LocalTerrainMemory


@pytest.mark.parametrize("u", [[.3,.6],[-.3,-.6],[0,.6],[.8,-1]])
def test_exact_midpoint_jacobian(u):
    pose, feat = np.array([.2,-.3,.7]), np.array([0,.2,-.1,0,.8])
    u = np.array(u)
    derivative = position_jacobian(pose,u,feat,.75)
    numerical = np.column_stack([(dynamics(pose,u+np.eye(2)[i]*1e-6,feat,.75)[0][:2]
        -dynamics(pose,u-np.eye(2)[i]*1e-6,feat,.75)[0][:2])/2e-6 for i in range(2)])
    np.testing.assert_allclose(derivative,numerical,atol=1e-9,rtol=1e-8)


@pytest.mark.parametrize("nominal", [[.3,0],[-.3,0],[0,.6],[.9,2.4]])
def test_real_qp_actuator_and_recheck(nominal):
    ctx=SafetyInput(low_input())
    p=SafetyProjector()
    out=p.project(ctx,nominal)
    assert out.model_feasible and out.qp_status == "solved"
    assert not p._verify(ctx,out.applied_control)
    assert max(abs(out.applied_control)) <= .60001
    assert out.residual["primal"] < 1e-5


def test_workspace_projection_changes_nominal():
    item=low_input()
    # Circle must stay within right boundary 0.16 -> centre at most 0.02.
    ctx=SafetyInput(item,np.array([[1.,0]]),np.array([.16]))
    p=SafetyProjector()
    out=p.project(ctx,[.3,0])
    assert out.safety_status == "projected" and out.applied_control[0] <= .10001
    assert not p._verify(ctx,out.applied_control)


def test_cbf_and_swept_peer_infeasible_never_claim_safe():
    item=low_input()
    peer=NeighborPrediction(np.tile([.3,0],(16,1)),np.full(16,.46),0,1)
    item=replace(item,neighbors=(IdentifiedNeighbor(3,peer),))
    out=SafetyProjector().project(SafetyInput(item),[.3,0])
    assert out.safety_status == "unverified_brake" and not out.model_feasible
    assert "neighbor" in out.reason and any("neighbor:3" in s for s in out.constraint_ids)


@pytest.mark.parametrize("kind", ["empty","blocked"])
def test_unknown_and_impassable_brake_not_safe(kind):
    memory=LocalTerrainMemory()
    if kind == "blocked": memory.observe(np.array([[0,0]]),np.array([[0,0,0,0,.1]]),0)
    out=SafetyProjector().project(SafetyInput(replace(low_input(),terrain=TerrainView(memory))),[.3,0])
    assert not out.model_feasible and out.safety_status == "unverified_brake"


@pytest.mark.parametrize("kind", ["hold","expired","timeout"])
def test_hold_expired_and_deadline_brake(kind):
    item=replace(low_input(),previous_control=np.array([.6,.8]))
    if kind == "hold": item=replace(item,intent=LocalIntent.for_hold(item.ego_pose,0))
    if kind == "expired": item=replace(item,now=2.,times=item.times+2)
    p=SafetyProjector(deadline_s=0 if kind == "timeout" else None)
    out=p.project(SafetyInput(item),[1.15,2.4])
    np.testing.assert_allclose(out.applied_control,[.3,.2])
    assert "brake" in out.safety_status
    assert p.execution_prediction(SafetyInput(item),out.applied_control) is not None


def test_snapshot_restore_and_version_rejection():
    p=SafetyProjector();ctx=SafetyInput(low_input())
    p.project(ctx,[.3,0]);saved=p.state_dict()
    q=SafetyProjector();q.load_state_dict(saved)
    np.testing.assert_allclose(p.project(ctx,[0,.6]).applied_control,q.project(ctx,[0,.6]).applied_control)
    bad=deepcopy(saved);bad["state_version"]=2
    with pytest.raises(ValueError): q.load_state_dict(bad)
    q.reset();assert q.last_control is None


def test_nonfinite_and_bad_workspace_reject_before_solver():
    p=SafetyProjector()
    with pytest.raises(ValueError): p.project(SafetyInput(low_input()),[np.nan,0])
    with pytest.raises(ValueError): SafetyInput(low_input(),np.ones((3,3)),np.ones(3))
    with pytest.raises(ValueError): p.project(SafetyInput(replace(low_input(),previous_control=np.array([np.nan,0]))),[0,0])


def test_solver_exception_is_classified_brake(monkeypatch):
    import osqp
    from test_planner_baselines import low_input
    item=low_input()
    def broken(*args,**kwargs):raise RuntimeError('injected solver failure')
    monkeypatch.setattr(osqp.OSQP,'setup',broken)
    result=SafetyProjector().project(SafetyInput(item),np.array([.3,0.]))
    assert result.qp_status=='solver_exception'
    assert result.reason=='qp_exception:RuntimeError'
    assert result.safety_status=='verified_brake'


def test_solved_inaccurate_does_not_pass(monkeypatch):
    import osqp
    solve=osqp.OSQP.solve
    def inaccurate(self,**kwargs):
        result=solve(self,**kwargs);result.info.status='solved inaccurate';return result
    monkeypatch.setattr(osqp.OSQP,'solve',inaccurate)
    result=SafetyProjector().project(SafetyInput(low_input()),[.3,0])
    assert result.reason=='qp:solved inaccurate' and result.safety_status=='verified_brake'


def test_deadline_after_solve_is_rejected(monkeypatch):
    import lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.safety as safety
    moments=iter([0.,.05,.2,.3])
    monkeypatch.setattr(safety.time,'monotonic',lambda:next(moments))
    result=SafetyProjector(deadline_s=.1).project(SafetyInput(low_input()),[.3,0])
    assert result.reason=='safety_timeout'
    np.testing.assert_array_equal(result.applied_control,[0,0])


def test_equivalent_workspace_scaling_keeps_proxy_clearance():
    item=low_input();projector=SafetyProjector()
    a=projector.project(SafetyInput(item,np.array([[1.,0]]),np.array([.16])),[.3,0])
    b=projector.project(SafetyInput(item,np.array([[2.,0]]),np.array([.32])),[.3,0])
    np.testing.assert_allclose(a.applied_control,b.applied_control,atol=1e-6)
    with pytest.raises(ValueError):SafetyInput(item,np.zeros((1,2)),np.zeros(1))
