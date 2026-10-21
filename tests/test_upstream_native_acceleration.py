"""Native arithmetic against scalar polygons and the stepwise physical model."""
from dataclasses import replace
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from _common import cfg_from_experiment
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_execution import (
    _control_error_reference, _limit_control_numpy_reference, control_error,
    limit_control_numpy, limit_control_torch,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.protocol import TerrainSnapshot
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_footprint import FootprintTerrain
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_mppi_precision import UpstreamMPPIKernelV2
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_planning import UpstreamIntent, UpstreamPlannerInput


@pytest.fixture(scope='module')
def cfg():
    return replace(cfg_from_experiment('configs/experiment/upstream_scale_s03_precision_v2.yaml'),
                   planner_device='cuda', simulated_latency=0., mppi_samples=8)


def test_native_projection_and_failure_priority_match_scalar_oracle(cfg):
    rng = np.random.default_rng(672)
    previous = np.array([(0.,0.),(.05,.2),(-.05,-.2),(.050000001,.35),
                         (-.050000001,-.35),(25/9,0.),(-.5,0.),(25/9-.9*.35,.35)])
    previous = np.repeat(previous, 24, axis=0)
    desired = rng.normal(size=previous.shape) * [2.,.4]
    expected = np.stack([_limit_control_numpy_reference(d,p,cfg.spec) for d,p in zip(desired,previous)])
    actual = np.stack([limit_control_numpy(d,p,cfg.spec) for d,p in zip(desired,previous)])
    np.testing.assert_allclose(actual,expected,atol=1e-10,rtol=1e-8)
    assert torch.cuda.is_available(), 'Required real CUDA projection regression'
    gpu = limit_control_torch(torch.tensor(desired,device='cuda'),torch.tensor(previous,device='cuda'),cfg.spec)
    np.testing.assert_allclose(gpu.cpu(),expected,atol=1e-10,rtol=1e-8)
    for u,p in zip(np.vstack([actual,desired]),np.vstack([previous,previous])):
        assert control_error(u,p,cfg.spec) == _control_error_reference(u,p,cfg.spec)


@pytest.mark.parametrize('features', [[0.,0.,0.,0.,1.],[.2,.01,-.02,.01,.75]])
def test_fused_rollout_matches_stepwise_with_motion_unknown_and_nonfinite_noise(cfg,features):
    xx,yy = np.meshgrid(np.arange(-1.,1.01,.2),np.arange(-1.,1.01,.2))
    xy = np.c_[xx.ravel(),yy.ravel()]
    terrain = FootprintTerrain(TerrainSnapshot(xy,np.tile(features,(len(xy),1)),
        np.zeros(len(xy)),cfg.spec.perception.support_radius),cfg.spec)
    requests = [UpstreamPlannerInput(0,i,np.array([x,0.,.3]),np.array(p),
        UpstreamIntent([3.,0.],False,0.,1.),terrain,(),np.array(p),0.,np.arange(41)*.2,
        (-100.,-100.,100.,100.),cfg.spec) for i,(x,p) in enumerate([
            (0.,(.050000001,.25)),(.8,(.2,0.)),(2.,(-.5,0.))])]
    kernel = UpstreamMPPIKernelV2(cfg)
    raw = kernel.tensor(np.random.default_rng(3).normal(size=(3,8,40,2))*[.4,.1])
    raw[0,0,3,0] = torch.nan
    packed,_ = kernel.pack(requests)
    actual = kernel.rollout(requests,raw,packed)
    assert hasattr(packed,'_rollout_query'), 'Exercise the fused path'
    with patch('lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_native.rollout_cuda',return_value=None):
        reference = kernel.rollout(requests,raw,packed)
    for a,b in zip(actual,reference):
        torch.testing.assert_close(a,b,atol=1e-10,rtol=1e-8)
    assert not actual[3][0,0] and not actual[4].all()
    # Cost/query consumers can mutate a returned trajectory. Cached features
    # must never conceal that mutation or certify its old support mask.
    actual[0][0,0,0,:2] = 1e6
    _,known = packed.lookup(actual[0][...,:2])
    assert not known[0,0,0]


def test_fused_braking_preserves_complete_applied_sequence(cfg):
    from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_execution import brake_sequence
    rng=np.random.default_rng(67)
    previous=np.zeros(2)
    for _ in range(40):
        previous=limit_control_numpy(rng.normal(size=2),previous,cfg.spec)
        last=previous.copy();expected=[]
        for _ in range(512):
            last=limit_control_numpy(np.zeros(2),last,cfg.spec)
            last[np.abs(last)<1e-10]=0.;expected.append(last.copy())
            if not last.any():break
        np.testing.assert_array_equal(brake_sequence(previous,cfg.spec),np.asarray(expected))


@pytest.mark.parametrize('previous',[(3.,0.),(0.,1.),(.01,.35)])
def test_fused_brake_rejects_invalid_previous_command(cfg,previous):
    from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_execution import brake_sequence
    with pytest.raises(ValueError,match='Previous command violates'):
        brake_sequence(previous,cfg.spec)
