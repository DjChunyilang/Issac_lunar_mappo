"""Observed crater classes share CPU/GPU risk and MPPI rejection semantics."""
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from _common import cfg_from_experiment
from test_upstream_planning import context
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_grid import GridMemory,GridTerrainBatch
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_mppi_calibration import UpstreamMPPIKernelV4,UpstreamMPPISettingsV3
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_factory import create_upstream_adapter
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_planning import Candidate,UpstreamIntent


@pytest.fixture(scope='module')
def cfg():
    return replace(cfg_from_experiment('configs/experiment/upstream_1km_crater_classes_v1.yaml'),planner_device='cpu')


class Atlas:
    def __init__(self,obstacles=()):
        self.identity='graded_test_map';self.extent=np.array([100.,100.]);self.res=.2;self.obstacles=obstacles

    def read(self,cells):
        xy=(cells+.5)*self.res-self.extent/2
        kind=np.zeros(len(cells),np.uint8)
        for x,y,r,bit in self.obstacles:
            delta=np.maximum(abs(xy-[x,y])-self.res/2,0)
            kind[np.sum(delta*delta,-1)<=r*r]|=bit
        values=np.zeros((len(cells),7),np.float32);values[:,4]=1.
        return values,kind,np.ones(len(cells),bool)


def memory(cfg,obstacles=()):
    atlas=Atlas(obstacles);mem=GridMemory(cfg.spec,cfg.obstacle_input)
    mem.update(np.zeros(3),np.zeros(3),0.,atlas)
    return mem,atlas


@pytest.mark.parametrize('bit,risk,kind',[(1,0.,None),(4,.1,'crater_medium'),(8,1.,'crater_large'),(2,1.,'obstacle')])
def test_class_risk_and_distances_ignore_small_craters(cfg,bit,risk,kind):
    mem,_=memory(cfg,[(0.,0.,.3,bit)]);view=mem.view()
    assert view.aggregate(np.zeros(3),[(-.5,.5,-.5,.5)])[0,0]==pytest.approx(risk)
    for channel in ('obstacle','crater_medium','crater_large'):
        distance,known=view.obstacle_distance([[0.,0.,0.]],kind=channel)
        expected=channel==kind or (bit==8 and channel=='obstacle')
        assert bool(distance[0]<0)==expected
        assert known[0]


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_graded_cpu_gpu_distance_known_and_selected_rows(cfg,device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    medium,_=memory(cfg,[(0.,0.,.3,4)])
    large,_=memory(cfg,[(0.,0.,.3,8)])
    views=[medium.view(),large.view()];batch=GridTerrainBatch(views,device)
    poses=np.array([[0.,0.,0.],[4.,0.,0.],[0.,4.,.6],[30.,0.,0.]])
    tensor=torch.as_tensor(np.stack([poses,poses]),device=device)
    for kind in ('obstacle','crater_medium','crater_large'):
        expected=np.stack([v.obstacle_distance(poses,kind=kind)[0] for v in views])
        np.testing.assert_allclose(batch.obstacle_distance(tensor,kind=kind).cpu(),expected,atol=1e-6)
        np.testing.assert_allclose(batch.select([1]).obstacle_distance(tensor[1:],kind=kind).cpu()[0],expected[1],atol=1e-6)
    features,known=batch.lookup(tensor[...,:2])
    for index,view in enumerate(views):
        expected,mask=view.lookup(poses[:,:2])
        np.testing.assert_allclose(features.cpu()[index],expected)
        np.testing.assert_array_equal(known.cpu()[index],mask)
    assert batch.data.shape[-1]==11


@pytest.mark.parametrize('bit,channel',[(4,'crater_medium'),(8,'crater_large')])
def test_hidden_crater_class_does_not_leak(cfg,bit,channel):
    free,_=memory(cfg)
    hidden,atlas=memory(cfg,[(15.3,0.,.1,bit)])
    pose=[[13.,0.,0.]]
    np.testing.assert_array_equal(hidden.view().obstacle_distance(pose,kind=channel)[0],
                                  free.view().obstacle_distance(pose,kind=channel)[0])
    hidden.update([1.,0.,0.],np.zeros(3),.2,atlas)
    assert hidden.view().obstacle_distance(pose,kind=channel)[0][0]<0


@pytest.mark.parametrize('bit', [1,4,8,2])
def test_mppi_medium_cost_and_large_hard_rejection(cfg,bit):
    mem,_=memory(cfg,[(0.,0.,.3,bit)]);item=context(cfg,mem.view())
    kernel=UpstreamMPPIKernelV4(cfg,UpstreamMPPISettingsV3(samples=2,iterations=1,device='cpu'))
    terrain,neighbors=kernel.pack([item])
    states=torch.zeros((1,1,41,3),dtype=torch.float64)
    controls=torch.zeros((1,1,40,2),dtype=torch.float64)
    parts,safe=kernel.cost([item],states,controls,terrain,neighbors)
    assert bool(safe[0,0])==(bit!=8)
    if bit==4:
        d=mem.view().obstacle_distance([[0.,0.,0.]],kind='crater_medium')[0][0]
        expected=40*cfg.spec.grid.dt*.5*np.clip(-d/cfg.obstacle_input.data['cost']['soft_band_m'],0,1)
        assert parts['crater_medium'][0,0]==pytest.approx(expected)
        assert parts['obstacle_overlap'][0,0]==0
    else:assert parts['crater_medium'][0,0]==0


def test_large_crater_rejects_entry_at_final_prediction_state(cfg):
    mem,_=memory(cfg,[(9.,0.,.3,8)]);item=context(cfg,mem.view())
    kernel=UpstreamMPPIKernelV4(cfg,UpstreamMPPISettingsV3(samples=2,iterations=1,device='cpu'))
    terrain,neighbors=kernel.pack([item])
    states=torch.zeros((1,2,41,3),dtype=torch.float64)
    states[0,1,-1,0]=9.
    controls=torch.zeros((1,2,40,2),dtype=torch.float64)
    _,safe=kernel.cost([item],states,controls,terrain,neighbors)
    assert safe.tolist()==[[True,False]]


@pytest.mark.parametrize('graded',[False,True])
def test_final_regenerated_trajectory_rejects_large_crater_only_in_v2(cfg,graded):
    selected=cfg if graded else cfg_from_experiment('configs/experiment/upstream_1km_obstacle_input_v1.yaml')
    local=replace(selected,planner_device='cpu',mppi_samples=2,mppi_iterations=1,
                  simulated_latency=0.,timing_version='held_previous_control_v1')
    mem,_=memory(local,[(5.,0.,.3,8 if graded else 1)])
    item=context(local,mem.view());planner=create_upstream_adapter(local)
    # finish regenerates a rate-limited trajectory and appends its braking tail.
    # This candidate bypasses the MPPI kernel and reproduces the final-review gap.
    command=np.tile([.6,0.],(40,1))
    trajectory,_=planner.trajectory_from_controls(item,command)
    assert trajectory.states[-1,0]>3.
    output=planner.finish(item,[Candidate(command,'sample')])
    assert output.valid is not graded
    if graded:
        assert output.diagnostics['candidate_rejection']=='large_crater'
        assert output.diagnostics['selected_source']=='brake'


@pytest.mark.parametrize('bit,valid',[(1,True),(4,True),(8,False)])
def test_hold_trajectory_uses_same_large_crater_final_review(cfg,bit,valid):
    local=replace(cfg,mppi_samples=2,mppi_iterations=1,simulated_latency=0.,timing_version='held_previous_control_v1')
    mem,_=memory(local,[(0.,0.,.3,bit)])
    item=context(local,mem.view(),intent=UpstreamIntent([0.,0.],True,0.,1.))
    planner=create_upstream_adapter(local)
    output=planner.prepare(item)
    assert output.valid is valid
    if not valid:assert output.diagnostics['execution_review_reason']=='large_crater'
