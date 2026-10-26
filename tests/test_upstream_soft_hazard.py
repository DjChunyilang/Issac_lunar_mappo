"""Cost/reward-only ablation: remove predictive interception, retain outcomes."""
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from _common import cfg_from_experiment
from upstream_local_closure import runtime_config
from upstream_curriculum_experiment import load_curriculum, timing_for
from test_upstream_planning import context, ScriptAdapter, terrain as terrain_fixture
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_planning import Candidate
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_environment import UpstreamEnvironment
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_footprint import FootprintReport
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_mppi import UpstreamMPPIKernel
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_reward import UpstreamReward, configured_reward_settings

BASE='configs/experiment/upstream_1km_soft_hazard_v1.yaml'
TRAIN='configs/training/upstream_1km_far_soft_hazard_v1.yaml'


def forbidden(*args,**kwargs):
    raise AssertionError('Predictive footprint path executed')


def test_explicit_mode_survives_runtime_overrides_and_rejects_old_state(monkeypatch):
    cfg,_=runtime_config(BASE,name='soft',seed=23,safety=True)
    assert cfg.execution_safety_version=='projection_disabled' and not cfg.predictive_footprint_checks
    cfg=replace(cfg,simulated_latency=0.,timing_version='held_previous_control_v1')
    terrain=terrain_fixture.__wrapped__(cfg);adapter=ScriptAdapter(cfg)
    monkeypatch.setattr(adapter.reviewer,'trajectory',forbidden)
    output=adapter.finish(context(cfg,terrain),[Candidate(np.zeros((40,2)),'updated_mean')])
    assert output.valid and not output.diagnostics['execution_review_passed']
    assert not output.diagnostics['execution_review_performed']
    assert not output.diagnostics['information_reserve_verified']
    old=ScriptAdapter(replace(cfg,hazard_response_version='strict_footprint_v1'))
    with pytest.raises(ValueError):adapter.load_state_dict(old.state_dict())


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_roughness_cost_uses_existing_query_and_keeps_hard_prescreen(device):
    cfg=replace(cfg_from_experiment(BASE),planner_device=device)
    strict=replace(cfg,hazard_response_version='strict_footprint_v1')
    item=context(cfg,terrain_fixture.__wrapped__(cfg))
    states=torch.zeros((1,2,41,3),device=device,dtype=torch.float64)
    states[:,1,:,0]=1
    controls=torch.zeros((1,2,40,2),device=device,dtype=torch.float64)
    class Terrain:
        calls=0
        def lookup(self,points):
            self.calls+=1
            f=torch.zeros((*points.shape[:-1],5),device=device,dtype=torch.float64)
            f[...,4]=1;f[...,3]=points[...,0]*cfg.spec.terrain.roughness_rms*.5
            return f,torch.ones(points.shape[:-1],device=device,dtype=torch.bool)
    class Neighbors:
        def risk_and_safe(self,s):
            return s.new_zeros(s.shape[:2]),torch.ones(s.shape[:2],device=device,dtype=torch.bool)
    a=Terrain();b=Terrain()
    old,safe_old=UpstreamMPPIKernel(strict).cost([item],states,controls,a,Neighbors())
    new,safe_new=UpstreamMPPIKernel(cfg).cost([item],states,controls,b,Neighbors())
    assert a.calls==b.calls==1
    assert torch.equal(safe_old,safe_new)
    assert new['terrain'][0,0]==old['terrain'][0,0]
    assert new['terrain'][0,1]>old['terrain'][0,1]


def test_environment_issues_limited_control_without_review_but_keeps_terrain_failure(monkeypatch):
    cfg=cfg_from_experiment(BASE)
    cfg=replace(cfg,simulation=replace(cfg.simulation,num_envs=1),mppi_samples=8)
    poses=np.zeros((1,4,3));poses[0,:,:2]=[[0,0],[12,0],[0,12],[12,12]]
    env=UpstreamEnvironment(cfg,poses);assert env.safety is None
    monkeypatch.setattr(env.reviewer,'trajectory',forbidden)
    limited=np.array([.04,0.]);p=env.local_pose(0,0)
    result,_=env._project_control((0,0,p,env.memory[0][0].view(),limited,'plan',limited,()))
    np.testing.assert_array_equal(result.control,limited)
    assert not result.verified and result.checked_candidates==0
    env.task.terrain_failure_terminal=True
    original=env._reports
    def reports(*,parking=True):
        if parking:return original(parking=True)
        return [[FootprintReport(False,'terrain_step')]*4]
    monkeypatch.setattr(env,'_reports',reports)
    env.tick()
    assert env.last_task['terminal_reason'].tolist()==['terrain_failure']
    assert env.last_task['learning_valid'].all()


def test_reward_penalizes_roughness_and_far_terrain_failure():
    cfg=cfg_from_experiment(BASE);config=load_curriculum(TRAIN)
    g,l=timing_for(config).discounts(.2);settings=configured_reward_settings(g,l,config)
    facts={'dmax':np.array([600.]),'dispersion':np.array([90000.]),
        'minimum_distance':np.array([100.]),'flatness_cost':np.ones(1),'hold_count':np.zeros(1)}
    task={'learning_valid':np.ones(1,bool),'ended_this_step':np.zeros(1,bool),'terminal_reason':np.array(['none'])}
    f=np.zeros((1,4,5));f[...,4]=1
    def reward(features,task):
        return UpstreamReward(cfg.spec,1,settings).advance(facts,facts,task,np.zeros((1,4,2)),features,np.ones((1,4)),np.zeros((1,4)))[0][0]
    baseline=reward(f,task);rock=f.copy();rock[...,3]=cfg.spec.terrain.roughness_rms*.5
    assert reward(rock,task)<baseline<0
    ended=dict(task,ended_this_step=np.ones(1,bool),terminal_reason=np.array(['terrain_failure']))
    assert reward(rock,ended)<0


def test_rejected_mppi_mean_cannot_bypass_prescreen_or_reuse_old_plan(monkeypatch):
    from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_mppi import UpstreamMPPIAdapter, UpstreamMPPISettings
    cfg=replace(cfg_from_experiment(BASE),planner_device='cpu',simulated_latency=0.,timing_version='held_previous_control_v1')
    item=context(cfg,terrain_fixture.__wrapped__(cfg))
    kernel=UpstreamMPPIKernel(cfg,UpstreamMPPISettings(samples=2,iterations=1,device='cpu'))
    adapter=UpstreamMPPIAdapter(cfg,kernel=kernel)
    monkeypatch.setattr(adapter.reviewer,'trajectory',forbidden)
    evaluate=kernel.evaluate
    def rejected(*args,**kwargs):
        result=list(evaluate(*args,**kwargs));result[5]=torch.zeros_like(result[5]);return tuple(result)
    monkeypatch.setattr(kernel,'evaluate',rejected)
    prior,_=adapter.trajectory_from_controls(item,np.zeros((40,2)))
    adapter._previous_trajectory=prior
    output=kernel.plan_batch([adapter],[item])[0]
    assert not output.valid
    assert output.feedback.emergency_brake
    assert not output.diagnostics['full_review_passed']
    assert output.diagnostics['selected_source']=='brake'
