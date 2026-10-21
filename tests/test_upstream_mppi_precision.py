"""Boundary-preserving sampling/actuator precision with real CPU/CUDA kernels."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import math
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from _common import cfg_from_experiment
from diagnose_upstream_execution import synthetic_local_frame
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_execution import (
    brake_sequence,control_error,limit_control_numpy,midpoint_numpy,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_factory import create_upstream_adapter,UpstreamPlannerPool,plan_batch
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_mppi import UpstreamMPPIKernel,UpstreamMPPISettings
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_mppi_precision import UpstreamMPPIKernelV2,UpstreamMPPIAdapterV2
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_planning import UpstreamIntent,UpstreamPlannerInput
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_timing import ActualControlHistory


@pytest.fixture(scope='module')
def cfg():
    return replace(cfg_from_experiment('configs/experiment/upstream_scale_s03_precision_v2.yaml'),
                   simulated_latency=0.,mppi_samples=8)


@pytest.fixture(scope='module')
def terrain(cfg):return synthetic_local_frame(np.zeros(3),cfg.spec,0.,query_version='batch_ties_v2')


@pytest.fixture(scope='module',autouse=True)
def threads():
    previous=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def request(cfg,terrain,previous=(.050000001,.25),*,agent=0,env=0,cap=None,hold=False):
    previous=np.asarray(previous,float)
    history=ActualControlHistory(env,agent,cfg.spec.fingerprint,[0.],np.zeros((1,3)),np.zeros((0,2)),np.zeros((0,2)),previous)
    return UpstreamPlannerInput(env,agent,np.zeros(3),previous,UpstreamIntent([.8,0.],hold,0.,1.,speed_cap=cap),
        terrain,(),previous,0.,np.arange(41)*.2,(-100.,-100.,100.,100.),cfg.spec,
        information_pose=np.zeros(3),information_time=0.,actual_history=history)


def reference(context,raw):
    """NumPy physical integration, independently iterating a single sample."""
    states=[context.ego_pose];controls=[];velocities=[];last=context.previous_control;stopping=False
    cap=context.spec.vehicle.forward_limit if context.intent.speed_cap is None else context.intent.speed_cap
    reverse=min(cap,context.spec.vehicle.reverse_limit);v=context.spec.vehicle;dt=context.spec.grid.dt
    for t,target in enumerate(np.asarray(raw,float)):
        target=target.copy();target[0]=np.clip(target[0],-reverse,cap)
        bound=math.ceil(abs(last[0])/(v.linear_deceleration*dt)-1e-10)+math.ceil(abs(last[1])/(v.angular_deceleration*dt)-1e-10)+1
        stopping|=bound>=40-t
        command=limit_control_numpy(np.zeros(2) if stopping else target,last,context.spec)
        command[abs(command)<=1e-10]=0.
        state,velocity=midpoint_numpy(states[-1],command,context.spec)
        states.append(state);controls.append(command);velocities.append(velocity);last=command
    return np.asarray(states),np.asarray(controls),np.asarray(velocities)


@pytest.mark.parametrize('previous',[(.050000001,.25),(-.050000001,.25),(.050000001,-.35),
    (-.050000001,-.35),(.05,.2),(-.05,-.2),(25/9,0.),(-.5,0.),(25/9-.9*.35,.35)])
def test_exact_feedback_and_all_actuator_boundaries_match_numpy_reference(cfg,terrain,previous):
    context=request(cfg,terrain,previous)
    assert not control_error(context.previous_control,context.previous_control,cfg.spec)
    kernel=UpstreamMPPIKernelV2(cfg);raw=kernel.tensor(np.random.default_rng(23).normal(size=(1,3,40,2))*[.25,.10])
    packed,_=kernel.pack([context]);states,controls,velocity,numeric,known=kernel.rollout([context],raw,packed)
    assert states.dtype==controls.dtype==velocity.dtype==torch.float64 and raw.dtype==torch.float32
    for k in range(3):
        expected=reference(context,raw[0,k].cpu().numpy())
        for actual,wanted in zip((states[0,k],controls[0,k],velocity[0,k]),expected):
            np.testing.assert_allclose(actual.cpu().numpy(),wanted,atol=1e-10,rtol=1e-8)
        previous=np.asarray(context.previous_control)
        for command in controls[0,k].cpu().numpy():
            assert not control_error(command,previous,cfg.spec);previous=command
    assert numeric.all() and known.shape==(1,3,41)
    np.testing.assert_array_equal(states[0,0,0].cpu(),context.ego_pose)


def test_original_failure_is_fixed_without_changing_legacy_kernel(cfg,terrain):
    context=request(cfg,terrain);legacy=UpstreamMPPIKernel(cfg);view,_=legacy.pack([context])
    with pytest.raises(ValueError,match='Previous command'):legacy.rollout([context],legacy.tensor(np.zeros((1,1,40,2))),view)
    fixed=UpstreamMPPIKernelV2(cfg);view,_=fixed.pack([context])
    states,controls,*_=fixed.rollout([context],fixed.tensor(np.zeros((1,1,40,2))),view)
    np.testing.assert_allclose(controls[0,0,0].numpy(),[.050000001,.21],atol=1e-10,rtol=1e-8)
    assert np.isfinite(states.numpy()).all()
    assert 'precision_version' not in legacy.metadata
    assert fixed.metadata['actuator_dtype']=='float64' and fixed.metadata['sampling_dtype']=='float32'


def test_lowered_cap_prescreen_retains_only_physical_braking_transient(cfg,terrain):
    context=request(cfg,terrain,(.2,0.),cap=.05);kernel=UpstreamMPPIKernelV2(cfg)
    views,peers=kernel.pack([context]);brake=brake_sequence(context.previous_control,cfg.spec)
    expected=np.maximum(.05,np.pad(abs(brake[:,0]),(0,40-len(brake))))
    np.testing.assert_allclose(peers.actuator_cap_curve_v2[0].cpu(),expected,atol=1e-10,rtol=1e-8)
    # The trajectory remains finite and follows the legal initial transient.
    raw=kernel.tensor(np.zeros((1,1,40,2)));states,controls,*_=kernel.rollout([context],raw,views)
    parts,safe=kernel.cost([context],states,controls,views,peers)
    assert safe.item() and all(torch.isfinite(value).all() for value in parts.values())
    bad=controls.clone();bad[0,0,10,0]=.051
    _,safe=kernel.cost([context],states,bad,views,peers)
    assert not safe.item()


def test_low_speed_cap_cannot_be_bypassed_with_high_yaw_polygon_epsilon(cfg,terrain):
    context=request(cfg,terrain,(.05,.2),cap=.05);kernel=UpstreamMPPIKernelV2(cfg)
    views,peers=kernel.pack([context]);raw=kernel.tensor(np.tile([.05,.35],(1,1,40,1)))
    states,controls,*_=kernel.rollout([context],raw,views)
    assert (controls[...,0]>.05+1e-10).any()
    _,safe=kernel.cost([context],states,controls,views,peers)
    assert not safe.item()
    from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_execution import rollout_numpy
    trajectory=rollout_numpy(context.ego_pose,controls[0,0].numpy(),cfg.spec)
    assert create_upstream_adapter(cfg)._review(context,trajectory)=='intent_speed_cap'


@pytest.mark.parametrize('seed',[23,31,47])
def test_real_v2_factory_raw_probability_state_and_restore(cfg,terrain,seed):
    config=replace(cfg,seed=seed);context=request(config,terrain)
    first=create_upstream_adapter(config);assert type(first) is UpstreamMPPIAdapterV2
    output=first.plan(context)
    assert np.isfinite(output.trajectory.states).all() and output.diagnostics['kernel']=='upstream_mppi_kernel_actuator_v2'
    samples=first.last_samples
    assert samples['raw_controls'].dtype==samples['raw_noise'].dtype==torch.float32
    assert samples['controls'].dtype==samples['states'].dtype==torch.float64
    torch.testing.assert_close(samples['raw_controls'],samples['proposal_mean'][None]+samples['raw_noise'],rtol=0,atol=0)
    restored=create_upstream_adapter(config);restored.load_state_dict(first.state_dict())
    left,right=first.plan(context),restored.plan(context)
    np.testing.assert_array_equal(left.trajectory.controls,right.trajectory.controls)
    torch.testing.assert_close(first.last_samples['raw_noise'],restored.last_samples['raw_noise'],rtol=0,atol=0)
    legacy=replace(config,planner_adapter='mppi_upstream_v1')
    with pytest.raises(ValueError,match='mismatch'):create_upstream_adapter(legacy).load_state_dict(first.state_dict())
    broken=deepcopy(first.state_dict());broken['adapter_state']['kernel']['actuator_dtype']='float32'
    with pytest.raises(ValueError,match='mismatch'):first.load_state_dict(broken)


def test_true_batch_permutation_and_sparse_reset_do_not_change_rng(cfg,terrain):
    contexts=[request(cfg,terrain,previous=(.050000001,.25),env=e,agent=a) for e in range(2) for a in range(4)]
    pool=UpstreamPlannerPool(cfg);outputs=pool.plan_batch(contexts)
    assert len(outputs)==8 and {o.diagnostics['batch_size'] for o in outputs}=={8}
    saved=pool.state_dict();restored=UpstreamPlannerPool(cfg);restored.load_state_dict(saved)
    left=pool.plan_batch(contexts);right=restored.plan_batch(contexts[::-1])[::-1]
    for a,b in zip(left,right):np.testing.assert_array_equal(a.trajectory.controls,b.trajectory.controls)
    pool.reset(0);other=create_upstream_adapter(cfg);state=next(x['state'] for x in pool.state_dict()['planners'] if x['identity']==(1,0))
    other.load_state_dict(state)
    a=pool.plan(contexts[4]);b=other.plan(contexts[4])
    np.testing.assert_array_equal(a.trajectory.controls,b.trajectory.controls)


def test_legacy_and_new_kernels_are_never_shared_or_mixed(cfg,terrain):
    cache={};old_cfg=replace(cfg,planner_adapter='mppi_upstream_v1')
    old=create_upstream_adapter(old_cfg,kernel_cache=cache);new=create_upstream_adapter(cfg,kernel_cache=cache)
    assert len(cache)==2 and old.batch_kernel is not new.batch_kernel
    contexts=[request(cfg,terrain,previous=(0.,0.),env=e) for e in range(2)]
    outputs=plan_batch([new,old],contexts)
    assert [o.diagnostics['adapter_id'] for o in outputs]==['mppi_upstream_v2','mppi_upstream_v1']
    assert old.state_version==1 and new.state_version==2
    with pytest.raises(ValueError,match='legacy'):UpstreamMPPIAdapterV2(cfg,kernel=old.batch_kernel)


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_split_precision_nonfinite_samples_and_empty_batch(cfg,terrain,device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('Actual CUDA required for strict acceptance')
    config=replace(cfg,planner_device=device);kernel=UpstreamMPPIKernelV2(config);context=request(config,terrain)
    assert kernel.plan_batch([],[])==[]
    raw=kernel.tensor(np.full((1,3,40,2),np.nan));view,_=kernel.pack([context])
    states,controls,velocities,numeric,known=kernel.rollout([context],raw,view)
    assert not numeric.any() and torch.isfinite(states).all() and torch.isfinite(controls).all()


@pytest.mark.parametrize('previous',[(.050000001,.25),(-.050000001,-.35),(25/9,0.)])
def test_real_cuda_injected_noise_matches_cpu_at_boundaries(cfg,terrain,previous):
    if not torch.cuda.is_available():pytest.skip('Actual CUDA required for strict acceptance')
    contexts=[request(cfg,terrain,previous,env=0),request(cfg,terrain,previous,env=1)]
    noise=np.random.default_rng(31).normal(size=(2,2,8,40,2)).astype(np.float32)*[.25,.10]
    cpu=UpstreamMPPIKernelV2(cfg);gpu_cfg=replace(cfg,planner_device='cuda');gpu=UpstreamMPPIKernelV2(gpu_cfg)
    cpu_handles=[create_upstream_adapter(cfg,kernel_cache={}) for _ in contexts]
    gpu_handles=[create_upstream_adapter(gpu_cfg,kernel_cache={}) for _ in contexts]
    left=cpu.plan_batch(cpu_handles,contexts,noise=noise);right=gpu.plan_batch(gpu_handles,contexts,noise=noise)
    for a,b,p,q in zip(left,right,cpu_handles,gpu_handles):
        np.testing.assert_allclose(a.trajectory.states,b.trajectory.states,atol=1e-5,rtol=1e-5)
        np.testing.assert_allclose(a.trajectory.controls,b.trajectory.controls,atol=1e-5,rtol=1e-5)
        np.testing.assert_allclose(p.last_samples['weights'],q.last_samples['weights'],atol=1e-5,rtol=1e-5)
        assert a.valid==b.valid and a.feedback.status==b.feedback.status
