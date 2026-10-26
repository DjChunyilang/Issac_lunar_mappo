"""CUDA swept checks retain CPU first-failure and local-support contracts."""
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from _common import cfg_from_experiment
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.protocol import TerrainSnapshot
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_footprint import FootprintTerrain,FootprintReviewer,FootprintNeighbor
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_flat_review import CudaSweepReviewer,create_reviewer
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_execution import rollout_numpy,terrain_gain_numpy


def test_cached_cuda_index_released_without_cyclic_gc():
    import gc
    import weakref
    s=cfg_from_experiment('configs/experiment/upstream_1km_cuda_review_v1.yaml').spec
    x,y=np.meshgrid(np.arange(-9.,9.01,.2),np.arange(-6.,6.01,.2),indexing='ij')
    xy=np.c_[x.ravel(),y.ravel()];f=np.zeros((len(xy),5))
    f[:,0]=.02*xy[:,0];f[:,1]=.02;f[:,4]=.9
    terrain=FootprintTerrain(TerrainSnapshot(xy,f,np.zeros(len(xy)),s.perception.support_radius),s)
    reviewer=CudaSweepReviewer(s)
    poses=np.c_[np.linspace(0.,.1,9),np.zeros(9),np.full(9,.17)]
    reviewer._strict_static_passes(poses,terrain)
    key=('cuda_sweep_index_v1',s.fingerprint)
    index=weakref.ref(terrain._review_witnesses[key])
    refreshed=terrain.with_observation_times(np.ones(len(xy)))
    was_enabled=gc.isenabled();gc.disable()
    try:
        del terrain
        assert index() is not None  # Timestamp-only refresh retains the index.
        actual=index().lookup(torch.tensor(poses[None,:,:2],device='cuda'))
        expected=refreshed.lookup(poses[:,:2])
        np.testing.assert_array_equal(actual[0].cpu()[0],expected[0])
        np.testing.assert_array_equal(actual[1].cpu()[0],expected[1])
        del refreshed
        assert index() is None  # Last geometry owner releases CUDA tensors now.
    finally:
        if was_enabled:gc.enable()


@pytest.mark.parametrize('case',['slope','rough','step','hole','impassable','workspace','neighbor','safe','threshold'])
def test_nonflat_moving_sweep_matches_cpu(case):
    assert torch.cuda.is_available(), 'This acceptance requires an actual CUDA device'
    torch.set_num_threads(1)
    s=cfg_from_experiment('configs/experiment/upstream_1km_cuda_review_v1.yaml').spec
    x,y=np.meshgrid(np.arange(-9.,9.01,.2),np.arange(-6.,6.01,.2),indexing='ij')
    xy=np.c_[x.ravel(),y.ravel()]
    f=np.zeros((len(xy),5));f[:,0]=.02*xy[:,0];f[:,1]=.02;f[:,4]=.9
    if case=='slope':f[:,1]=s.terrain.driving_slope+.001
    if case=='threshold':f[:,1]=s.terrain.driving_slope+1e-8
    if case in {'rough','step'}:f[np.linalg.norm(xy-[1.35,.9],axis=-1)<.4,0]+=.5
    if case=='impassable':f[np.linalg.norm(xy-[1.35,.9],axis=-1)<.4,4]=0.
    if case=='hole':
        keep=np.linalg.norm(xy-[1.35,.9],axis=-1)>.4;xy,f=xy[keep],f[keep]
    terrain=FootprintTerrain(TerrainSnapshot(xy,f,np.zeros(len(xy)),s.perception.support_radius),s)
    def gain(p):
        feat,known=terrain.lookup(p[:2]);assert known[0]
        return terrain_gain_numpy(p,feat[0],s)
    u=np.tile([.1,.002],(3,1))
    trajectory=rollout_numpy([0.,0.,.17],u,s,gain=gain)
    kwargs={}
    if case=='workspace':kwargs['workspace']=np.array([[-10.,-10.],[s.vehicle.effective_radius+.015,10.]])
    if case=='neighbor':kwargs['neighbors']=[FootprintNeighbor.for_spec(1,[0.,1.],[[10.,0.]]*2,[7.,14.],s)]
    expected=FootprintReviewer(s).trajectory(trajectory,u[0],terrain,**kwargs)
    actual=CudaSweepReviewer(s).trajectory(trajectory,u[0],terrain,**kwargs)
    assert actual==expected


def test_candidate_explicit_identity_and_copy():
    from copy import deepcopy
    a=cfg_from_experiment('configs/experiment/upstream_1km_curriculum_v1.yaml')
    b=cfg_from_experiment('configs/experiment/upstream_1km_cuda_review_v1.yaml')
    assert a.review_acceleration_version=='flat_frame_pose_cache_v3'
    assert a.metadata!=b.metadata
    reviewer=create_reviewer(b.spec,b.review_acceleration_version)
    assert isinstance(deepcopy(reviewer),CudaSweepReviewer)


@pytest.mark.parametrize('case',['safe','late_static','control_priority'])
def test_full_horizon_batch_preserves_first_failure(case):
    s=cfg_from_experiment('configs/experiment/upstream_1km_cuda_review_v1.yaml').spec
    x,y=np.meshgrid(np.arange(-9.,9.01,.2),np.arange(-6.,6.01,.2),indexing='ij')
    xy=np.c_[x.ravel(),y.ravel()];f=np.zeros((len(xy),5))
    f[:,0]=.02*xy[:,0];f[:,1]=.02;f[:,4]=.9
    if case!='safe':f[xy[:,0]>3.3,0]+=.5
    terrain=FootprintTerrain(TerrainSnapshot(xy,f,np.zeros(len(xy)),s.perception.support_radius),s)
    def gain(p):
        feature,known=terrain.lookup(p[:2]);assert known[0]
        return terrain_gain_numpy(p,feature[0],s)
    controls=np.tile([.2,.002],(40,1))
    trajectory=rollout_numpy([0.,0.,.17],controls,s,gain=gain)
    if case=='control_priority':trajectory.controls[1,0]=9.
    expected=FootprintReviewer(s).trajectory(trajectory,controls[0],terrain)
    actual=CudaSweepReviewer(s).trajectory(trajectory,controls[0],terrain)
    assert actual==expected
    if case=='safe':assert actual.passed
    if case=='late_static':assert not actual.passed and actual.checked_poses>9
    if case=='control_priority':assert not actual.passed and actual.checked_poses==9


def test_static_proof_cache_reuses_exact_poses_but_not_changed_terrain():
    from unittest.mock import patch
    from dataclasses import replace
    s=cfg_from_experiment('configs/experiment/upstream_1km_cuda_review_v1.yaml').spec
    x,y=np.meshgrid(np.arange(-9.,9.01,.2),np.arange(-6.,6.01,.2),indexing='ij')
    xy=np.c_[x.ravel(),y.ravel()];f=np.zeros((len(xy),5))
    f[:,0]=.02*xy[:,0];f[:,1]=.02;f[:,4]=.9
    snapshot=TerrainSnapshot(xy,f,np.zeros(len(xy)),s.perception.support_radius)
    terrain=FootprintTerrain(snapshot,s);r=CudaSweepReviewer(s)
    poses=np.c_[np.linspace(0.,.1,9),np.zeros(9),np.full(9,.17)]
    with patch.object(r,'_uncached_static_passes',wraps=r._uncached_static_passes) as calls:
        first=r._strict_static_passes(poses,terrain);assert first.all()
        assert np.array_equal(first,r._strict_static_passes(poses,terrain))
        refreshed=terrain.with_observation_times(np.ones(len(xy)))
        assert np.array_equal(first,r._strict_static_passes(poses,refreshed))
        assert calls.call_count==1
        f[:,4]=0.
        changed=FootprintTerrain(replace(snapshot,features=f),s)
        assert not r._strict_static_passes(poses,changed).any()
        assert calls.call_count==2
