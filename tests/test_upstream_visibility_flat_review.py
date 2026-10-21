"""Legal transit geometry and unchanged full-footprint regular-grid decisions."""
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from _common import cfg_from_experiment
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_visibility import visibility_waypoint
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_flat_review import FlatGridPoseReviewer
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_footprint import FootprintTerrain,FootprintReviewer
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.protocol import TerrainSnapshot


def spec():return cfg_from_experiment('configs/experiment/upstream_scale_s04_visibility_recovery_cuda_v2.yaml').spec


def field(s,angle=0.,height=0.,hole=False,hazard=False):
    x,y=np.meshgrid(np.linspace(-15,15,151),np.linspace(-8,8,81),indexing='ij')
    xy=np.c_[x.ravel(),y.ravel()];c,t=np.cos(angle),np.sin(angle);xy=xy@np.array([[c,t],[-t,c]])
    features=np.zeros((len(xy),5));features[:,0]=height;features[:,4]=1.
    if hazard:features[np.linalg.norm(xy,axis=-1)<.3,0]=.5
    if hole:
        keep=np.linalg.norm(xy,axis=-1)>.21;xy=xy[keep];features=features[keep]
    return FootprintTerrain(TerrainSnapshot(xy,features,np.zeros(len(xy)),s.perception.support_radius),s)


@pytest.mark.parametrize('angle,height',[(0.,0.),(.73,0.),(-2.8,1234.)])
def test_regular_flat_witness_matches_independent_full_queries(angle,height):
    torch.set_num_threads(1);s=spec();terrain=field(s,angle,height)
    original=FootprintReviewer(s);fast=FlatGridPoseReviewer(s);rng=np.random.default_rng(31)
    for _ in range(16):
        pose=np.r_[rng.uniform([-12.,-7.],[12.,7.]),rng.uniform(-np.pi,np.pi)]
        for parking in (False,True):
            a=original.pose(pose,terrain,parking=parking);b=fast.pose(pose,terrain,parking=parking)
            assert (a.passed,a.reason,a.checked_poses)==(b.passed,b.reason,b.checked_poses)
            if a.passed:np.testing.assert_allclose([a.slope,a.roughness_rms,a.maximum_step,a.minimum_body_clearance],
                [b.slope,b.roughness_rms,b.maximum_step,b.minimum_body_clearance],atol=1e-10,rtol=1e-8)
    assert fast._witness is not None


@pytest.mark.parametrize('hole,hazard',[(True,False),(False,True)])
def test_flat_witness_cannot_fill_unknown_holes_or_hide_a_body_obstacle(hole,hazard):
    s=spec();terrain=field(s,hole=hole,hazard=hazard);fast=FlatGridPoseReviewer(s)
    a=FootprintReviewer(s).pose([0.,0.,0.],terrain);b=fast.pose([0.,0.,0.],terrain)
    assert not a.passed and (b.passed,b.reason)==(a.passed,a.reason) and fast._witness is None


def segment_clear(a,b,center,radius):
    delta=b-a;t=np.clip(np.dot(center-a,delta)/max(np.dot(delta,delta),1e-20),0.,1.)
    return np.linalg.norm(a+t*delta-center)>=radius-1e-8


def test_visibility_progresses_around_large_known_body_and_binds_sender_identity():
    position=np.array([-12.,0.]);goal=np.array([7.2,0.]);circles=[(0,np.zeros(2),6.548968),(1,np.array([0.,7.2]),6.548968)]
    for _ in range(50):
        waypoint,detour=visibility_waypoint(position,goal,circles)
        assert waypoint is not None and np.linalg.norm(waypoint-position)<=4.+1e-10
        for _,center,radius in circles:assert segment_clear(position,waypoint,center,radius)
        reversed_goal,_=visibility_waypoint(position,goal,list(reversed(circles)))
        np.testing.assert_allclose(waypoint,reversed_goal,atol=1e-10)
        position=waypoint
        if np.linalg.norm(position-goal)<1e-8:break
    assert np.linalg.norm(position-goal)<1e-8
    assert visibility_waypoint([0.,0.],goal,circles)[0] is None
