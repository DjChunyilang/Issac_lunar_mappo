"""A complete latest frame may certify flat support; older hazards still veto."""
from dataclasses import replace

import numpy as np
import pytest

from test_upstream_visibility_flat_review import field, spec
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_flat_review import FlatFramePoseReviewer,FlatGridPoseReviewer
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_footprint import FootprintTerrain,FootprintReviewer


def mixed(s,hazard=False):
    old=field(s,angle=.37).snapshot();new=field(s,angle=-.68).snapshot()
    xy=np.vstack((old.xy,new.xy));features=np.vstack((old.features,new.features))
    if hazard:features[0,0]=1.
    return FootprintTerrain(replace(new,xy=xy,features=features,
        observed_at=np.r_[np.zeros(len(old.xy)),np.ones(len(new.xy))]),s)


def test_complete_rotated_frame_with_old_memory_matches_full_reference():
    s=spec();terrain=mixed(s);old=FlatGridPoseReviewer(s);fast=FlatFramePoseReviewer(s);reference=FootprintReviewer(s)
    poses=[[0.,0.,0.],[2.,-1.,.6],[10.,5.,1.4],[30.,30.,0.]]
    for pose in poses:
        for parking in (False,True):
            a=reference.pose(pose,terrain,parking=parking);b=fast.pose(pose,terrain,parking=parking)
            assert (a.passed,a.reason,a.checked_poses)==(b.passed,b.reason,b.checked_poses)
            if a.passed:np.testing.assert_allclose([a.slope,a.roughness_rms,a.maximum_step,a.minimum_body_clearance],
                [b.slope,b.roughness_rms,b.maximum_step,b.minimum_body_clearance],atol=1e-10,rtol=1e-8)
    old.pose([0.,0.,0.],terrain)
    assert old._witness is None and fast._witness is not None


@pytest.mark.parametrize('kind',['old_hazard','unknown_hole'])
def test_frame_proof_cannot_hide_old_hazard_or_fill_a_hole(kind):
    s=spec();terrain=mixed(s,hazard=kind=='old_hazard')
    if kind=='unknown_hole':
        snapshot=terrain.snapshot();keep=np.linalg.norm(snapshot.xy,axis=1)>.25
        terrain=FootprintTerrain(replace(snapshot,xy=snapshot.xy[keep],features=snapshot.features[keep],
            observed_at=snapshot.observed_at[keep]),s)
    fast=FlatFramePoseReviewer(s);a=FootprintReviewer(s).pose([0.,0.,0.],terrain);b=fast.pose([0.,0.,0.],terrain)
    assert fast._witness is None and (a.passed,a.reason)==(b.passed,b.reason)
