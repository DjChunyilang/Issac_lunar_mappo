"""Exact pose caching preserves every time-dependent full-review decision."""
from dataclasses import replace
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from _common import cfg_from_experiment
from diagnose_upstream_execution import synthetic_local_frame
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_execution import rollout_numpy
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_footprint import FootprintReviewer,FootprintTerrain,FootprintNeighbor
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_review_cache import PoseCachedReviewer


@pytest.fixture(scope='module')
def cfg():return cfg_from_experiment('configs/experiment/upstream_scale_s03.yaml')


@pytest.fixture(scope='module')
def terrain(cfg):return synthetic_local_frame(np.zeros(3),cfg.spec,0.,query_version='batch_ties_v2')


def test_stationary_pose_queries_are_reused_but_all_checks_remain(cfg,terrain):
    trajectory=rollout_numpy(np.zeros(3),np.zeros((40,2)),cfg.spec)
    lookup=FootprintTerrain.lookup;counts={};mode=None
    def counted(owner,*args,**kwargs):
        counts[mode]=counts.get(mode,0)+1
        return lookup(owner,*args,**kwargs)
    reports=[]
    with patch.object(FootprintTerrain,'lookup',counted):
        for mode,reviewer in [('reference',FootprintReviewer(cfg.spec)),('cached',PoseCachedReviewer(cfg.spec))]:
            reports.append(reviewer.trajectory(trajectory,[0.,0.],terrain,require_stop=True))
    assert reports[0]==reports[1] and reports[1].passed and reports[1].checked_poses==360
    assert counts=={'reference':400,'cached':41}


@pytest.mark.parametrize('case',['move','turn','workspace','unknown','growing_peer','parking_slope'])
def test_cached_report_equals_uncached_report_for_dynamic_and_failure_cases(cfg,terrain,case):
    controls=np.zeros((40,2));kwargs={};view=terrain
    if case in {'move','turn'}:
        controls[:4,0]=[.04,.08,.12,.04]
        if case=='turn':controls[:2,1]=[.03,.04]
    if case=='workspace':kwargs['workspace']=[[-1.,-1.],[1.,1.]]
    if case=='unknown':
        snapshot=terrain.snapshot();mask=snapshot.xy[:,0]>.5
        view=FootprintTerrain(replace(snapshot,xy=snapshot.xy[mask],features=snapshot.features[mask],
            observed_at=snapshot.observed_at[mask]),cfg.spec,query_version='batch_ties_v2')
    if case=='growing_peer':
        kwargs['neighbors']=(FootprintNeighbor.for_spec(1,[0.,8.],[[10.,0.]]*2,
            [2*cfg.spec.vehicle.effective_radius,11.],cfg.spec),)
    if case=='parking_slope':
        snapshot=terrain.snapshot();features=snapshot.features.copy();features[:,1]=np.tan(np.deg2rad(7.))
        features[:,0]=snapshot.xy[:,0]*features[:,1]
        view=FootprintTerrain(replace(snapshot,features=features),cfg.spec,query_version='batch_ties_v2')
        kwargs['parking_end']=True
    trajectory=rollout_numpy(np.zeros(3),controls,cfg.spec)
    # A sloped trajectory must be generated from its local gain, not falsely
    # use a flat velocity. Here it is stationary, so both are identical.
    old=FootprintReviewer(cfg.spec).trajectory(trajectory,[0.,0.],view,require_stop=True,**kwargs)
    new=PoseCachedReviewer(cfg.spec).trajectory(trajectory,[0.,0.],view,require_stop=True,**kwargs)
    assert new==old
    if case=='growing_peer':assert not new.passed and new.reason=='neighbor_swept_clearance' and new.checked_poses>200
    if case=='parking_slope':assert not new.passed and new.reason=='parking_slope'


def test_cache_is_discarded_between_views_and_after_exception(cfg,terrain):
    reviewer=PoseCachedReviewer(cfg.spec);trajectory=rollout_numpy(np.zeros(3),np.zeros((40,2)),cfg.spec)
    assert reviewer.trajectory(trajectory,[0.,0.],terrain).passed
    snapshot=terrain.snapshot();features=snapshot.features.copy();features[:,4]=0.
    hazardous=FootprintTerrain(replace(snapshot,features=features),cfg.spec,query_version='batch_ties_v2')
    assert reviewer.trajectory(trajectory,[0.,0.],hazardous).reason=='impassable_footprint'
    with pytest.raises(ValueError):reviewer.trajectory(trajectory,[0.,0.],terrain,workspace=[0.])
    assert reviewer._scope.get() is None
    assert reviewer.trajectory(trajectory,[0.,0.],terrain).passed


def test_old_configuration_identity_has_no_cache_and_rejects_new_state(cfg,terrain):
    from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_factory import UpstreamPlannerPool
    from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_planning import PlanningAdapter
    old=cfg_from_experiment('configs/experiment/upstream_scale_s02.yaml')
    assert 'review_acceleration_version' not in old.metadata
    assert type(PlanningAdapter(old).reviewer) is FootprintReviewer
    assert type(PlanningAdapter(cfg).reviewer) is PoseCachedReviewer
    with pytest.raises(ValueError,match='mismatch'):UpstreamPlannerPool(old).load_state_dict(UpstreamPlannerPool(cfg).state_dict())
