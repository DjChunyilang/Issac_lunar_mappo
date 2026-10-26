"""Shared soft queries and compact transfers preserve actual planning results."""
from dataclasses import replace
import numpy as np
import pytest
import torch

from test_upstream_crater_grid import memory
from test_upstream_mppi_body_geometry import cfg_from_experiment, context
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_factory import create_upstream_adapter
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_grid import GridTerrainBatch


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_shared_soft_channels_are_bitwise_equal_including_reordered_rows(device):
    assert device!='cuda' or torch.cuda.is_available()
    cfg=cfg_from_experiment('configs/experiment/upstream_1km_mppi_body_geometry_v1.yaml')
    kernel=create_upstream_adapter(cfg).batch_kernel
    views=[kernel.planning_terrain(memory(cfg,obs)[0].view()) for obs in
           ([(0.,0.,.3,4)],[(4.2,0.,.9,8),(0.,3.,.3,2)],[])]
    batch=GridTerrainBatch(views,device)
    rng=np.random.default_rng(52)
    # Cross the 4096-pose block and row boundaries, including unknown tiles.
    poses=torch.tensor(rng.uniform([-25,-25,-3.14],[25,25,3.14],(3,1400,3)),device=device)
    for selected,points in [(batch,poses),(batch.select([2,0,1]),poses[[2,0,1]])]:
        actual=selected.obstacle_distances(points)
        for i,kind in enumerate(('obstacle','crater_medium')):
            assert torch.equal(actual[...,i],selected.obstacle_distance(points,kind=kind))


@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('blocked',[False,True])
def test_compact_transfer_preserves_controls_state_and_rng(device,blocked):
    assert device!='cuda' or torch.cuda.is_available()
    torch.set_num_threads(1)
    cfg=replace(cfg_from_experiment('configs/experiment/upstream_1km_mppi_body_geometry_v1.yaml'),
                planner_device=device,mppi_samples=8,mppi_iterations=2,simulated_latency=0.,
                timing_version='held_previous_control_v1')
    view=memory(cfg,[(0.,0.,.9,8)] if blocked else [(5.,3.,.4,4)])[0].view()
    item=context(cfg,view)
    a,b=create_upstream_adapter(cfg),create_upstream_adapter(cfg)
    noise=np.random.default_rng(7).normal(size=(2,1,8,40,2))*[.2,.1]
    full=a.batch_kernel.plan_batch([a],[item],noise=noise,record_samples=True)[0]
    compact=b.batch_kernel.plan_batch([b],[item],noise=noise,record_samples=False)[0]
    assert full.valid==compact.valid
    assert full.feedback.reason==compact.feedback.reason
    np.testing.assert_array_equal(full.trajectory.controls,compact.trajectory.controls)
    np.testing.assert_array_equal(full.trajectory.states,compact.trajectory.states)
    assert full.diagnostics.get('selected_cost_parts')==compact.diagnostics.get('selected_cost_parts')
    assert a.last_samples is not None and b.last_samples is None
    assert compact.diagnostics['host_transfer_bytes']<full.diagnostics['host_transfer_bytes']
    assert torch.equal(a.generator.get_state(),b.generator.get_state())
    if a.warm is not None:np.testing.assert_array_equal(a.warm,b.warm)


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_sat_observed_cells_cpu_gpu_and_reordered_rows(device):
    from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_grid import _intersection_areas
    cfg=cfg_from_experiment('configs/experiment/upstream_1km_mppi_grid_sat_v1.yaml')
    kernel=create_upstream_adapter(cfg).batch_kernel
    original=[memory(cfg,obs)[0].view() for obs in
              ([(0.,0.,.4,8),(5.,3.,.9,8)],[],[(0.,0.,.4,4)])]
    views=[kernel.planning_terrain(v) for v in original]
    batch=GridTerrainBatch(views,device)
    rng=np.random.default_rng(53)
    poses=rng.uniform([-8,-8,-3.14],[8,8,3.14],(3,40,3))
    half=np.array([cfg.spec.vehicle.length,cfg.spec.vehicle.width])/2
    expected=[]
    for view,points in zip(views,poses):
        shapes=view.large_crater_rectangles()
        corners=shapes[:,None,:2]+np.array([[-1,-1],[1,-1],[1,1],[-1,1]])[None]*shapes[:,None,2:]
        hits=[]
        for p in points:
            c,s=np.cos(p[2]),np.sin(p[2]);body=(corners-p[:2])@np.array([[c,-s],[s,c]])
            hits.append(bool((_intersection_areas(body,(-half[0],half[0],-half[1],half[1]))>1e-10).any()))
        expected.append(hits)
        np.testing.assert_array_equal(view.obstacle_distance(points,kind='crater_large')[0]<0,hits)
    tensor=torch.tensor(poses,device=device)
    np.testing.assert_array_equal((batch.obstacle_distance(tensor,kind='crater_large')<0).cpu(),expected)
    ids=[2,0,1]
    np.testing.assert_array_equal((batch.select(ids).obstacle_distance(tensor[ids],kind='crater_large')<0).cpu(),np.array(expected)[ids])
    assert not hasattr(original[0],'large_crater_query')
    legacy=create_upstream_adapter(cfg_from_experiment('configs/experiment/upstream_1km_mppi_body_geometry_v1.yaml'))
    assert 'large_crater_query' not in legacy.batch_kernel.metadata['settings']
    with pytest.raises(ValueError):create_upstream_adapter(cfg).load_state_dict(legacy.state_dict())


def test_sat_touch_and_containment():
    from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_grid import _rectangle_hits_numpy
    shape=np.array([[0.,0.,.2,.3]])
    half=np.array([2.5075,1.6]);touch=half[0]+.2
    assert _rectangle_hits_numpy([[0,0,0],[touch,0,0],[touch-1e-6,0,0],[touch+1e-6,0,0]],shape,half).tolist()==[True,False,True,False]


@pytest.mark.parametrize('sat',[False,True])
def test_final_review_omits_unused_known_query_without_changing_collision(sat,monkeypatch):
    name='upstream_1km_mppi_grid_sat_v1' if sat else 'upstream_1km_mppi_body_geometry_v1'
    cfg=replace(cfg_from_experiment('configs/experiment/'+name+'.yaml'),planner_device='cpu')
    view=memory(cfg,[(4.2,0.,.9,8)])[0].view();planner=create_upstream_adapter(cfg)
    local=planner.batch_kernel.planning_terrain(view)
    points=np.array([[0.,0.,0.],[1.,0.,0.]])
    expected,known=local.obstacle_distance(points,kind='crater_large')
    assert known.all()
    def unused(*args):raise AssertionError('Discarded known footprint was computed')
    monkeypatch.setattr(local,'footprint_known',unused)
    actual,known=local.obstacle_distance(points,kind='crater_large',include_known=False)
    np.testing.assert_array_equal(actual,expected);assert known is None
    monkeypatch.setattr(view,'footprint_known',unused)
    for point,distance in zip(points,expected):
        item=context(cfg,view,ego_pose=point)
        trajectory,_=planner.trajectory_from_controls(item,np.zeros((40,2)))
        assert planner._review(item,trajectory)==('large_crater' if distance<0 else '')
