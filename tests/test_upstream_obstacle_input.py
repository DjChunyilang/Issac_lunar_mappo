"""Behavioral contracts for fixed-grid perception and observed-obstacle feedback."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from _common import cfg_from_experiment
from test_upstream_planning import context
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_grid import GridMemory,GridTerrainBatch,_intersection_areas
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_observation import UpstreamObservationEncoder
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_reward import UpstreamReward,actual_facts
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_mppi_calibration import UpstreamMPPIKernelV4,UpstreamMPPISettingsV3
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_queue import PendingPlan
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_planning import UpstreamIntent
from lunar_rover_tasks.terrain_data.scene_pack import ScenePack
from lunar_rover_tasks.terrain_data.obstacle_atlas import build_obstacle_atlas,ObstacleAtlas

BASE='configs/experiment/upstream_1km_obstacle_input_v1.yaml'


@pytest.fixture(scope='module')
def cfg():return cfg_from_experiment(BASE)


class Atlas:
    def __init__(self,obstacles=((0.,0.,.3),)):
        self.identity='test_map';self.extent=np.array([100.,100.]);self.res=.2;self.obstacles=obstacles;self.reads=0

    def read(self,cells):
        self.reads+=len(cells);xy=(cells+.5)*self.res-self.extent/2
        kind=np.zeros(len(cells),np.uint8)
        for x,y,r in self.obstacles:
            d=np.maximum(abs(xy-[x,y])-self.res/2,0)
            kind[np.sum(d*d,-1)<=r*r]=2
        values=np.zeros((len(cells),7),np.float32);values[:,4]=1.
        return values,kind,np.ones(len(cells),bool)


def memory(cfg,atlas=None,pose=None):
    atlas=Atlas() if atlas is None else atlas
    mem=GridMemory(cfg.spec,cfg.obstacle_input)
    mem.update(np.zeros(3) if pose is None else pose,np.zeros(3),0.,atlas)
    return mem,atlas


def test_static_reobservation_and_translation_are_incremental(cfg):
    mem,atlas=memory(cfg);first=mem.view();reads=atlas.reads
    for i in range(4):mem.update(np.zeros(3),np.zeros(3),.2*(i+1),atlas)
    assert atlas.reads==reads and mem.view() is first
    mem.update(np.array([.8,0.,0.]),np.zeros(3),1.,atlas)
    assert 0<atlas.reads-reads<800
    assert first.lookup([[15.1,0.]])[1].tolist()==[False]
    assert mem.view().lookup([[15.1,0.]])[1].tolist()==[True]
    assert not any(isinstance(x,Atlas) for x in vars(first).values())
    assert all(not t.values.flags.writeable for t in first.tiles.values())


def test_unknown_and_hidden_obstacles_do_not_leak_into_distance(cfg):
    free,_=memory(cfg,Atlas(()))
    hidden,_=memory(cfg,Atlas(((15.3,0.,.1),)))
    np.testing.assert_equal(free.view().obstacle_distance([[13,0,0]])[0],hidden.view().obstacle_distance([[13,0,0]])[0])
    assert not hidden.view().lookup([[15.3,0]])[1][0]
    assert hidden.view().aggregate(np.zeros(3),[(15.,16.,-.5,.5)])[0,2]==0
    # Fine cell fits, but its 0.8 m source cost support extends beyond the FOV.
    assert not hidden.view().lookup([[14.9,0.]])[1][0]
    atlas=Atlas(((15.3,0.,.1),));hidden.update([1.,0.,0.],np.zeros(3),.2,atlas)
    assert hidden.view().obstacle_distance([[13.,0.,0.]])[0][0]<0


def test_full_cell_visibility_under_rotation(cfg):
    angle=.43;mem,_=memory(cfg,Atlas(()),np.array([0.,0.,angle]));v=mem.view()
    for key,tile in v.tiles.items():
        yy,xx=np.nonzero(tile.known);ty,tx=divmod(key,int(v.tile_shape[0]))
        centers=(np.c_[xx+tx*v.tile_size,yy+ty*v.tile_size]+.5)*v.res-v.extent/2
        c,s=np.cos(angle),np.sin(angle);body=centers@np.array([[c,-s],[s,c]])
        allowance=v.res/2*(abs(c)+abs(s))
        assert (abs(body[:,0])<=15-allowance+1e-9).all()
        assert (abs(body[:,1])<=8-allowance+1e-9).all()


def test_region_area_and_between_node_obstacle(cfg):
    mem,_=memory(cfg,Atlas(((1.37,.17,.3),)))
    # A whole region retains an obstacle even when representative endpoints miss it.
    obs=mem.view().aggregate(np.zeros(3),[(.5,2.5,-.5,.5)])[0]
    assert obs[0]==1 and obs[2]==pytest.approx(1)
    partial=mem.view().aggregate(np.zeros(3),[(14.,16.,-1.,1.)])[0]
    assert partial[2]==pytest.approx(.4)
    polygons=np.array([[[0,0],[1,0],[1,1],[0,1]],[[3,3],[4,3],[4,4],[3,4]]],float)
    np.testing.assert_allclose(_intersection_areas(polygons,(.5,1.5,.5,1.5)),[.25,0])
    rotated=mem.view().aggregate(np.array([0.,0.,.31]),[(-1.,1.,-1.,1.)])[0]
    assert rotated[2]==pytest.approx(1)


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_grid_cpu_device_queries_distance_and_upload_reuse(cfg,device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    mem,_=memory(cfg);view=mem.view();batch=GridTerrainBatch([view],device)
    xy=np.array([[0,0],[.39,.11],[-9.23,2.52],[16.1,0.]])
    got,known=batch.lookup(torch.tensor(xy[None],device=device))
    expected,support=view.lookup(xy)
    np.testing.assert_allclose(got.cpu()[0],expected);np.testing.assert_equal(known.cpu()[0],support)
    poses=np.array([[0,0,0],[4,0,0],[0,4,0],[0,4,np.pi/2],[30,0,.1]])
    distance=batch.obstacle_distance(torch.tensor(poses[None],device=device)).cpu()[0]
    np.testing.assert_allclose(distance,view.obstacle_distance(poses)[0],atol=1e-6)
    assert distance[1]<distance[2] and distance[3]<distance[2]
    again=GridTerrainBatch([view],device,batch.cache)
    assert again.upload_bytes==0
    selected=again.select([0]);np.testing.assert_equal(selected.lookup(torch.tensor(xy[None],device=device))[1].cpu()[0],support)


def test_eviction_checkpoint_and_queued_snapshot_keep_observed_history(cfg):
    mem,atlas=memory(cfg);initial=mem.view();old=initial.lookup([[0.,0.]])
    for i,x in enumerate([10.,20.,30.,-10.,-20.,-30.]):mem.update([x,0.,0.],np.zeros(3),i+1.,atlas)
    mem.update([0.,20.,0.],np.zeros(3),8.,atlas)
    assert len(mem.tiles)<=64 and mem.stats['evicted_tiles']>0
    np.testing.assert_equal(initial.lookup([[0.,0.]])[0],old[0])
    restored=GridMemory(cfg.spec,cfg.obstacle_input);restored.load_state_dict(deepcopy(mem.state_dict()))
    points=np.array([[-30.,0.],[0.,0.],[30.,0.]])
    np.testing.assert_equal(restored.view().lookup(points)[1],mem.view().lookup(points)[1])
    ticket=UpstreamIntent([1.,0.],False,0.,cfg.spec.grid.request_validity)
    pending=PendingPlan.capture(cfg,0,0,0,np.zeros(3),np.zeros(2),np.zeros(2),ticket,initial,(),0.)
    again=PendingPlan.restore(cfg,pending.state_dict())
    np.testing.assert_equal(again.terrain.lookup([[0.,0.]])[0],old[0])
    with pytest.raises(ValueError):
        restored.load_state_dict({**mem.state_dict(),'input':{}})


def test_mppi_soft_cost_direction_and_same_evaluation_function(cfg):
    mem,_=memory(cfg);view=mem.view();item=context(cfg,view)
    kernel=UpstreamMPPIKernelV4(replace(cfg,planner_device='cpu'),UpstreamMPPISettingsV3(samples=2,iterations=1,device='cpu'))
    terrain,neighbors=kernel.pack([item])
    states=torch.zeros((1,3,41,3),dtype=torch.float64)
    states[0,1,:,1]=3.1;states[0,2,:,1]=6.
    controls=torch.zeros((1,3,40,2),dtype=torch.float64)
    parts,safe=kernel.cost([item],states,controls,terrain,neighbors)
    assert parts['obstacle_overlap'][0,0]>parts['obstacle_overlap'][0,1]
    assert parts['obstacle_near'][0,1]>parts['obstacle_near'][0,2]
    assert safe.all()  # Obstacle cost itself does not introduce hard rejection.
    assert parts['obstacle_near'][0,0].item()==pytest.approx(8*3.)
    single,_=kernel.cost([item],states[:,:1],controls[:,:1],terrain,neighbors)
    assert single['obstacle_near'][0,0]==parts['obstacle_near'][0,0]


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_real_mppi_prediction_and_cost_share_grid_device(cfg,device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    mem,_=memory(cfg);item=context(cfg,mem.view())
    kernel=UpstreamMPPIKernelV4(replace(cfg,planner_device=device),UpstreamMPPISettingsV3(samples=2,iterations=1,device=device))
    raw=torch.zeros((1,2,40,2),device=device)
    output=kernel.evaluate([item],raw)
    assert torch.isfinite(output[3]).all()
    assert output[0].shape==(1,2,41,3) and 'obstacle_near' in output[-1]


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_full_planner_selects_and_rescores_grid_trajectory(cfg,device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_factory import create_upstream_adapter
    local=replace(cfg,planner_device=device,mppi_samples=2,mppi_iterations=1,simulated_latency=0.,timing_version='held_previous_control_v1')
    mem,_=memory(local)
    planner=create_upstream_adapter(local)
    output=planner.plan(context(local,mem.view()))
    assert output.valid
    parts=output.diagnostics['selected_cost_parts']
    assert 'obstacle_near' in parts and output.total_cost==pytest.approx(sum(parts.values()))


def test_partial_planner_batches_keep_other_rover_tiles_uploaded(cfg):
    first,atlas=memory(cfg)
    second=GridMemory(cfg.spec,cfg.obstacle_input)
    second.update(np.zeros(3),np.array([1.,2.,0.]),0.,atlas)
    batch=GridTerrainBatch([first.view(),second.view()],'cpu',max_owners=2)
    one=GridTerrainBatch([first.view()],'cpu',batch.cache,max_owners=2)
    two=GridTerrainBatch([second.view()],'cpu',one.cache,max_owners=2)
    assert one.upload_bytes==two.upload_bytes==0


def test_reward_uses_executed_distance_and_dt_team_mean(cfg):
    from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_footprint import FootprintReport
    reward=UpstreamReward(cfg.spec,1,obstacle_input=cfg.obstacle_input)
    poses=np.zeros((1,4,3));poses[0,:,0]=np.arange(4)*10
    velocity=np.zeros((1,4,2));reports=[[FootprintReport(True) for _ in range(4)]]
    facts=actual_facts(cfg.spec,poses,velocity,reports,np.zeros(1,dtype=int))
    task={'learning_valid':np.ones(1,bool),'ended_this_step':np.zeros(1,bool),'terminal_reason':np.array(['none'])}
    f=np.zeros((1,4,5));f[...,4]=1
    _,terms=reward.advance(facts,facts,task,velocity,f,np.ones((1,4)),np.zeros((1,4)),obstacle_distances=[[-1,-.5,.5,1]])
    assert terms['obstacle_near'][0]==pytest.approx(-.2*.2*(1+1+.25)/4)
    assert terms['obstacle_overlap'][0]==pytest.approx(-.2*2*(1+.5)/4)


def test_atlas_source_geometry_ground_scale_and_y_orientation(cfg,tmp_path):
    scene=object.__new__(ScenePack);scene.path=tmp_path/'scene.json'
    n=32;spacing=.1;axis=(np.arange(n)+.5)*spacing-n*spacing/2
    x,y=np.meshgrid(axis,axis[::-1]);height=.05*x+.02*y
    features=np.stack([height,np.full_like(x,.05),np.full_like(x,.02),np.zeros_like(x)]).astype(np.float32)
    entry={'accepted':True,'affects_target':True,'diameter_m':.6,'local_xy_m':[.4,.8]}
    outside={**entry,'local_xy_m':[-2.,0.],'diameter_m':.1}
    scene.meta={'extent_xy_m':[3.2,3.2],'shape':[n,n],'feature_catalog':{'craters':[entry,outside],'rocks':[entry]},'files':{}}
    scene.path.write_text(json.dumps(scene.meta));before=scene.path.read_bytes()
    scene.arrays={'features':features,'metrics_valid':np.ones((n,n),bool),'height_valid':np.ones((n,n),bool)}
    path=build_obstacle_atlas(scene,tmp_path/'derived',cfg.obstacle_input,cfg.spec)
    atlas=ObstacleAtlas(path,cfg.obstacle_input,scene)
    assert scene.path.read_bytes()==before
    values,kind,known=atlas.read(np.array([[10,12],[8,4]]))
    assert kind[0]==3 and kind[1]==0 and known.all()
    assert values[0,0]>values[1,0]
    assert (values[:,6]>values[:,5]).all()  # Same slope, stricter parking threshold.
    assert values[:,5].max()<1  # A linear slope is not a step.
    with pytest.raises(FileExistsError):build_obstacle_atlas(scene,path.parent,cfg.obstacle_input,cfg.spec)
