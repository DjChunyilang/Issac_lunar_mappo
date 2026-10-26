"""MPPI geometry tuning removes extra padding without changing sensed cells."""
from dataclasses import replace
from pathlib import Path
import sys
import numpy as np
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from _common import cfg_from_experiment
from test_upstream_crater_grid import memory
from test_upstream_planning import context
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_factory import create_upstream_adapter
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_planning import UpstreamIntent


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_body_clear_pose_is_plannable_but_actual_entry_is_rejected(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.fail('CUDA verification required')
    old=replace(cfg_from_experiment('configs/experiment/upstream_1km_crater_classes_v1.yaml'),planner_device=device,mppi_samples=2,mppi_iterations=1,simulated_latency=0.,timing_version='held_previous_control_v1')
    new=replace(cfg_from_experiment('configs/experiment/upstream_1km_mppi_body_geometry_v1.yaml'),planner_device=device,mppi_samples=2,mppi_iterations=1,simulated_latency=0.,timing_version='held_previous_control_v1')
    assert old.spec.fingerprint==new.spec.fingerprint and old.obstacle_input.metadata==new.obstacle_input.metadata
    mem,_=memory(old,[(4.2,0.,.9,8)]);view=mem.view();original_points=view.points.copy()
    item=context(new,view,intent=UpstreamIntent([0.,0.],True,0.,1.))
    a,b=create_upstream_adapter(old),create_upstream_adapter(new)
    states=torch.zeros((1,2,41,3),dtype=torch.float64,device=device);states[:,1,-1,0]=1.
    controls=torch.zeros((1,2,40,2),dtype=torch.float64,device=device)
    for adapter,expected in [(a,[[False,False]]),(b,[[True,False]])]:
        terrain,neighbors=adapter.batch_kernel.pack([item]);_,safe=adapter.batch_kernel.cost([item],states,controls,terrain,neighbors)
        assert safe.tolist()==expected
    # The final regenerated HOLD review agrees with the batched screening.
    assert not a.prepare(item).valid
    assert b.prepare(item).valid
    invaded=context(new,view,intent=UpstreamIntent([1.,0.],True,0.,1.),ego_pose=[1.,0.,0.])
    assert not create_upstream_adapter(new).prepare(invaded).valid
    np.testing.assert_array_equal(view.points,original_points)
    local=b.batch_kernel.planning_terrain(view)
    assert local.tiles is view.tiles and local is not view
    half=np.array([new.spec.vehicle.length,new.spec.vehicle.width])/2
    assert (local.points.max(0)<half).all()
    # Every cell corner is inside a covering disk, with no omitted body strip.
    nx=int(np.ceil(new.spec.vehicle.length/.2));ny=int(np.ceil(new.spec.vehicle.width/.2))
    corners=np.stack(np.meshgrid(np.linspace(-half[0],half[0],nx+1),np.linspace(-half[1],half[1],ny+1)),axis=-1).reshape(-1,2)
    assert np.linalg.norm(corners[:,None]-local.points[None],axis=-1).min(-1).max()<=local.cover_radius+1e-12


def test_geometry_parameters_keep_legacy_identity_and_reject_silent_restore():
    old=cfg_from_experiment('configs/experiment/upstream_1km_crater_classes_v1.yaml')
    new=cfg_from_experiment('configs/experiment/upstream_1km_mppi_body_geometry_v1.yaml')
    a,b=create_upstream_adapter(old),create_upstream_adapter(new)
    assert 'footprint_margin_m' not in a.batch_kernel.metadata['settings']
    assert 'footprint_cover_spacing_m' not in a.batch_kernel.metadata['settings']
    assert b.batch_kernel.metadata['settings']['footprint_margin_m']==0.
    with pytest.raises(ValueError):b.load_state_dict(a.state_dict())
