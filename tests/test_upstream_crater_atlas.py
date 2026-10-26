"""Crater diameter classes and derived driving terrain preserve the v1 evidence."""
from pathlib import Path
from copy import deepcopy
import json
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from _common import cfg_from_experiment
from lunar_rover_tasks.terrain_data.metrics import local_metrics
from lunar_rover_tasks.terrain_data.scene_pack import ScenePack
from lunar_rover_tasks.terrain_data.obstacle_atlas import build_obstacle_atlas, ObstacleAtlas


@pytest.fixture(scope='module')
def cfg():
    return cfg_from_experiment('configs/experiment/upstream_1km_crater_classes_v1.yaml')


def synthetic_scene(path):
    n=160; spacing=.1; axis=(np.arange(n)+.5)*spacing-n*spacing/2
    x,y=np.meshgrid(axis,axis[::-1])
    base=.01*x+.02*y
    rocks=.3*np.exp(-((x+4)**2+(y+3)**2)/.1)
    fractal=.003*np.sin(x)*np.cos(y)
    craters=-.15*np.exp(-((x+2)**2+(y-3)**2)/.2)
    height=base+rocks+fractal+craters
    valid=np.ones((n,n),bool)
    metrics=local_metrics(height,valid,spacing,spacing,4)
    entry={'accepted':True,'affects_target':True}
    catalogue=[{**entry,'diameter_m':d,'local_xy_m':[cx,3.]}
               for cx,d in [(-5,.4),(-2,.5),(1,1.799),(4,1.8)]]
    catalogue.append({**entry,'accepted':False,'diameter_m':3.,'local_xy_m':[0.,-3.]})
    scene=object.__new__(ScenePack);scene.path=path
    scene.meta={'extent_xy_m':[16.,16.],'shape':[n,n],
                'processing_spacing_xy_m':[spacing,spacing],'metric_baseline_xy_m':[.8,.8],
                'feature_catalog':{'craters':catalogue,'rocks':[
                    {**entry,'diameter_m':.6,'local_xy_m':[-4.,-3.]}]},'files':{}}
    path.write_text(json.dumps(scene.meta))
    scene.arrays={'features':np.stack([height,metrics['slope_x'],metrics['slope_y'],metrics['roughness']]).astype(np.float32),
                  'metrics_valid':metrics['valid'],'height_valid':valid,
                  'base_height':base.astype(np.float32),'layer_rocks':rocks.astype(np.float32),
                  'layer_fractal':fractal.astype(np.float32),'layer_craters':craters.astype(np.float32)}
    return scene


@pytest.fixture
def atlas(cfg,tmp_path):
    scene=synthetic_scene(tmp_path/'scene.json')
    original={k:v.copy() for k,v in scene.arrays.items()}
    path=build_obstacle_atlas(scene,tmp_path/'v2',cfg.obstacle_input,cfg.spec)
    for key,value in original.items():
        np.testing.assert_array_equal(scene.arrays[key],value)
    return ObstacleAtlas(path,cfg.obstacle_input,scene),scene


def test_diameter_boundaries_and_legacy_configuration(cfg,atlas):
    current,_=atlas
    assert cfg.obstacle_input.graded_craters
    assert not cfg_from_experiment('configs/experiment/upstream_1km_obstacle_input_v1.yaml').obstacle_input.graded_craters
    points=np.array([[-5,3],[-2,3],[1,3],[4,3],[-4,-3],[0,-3]])
    cells=np.floor((points+current.extent/2)/current.res).astype(int)
    _,kind,known=current.read(cells)
    np.testing.assert_array_equal(kind,[1,4,4,8,2,0])
    assert known.all()
    assert current.meta['counts']=={'craters':4,'rocks':1,'small_craters':1,'medium_craters':2,'large_craters':1}


def test_curriculum_deepcopy_shares_immutable_atlas(atlas):
    current,_=atlas
    copied=deepcopy({'atlases':[current,current]})
    assert copied['atlases'][0] is current
    assert copied['atlases'][1] is current
    assert not current.driving_features.flags.writeable
    assert not current.ground.flags.writeable


def test_native_surface_removes_only_craters_and_keeps_parking_original(cfg,atlas,tmp_path):
    current,scene=atlas
    a=scene.arrays
    expected=a['base_height']+a['layer_rocks']+a['layer_fractal']
    metrics=local_metrics(expected,a['height_valid'],.1,.1,4)
    np.testing.assert_allclose(current.driving_features[0],expected,atol=1e-7)
    for i,key in enumerate(('slope_x','slope_y','roughness'),1):
        np.testing.assert_allclose(current.driving_features[i],metrics[key],atol=1e-7,equal_nan=True)
    np.testing.assert_array_equal(current.driving_valid,metrics['valid'])
    old=cfg_from_experiment('configs/experiment/upstream_1km_obstacle_input_v1.yaml')
    old_path=build_obstacle_atlas(scene,tmp_path/'v1',old.obstacle_input,old.spec)
    legacy=ObstacleAtlas(old_path,old.obstacle_input,scene)
    np.testing.assert_array_equal(current.ground[...,6],legacy.ground[...,6])
    assert np.max(legacy.ground[...,5]-current.ground[...,5])>.1
    # Rock remains a significant slope feature in the new driving surface.
    assert np.nanmax(np.linalg.norm(current.driving_features[1:3],axis=0))>.1


def test_native_query_orientation_interpolation_score_and_unknown(atlas):
    current,scene=atlas
    points=np.array([[-6.13,5.27],[3.24,-4.36],[0,0],[-8,0],[0,8],[np.nan,0]])
    features,known=current.lookup_driving(points)
    assert known.tolist()==[True,True,True,False,False,False]
    # Independent scene interpolation checks north-to-south native raster orientation.
    from lunar_rover_tasks.terrain_data.scene_pack import strict_sample
    rows=(8-points[:3,1])/.1-.5;cols=(points[:3,0]+8)/.1-.5
    expected=np.stack([strict_sample(values,rows,cols)[0] for values in current.driving_features],-1)
    np.testing.assert_allclose(features[:3,:4],expected,atol=1e-7)
    np.testing.assert_allclose(features[:3,4],np.exp(-np.linalg.norm(expected[:,1:3],axis=-1)/.6-expected[:,3]/.05),atol=1e-7)
    np.testing.assert_array_equal(features[~known],0)
    shaped,_=current.lookup_driving(points[:2].reshape(1,2,2))
    assert shaped.shape==(1,2,5)


def test_exact_large_entry_excludes_tangency_margin_and_other_classes(atlas):
    current,_=atlas
    poses=np.array([[5.9,3,0],[5.9-1e-7,3,0],[5.9+1e-7,3,0],
                    [5.4,3,np.pi/2],[5.4-1e-7,3,np.pi/2],[-2,3,0],[-4,-3,0],
                    [4+1+.9/np.sqrt(2),3+.5+.9/np.sqrt(2),0]])
    result=current.exact_large_crater_overlap(poses.reshape(2,4,3),2.,1.)
    np.testing.assert_array_equal(result,[[False,True,False,False],[True,False,False,False]])
    # No extra 0.3 m clearance margin is included.
    assert not current.exact_large_crater_overlap([[6.,3,0]],2.,1.)[0]
    with pytest.raises(FileExistsError):
        build_obstacle_atlas(atlas[1],current.path.parent,
                             cfg_from_experiment('configs/experiment/upstream_1km_crater_classes_v1.yaml').obstacle_input,
                             cfg_from_experiment('configs/experiment/upstream_1km_crater_classes_v1.yaml').spec)
