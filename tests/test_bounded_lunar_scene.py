"""Large-map path must preserve global feature geometry and masked interpolation."""
import numpy as np
import torch
import yaml
from pathlib import Path
import rasterio
from rasterio.transform import from_origin
from lunar_rover_tasks.terrain_data.scene_pack import build_scene, ScenePack
from lunar_rover_tasks.terrain_data.scene_build_bounded import build_scene_bounded
from lunar_rover_tasks.terrain_data.batched import SceneBank


def test_strips_match_global_render_and_masks(tmp_path):
    p=tmp_path/'native.tif'
    y,x=np.indices((20,20))
    with rasterio.open(p,'w',driver='GTiff',width=20,height=20,count=1,dtype='float32',
                       crs='+proj=stere +lat_0=-90 +lon_0=0 +R=1737400 +units=m +no_defs',
                       transform=from_origin(1000,2000,5,5),nodata=-9999) as ds:
        z=(100+.2*x-.3*y+.02*np.sin(x)).astype('float32');z[9,9]=-9999
        ds.write(z,1)
    spec=yaml.safe_load(Path('configs/terrain/nasa_sfd_v2.yaml').read_text())
    args=dict(source_id='test',geographic_group='test',pixel_window=[7,7,6,6],processing_spacing_m=.1,
              metric_baseline_m=.8,source_half_width=2,enhancement=spec,quality_files={'height_error':p})
    dense=ScenePack(build_scene(p,tmp_path/'dense',**args))
    striped=ScenePack(build_scene_bounded(p,tmp_path/'strip',stripe_rows=37,**args))
    for name in dense.arrays:
        np.testing.assert_allclose(striped.arrays[name],dense.arrays[name],rtol=1e-5,atol=1e-6,equal_nan=True,err_msg=name)
    for kind in ('craters','rocks'):
        assert striped.meta['feature_catalog'][kind]==dense.meta['feature_catalog'][kind]
    a,b=SceneBank([striped.path]),SceneBank([striped.path],storage='mmap',max_bytes=100)
    a.prefetch([0]);b.prefetch([0])
    xy=(torch.rand((3,4,123,2),generator=torch.Generator().manual_seed(23))-.5)*35
    xy[0,0,0]=float('nan');xy[0,0,1]=float('inf')
    ids=torch.zeros(3,dtype=torch.long)
    qa,qb=a.query(xy,ids),b.query(xy,ids)
    for field in ('features','height_valid','metrics_valid','height_error','error_valid'):
        torch.testing.assert_close(getattr(qa,field),getattr(qb,field),atol=2e-5,rtol=2e-5,equal_nan=True)
