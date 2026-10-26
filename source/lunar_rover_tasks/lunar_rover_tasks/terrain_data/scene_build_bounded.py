"""Strip-wise construction of the existing immutable NASA scene-pack contract.

The catalogue/probability grid is global. Strips only bound rasterization and
metric working memory; feature density, random streams and seam geometry remain
those of nasa_sfd_v2. This builder requires aligned source quality products.
"""
from copy import deepcopy
import gc
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio import Affine
from rasterio.windows import Window

from .scene_pack import SCHEMA, file_hash, strict_sample
from .metrics import local_metrics
from .nasa_sfd import generate_catalog, render_catalog


def build_scene_bounded(dtm, output, *, source_id, geographic_group, pixel_window,
                        processing_spacing_m, metric_baseline_m, source_half_width,
                        quality_files=None, error_semantics='unavailable', enhancement=None,
                        source_url=None, role='development', stripe_rows=128):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f'Scene packs are immutable: {output}')
    if role not in ('development', 'validation', 'test') or stripe_rows < 1:
        raise ValueError('Invalid role or strip size')
    if not enhancement or enhancement['method'] != 'nasa_sfd_v2':
        raise ValueError('Bounded builder requires the existing NASA SFD v2 model')
    col, row, width, height = map(int, pixel_window)
    if list(pixel_window) != [col,row,width,height] or min(width,height) <= 0 or min(col,row) < 0:
        raise ValueError('Invalid native window')
    spacing = float(processing_spacing_m)
    if spacing <= 0 or metric_baseline_m < 2*spacing or source_half_width < 1:
        raise ValueError('Invalid processing or analysis scale')
    with rasterio.open(dtm) as ds:
        if (not ds.crs or not ds.crs.is_projected or ds.crs.linear_units != 'metre'
                or ds.transform.b or ds.transform.d or ds.transform.a <= 0 or ds.transform.e >= 0
                or col+width > ds.width or row+height > ds.height):
            raise ValueError('Expected an in-bounds north-up projected metre window')
        dx,dy=ds.res
        w,h=round(width*dx/spacing),round(height*dy/spacing)
        sx,sy=width*dx/w,height*dy/h
        bx,by=[int(np.ceil(enhancement['generation_buffer_m']/v)) for v in (sx,sy)]
        halo=max(source_half_width+2,int(np.ceil(enhancement['generation_buffer_m']/min(dx,dy)))+2)
        c0,r0=max(0,col-halo),max(0,row-halo)
        c1,r1=min(ds.width,col+width+halo),min(ds.height,row+height+halo)
        window=Window(c0,r0,c1-c0,r1-r0)
        native=ds.read(1,window=window,masked=True).astype(float).filled(np.nan)*ds.scales[0]+ds.offsets[0]
        z_origin=float(np.nanmedian(native))
        native-=z_origin
        transform=rasterio.windows.transform(Window(col,row,width,height),ds.transform)*Affine.scale(width/w,height/h)
        native_transform=ds.transform
        crs_wkt=ds.crs.to_wkt()
    output.mkdir(parents=True)
    arrays={}
    def allocate(name,shape,dtype=np.float32):
        a=np.lib.format.open_memmap(output/f'{name}.npy',mode='w+',dtype=dtype,shape=shape)
        arrays[name]=a
        return a
    base=allocate('base_height',(h,w))
    valid=allocate('height_valid',(h,w),bool)
    features=allocate('features',(4,h,w))
    metrics_valid=allocate('metrics_valid',(h,w),bool)
    source=allocate('source_features',(3,h,w))
    layers={k:allocate('layer_'+k,(h,w)) for k in ('craters','rocks')}
    ph,pw=h+2*by,w+2*bx
    padded=np.lib.format.open_memmap(output/'_padded.npy',mode='w+',dtype=np.float32,shape=(ph,pw))
    padded_valid=np.lib.format.open_memmap(output/'_padded_valid.npy',mode='w+',dtype=bool,shape=(ph,pw))
    for start in range(0,ph,stripe_rows):
        end=min(ph,start+stripe_rows)
        rr,cc=np.meshgrid((np.arange(start,end)-by+.5)*sy/dy+row-r0-.5,
                         (np.arange(pw)-bx+.5)*sx/dx+col-c0-.5,indexing='ij')
        padded[start:end],padded_valid[start:end]=strict_sample(native,rr,cc)
    for start in range(0,h,stripe_rows):
        end=min(h,start+stripe_rows)
        base[start:end]=padded[by+start:by+end,bx:bx+w]
        valid[start:end]=padded_valid[by+start:by+end,bx:bx+w]
    print(json.dumps({'phase':'global_catalogue','output':str(output),'shape':[h,w]}),flush=True)
    catalog,probabilities=generate_catalog(padded_valid,(sx,sy),enhancement)
    for name,value in probabilities.items():
        np.save(output/f'probability_{name}.npy',value,allow_pickle=False)
    del probabilities
    gc.collect()
    for start in range(0,ph,stripe_rows):
        end=min(ph,start+stripe_rows)
        y0,y1=(ph-end)*sy,(ph-start)*sy
        sub={'generation':{},'craters':[],'rocks':[]}
        for kind in ('craters','rocks'):
            for item in catalog[kind]:
                support=item['diameter_m']*(1 if kind=='craters' else .5)
                if item['y_m']+support >= y0 and item['y_m']-support <= y1:
                    sub[kind].append(dict(item,y_m=item['y_m']-y0))
        final,local_layers=render_catalog(padded[start:end],padded_valid[start:end],(sx,sy),sub,enhancement)
        # Profiles extend across strips; render_catalog evaluates global physical
        # centres translated into this strip, without rejecting clipped edges.
        lo,hi=max(start,by),min(end,by+h)
        if lo < hi:
            target=slice(lo-by,hi-by); part=(slice(lo-start,hi-start),slice(bx,bx+w))
            features[0,target]=final[part]
            for k in layers: layers[k][target]=local_layers[k][part]
        del final,local_layers,sub
    del padded,padded_valid
    (output/'_padded.npy').unlink(); (output/'_padded_valid.npy').unlink()
    gc.collect()
    print(json.dumps({'phase':'strip_metrics','output':str(output)}),flush=True)
    half=max(1,int(np.ceil(metric_baseline_m/(2*min(sx,sy)))))
    native_metrics=local_metrics(native,np.isfinite(native),dx,dy,source_half_width)
    refs={'dtm':{'path':str(Path(dtm).resolve()),'sha256':file_hash(dtm),'url':source_url}}
    quality={}
    for name,path in (quality_files or {}).items():
        with rasterio.open(path) as ds:
            if ds.transform != native_transform or ds.crs.to_wkt() != crs_wkt:
                raise ValueError('Bounded quality sampling requires aligned source products')
            v=ds.read(1,window=window,masked=True).astype(float).filled(np.nan)*ds.scales[0]+ds.offsets[0]
            v[v<0]=np.nan
        quality[name]=(v,allocate('quality_'+name,(h,w)))
        refs[name]={'path':str(Path(path).resolve()),'sha256':file_hash(path)}
    for start in range(0,h,stripe_rows):
        end=min(h,start+stripe_rows)
        lo,hi=max(0,start-half),min(h,end+half)
        m=local_metrics(features[0,lo:hi],valid[lo:hi],sx,sy,half)
        part=slice(start-lo,end-lo)
        for k,name in enumerate(('slope_x','slope_y','roughness')):
            features[k+1,start:end]=m[name][part]
        metrics_valid[start:end]=m['valid'][part]
        rr,cc=np.meshgrid((np.arange(start,end)+.5)*sy/dy+row-r0-.5,
                         (np.arange(w)+.5)*sx/dx+col-c0-.5,indexing='ij')
        for k,name in enumerate(('slope_x','slope_y','roughness')):
            source[k,start:end]=strict_sample(native_metrics[name],rr,cc)[0]
        for v,target in quality.values(): target[start:end]=strict_sample(v,rr,cc,order=0)[0]
    for a in arrays.values(): a.flush()
    with rasterio.open(output/'height.tif','w',driver='GTiff',height=h,width=w,count=1,dtype='float32',
                       crs=crs_wkt,transform=transform,nodata=np.nan,compress='deflate',tiled=True) as dst:
        for start in range(0,h,stripe_rows):
            end=min(h,start+stripe_rows)
            dst.write(features[0,start:end],1,window=Window(0,start,w,end-start))
        dst.update_tags(height_origin_m=z_origin,representation='local_height; add height_origin_m for source datum')
    ex,ey=width*dx,height*dy
    lower_left=transform*(0,h)
    for kind in ('craters','rocks'):
        for item in catalog[kind]:
            item['x_m']-=bx*sx; item['y_m']-=by*sy
            x,y=item['x_m'],item['y_m']; d=item['diameter_m']
            support=d*(1 if kind=='craters' else .5)
            item.update(centre_in_target=bool(0<=x<ex and 0<=y<ey),
                        affects_target=bool(np.hypot(max(-x,0,x-ex),max(-y,0,y-ey))<support),
                        local_xy_m=[x-ex/2,y-ey/2],projected_xy_m=[lower_left[0]+x,lower_left[1]+y])
            if kind=='craters': item.update(depth_m=enhancement[kind]['depth_to_diameter']*d,
                                            rim_height_m=enhancement[kind]['rim_to_diameter']*d,
                                            profile='c1_polynomial_hypothesis_v1')
            else: item.update(height_m=enhancement[kind]['height_to_diameter']*d,profile='upper_half_ellipsoid_height_envelope_v1')
    catalog['generation'].update(grid_origin_lower_left_m=[-bx*sx,-by*sy],
        probability_transform=list(transform*Affine.translation(-bx,-by))[:6],buffer_xy_m=[bx*sx,by*sy],
        target_crop_rc=[by,bx,h,w],target_area_m2=float(np.count_nonzero(valid)*sx*sy))
    for name,data in [('catalog_generation',catalog),('catalog_target',{k:[x for x in catalog[k] if x['centre_in_target']] for k in ('craters','rocks')})]:
        (output/f'{name}.json').write_text(json.dumps(data,indent=2,allow_nan=False))
    files={}
    for path in sorted(output.iterdir()):
        item={'file':path.name,'sha256':file_hash(path)}
        if path.suffix=='.npy':
            a=np.load(path,mmap_mode='r'); item.update(shape=list(a.shape),dtype=str(a.dtype))
        files['height_geotiff' if path.name=='height.tif' else path.stem]=item
    meta=dict(schema=SCHEMA,source_id=source_id,geographic_group=geographic_group,role=role,sources=refs,
              pixel_window=[col,row,width,height],crs_wkt=crs_wkt,native_transform=list(native_transform)[:6],
              transform=list(transform)[:6],origin_projected_xy_m=list(transform*(w/2,h/2)),height_origin_m=z_origin,
              native_spacing_xy_m=[dx,dy],processing_spacing_xy_m=[sx,sy],
              metric_baseline_xy_m=[2*half*sx,2*half*sy],source_metric_baseline_xy_m=[2*source_half_width*dx,2*source_half_width*dy],
              effective_support_m=None,source_support_caveat='native grid spacing is not effective measurement resolution',
              shape=[h,w],extent_xy_m=[ex,ey],height_interpolation='bilinear_strict_support',
              quality_sampling='nearest_source_attribute_no_uncertainty_propagation',error_semantics=error_semantics,
              classification='real_plus_hypothesis',enhancement=enhancement,feature_catalog=catalog,files=files,
              feature_names=['relative_height_m','simulation_slope_x','simulation_slope_y','simulation_detrended_rms_m'],
              construction={'method':'global_catalog_strip_raster_v1','stripe_rows':stripe_rows})
    (output/'scene.json').write_text(json.dumps(meta,indent=2,allow_nan=False))
    return output/'scene.json'
