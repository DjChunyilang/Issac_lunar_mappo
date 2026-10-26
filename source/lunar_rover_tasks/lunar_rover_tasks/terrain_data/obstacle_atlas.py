"""Immutable derived obstacle labels and coarse ground metrics; no new terrain RNG."""
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from .metrics import local_metrics
from .scene_pack import ScenePack, file_hash

SCHEMA = 'local_obstacle_atlas_v1'
GRADED_SCHEMA = 'local_obstacle_atlas_v2'


def _driving_surface(scene, output):
    """Remove the synthetic crater layer without changing rocks or source terrain."""
    a = scene.arrays
    h, w = scene.meta['shape']
    sx, sy = scene.meta['processing_spacing_xy_m']
    half_xy = np.asarray(scene.meta['metric_baseline_xy_m']) / (2 * np.array([sx, sy]))
    if not np.allclose(half_xy, round(half_xy[0])) or half_xy[0] < 1:
        raise ValueError('Driving metrics require the original square analysis window')
    half = round(half_xy[0])
    layers = ['base_height', 'layer_rocks']
    if 'layer_fractal' in a:
        layers.append('layer_fractal')
    features = np.lib.format.open_memmap(output/'driving_features.npy', mode='w+', dtype=np.float32, shape=(4,h,w))
    valid = np.lib.format.open_memmap(output/'driving_valid.npy', mode='w+', dtype=np.bool_, shape=(h,w))
    for start in range(0, h, 128):
        end = min(h, start+128)
        lo, hi = max(0, start-half), min(h, end+half)
        height = np.asarray(a[layers[0]][lo:hi], dtype=np.float32).copy()
        for name in layers[1:]:
            height += a[name][lo:hi]
        metrics = local_metrics(height, a['height_valid'][lo:hi], sx, sy, half)
        part = slice(start-lo, end-lo)
        features[0,start:end] = height[part]
        for i, name in enumerate(('slope_x', 'slope_y', 'roughness'), 1):
            features[i,start:end] = metrics[name][part]
        valid[start:end] = metrics['valid'][part]
    features.flush(); valid.flush()
    return features, valid, layers


def _ground_block(raw, height_valid, metrics_valid, gx, bx, by, dx, dy, data, terrain):
    rows = raw.shape[1]//by
    block=raw.reshape(4,rows,by,gx,bx).transpose(1,3,2,4,0).reshape(rows,gx,by*bx,4)
    good=np.asarray(metrics_valid).reshape(rows,by,gx,bx).all((1,3))
    good &= np.asarray(height_valid).reshape(rows,by,gx,bx).all((1,3))
    good &= np.isfinite(block).all((2,3))
    block=np.nan_to_num(block)
    magnitude=np.linalg.norm(block[...,1:3],axis=-1)
    index=magnitude.argmax(-1)
    slope=np.take_along_axis(block[...,1:3],index[...,None,None],axis=2)[...,0,:]
    height=block[...,0].mean(-1); rms=block[...,3].max(-1)
    avg=block[...,1:3].mean(-2)
    ox,oy=np.meshgrid(dx,dy)
    residual=block[...,0]-height[...,None]-avg[...,0,None]*ox.ravel()-avg[...,1,None]*oy.ravel()
    step=abs(residual).max(-1)
    score=np.exp(-magnitude/data['ground']['slope_score_scale']-block[...,3]/data['ground']['rms_score_scale_m']).min(-1)
    base=(rms/terrain.roughness_rms)**2+(step/terrain.step_height)**2+((1-score)/.75)**2
    max_slope=magnitude.max(-1)
    drive=np.minimum(100,base+(max_slope/terrain.driving_slope)**2)
    park=np.minimum(100,base+(max_slope/terrain.parking_slope)**2)
    out=np.concatenate([height[...,None],slope,rms[...,None],score[...,None],drive[...,None],park[...,None]],-1)
    return out, good


def build_obstacle_atlas(scene, output, settings, spec):
    """Stream native blocks, preserving source files and accepted catalogue geometry."""
    scene = scene if isinstance(scene, ScenePack) else ScenePack(scene)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    data = settings.data
    res, ground_res = data['labels']['resolution_m'], data['ground']['resolution_m']
    ex, ey = scene.meta['extent_xy_m']; h, w = scene.meta['shape']
    nx, ny = round(ex/res), round(ey/res)
    gx, gy = round(ex/ground_res), round(ey/ground_res)
    bx, by = w//gx, h//gy
    if (w % gx or h % gy or not np.isclose(ex/nx, res) or not np.isclose(ey/ny, res)):
        raise ValueError('Native, ground and label grids must align exactly')
    labels = np.lib.format.open_memmap(output/'obstacle_kind.npy', mode='w+', dtype=np.uint8, shape=(ny,nx))
    labels[:] = 0
    counts = {}
    graded = settings.graded_craters
    if graded:
        counts.update(small_craters=0, medium_craters=0, large_craters=0)
    for category, bit in [('craters', 1), ('rocks', 2)]:
        counts[category] = 0
        for item in scene.meta['feature_catalog'][category]:
            if not item['accepted'] or not item['affects_target']:
                continue
            if graded and category == 'craters':
                thresholds = data['crater_classes']['thresholds']
                size = ('small' if item['diameter_m'] < thresholds['small_below_m'] else
                        'medium' if item['diameter_m'] < thresholds['large_from_m'] else 'large')
                bit = {'small':1, 'medium':4, 'large':8}[size]
            x, y = np.asarray(item['local_xy_m']) + [ex/2,ey/2]
            radius = item['diameter_m']/2
            x0,x1 = max(0,int(np.floor((x-radius)/res))), min(nx,int(np.floor((x+radius)/res))+1)
            y0,y1 = max(0,int(np.floor((y-radius)/res))), min(ny,int(np.floor((y+radius)/res))+1)
            if x0>=x1 or y0>=y1:continue
            xx=(np.arange(x0,x1)+.5)*res; yy=(np.arange(y0,y1)+.5)*res
            # Exact circle versus axis-aligned cell intersection, including tangency.
            hit = np.maximum(abs(xx[None,:]-x)-res/2,0)**2 + np.maximum(abs(yy[:,None]-y)-res/2,0)**2 <= radius**2
            labels[y0:y1,x0:x1] |= hit.astype(np.uint8)*bit
            counts[category] += 1
            if graded and category == 'craters':
                counts[size+'_craters'] += 1
    labels.flush()
    ground = np.lib.format.open_memmap(output/'ground.npy',mode='w+',dtype=np.float32,shape=(gy,gx,7))
    valid = np.lib.format.open_memmap(output/'ground_valid.npy',mode='w+',dtype=np.bool_,shape=(gy,gx))
    a = scene.arrays; terrain=spec.terrain
    if graded:
        driving_features, driving_valid, driving_layers = _driving_surface(scene, output)
    dx=(np.arange(bx)-(bx-1)/2)*ex/w
    # Native raster y runs north-to-south; atlas y increases with world y.
    dy=-(np.arange(by)-(by-1)/2)*ey/h
    for y0 in range(0,gy,16):
        y1=min(gy,y0+16)
        raw=np.asarray(a['features'][:,y0*by:y1*by],dtype=np.float32)
        hv = a['height_valid'][y0*by:y1*by]
        out, good = _ground_block(raw, hv, a['metrics_valid'][y0*by:y1*by], gx,bx,by,dx,dy,data,terrain)
        if graded:
            original_parking = out[...,6].copy()
            out, drive_good = _ground_block(np.asarray(driving_features[:,y0*by:y1*by]), hv,
                                           driving_valid[y0*by:y1*by], gx,bx,by,dx,dy,data,terrain)
            out[...,6] = original_parking
            good &= drive_good
        ground[gy-y1:gy-y0]=np.where(good[...,None],out,0)[::-1]
        valid[gy-y1:gy-y0]=good[::-1]
    ground.flush();valid.flush()
    names = ['obstacle_kind','ground','ground_valid']
    if graded:
        names += ['driving_features','driving_valid']
    files={name:{'file':name+'.npy','sha256':file_hash(output/(name+'.npy'))} for name in names}
    manifest={'schema':GRADED_SCHEMA if graded else SCHEMA,'source_scene':str(scene.path.resolve()),'source_scene_sha256':file_hash(scene.path),
              'source_files':scene.meta['files'],'input_spec':settings.metadata,'vehicle_terrain':{
                  'driving_slope':terrain.driving_slope,'parking_slope':terrain.parking_slope,
                  'roughness_rms':terrain.roughness_rms,'step_height':terrain.step_height},
              'extent_xy_m':[ex,ey],'resolution_m':res,'ground_resolution_m':ground_res,'files':files,
              'counts':counts,'occupied_fraction':float(np.count_nonzero(labels)/labels.size),
              'valid_ground_fraction':float(np.count_nonzero(valid)/valid.size)}
    if graded:
        manifest['driving_surface'] = {'layers':driving_layers, 'removed_layer':'layer_craters',
                                      'native_spacing_xy_m':scene.meta['processing_spacing_xy_m'],
                                      'metric_baseline_xy_m':scene.meta['metric_baseline_xy_m'],
                                      'score_scales':data['ground'],
                                      'parking_surface':'original_terrain'}
    (output/'obstacle.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    return output/'obstacle.json'


class ObstacleAtlas:
    """Sensor-owned arrays. A local view never retains this object."""
    def __deepcopy__(self, memo):
        # Environment restores may copy a curriculum that owns this immutable
        # atlas. Keep its read-only mmap arrays shared instead of copying GBs.
        memo[id(self)] = self
        return self

    def __init__(self,path,settings,scene):
        self.path=Path(path);self.meta=json.loads(self.path.read_text())
        self.graded_craters = settings.graded_craters
        if (self.meta['schema']!=(GRADED_SCHEMA if self.graded_craters else SCHEMA) or self.meta['input_spec']!=settings.metadata
                or self.meta['source_scene_sha256']!=file_hash(scene.path) or self.meta['source_files']!=scene.meta['files']):
            raise ValueError('Obstacle atlas source/configuration mismatch')
        self.identity=file_hash(self.path)
        for name,item in self.meta['files'].items():
            p=self.path.parent/item['file']
            if p.parent.resolve()!=self.path.parent.resolve() or file_hash(p)!=item['sha256']:
                raise ValueError('Obstacle atlas checksum mismatch')
            setattr(self,name,np.load(p,mmap_mode='r',allow_pickle=False))
        self.res=self.meta['resolution_m'];self.ratio=round(self.meta['ground_resolution_m']/self.res)
        self.extent=np.asarray(self.meta['extent_xy_m'])
        if self.graded_craters:
            self._score_scales = settings.data['ground']
            self._spacing = np.asarray(self.meta['driving_surface']['native_spacing_xy_m'])
            threshold = settings.data['crater_classes']['thresholds']['large_from_m']
            self._large_craters = np.asarray([item['local_xy_m']+[item['diameter_m']/2]
                for item in scene.meta['feature_catalog']['craters']
                if item['accepted'] and item['affects_target'] and item['diameter_m'] >= threshold], dtype=float).reshape(-1,3)
            self._large_tree = cKDTree(self._large_craters[:,:2]) if len(self._large_craters) else None

    def read(self,cells):
        x,y=cells.T;inside=(x>=0)&(y>=0)&(x<self.obstacle_kind.shape[1])&(y<self.obstacle_kind.shape[0])
        values=np.zeros((len(cells),7),np.float32);kind=np.zeros(len(cells),np.uint8);known=np.zeros(len(cells),bool)
        i=np.flatnonzero(inside);gx=x[i]//self.ratio;gy=y[i]//self.ratio
        values[i]=self.ground[gy,gx];kind[i]=self.obstacle_kind[y[i],x[i]];known[i]=self.ground_valid[gy,gx]
        return values,kind,known

    def lookup_driving(self, xy):
        """Strict bilinear native features, then the existing DenseSensor score formula."""
        if not self.graded_craters:
            raise ValueError('A crater-class atlas is required for the derived driving surface')
        xy = np.asarray(xy, dtype=float)
        shape = xy.shape[:-1]
        points = xy.reshape(-1,2)
        cols = (points[:,0]+self.extent[0]/2)/self._spacing[0]-.5
        rows = (self.extent[1]/2-points[:,1])/self._spacing[1]-.5
        h, w = self.driving_valid.shape
        known = np.isfinite(points).all(-1)&(cols>=0)&(cols<=w-1)&(rows>=0)&(rows<=h-1)
        cols = np.clip(np.where(known,cols,0),0,w-1)
        rows = np.clip(np.where(known,rows,0),0,h-1)
        x0,y0 = np.floor(cols).astype(int),np.floor(rows).astype(int)
        tx,ty = cols-x0,rows-y0
        native = np.zeros((len(points),4), dtype=float)
        for xx,yy,weight in ((x0,y0,(1-tx)*(1-ty)), (np.minimum(x0+1,w-1),y0,tx*(1-ty)),
                             (x0,np.minimum(y0+1,h-1),(1-tx)*ty),
                             (np.minimum(x0+1,w-1),np.minimum(y0+1,h-1),tx*ty)):
            values = np.asarray(self.driving_features[:,yy,xx]).T
            support = self.driving_valid[yy,xx]&np.isfinite(values).all(-1)
            known &= support | (weight==0)
            native += np.where(support[:,None],values,0)*weight[:,None]
        score = np.exp(-np.linalg.norm(native[:,1:3],axis=-1)/self._score_scales['slope_score_scale']
                       -native[:,3]/self._score_scales['rms_score_scale_m'])
        result = np.column_stack((native,score))
        result[~known] = 0
        return result.reshape((*shape,5)), known.reshape(shape)

    def exact_large_crater_overlap(self, poses, length, width):
        """Environment truth: positive-area mouth/body overlap, with no safety margin."""
        if not self.graded_craters:
            raise ValueError('A crater-class atlas is required for large-crater entry')
        poses = np.asarray(poses, dtype=float)
        shape = poses.shape[:-1]
        flat = poses.reshape(-1,3)
        result = np.zeros(len(flat),bool)
        if self._large_tree is None or not len(flat):
            return result.reshape(shape)
        half = np.array([length,width],dtype=float)/2
        radius = np.linalg.norm(half)+self._large_craters[:,2].max()
        candidates = self._large_tree.query_ball_point(flat[:,:2],radius)
        for i, ids in enumerate(candidates):
            if not ids:
                continue
            craters = self._large_craters[ids]
            delta = craters[:,:2]-flat[i,:2]
            c,s = np.cos(flat[i,2]),np.sin(flat[i,2])
            body = delta @ np.array([[c,-s],[s,c]])
            distance = np.linalg.norm(np.maximum(abs(body)-half,0),axis=-1)
            # The tolerance only absorbs floating-point tangency, not a physical buffer.
            result[i] = np.any(distance < craters[:,2]-1e-12)
        return result.reshape(shape)
