"""Private fixed-cell memory and detached CPU/CUDA grid queries.

Only GridMemory.update owns an atlas. GridTerrain stores revealed cells only;
copy-on-write tiles keep queued plans and pre-step rewards immutable.
"""
from dataclasses import dataclass
from collections import OrderedDict
from types import SimpleNamespace
import math
import time

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt


@dataclass(frozen=True)
class GridTile:
    values: np.ndarray  # height,gx,gy,RMS,score,drive,park,obstacle distance[,medium distance,large distance]
    known: np.ndarray
    kind: np.ndarray
    scanned: np.ndarray

    def __post_init__(self):
        for value in (self.values,self.known,self.kind,self.scanned):value.setflags(write=False)

    def copy_arrays(self):
        return [a.copy() for a in (self.values,self.known,self.kind,self.scanned)]


def cover_points(spec,settings,*,spacing=None,margin=None,centered=False):
    v=spec.vehicle
    spacing=settings.data['geometry']['cover_spacing_m'] if spacing is None else spacing
    margin=v.clearance_margin if margin is None else margin
    length=v.length+2*margin;width=v.width+2*margin
    nx,ny=math.ceil(length/spacing),math.ceil(width/spacing)
    if centered:
        # Each disk covers one rectangle cell, including all four corners.
        # Edge-centered disks needlessly extend a full radius outside the body.
        x=(np.arange(nx)+.5)*length/nx-length/2
        y=(np.arange(ny)+.5)*width/ny-width/2
    else:
        x=np.linspace(-length/2,length/2,nx+1)
        y=np.linspace(-width/2,width/2,ny+1)
    points=np.stack(np.meshgrid(x,y,indexing='xy'),-1).reshape(-1,2)
    radius=(math.hypot(length/nx,width/ny)/2 if centered
            else math.hypot(x[1]-x[0],y[1]-y[0])/2)
    return points,radius


def _distance_channel(settings,kind):
    if kind=='obstacle':return 7
    if settings.graded_craters and kind in {'crater_medium','crater_large'}:
        return 8 if kind=='crater_medium' else 9
    raise ValueError('Unsupported obstacle distance kind: '+kind)


class GridTerrain:
    version='private_observed_grid_v1'
    query_version='fixed_cell_lookup_v1'

    def __init__(self,tiles,anchor,extent,settings,spec,revision,timestamp,map_identity):
        self.tiles=dict(tiles);self.anchor=np.asarray(anchor,float).copy();self.anchor.setflags(write=False)
        self.extent=np.asarray(extent,float).copy();self.extent.setflags(write=False)
        self.settings=settings;self.spec=spec;self.profile_fingerprint=spec.fingerprint
        self.value_channels=10 if settings.graded_craters else 8
        self.revision=revision;self.timestamp=timestamp;self.map_identity=map_identity
        self.res=settings.data['labels']['resolution_m'];self.tile_size=settings.data['memory']['tile_cells']
        self.shape=np.rint(self.extent/self.res).astype(int)
        self.tile_shape=np.ceil(self.shape/self.tile_size).astype(int)
        self.points,self.cover_radius=cover_points(spec,settings)
        self.cap=settings.data['cost']['soft_band_m']+self.cover_radius+self.res*2

    def readonly_snapshot(self):
        # Identity checks need only observation times; never export unseen atlas data.
        return SimpleNamespace(observed_at=np.array([self.timestamp]))

    def snapshot(self):
        return {'version':self.version,'input':self.settings.metadata,'profile':self.profile_fingerprint,
                'tiles':self.tiles,'anchor':self.anchor,'extent':self.extent,'revision':self.revision,
                'timestamp':self.timestamp,'map_identity':self.map_identity}

    @classmethod
    def restore(cls,state,cfg):
        if (cfg.obstacle_input is None or state['version']!=cls.version or state['input']!=cfg.obstacle_input.metadata
                or state['profile']!=cfg.spec.fingerprint):raise ValueError('Queued grid snapshot identity mismatch')
        tiles={key:GridTile(*tile.copy_arrays()) for key,tile in state['tiles'].items()}
        return cls(tiles,state['anchor'],state['extent'],cfg.obstacle_input,cfg.spec,state['revision'],state['timestamp'],state['map_identity'])

    def _cells(self,points):
        return np.floor((np.atleast_2d(points)+self.anchor[:2]+self.extent/2)/self.res+1e-10).astype(np.int64)

    def read_cells(self,cells):
        cells=np.asarray(cells,dtype=np.int64);n=len(cells)
        values=np.zeros((n,self.value_channels),np.float32);values[:,7:]=self.cap
        known=np.zeros(n,bool);kind=np.zeros(n,np.uint8)
        inside=((cells>=0)&(cells<self.shape)).all(-1)
        tc=cells//self.tile_size;keys=tc[:,1]*self.tile_shape[0]+tc[:,0]
        for key in np.unique(keys[inside]):
            tile=self.tiles.get(int(key))
            if tile is None:continue
            ids=np.flatnonzero(inside&(keys==key));x,y=(cells[ids]%self.tile_size).T
            values[ids]=tile.values[y,x];known[ids]=tile.known[y,x];kind[ids]=tile.kind[y,x]
        return values,known,kind

    def lookup(self,xy):
        data,known,_=self.read_cells(self._cells(xy))
        return np.where(known[:,None],data[:,:5],0).astype(float),known

    def obstacle_distance(self,poses,*,kind='obstacle',include_known=True):
        """Return distance (SAT: hard sign) and optional known-footprint flags."""
        if kind=='crater_large' and getattr(self,'large_crater_query',None)=='observed_grid_sat_v1':
            poses=np.atleast_2d(np.asarray(poses,float))
            half=np.array([self.spec.vehicle.length,self.spec.vehicle.width])/2
            hit=_rectangle_hits_numpy(poses,self.large_crater_rectangles(),half)
            # Hard-query sign only: deliberately not a metric soft distance.
            known=np.array([self.footprint_known(p) for p in poses]) if include_known else None
            return np.where(hit,-1.,1.),known
        channel=_distance_channel(self.settings,kind)
        poses=np.atleast_2d(np.asarray(poses,float));result=[];coverage=[]
        for pose in poses:
            c,s=np.cos(pose[2]),np.sin(pose[2])
            points=self.points@np.array([[c,s],[-s,c]])+pose[:2]
            values,known,_=self.read_cells(self._cells(points))
            # Conservative exterior-distance approximation from an oriented
            # rectangle disk cover. Interior depth remains a soft-cost proxy.
            d=np.min(values[:,channel])-self.cover_radius-self.res/math.sqrt(2)
            result.append(np.clip(d,-self.settings.data['cost']['soft_band_m'],self.settings.data['cost']['soft_band_m']))
            if include_known:coverage.append(self.footprint_known(pose))
        return np.asarray(result),np.asarray(coverage) if include_known else None

    def large_crater_rectangles(self):
        """Exact union of revealed large-crater cells; cached per immutable view."""
        if not hasattr(self,'_large_crater_rectangles'):
            cells=[]
            for key,tile in self.tiles.items():
                y,x=np.nonzero(tile.known & ((tile.kind & 8)!=0))
                ty,tx=divmod(key,int(self.tile_shape[0]))
                cells.append(np.column_stack((x+tx*self.tile_size,y+ty*self.tile_size)))
            merged=_merge_occupied_cells(np.concatenate(cells) if cells else np.empty((0,2),dtype=int))
            shapes=np.empty_like(merged)
            shapes[:,:2]=(merged[:,:2]+merged[:,2:])*(.5*self.res)-self.extent/2-self.anchor[:2]
            shapes[:,2:]=(merged[:,2:]-merged[:,:2])*(.5*self.res)
            shapes.setflags(write=False);self._large_crater_rectangles=shapes
        return self._large_crater_rectangles

    def footprint_known(self,pose):
        v=self.spec.vehicle;hx=v.length/2+v.clearance_margin;hy=v.width/2+v.clearance_margin
        c,s=np.cos(pose[2]),np.sin(pose[2]);rotation=np.array([[c,s],[-s,c]])
        corners=np.array([[-hx,-hy],[hx,-hy],[hx,hy],[-hx,hy]])@rotation+pose[:2]
        lo=self._cells(corners.min(0)[None])[0];hi=self._cells(corners.max(0)[None])[0]
        x,y=np.meshgrid(np.arange(lo[0],hi[0]+1),np.arange(lo[1],hi[1]+1),indexing='xy')
        cells=np.c_[x.ravel(),y.ravel()];_,known,_=self.read_cells(cells)
        if known.all():return True
        centers=(cells[~known]+.5)*self.res-self.extent/2-self.anchor[:2]
        body=(centers-pose[:2])@rotation.T;half=self.res/2
        corner=np.array([[-half,-half],[half,-half],[half,half],[-half,half]])@rotation.T
        areas=_intersection_areas(body[:,None,:]+corner[None],(-hx,hx,-hy,hy))
        return not (areas>1e-10).any()

    def aggregate(self,pose,regions,parking=False):
        """Batch exact areas; clip only boundary cells, reuse stationary queries."""
        pose=np.asarray(pose,float);bounds=np.asarray(regions,float)
        cache_key=(pose.tobytes(),bounds.tobytes(),parking)
        cache=getattr(self,'_aggregation_cache',{})
        if cache_key in cache:return cache[cache_key]
        c,s=np.cos(pose[2]),np.sin(pose[2]);rotation=np.array([[c,s],[-s,c]])
        midpoint=np.c_[(bounds[:,0]+bounds[:,1])/2,(bounds[:,2]+bounds[:,3])/2]
        half=np.c_[(bounds[:,1]-bounds[:,0])/2,(bounds[:,3]-bounds[:,2])/2]
        world=midpoint@rotation+pose[:2]
        extent=half@np.array([[abs(c),abs(s)],[abs(s),abs(c)]])
        lo=self._cells(world-extent);hi=self._cells(world+extent)
        groups=[]
        for low,high in zip(lo,hi):
            xx,yy=np.meshgrid(np.arange(low[0],high[0]+1),np.arange(low[1],high[1]+1),indexing='xy')
            groups.append(np.c_[xx.ravel(),yy.ravel()])
        cells=np.concatenate(groups);ids=np.repeat(np.arange(len(bounds)),[len(g) for g in groups])
        centers=(cells+.5)*self.res-self.extent/2-self.anchor[:2]
        body=(centers-pose[:2])@rotation.T
        allowance=self.res/2*(abs(c)+abs(s))
        delta=abs(body-midpoint[ids])
        # Strict SAT on both rectangle and world axes excludes zero-area touches.
        intersects=(delta<half[ids]+allowance-1e-12).all(-1)
        intersects&=(abs(centers-world[ids])<extent[ids]+self.res/2-1e-12).all(-1)
        cells,ids,body,delta=cells[intersects],ids[intersects],body[intersects],delta[intersects]
        interior=(delta<=half[ids]-allowance+1e-12).all(-1)
        values,known,_=self.read_cells(cells)
        total=np.bincount(ids,minlength=len(bounds))
        observed=np.bincount(ids,weights=known,minlength=len(bounds))
        mixed=(observed>0)&(observed<total)
        # Entirely known regions need no polygon areas. For mixed regions,
        # integrate the unknown complement, clipping its boundary cells only.
        unknown=~known&mixed[ids]
        unknown_area=np.bincount(ids[unknown&interior],minlength=len(bounds))*self.res**2
        boundary=unknown&~interior
        if boundary.any():
            r=self.res/2;corner=np.array([[-r,-r],[r,-r],[r,r],[-r,r]])@rotation.T
            areas=_intersection_areas(body[boundary,None,:]+corner[None],bounds[ids[boundary]])
            unknown_area+=np.bincount(ids[boundary],weights=areas,minlength=len(bounds))
        result=np.zeros((len(bounds),3),float)
        if known.any():
            band=self.settings.data['cost']['soft_band_m'];d=values[known,7]-self.res/math.sqrt(2)
            risk=np.clip(1-d/band,0,1)**2
            if self.settings.graded_craters:
                medium=values[known,8]-self.res/math.sqrt(2)
                risk=np.maximum(risk,self.settings.data['crater_classes']['medium_risk']*np.clip(1-medium/band,0,1)**2)
            np.maximum.at(result[:,0],ids[known],risk)
            cost=values[known,5]
            if parking:cost=np.maximum(cost,values[known,6])
            np.maximum.at(result[:,1],ids[known],np.minimum(cost,1))
        area=(bounds[:,1]-bounds[:,0])*(bounds[:,3]-bounds[:,2])
        result[:,2]=np.where(observed==0,0,np.clip(1-unknown_area/area,0,1))
        result.setflags(write=False)
        if len(cache)>=2:cache.pop(next(iter(cache)))
        cache[cache_key]=result;self._aggregation_cache=cache
        return result


def _intersection_areas(polygons,box):
    """Vectorized Sutherland-Hodgman clipping of N cell squares to one rectangle."""
    n=len(polygons);p=np.zeros((n,8,2));p[:,:4]=polygons;counts=np.full(n,4,int)
    slots=np.arange(8)[None,:];rows=np.arange(n)[:,None]
    bounds=np.broadcast_to(np.asarray(box),(n,4))
    for axis,column,sign in ((0,0,1),(0,1,-1),(1,2,1),(1,3,-1)):
        bound=bounds[:,column,None]
        valid=slots<counts[:,None]
        prev=p[rows,(slots-1)%np.maximum(counts[:,None],1)]
        pin=(prev[...,axis]-bound)*sign>=-1e-12;qin=(p[...,axis]-bound)*sign>=-1e-12
        cross=valid&(pin!=qin);keep=valid&qin
        den=p[...,axis]-prev[...,axis]
        alpha=np.divide(bound-prev[...,axis],den,out=np.zeros_like(den),where=abs(den)>1e-20)
        intersections=prev+alpha[...,None]*(p-prev)
        emits=np.stack([cross,keep],-1).reshape(n,16)
        points=np.stack([intersections,p],-2).reshape(n,16,2)
        indices=np.cumsum(emits,axis=1)-1
        out=np.zeros_like(p);rr,ss=np.nonzero(emits)
        out[rr,indices[rr,ss]]=points[rr,ss]
        counts=emits.sum(1);p=out
    nxt=p[rows,(slots+1)%np.maximum(counts[:,None],1)]
    cross=p[...,0]*nxt[...,1]-p[...,1]*nxt[...,0]
    return abs(np.where(slots<counts[:,None],cross,0).sum(1))/2


class GridMemory:
    version='private_observed_grid_v1'

    def __init__(self,spec,settings):
        self.spec=spec;self.settings=settings
        self.value_channels=10 if settings.graded_craters else 8
        ratio=round(settings.data['ground']['resolution_m']/settings.data['labels']['resolution_m'])
        self._child_offsets=np.array([(x,y) for y in range(ratio) for x in range(ratio)],np.int64)
        self.reset()

    def reset(self):
        self.tiles={};self.last_seen={};self.revision=0;self._view=None;self.anchor=None;self.identity=None
        self._visible_pose=None;self._visible_keys=();self._visible_count=0
        self.stats={'frames':0,'visible_cells':0,'new_cells':0,'atlas_cells_read':0,'evicted_tiles':0,'update_s':0.}

    def update(self,pose,anchor,now,atlas):
        start=time.perf_counter();d=self.settings.data;res=d['labels']['resolution_m'];size=d['memory']['tile_cells']
        if self.identity is not None and (self.identity!=atlas.identity or not np.array_equal(self.anchor,anchor)):
            raise ValueError('Grid map/episode frame changed without reset')
        self.identity=atlas.identity;self.anchor=np.asarray(anchor).copy();self.extent=atlas.extent.copy()
        array=np.asarray(pose);pose_key=(array.dtype.str,array.tobytes())
        if self._view is not None and pose_key==self._visible_pose:
            for key in self._visible_keys:self.last_seen[key]=now
            self.stats['frames']+=1;self.stats['visible_cells']+=self._visible_count
            self.stats['update_s']+=time.perf_counter()-start
            return self._view
        shape=np.rint(atlas.extent/res).astype(int);tile_shape=np.ceil(shape/size).astype(int)
        c,s=np.cos(pose[2]),np.sin(pose[2]);p=d['perception']
        corners=np.array([[-p['reverse_m'],-p['lateral_m']],[p['forward_m'],-p['lateral_m']],
                          [p['forward_m'],p['lateral_m']],[-p['reverse_m'],p['lateral_m']]])@np.array([[c,s],[-s,c]])+pose[:2]
        lo=np.maximum(0,np.floor((corners.min(0)+atlas.extent/2)/res).astype(int))
        hi=np.minimum(shape-1,np.floor((corners.max(0)+atlas.extent/2)/res).astype(int))
        # A coarse cost aggregates an entire source cell. Reveal its children
        # only after that whole source support is visible; boundary cells must
        # not expose slope/hazard evidence from outside the field of view.
        ground_res=d['ground']['resolution_m'];ratio=round(ground_res/res)
        gx,gy=np.meshgrid(np.arange(lo[0]//ratio,hi[0]//ratio+1),
                          np.arange(lo[1]//ratio,hi[1]//ratio+1),indexing='xy')
        coarse=np.c_[gx.ravel(),gy.ravel()]
        ground_world=(coarse+.5)*ground_res-atlas.extent/2
        ground_body=(ground_world-pose[:2])@np.array([[c,-s],[s,c]])
        ground_allowance=ground_res/2*(abs(c)+abs(s))
        visible=(ground_body[:,0]>=-p['reverse_m']+ground_allowance-1e-10)&(ground_body[:,0]<=p['forward_m']-ground_allowance+1e-10)&(abs(ground_body[:,1])<=p['lateral_m']-ground_allowance+1e-10)
        cells=(coarse[visible,None,:]*ratio+self._child_offsets[None]).reshape(-1,2)
        cells=cells[((cells>=lo)&(cells<=hi)).all(-1)]
        world=(cells+.5)*res-atlas.extent/2
        body=(world-pose[:2])@np.array([[c,-s],[s,c]])
        allowance=res/2*(abs(c)+abs(s))
        visible=(body[:,0]>=-p['reverse_m']+allowance-1e-10)&(body[:,0]<=p['forward_m']-allowance+1e-10)&(abs(body[:,1])<=p['lateral_m']-allowance+1e-10)
        cells=cells[visible]
        cells=cells[np.argsort(cells[:,1]*shape[0]+cells[:,0])]
        keys=(cells[:,1]//size)*tile_shape[0]+cells[:,0]//size
        current=set(map(int,np.unique(keys)));capacity=d['memory']['capacity_cells']//size**2
        if len(current)>capacity:raise ValueError('Grid capacity cannot retain current field of view')
        tiles=dict(self.tiles);changed=set();created=set();evicted=set();obstacle_changes=set()
        for key in current:
            self.last_seen[key]=now
            if key not in tiles:
                values=np.zeros((size,size,self.value_channels),np.float32);values[...,7:]=d['cost']['soft_band_m']+math.hypot(d['geometry']['cover_spacing_m'],d['geometry']['cover_spacing_m'])
                tiles[key]=GridTile(values,np.zeros((size,size),bool),np.zeros((size,size),np.uint8),np.zeros((size,size),bool))
                changed.add(key);created.add(key)
        candidates=sorted(set(tiles)-current,key=lambda k:(self.last_seen[k],k))
        while len(tiles)>capacity:
            key=candidates.pop(0)
            if (tiles[key].known&(tiles[key].kind!=0)).any():obstacle_changes.add(key)
            del tiles[key];del self.last_seen[key];evicted.add(key)
        dirty={};pending=[];locations=[]
        for key in current:
            subset=cells[keys==key];x,y=(subset%size).T;tile=tiles[key]
            fresh=~tile.scanned[y,x]
            if fresh.any():
                dirty[key]=tile.copy_arrays();pending.append(subset[fresh]);locations.append((key,x[fresh],y[fresh]))
        new_count=sum(map(len,pending))
        if pending:
            values,kind,known=atlas.read(np.concatenate(pending));offset=0
            for key,x,y in locations:
                count=len(x);v,k,t,sc=dirty[key];sl=slice(offset,offset+count)
                v[y,x,:7]=np.where(known[sl,None],values[sl],0);k[y,x]=known[sl];t[y,x]=np.where(known[sl],kind[sl],0);sc[y,x]=True
                if t[y,x].any():obstacle_changes.add(key)
                tiles[key]=GridTile(v,k,t,sc);changed.add(key);offset+=count
        # Update a bounded halo using only observed occupied cells, never atlas labels.
        # Newly scanned free/invalid cells do not alter the binary obstacle
        # panel. Their existing distance channel already covers the whole tile.
        affected=set(created)
        for key in obstacle_changes:
            ty,tx=divmod(key,int(tile_shape[0]))
            for oy in (-1,0,1):
                for ox in (-1,0,1):
                    xx,yy=tx+ox,ty+oy
                    candidate=yy*int(tile_shape[0])+xx
                    if 0<=xx<tile_shape[0] and 0<=yy<tile_shape[1] and candidate in tiles:affected.add(candidate)
        distance_masks=((7,2|8),(8,4),(9,8)) if self.settings.graded_craters else ((7,255),)
        for key in affected:
            ty,tx=divmod(key,int(tile_shape[0]));labels=np.zeros((size*3,size*3),np.uint8)
            for oy in (-1,0,1):
                for ox in (-1,0,1):
                    xx,yy=tx+ox,ty+oy
                    other=tiles.get(yy*int(tile_shape[0])+xx) if 0<=xx<tile_shape[0] and 0<=yy<tile_shape[1] else None
                    if other is not None:labels[(oy+1)*size:(oy+2)*size,(ox+1)*size:(ox+2)*size]=np.where(other.known,other.kind,0)
            v,k,t,sc=tiles[key].copy_arrays()
            cap=d['cost']['soft_band_m']+math.hypot(d['geometry']['cover_spacing_m'],d['geometry']['cover_spacing_m'])
            for channel,mask in distance_masks:
                panel=(labels&mask)!=0
                if panel.any():
                    signed=distance_transform_edt(~panel)*res
                    signed[panel]=-distance_transform_edt(panel)[panel]*res
                    v[...,channel]=np.minimum(cap,signed[size:2*size,size:2*size])
                else:v[...,channel]=cap
            tiles[key]=GridTile(v,k,t,sc)
        self.tiles=tiles
        if changed or evicted or self._view is None:
            self.revision+=1
            self._view=GridTerrain(tiles,anchor,atlas.extent,self.settings,self.spec,self.revision,now,atlas.identity)
        self._visible_pose=pose_key;self._visible_keys=tuple(current);self._visible_count=len(cells)
        self.stats['frames']+=1;self.stats['visible_cells']+=len(cells);self.stats['new_cells']+=new_count
        self.stats['atlas_cells_read']+=new_count;self.stats['evicted_tiles']+=len(evicted);self.stats['update_s']+=time.perf_counter()-start
        return self.view()

    def view(self):
        if self._view is None:raise ValueError('Grid memory has not received its first observation')
        return self._view

    def state_dict(self):
        return {'version':self.version,'input':self.settings.metadata,'profile':self.spec.fingerprint,
                'tiles':self.tiles,'last_seen':self.last_seen,'revision':self.revision,'anchor':self.anchor,
                'extent':self.extent,'identity':self.identity,'timestamp':self.view().timestamp,'stats':dict(self.stats)}

    def load_state_dict(self,state):
        if (state['version']!=self.version or state['input']!=self.settings.metadata or state['profile']!=self.spec.fingerprint):
            raise ValueError('Grid memory checkpoint identity mismatch')
        if self.identity is not None and state['identity']!=self.identity:
            raise ValueError('Grid memory checkpoint map mismatch')
        limit=self.settings.data['memory']['capacity_cells']//self.settings.data['memory']['tile_cells']**2
        if len(state['tiles'])>limit or set(state['tiles'])!=set(state['last_seen']):raise ValueError('Invalid grid tile capacity/state')
        self.tiles={key:GridTile(*tile.copy_arrays()) for key,tile in state['tiles'].items()}
        self.last_seen=dict(state['last_seen']);self.revision=state['revision'];self.anchor=np.asarray(state['anchor']).copy()
        self.extent=np.asarray(state['extent']).copy();self.identity=state['identity'];self.stats=dict(state['stats'])
        self._visible_pose=None;self._visible_keys=();self._visible_count=0
        self._view=GridTerrain(self.tiles,self.anchor,self.extent,self.settings,self.spec,self.revision,state['timestamp'],self.identity)


class GridTerrainBatch:
    version='fixed_grid_cuda_lookup_v1'

    def __init__(self,views,device,cache=None,*,max_owners=None):
        self.device=torch.device(device)
        if self.device.type=='cuda' and self.device.index is None:
            self.device=torch.device('cuda',torch.cuda.current_device())
        self._views=tuple(views);self.batch_size=len(views)
        self.upload_bytes=0;self._rollout_query=None
        self.ambiguous_queries=self.boundary_queries=self.max_pair_entries=0;self.cpu_tie_query_s=0.
        self.cache=OrderedDict() if cache is None else cache
        limit=len(views) if max_owners is None else max_owners
        data=[];lookups=[];defaults=[]
        self.tile_size=views[0].tile_size
        self.value_channels=views[0].value_channels
        lut_shape=np.max([v.tile_shape[::-1] for v in views],axis=0)
        for view in views:
            if view.tile_size!=self.tile_size:raise ValueError('Mixed grid tile sizes')
            if view.value_channels!=self.value_channels:raise ValueError('Mixed grid crater semantics')
            owner=(view.map_identity,view.anchor.tobytes())
            previous=self.cache.pop(owner,{})
            current={}
            offset=len(data);defaults.append(offset)
            keys=sorted(view.tiles);lookup=np.full(tuple(lut_shape),offset,np.int64)
            empty=np.zeros((view.tile_size,view.tile_size,self.value_channels+1),np.float32);empty[...,7:self.value_channels]=view.cap
            data.append(torch.tensor(empty,device=self.device))
            for i,key in enumerate(keys,1):
                tile=view.tiles[key]
                item=previous.get(key)
                if item is None or item[0] is not tile:
                    raw=np.concatenate([tile.values,tile.known[...,None]],-1)
                    tensor=torch.tensor(raw,device=self.device,dtype=torch.float32)
                    item=(tile,tensor);self.upload_bytes+=raw.nbytes
                current[key]=item
                data.append(item[1]);ty,tx=divmod(key,int(view.tile_shape[0]));lookup[ty,tx]=offset+i
            lookups.append(lookup)
            self.cache[owner]=current
        while len(self.cache)>limit:self.cache.popitem(last=False)
        self.data=torch.stack(data)
        self.lut=torch.tensor(np.stack(lookups),device=self.device)
        self._row_ids=torch.arange(len(views),device=self.device)
        self._defaults=torch.tensor(defaults,device=self.device)
        self._origins=torch.tensor(np.stack([v.anchor[:2]+v.extent/2 for v in views]),device=self.device)
        self._res=torch.tensor([v.res for v in views],device=self.device,dtype=torch.float64)
        self._shape=torch.tensor(np.stack([v.shape for v in views]),device=self.device)
        self._points={dtype:torch.tensor(np.stack([v.points for v in views]),device=self.device,dtype=dtype)
                      for dtype in (torch.float32,torch.float64)}
        # These operations previously used Python scalars on float32 distances.
        self._radius=torch.tensor([v.cover_radius for v in views],device=self.device,dtype=torch.float32)
        self._margin=torch.tensor([v.res/math.sqrt(2) for v in views],device=self.device,dtype=torch.float32)
        self._band=torch.tensor([v.settings.data['cost']['soft_band_m'] for v in views],device=self.device,dtype=torch.float32)
        modes={getattr(v,'large_crater_query',None) for v in views}
        if len(modes)!=1:raise ValueError('Mixed large-crater query semantics')
        self._large_crater_index=(ObservedCraterBatch(views,self.device)
                                 if modes=={'observed_grid_sat_v1'} else None)

    def select(self,indices):
        out=object.__new__(type(self));out.__dict__.update(self.__dict__)
        out._row_ids=self._row_ids[indices];out._views=tuple(self._views[i] for i in indices);out.batch_size=len(indices)
        return out

    def _cell_indices_rows(self,rows,xy):
        """Shared cell addressing for channels queried at identical positions."""
        cells=torch.floor((xy.double()+self._origins[rows,None])/self._res[rows,None,None]+1e-10).long()
        inside=((cells>=0)&(cells<self._shape[rows,None])).all(-1)
        tx=(cells[...,0]//self.tile_size).clamp(0,self.lut.shape[2]-1)
        ty=(cells[...,1]//self.tile_size).clamp(0,self.lut.shape[1]-1)
        index=torch.where(inside,self.lut[rows[:,None],ty,tx],self._defaults[rows,None])
        y=cells[...,1]%self.tile_size;x=cells[...,0]%self.tile_size
        return index,y,x

    def _read_rows(self,rows,xy,channel=None):
        """Read [row,point,xy] without a per-rover dispatch or host constant copy."""
        index,y,x=self._cell_indices_rows(rows,xy)
        if channel is not None:return self.data[index,y,x,channel]
        return self.data[index,y,x]

    def _read(self,xy,channel=None):
        shape=xy.shape[:-1]
        values=self._read_rows(self._row_ids,xy.reshape(self.batch_size,-1,2),channel)
        return values.reshape(*shape,*(() if channel is not None else (self.value_channels+1,)))

    def lookup(self,xy):
        values=self._read(xy)
        known=values[...,self.value_channels]>.5
        return torch.where(known[...,None],values[...,:5],0).to(xy.dtype),known

    def ground_cost(self,xy):
        return self._read(xy,5)

    def lookup_ground(self,xy):
        values=self._read(xy);known=values[...,self.value_channels]>.5
        return torch.where(known[...,None],values[...,:5],0).to(xy.dtype),known,values[...,5]

    def obstacle_distance(self,poses,*,kind='obstacle'):
        if kind=='crater_large' and self._large_crater_index is not None:
            return torch.where(self._large_crater_index.hits(poses,self._row_ids),
                               -torch.ones_like(poses[...,0]),torch.ones_like(poses[...,0]))
        channel=_distance_channel(self._views[0].settings,kind)
        flat=poses.reshape(-1,3);per_row=poses[0].numel()//3;chunks=[]
        for start in range(0,len(flat),4096):
            p=flat[start:start+4096]
            rows=self._row_ids[torch.arange(start,start+len(p),device=self.device)//per_row]
            points=self._points[poses.dtype][rows];c,s=p[:,2].cos(),p[:,2].sin()
            xy=p[:,None,:2]+torch.stack([c[:,None]*points[:,:,0]-s[:,None]*points[:,:,1],
                                       s[:,None]*points[:,:,0]+c[:,None]*points[:,:,1]],-1)
            signed=self._read_rows(rows,xy,channel).amin(-1)-self._radius[rows]-self._margin[rows]
            band=self._band[rows];chunks.append(signed.clamp(-band,band))
        return torch.cat(chunks).reshape(poses.shape[:-1])

    def obstacle_distances(self,poses,*,kinds=('obstacle','crater_medium')):
        """Original disk-cover distances, with one transform/addressing pass.

        Each channel retains the original gather, reduction and subtraction
        order; this changes computation sharing, not the soft-risk definition.
        """
        channels=[_distance_channel(self._views[0].settings,kind) for kind in kinds]
        flat=poses.reshape(-1,3);per_row=poses[0].numel()//3;chunks=[]
        for start in range(0,len(flat),4096):
            p=flat[start:start+4096]
            rows=self._row_ids[torch.arange(start,start+len(p),device=self.device)//per_row]
            points=self._points[poses.dtype][rows];c,s=p[:,2].cos(),p[:,2].sin()
            xy=p[:,None,:2]+torch.stack([c[:,None]*points[:,:,0]-s[:,None]*points[:,:,1],
                                       s[:,None]*points[:,:,0]+c[:,None]*points[:,:,1]],-1)
            index,y,x=self._cell_indices_rows(rows,xy)
            radius,margin,band=self._radius[rows],self._margin[rows],self._band[rows]
            chunks.append(torch.stack([(self.data[index,y,x,ch].amin(-1)-radius-margin).clamp(-band,band)
                                       for ch in channels],-1))
        return torch.cat(chunks).reshape(*poses.shape[:-1],len(channels))


def _merge_occupied_cells(cells):
    """Exact union of occupied cells: merge equal horizontal runs vertically."""
    if not len(cells):
        return np.empty((0, 4), dtype=float)
    cells = cells[np.lexsort((cells[:, 0], cells[:, 1]))]
    rectangles = []
    active = {}
    previous_y = None
    for row in np.split(cells, np.flatnonzero(np.diff(cells[:, 1])) + 1):
        y = int(row[0, 1])
        spans = [(int(g[0, 0]), int(g[-1, 0]) + 1) for g in np.split(row, np.flatnonzero(np.diff(row[:, 0]) != 1) + 1)]
        if previous_y is None or y != previous_y + 1:
            rectangles.extend(active.values())
            active = {}
        new = {}
        for span in spans:
            rect = active.pop(span, [span[0], y, span[1], y + 1])
            rect[3] = y + 1
            new[span] = rect
        rectangles.extend(active.values())
        active = new
        previous_y = y
    rectangles.extend(active.values())
    return np.asarray(rectangles, dtype=float)

def _rectangle_hits_numpy(poses, shapes, half):
    poses = np.asarray(poses, dtype=float).reshape(-1, 3)
    if not len(shapes):
        return np.zeros(len(poses), dtype=bool)
    hit = []
    for start in range(0, len(poses), 2048):
        p = poses[start:start + 2048]
        d = shapes[None, :, :2] - p[:, None, :2]
        c, s = (np.cos(p[:, 2, None]), np.sin(p[:, 2, None]))
        u = c * d[..., 0] + s * d[..., 1]
        v = -s * d[..., 0] + c * d[..., 1]
        a, b = (abs(c), abs(s))
        x, y = (shapes[None, :, 2], shapes[None, :, 3])
        overlap = (abs(d[..., 0]) < half[0] * a + half[1] * b + x - 1e-12) & (abs(d[..., 1]) < half[0] * b + half[1] * a + y - 1e-12) & (abs(u) < half[0] + x * a + y * b - 1e-12) & (abs(v) < half[1] + x * b + y * a - 1e-12)
        hit.append(overlap.any(-1))
    return np.concatenate(hit)

class ObservedCraterBatch:
    """Spatial bins and exact rectangle SAT on this rover's observed cells."""
    bin_size = 4.0

    def __init__(self, views, device):
        self.device = torch.device(device)
        self.half = np.array([views[0].spec.vehicle.length, views[0].spec.vehicle.width]) / 2
        reach = float(np.linalg.norm(self.half))
        collections = []
        for view in views:
            shapes = view.large_crater_rectangles()
            if len(shapes):
                ext = shapes[:, 2:]
                low = np.floor((shapes[:, :2] - ext - reach) / self.bin_size).astype(int)
                high = np.floor((shapes[:, :2] + ext + reach) / self.bin_size).astype(int)
                origin = low.min(0)
                dims = high.max(0) - origin + 1
            else:
                low = high = np.empty((0, 2), dtype=int)
                origin = np.zeros(2, dtype=int)
                dims = np.ones(2, dtype=int)
            bins = {}
            for i, (lo, hi) in enumerate(zip(low, high), 1):
                for y in range(lo[1], hi[1] + 1):
                    for x in range(lo[0], hi[0] + 1):
                        bins.setdefault((y - origin[1], x - origin[0]), []).append(i)
            collections.append((shapes, origin, dims, bins))
        nshape = max((len(q[0]) for q in collections)) + 1
        nx = max((q[2][0] for q in collections))
        ny = max((q[2][1] for q in collections))
        nk = max([1] + [len(v) for q in collections for v in q[3].values()])
        shape_array = np.zeros((len(views), nshape, 4))
        lut = np.zeros((len(views), ny, nx, nk), dtype=np.int64)
        for row, (shapes, origin, dims, bins) in enumerate(collections):
            shape_array[row, 1:len(shapes) + 1] = shapes
            for (y, x), ids in bins.items():
                lut[row, y, x, :len(ids)] = ids
        self.shapes = torch.as_tensor(shape_array, device=device)
        self.lut = torch.as_tensor(lut, device=device)
        self.origins = torch.as_tensor(np.stack([q[1] for q in collections]), device=device)
        self.dims = torch.as_tensor(np.stack([q[2] for q in collections]), device=device)
        self.statistics = dict(shape_counts=[len(q[0]) for q in collections], maximum_bin_shapes=nk, lookup_bytes=lut.nbytes, shape_bytes=shape_array.nbytes)

    def hits(self, poses, row_ids=None):
        if row_ids is None:
            row_ids = torch.arange(len(poses), device=self.device)
        flat = poses.reshape(-1, 3)
        per_row = poses[0].numel() // 3
        result = []
        for start in range(0, len(flat), 8192):
            p = flat[start:start + 8192].double()
            rows = row_ids[torch.arange(start, start + len(p), device=self.device) // per_row]
            cell = torch.floor(p[:, :2] / self.bin_size).long() - self.origins[rows]
            inside = ((cell >= 0) & (cell < self.dims[rows])).all(-1)
            ids = self.lut[rows, cell[:, 1].clamp(0, self.lut.shape[1] - 1), cell[:, 0].clamp(0, self.lut.shape[2] - 1)]
            shapes = self.shapes[rows[:, None], ids]
            d = shapes[..., :2] - p[:, None, :2]
            c, s = (p[:, 2, None].cos(), p[:, 2, None].sin())
            u = c * d[..., 0] + s * d[..., 1]
            v = -s * d[..., 0] + c * d[..., 1]
            a, b = (c.abs(), s.abs())
            x, y = (shapes[..., 2], shapes[..., 3])
            overlap = (d[..., 0].abs() < self.half[0] * a + self.half[1] * b + x - 1e-12) & (d[..., 1].abs() < self.half[0] * b + self.half[1] * a + y - 1e-12) & (u.abs() < self.half[0] + x * a + y * b - 1e-12) & (v.abs() < self.half[1] + x * b + y * a - 1e-12)
            result.append((overlap & (ids != 0) & inside[:, None]).any(-1))
        return torch.cat(result).reshape(poses.shape[:-1])
