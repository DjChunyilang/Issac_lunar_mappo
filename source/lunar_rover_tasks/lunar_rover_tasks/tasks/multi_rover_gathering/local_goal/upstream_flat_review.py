"""Exact regular-flat local support witness; all dynamic gates remain active."""
import numpy as np

from .upstream_footprint import FootprintTerrain,FootprintReviewer,FootprintReport,QUERY_V2
from .upstream_review_cache import PoseCachedReviewer
from .upstream_execution import control_error,midpoint_numpy

ACCELERATION_VERSION='flat_grid_pose_cache_v2'


class FlatGridPoseReviewer(PoseCachedReviewer):
    acceleration_version=ACCELERATION_VERSION

    def __init__(self,spec,**kwargs):
        super().__init__(spec,**kwargs);self._witness_key=None;self._witness=None;self._witness_terrain=None
        lo=self.points.min(0);hi=self.points.max(0)
        self._corners=np.array([[lo[0],lo[1]],[lo[0],hi[1]],[hi[0],lo[1]],[hi[0],hi[1]]])

    def _prepare_witness(self,terrain):
        snapshot=terrain.readonly_snapshot();xy=snapshot.xy;f=snapshot.features
        if len(xy)<4 or not (f[:,0]==f[0,0]).all() or (f[:,1:4]!=0.).any() or (f[:,4]!=1.).any():return None
        return self._grid_witness(terrain,xy)

    def _grid_witness(self,terrain,xy):
        snapshot=terrain.readonly_snapshot()
        if len(xy)<4:return None
        difference=xy[1:min(33,len(xy))]-xy[0];length=np.linalg.norm(difference,axis=-1)
        if not (length>1e-12).any():return None
        length=np.where(length>1e-12,length,np.inf);i=int(length.argmin());spacing=length[i]
        y_axis=difference[i]/spacing;x_axis=np.array([y_axis[1],-y_axis[0]])
        basis=np.column_stack((x_axis,y_axis));coordinates=(xy-xy[0])@basis
        scaled=coordinates/spacing
        if np.max(np.abs(scaled))>2**40:return None
        indices=np.rint(scaled).astype(np.int64)
        error=float(np.linalg.norm(coordinates-indices*spacing,axis=-1).max())
        if error>1e-7 or spacing/np.sqrt(2)+error>snapshot.support_radius-1e-10:return None
        low=indices.min(0);high=indices.max(0);shape=high-low+1
        if int(shape[0])*int(shape[1])>len(xy):return None
        slots=(indices[:,0]-low[0])*shape[1]+indices[:,1]-low[1]
        if len(np.unique(slots))!=int(shape[0])*int(shape[1]):return None
        bounds=np.stack((low,high))*spacing
        if (np.diff(bounds,axis=0)[0] < np.ptp(self.points,axis=0)).any():return None
        center=xy[0]+bounds.mean(0)@basis.T;yaw=float(np.arctan2(x_axis[1],x_axis[0]))
        reports={p:FootprintReviewer.pose(self,[*center,yaw],terrain,parking=p) for p in (False,True)}
        if any(r.reason=='unknown_footprint' for r in reports.values()):return None
        return xy[0],basis,bounds,reports

    def _ensure_witness(self,terrain):
        key=(id(terrain),id(terrain._xy),id(terrain._features),terrain.support_radius,terrain.query_version,terrain.profile_fingerprint)
        if key!=self._witness_key:
            witness_key=(self.acceleration_version,self._fingerprint,self.version)
            if witness_key not in terrain._review_witnesses:
                terrain._review_witnesses[witness_key]=self._prepare_witness(terrain)
            self._witness_key=key;self._witness_terrain=terrain;self._witness=terrain._review_witnesses[witness_key]

    @staticmethod
    def _neighbor_margins(position, peers, radii, allowances):
        return np.linalg.norm(position-peers,axis=-1)-(radii+allowances-1e-8)

    def _flat_trajectory(self,trajectory,previous,terrain,neighbors,workspace,require_stop,parking_end):
        """Batch full moving sweeps inside an exact flat support witness.

        Every original control and sample is checked. If a static/dynamics
        check or numerical boundary needs the scalar ordering, defer to it.
        Only the original dynamic first-failure reduction is accelerated.
        """
        if self.version != 'footprint_review_v2':return None
        if any(a.dtype != np.float64 for a in (trajectory.states,trajectory.controls,trajectory.velocities)):return None
        self._ensure_witness(terrain)
        if self._witness is None:return None
        origin,basis,bounds,reports=self._witness
        if not reports[False].passed:return None
        from .upstream_native import sequence_valid_cpu
        if previous.shape != (2,) or not np.isfinite(previous).all():return None
        if not sequence_valid_cpu(trajectory.controls,previous,self.spec):return None
        u=trajectory.controls;states=trajectory.states;dt=self.spec.grid.dt
        # A flat witness has exactly unit traversability and zero gradients.
        atol,rtol=1e-10,1e-8
        if not np.array_equal(trajectory.velocities,u):return None
        mid=states[:-1,2]+.5*dt*u[:,1]
        expected=states[:-1]+dt*np.column_stack((u[:,0]*np.cos(mid),u[:,0]*np.sin(mid),u[:,1]))
        if not np.allclose(expected,states[1:],atol=atol,rtol=rtol):return None
        if (abs(abs(expected-states[1:])-(atol+rtol*abs(states[1:])))<1e-12).any():return None
        # Preserve the independent velocity-ratio gate before dynamics.
        nonzero=abs(u)>1e-10
        if ((abs(u)<1e-10)&(abs(trajectory.velocities)>1e-8)).any():return None
        ratios=np.divide(trajectory.velocities,u,out=np.ones_like(u),where=nonzero)
        if ((ratios<.25-1e-7)|(ratios>1+1e-7)).any():return None
        if ((nonzero.all(-1)) & (np.ptp(ratios,axis=-1)>1e-7)).any():return None
        delta=np.diff(states,axis=0);delta[:,2]=(delta[:,2]+np.pi)%(2*np.pi)-np.pi
        displacement=np.linalg.norm(delta[:,:2],axis=-1)+self.sweep_radius*abs(delta[:,2])
        relative=np.zeros(len(u))
        for peer in neighbors:
            if trajectory.times[0]<peer.times[0]-1e-9 or trajectory.times[-1]>peer.times[-1]+1e-9:return None
            xy=np.column_stack([np.interp(trajectory.times,peer.times,peer.positions[:,j]) for j in range(2)])
            radii=np.interp(trajectory.times,peer.times,peer.radii)
            relative=np.maximum(relative,np.linalg.norm(np.diff(xy,axis=0),axis=-1)+abs(np.diff(radii)))
        scaled=(displacement+relative)/self.spec.vehicle.sweep_spacing
        if ((scaled>=8.-1e-9)&(abs(scaled-np.rint(scaled))<1e-9)).any():return None
        intervals=np.maximum(8,np.ceil(scaled).astype(int))
        grids={int(n):np.linspace(0.,1.,int(n)+1) for n in np.unique(intervals)}
        alpha=np.concatenate([grids[int(n)] for n in intervals])
        step=np.repeat(np.arange(len(u)),intervals+1)
        poses=states[step]+alpha[:,None]*delta[step]
        c,s=np.cos(poses[:,2]),np.sin(poses[:,2])
        rotation=np.stack((c,s,-s,c),axis=-1).reshape(-1,2,2)
        local=(self._corners@rotation+poses[:,None,:2]-origin)@basis
        if not ((local>=bounds[0]+1e-9).all() and (local<=bounds[1]-1e-9).all()):return None
        first_workspace=first_neighbor=len(poses)
        if workspace is not None:
            radius=self.spec.vehicle.effective_radius
            gaps=np.concatenate((poses[:,:2]-radius-workspace[0],workspace[1]-poses[:,:2]-radius),axis=-1)
            if (abs(gaps)<1e-9).any():return None
            failed=np.flatnonzero((gaps<0.).any(-1))
            if len(failed):first_workspace=int(failed[0])
        times=trajectory.times[step]+alpha*dt
        allowance=np.repeat((np.linalg.norm(delta[:,:2],axis=-1)+relative)/(2*intervals),intervals+1)
        for peer in neighbors:
            xy=np.column_stack([np.interp(times,peer.times,peer.positions[:,j]) for j in range(2)])
            radii=np.interp(times,peer.times,peer.radii)
            margin=np.linalg.norm(poses[:,:2]-xy,axis=-1)-(radii+allowance-1e-8)
            if (abs(margin)<1e-9).any():return None
            failed=np.flatnonzero(margin<0.)
            if len(failed):first_neighbor=min(first_neighbor,int(failed[0]))
        if first_workspace<len(poses) and first_workspace<=first_neighbor:
            return FootprintReport(False,'workspace_clearance',checked_poses=first_workspace+1)
        if first_neighbor<len(poses):return FootprintReport(False,'neighbor_swept_clearance',checked_poses=first_neighbor+1)
        if require_stop and np.max(abs(u[-1]))>1e-8:
            return FootprintReport(False,'missing_stationary_tail',checked_poses=len(poses))
        if parking_end:
            end=self.pose(states[-1],terrain,parking=True)
            if not end.passed:return end
        return FootprintReport(True,checked_poses=len(poses))

    def _stationary_trajectory(self,trajectory,previous,terrain,neighbors,workspace,parking_end):
        """Batch every original future sweep sample of an exactly stopped path.

        Static geometry is proved once; growing neighbor uncertainty is still
        checked at every original time. Unknown terrain, motion, numerical
        boundaries and unsupported forecasts use the general reviewer.
        """
        if (self.version!='footprint_review_v2' or np.any(trajectory.controls)
                or np.any(trajectory.velocities)
                or not (trajectory.states==trajectory.states[:1]).all()):return None
        if control_error(trajectory.controls[0],previous,self.spec):return None
        pose=trajectory.states[0]
        expected,_=midpoint_numpy(pose,trajectory.controls[0],self.spec)
        if not np.array_equal(expected,pose):return None
        self._ensure_witness(terrain)
        if self._witness is None:return None
        origin,basis,bounds,reports=self._witness
        if not reports[False].passed:return None
        c,s=np.cos(pose[2]),np.sin(pose[2])
        local=(self._corners@np.array([[c,s],[-s,c]])+pose[:2]-origin)@basis
        if not ((local>=bounds[0]+1e-9).all() and (local<=bounds[1]-1e-9).all()):return None
        # Match the center-support/dynamics check before the first swept pose.
        if not terrain.lookup(pose[:2])[1][0]:return None
        radius=self.spec.vehicle.effective_radius
        count=len(trajectory.controls);relative=np.zeros(count)
        for peer in neighbors:
            if trajectory.times[0]<peer.times[0]-1e-9 or trajectory.times[-1]>peer.times[-1]+1e-9:return None
            positions=np.column_stack([np.interp(trajectory.times,peer.times,peer.positions[:,j]) for j in range(2)])
            radii=np.interp(trajectory.times,peer.times,peer.radii)
            relative=np.maximum(relative,np.linalg.norm(np.diff(positions,axis=0),axis=-1)+abs(np.diff(radii)))
        if workspace is not None and ((pose[:2]-radius<workspace[0]).any() or
                                      (pose[:2]+radius>workspace[1]).any()):
            return FootprintReport(False,'workspace_clearance',checked_poses=1)
        scaled=relative/self.spec.vehicle.sweep_spacing
        if ((scaled>=8.-1e-9) & (abs(scaled-np.rint(scaled))<1e-9)).any():return None
        intervals=np.maximum(8,np.ceil(scaled).astype(int))
        grids={int(n):np.linspace(0.,1.,int(n)+1) for n in np.unique(intervals)}
        times=np.concatenate([trajectory.times[i]+grids[int(n)]*self.spec.grid.dt for i,n in enumerate(intervals)])
        allowances=np.repeat(relative/(2*intervals),intervals+1)
        if neighbors:
            positions=np.stack([np.column_stack([np.interp(times,p.times,p.positions[:,j]) for j in range(2)]) for p in neighbors])
            radii=np.stack([np.interp(times,p.times,p.radii) for p in neighbors])
            margins=self._neighbor_margins(pose[:2],positions,radii,allowances)
            if (abs(margins)<1e-9).any():return None
            failures=np.flatnonzero((margins<0.).any(0))
            if len(failures):return FootprintReport(False,'neighbor_swept_clearance',checked_poses=int(failures[0])+1)
        if parking_end:
            end=self.pose(pose,terrain,parking=True)
            if not end.passed:return end
        return FootprintReport(True,checked_poses=len(times))

    def _clear_sweep(self,start,delta,alphas,terrain,workspace,neighbors,time,relative_bound,intervals):
        if (type(terrain) is not FootprintTerrain or terrain.profile_fingerprint!=self._fingerprint
                or terrain.query_version!=QUERY_V2):return False
        self._ensure_witness(terrain)
        if self._witness is None:return False
        origin,basis,bounds,reports=self._witness
        if not reports[False].passed:return False
        poses=start+alphas[:,None]*delta
        c,s=np.cos(poses[:,2]),np.sin(poses[:,2])
        rotation=np.stack((c,s,-s,c),axis=-1).reshape(-1,2,2)
        corners=self._corners@rotation+poses[:,None,:2]
        local=(corners-origin)@basis
        # Boundary rounding never creates a new acceptance; use scalar checks.
        if not ((local>=bounds[0]+1e-9).all() and (local<=bounds[1]-1e-9).all()):return False
        # The whole segment has certified static terrain. Determine the first
        # dynamic failure in the original pose/workspace/neighbor order instead
        # of discarding this batch and repeating it point by point.
        first_workspace=len(poses)
        if workspace is not None:
            radius=self.spec.vehicle.effective_radius
            gaps=np.concatenate((poses[:,:2]-radius-workspace[0],workspace[1]-poses[:,:2]-radius),axis=-1)
            if (abs(gaps)<1e-9).any():return False
            failures=np.flatnonzero((gaps<0.).any(-1))
            if len(failures):first_workspace=int(failures[0])
        times=time+alphas*self.spec.grid.dt
        allowance=(np.linalg.norm(delta[:2])+relative_bound)/(2*intervals)
        first_neighbor=len(poses)
        for neighbor in neighbors:
            if times[0]<neighbor.times[0]-1e-9 or times[-1]>neighbor.times[-1]+1e-9:return False
            positions=np.column_stack([np.interp(times,neighbor.times,neighbor.positions[:,j]) for j in range(2)])
            radii=np.interp(times,neighbor.times,neighbor.radii)
            gaps=np.linalg.norm(poses[:,:2]-positions,axis=-1)-(radii+allowance-1e-8)
            if (abs(gaps)<1e-9).any():return False
            failures=np.flatnonzero(gaps<0.)
            if len(failures):first_neighbor=min(first_neighbor,int(failures[0]))
        first=min(first_workspace,first_neighbor)
        if first<len(poses):
            reason='workspace_clearance' if first_workspace<=first_neighbor else 'neighbor_swept_clearance'
            return FootprintReport(False,reason,checked_poses=first+1)
        return True

    def pose(self,pose,terrain,*,parking=False):
        pose=np.asarray(pose,float)
        if (pose.shape!=(3,) or not np.isfinite(pose).all() or type(terrain) is not FootprintTerrain
                or terrain.profile_fingerprint!=self._fingerprint or terrain.query_version!=QUERY_V2):
            return super().pose(pose,terrain,parking=parking)
        self._ensure_witness(terrain)
        if self._witness is not None:
            origin,basis,bounds,reports=self._witness;c,s=np.cos(pose[2]),np.sin(pose[2])
            corners=self._corners@np.array([[c,s],[-s,c]])+pose[:2]
            local=(corners-origin)@basis
            if (local>=bounds[0]).all() and (local<=bounds[1]).all():return reports[bool(parking)]
        return super().pose(pose,terrain,parking=parking)


class FlatFramePoseReviewer(FlatGridPoseReviewer):
    """Use a complete observed frame inside a mixed-orientation flat memory.

    Every retained point must have the same flat safe attributes. A complete
    regular subset proves support inside its rectangle; remaining observations
    cannot invalidate that proof or hide a hazard because they were checked too.
    Outside that rectangle the original full query remains authoritative.
    """
    acceleration_version='flat_frame_pose_cache_v3'

    def _prepare_witness(self,terrain):
        snapshot=terrain.readonly_snapshot();xy=snapshot.xy;f=snapshot.features
        if len(xy)<4 or not (f[:,0]==f[0,0]).all() or (f[:,1:4]!=0.).any() or (f[:,4]!=1.).any():return None
        for timestamp in np.unique(snapshot.observed_at)[-3:][::-1]:
            witness=self._grid_witness(terrain,xy[snapshot.observed_at==timestamp])
            if witness is not None:return witness
        return self._grid_witness(terrain,xy)


def create_reviewer(spec,acceleration='pose_scope_cache_v1'):
    if acceleration==ACCELERATION_VERSION:return FlatGridPoseReviewer(spec)
    if acceleration=='flat_frame_pose_cache_v3':return FlatFramePoseReviewer(spec)
    if acceleration=='pose_scope_cache_v1':return PoseCachedReviewer(spec)
    if acceleration=='disabled':return FootprintReviewer(spec)
    raise ValueError('Unsupported full-footprint reviewer')
