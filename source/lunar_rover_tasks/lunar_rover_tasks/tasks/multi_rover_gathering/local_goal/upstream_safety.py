"""Profile-bound discrete CBF/QP with full nonlinear footprint/stop review.

Only detached local samples and sender predictions are consumed. The actuator
union is solved as three 2-D convex QPs; there is no safety slack. Linearized
support discs cannot replace the independent plane/clearance/sweep checks.
"""
from copy import deepcopy
from dataclasses import dataclass
import time

import numpy as np
from scipy import sparse

from .upstream_execution import _polygon,control_error,limit_control_numpy,midpoint_numpy,rollout_numpy,stop_envelope,terrain_gain_numpy
from .upstream_footprint import FootprintTerrain,FootprintNeighbor
from .upstream_review_cache import PoseCachedReviewer

SAFETY_VERSION='footprint_discrete_cbf_upstream_v1'


def midpoint_jacobian(pose,control,gain,dt):
    v,w=control;angle=pose[2]+.5*dt*w*gain;c,s=np.cos(angle),np.sin(angle)
    return np.array([[dt*gain*c,-.5*dt**2*v*gain**2*s],
                     [dt*gain*s,.5*dt**2*v*gain**2*c],[0.,dt*gain]])


@dataclass(frozen=True)
class SafetyContext:
    pose: np.ndarray
    previous_control: np.ndarray
    terrain: FootprintTerrain
    neighbors: tuple
    now: float
    workspace: object = None
    speed_cap: float | None = None
    hold: bool = False


@dataclass(frozen=True)
class SafetyDecision:
    nominal_control: np.ndarray
    control: np.ndarray
    verified: bool
    status: str
    reason: str
    qp_status: str
    residual: dict
    constraint_sources: tuple
    iterations: int
    prediction: object
    seconds: float


class UpstreamSafetyProjector:
    def __init__(self,spec,*,reviewer=None):
        import osqp
        if osqp.__version__!='1.0.4':raise RuntimeError('Safety projection requires osqp==1.0.4')
        self.spec=spec;self.reviewer=reviewer or PoseCachedReviewer(spec);self.last_control=None
        if self.reviewer.spec.fingerprint!=spec.fingerprint:raise ValueError('Safety reviewer profile mismatch')

    @property
    def metadata(self):return {'version':SAFETY_VERSION,'state_version':1,'profile_fingerprint':self.spec.fingerprint,
        'solver':'osqp_1.0.4','maximum_linearizations':3,'tolerance':1e-6,'safety_slack':False}

    def reset(self):self.last_control=None

    def state_dict(self):return {'metadata':self.metadata,'last_control':deepcopy(self.last_control)}

    def load_state_dict(self,state):
        if not isinstance(state,dict) or state.get('metadata')!=self.metadata:raise ValueError('Safety snapshot mismatch')
        control=state.get('last_control')
        if control is not None and control_error(control,control,self.spec):raise ValueError('Illegal safety warm control')
        self.last_control=None if control is None else np.asarray(control,float).copy()

    def validate(self,context):
        if type(context) is not SafetyContext or type(context.terrain) is not FootprintTerrain:
            raise ValueError('Safety requires typed detached local inputs')
        if context.terrain.profile_fingerprint!=self.spec.fingerprint:raise ValueError('Safety terrain/profile mismatch')
        for value,shape in ((context.pose,(3,)),(context.previous_control,(2,))):
            if np.shape(value)!=shape or not np.isfinite(value).all():raise ValueError('Safety actual state is invalid')
        if control_error(context.previous_control,context.previous_control,self.spec):raise ValueError('Safety previous control is illegal')
        if not np.isfinite(context.now) or context.now<0 or type(context.hold) is not bool:raise ValueError('Invalid safety clock/HOLD')
        if context.speed_cap is not None and (not np.isfinite(context.speed_cap) or context.speed_cap<0
                or context.speed_cap>self.spec.vehicle.forward_limit):raise ValueError('Invalid execution speed bound')
        if len({n.sender_id for n in context.neighbors})!=len(context.neighbors):raise ValueError('Duplicate safety sender')
        for peer in context.neighbors:
            if type(peer) is not FootprintNeighbor or peer.contract_error(self.spec,self.spec.fingerprint):raise ValueError('Invalid safety sender prediction')
        if (context.terrain.readonly_snapshot().observed_at>context.now+1e-10).any():raise ValueError('Safety cannot read future terrain')
        if context.workspace is not None:
            box=np.asarray(context.workspace)
            if box.shape!=(2,2) or not np.isfinite(box).all() or (box[1]<=box[0]).any():raise ValueError('Invalid safety workspace')

    def _gain(self,context):
        last_key = None; last_gain = None
        def gain(pose):
            nonlocal last_key, last_gain
            key = np.asarray(pose, dtype=float).tobytes()
            if key == last_key:return last_gain
            f,known=context.terrain.lookup(pose[:2])
            last_gain = terrain_gain_numpy(pose,f[0],self.spec) if known[0] else .25
            last_key = key
            return last_gain
        gain._stationary_repeatable = True
        return gain

    def _geometry_constraints(self,context,reference):
        """One immutable geometry block per linearization, shared by all modes."""
        rows=[];upper=[];sources=set()
        gain=self._gain(context)(context.pose);end,_=midpoint_numpy(context.pose,reference,self.spec,gain=gain)
        jac=midpoint_jacobian(context.pose,reference,gain,self.spec.grid.dt)
        snapshot=context.terrain.readonly_snapshot();tree=context.terrain._tree
        if not len(snapshot.xy):raise ValueError('unknown_footprint')
        panels=[]
        for alpha in np.linspace(0.,1.,9):
            pose=context.pose+alpha*(end-context.pose)
            c,s=np.cos(pose[2]),np.sin(pose[2]);local=self.reviewer.points
            offsets=local@np.array([[c,s],[-s,c]])
            panels.append((alpha,pose,offsets,pose[:2]+offsets))
        distances_all,indices_all=tree.query(np.concatenate([x[3] for x in panels]))
        distances_all=distances_all.reshape(9,-1);indices_all=indices_all.reshape(9,-1)
        initial_neighbors=[peer.at(context.now) for peer in context.neighbors]
        for panel,(alpha,pose,offsets,points) in enumerate(panels):
            j=alpha*jac
            distances,indices=distances_all[panel],indices_all[panel]
            # All observed support is local. Unsafe sampled attributes are not
            # permitted support discs; full nonlinear review resolves ties.
            f=snapshot.features[indices];usable=(f[:,4]>=self.spec.terrain.minimum_traversability)
            usable&=np.linalg.norm(f[:,1:3],axis=-1)<=self.spec.terrain.driving_slope+1e-10
            usable&=f[:,3]<=self.spec.terrain.roughness_rms+1e-10
            radius=np.where(usable,snapshot.support_radius,0.)
            offset=points-snapshot.xy[indices]
            point_jac=j[:2][None]+np.stack([-offsets[:,1],offsets[:,0]],-1)[:,:,None]*j[2]
            gradients=-2*np.einsum('ni,nij->nj',offset,point_jac)
            values=radius**2-distances**2
            rows.append(-gradients);upper.append(values-gradients@reference);sources.add('terrain_support')
            for peer,(initial_position,initial_radius) in zip(context.neighbors,initial_neighbors):
                position,radius=peer.at(context.now+alpha*self.spec.grid.dt)
                relative=pose[:2]-position;gradient=2*relative@j[:2]
                initial=np.sum((context.pose[:2]-initial_position)**2)-initial_radius**2
                minimum=max(0.,.8*initial) if alpha==1. else 0.
                rows.append((-gradient)[None]);upper.append(np.array([relative@relative-radius**2-minimum-gradient@reference]))
                sources.add(f'neighbor:{peer.sender_id}')
            if context.workspace is not None:
                for axis in range(2):
                    for direction in (-1.,1.):
                        row=direction*j[axis];boundary=direction*np.asarray(context.workspace)[int(direction>0),axis]-self.spec.vehicle.effective_radius
                        rows.append(row[None]);upper.append(np.array([boundary-direction*pose[axis]+row@reference]));sources.add('workspace')
        return np.concatenate(rows),np.concatenate(upper),sources

    def _constraints(self,context,reference,mode,*,geometry=None):
        a,b=_polygon(np.asarray(context.previous_control),self.spec,mode,1e-9)
        rows=list(a);upper=list(b);sources=['actuator']*len(b)
        if context.speed_cap is not None:
            rows.extend([[1.,0.],[-1.,0.]]);upper.extend([context.speed_cap,context.speed_cap]);sources.extend(['motion_bound']*2)
        base_count=len(upper)
        if geometry is None:geometry=self._geometry_constraints(context,reference)
        ga,gb,gsources=geometry
        sources.extend(gsources)
        a=np.concatenate((np.asarray(rows),ga));b=np.concatenate((np.asarray(upper),gb))
        # Thousands of support discs are inactive throughout the small
        # physical control polygon. Prove redundancy against every vertex;
        # removing only those rows keeps the feasible set unchanged and avoids
        # fixed-rho OSQP conditioning on thousands of unnecessary constraints.
        pairs=np.asarray([(j,i) for i in range(base_count) for j in range(i)])
        systems=a[pairs];nonsingular=np.abs(np.linalg.det(systems))>1e-15
        vertices=np.linalg.solve(systems[nonsingular],b[pairs[nonsingular]][...,None])[...,0]
        # Keep the exact per-vertex matrix-vector reduction used by the solver
        # authority; batch GEMM can round the redundant-row boundary differently.
        vertices=np.asarray([v for v in vertices if (a[:base_count]@v<=b[:base_count]+1e-12).all()])
        if len(vertices):
            maximum=(a[base_count:]@np.asarray(vertices).T).max(-1)
            keep=np.r_[np.ones(base_count,bool),maximum>b[base_count:]-1e-10]
            a,b=a[keep],b[keep]
        else:a,b=a[:base_count],b[:base_count]
        return a,b,tuple(sorted(set(sources)))

    def _verify(self,context,control,*,cbf=True):
        reason=control_error(control,context.previous_control,self.spec,tolerance=1e-8)
        if reason:return reason,None
        if context.speed_cap is not None and abs(control[0])>context.speed_cap+1e-10:return 'motion_bound',None
        envelope=stop_envelope(context.pose,context.previous_control,control,self.spec,
            information_time=context.now,latency=0.,gain=self._gain(context),pad_to_horizon=True)
        if not envelope.fits_horizon:return 'stop_horizon_exceeded',None
        prediction=envelope.trajectory
        report=self.reviewer.trajectory(prediction,context.previous_control,context.terrain,
            neighbors=context.neighbors,workspace=context.workspace,require_stop=True,conservative_reserve=True)
        if not report.passed:return report.reason,None
        if cbf:
            end,_=midpoint_numpy(context.pose,control,self.spec,gain=self._gain(context)(context.pose))
            for peer in context.neighbors:
                p0,r0=peer.at(context.now);p1,r1=peer.at(context.now+self.spec.grid.dt)
                if np.sum((end[:2]-p1)**2)-r1**2<.8*(np.sum((context.pose[:2]-p0)**2)-r0**2)-1e-6:
                    return f'cbf_recheck:{peer.sender_id}',None
        return '',prediction

    def project(self,context,nominal):
        import osqp
        self.validate(context);start=time.perf_counter();nominal=np.asarray(nominal,float)
        if nominal.shape!=(2,) or not np.isfinite(nominal).all():raise ValueError('Invalid safety nominal control')
        reference=limit_control_numpy(nominal,context.previous_control,self.spec)
        status='';reason='';residual={};sources=();attempts=0;chosen=None;prediction=None
        if not context.hold:
            for _ in range(3):
                candidates=[];geometry=None
                for mode in range(3):
                    try:
                        if geometry is None:geometry=self._geometry_constraints(context,reference)
                        a,b,ids=self._constraints(context,reference,mode,geometry=geometry)
                    except ValueError as error:
                        reason=str(error);break
                    scale=np.array([
                        self.spec.vehicle.linear_acceleration*self.spec.grid.dt,
                        self.spec.vehicle.angular_acceleration*self.spec.grid.dt])
                    weights=1/scale**2
                    # Solve in normalized control coordinates and normalize
                    # rows. Fixed rho then has comparable scales for tiny
                    # terrain derivatives and physical actuator constraints.
                    scaled=a*scale;norm=np.linalg.norm(scaled,axis=1)
                    rows=(norm>1e-14)|(b<0.)
                    divisor=np.where(norm[rows]>1e-14,norm[rows],1.)
                    matrix=scaled[rows]/divisor[:,None];bound=b[rows]/divisor
                    solver=osqp.OSQP();attempts+=1
                    solver.setup(P=sparse.eye(2,format='csc'),q=-nominal/scale,A=sparse.csc_matrix(matrix),
                        l=np.full(len(bound),-np.inf),u=bound,eps_abs=1e-6,eps_rel=1e-6,max_iter=4000,
                        adaptive_rho=False,polishing=True,verbose=False)
                    if self.last_control is not None:solver.warm_start(x=self.last_control/scale)
                    result=solver.solve(raise_error=False);status=result.info.status
                    if status!='solved' or result.x is None or not np.isfinite(result.x).all():
                        reason='qp:'+status;continue
                    control=limit_control_numpy(result.x*scale,context.previous_control,self.spec)
                    if np.max(a@control-b)>1e-6:reason='linear_constraint_residual';continue
                    candidates.append((float(np.sum(weights*(control-nominal)**2)),control,ids,
                        {'primal':float(result.info.prim_res),'dual':float(result.info.dual_res)}))
                if not candidates:break
                _,reference,sources,residual=min(candidates,key=lambda x:x[0])
                reason,prediction=self._verify(context,reference)
                if not reason:chosen=reference;status='solved';break
        if chosen is None:
            chosen=limit_control_numpy([0.,0.],context.previous_control,self.spec)
            brake_reason,prediction=self._verify(context,chosen,cbf=False)
            reason=reason or ('hold_request' if context.hold else 'no_verified_qp_solution')
            if brake_reason:reason+=';'+brake_reason
            execution_status='unverified_brake' if brake_reason else 'verified_brake'
        else:execution_status='unchanged' if np.max(abs(chosen-nominal))<=1e-6 else 'projected'
        self.last_control=chosen.copy()
        return SafetyDecision(nominal.copy(),chosen.copy(),prediction is not None,execution_status,reason,status,
            residual,sources,attempts,prediction,time.perf_counter()-start)

    def execution_prediction(self,context,control):
        self.validate(context)
        return self._verify(context,np.asarray(control,float),cbf=False)[1]
