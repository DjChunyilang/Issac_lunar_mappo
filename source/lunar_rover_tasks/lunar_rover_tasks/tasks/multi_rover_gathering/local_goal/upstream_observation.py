"""S03 private-memory Actor and centralized Critic with unchanged N1 layout.

Actor: 15 ego + 3x17 neighbors + 112x2 terrain + 5 aggregation +
115 local intent/feedback + one task clock = 411. Dense SensorFrame is never
decoded from network input. Critic: historical 991-slot backbone layout with
new-scale normalization and nine reserved zero Oracle slots.
"""
from copy import deepcopy
from dataclasses import dataclass,field,replace
import math

import numpy as np
import torch

from .contracts import PlannerFeedback,wrap
from .upstream_communication import MESSAGE_V2,ReceivedMessage,ReceivedSnapshot,validate_actuator_state
from .upstream_footprint import FootprintTerrain
from .upstream_grid import GridTerrain
from .upstream_obstacle_spec import OBSERVATION_VERSION as GRID_OBSERVATION_VERSION
from .upstream_planning import UpstreamIntent
from .upstream_spec import UpstreamSpec

OBSERVATION_VERSION="dense_actor_clock_upstream_v1"
ACTOR_DIM=411
CRITIC_DIM=991
STATUS={"reset":0.,"updated_mean":.2,"sample":.3,"line":.2,"quintic":.2,"nmpc":.2,
        "primitive":.2,"feasible_prefix":.5,"previous_plan":.6,"hold":.8,"brake":1.,"emergency_brake":1.}


def _status(value):
    if value.startswith('primitive:'):
        suffix=value.removeprefix('primitive:')
        if not suffix.isdigit() or not 0<=int(suffix)<47:raise ValueError("Invalid native primitive feedback")
        return STATUS['primitive']
    if value not in STATUS:raise ValueError("Unknown public planner feedback status")
    return STATUS[value]


def _owned(value,shape,dtype=float):
    result=np.asarray(value,dtype=dtype).copy()
    if result.shape!=shape or not np.isfinite(result).all():raise ValueError("Invalid versioned observation array")
    result.setflags(write=False)
    return result


def _rotate(value,yaw):
    c,s=math.cos(yaw),math.sin(yaw)
    return np.asarray(value)@np.array([[c,s],[-s,c]])


@dataclass(frozen=True)
class ObservationInput:
    spec: UpstreamSpec
    env_id: int
    agent_id: int
    episode: int
    now: float
    episode_seconds: float
    ego_pose: np.ndarray
    actual_velocity: np.ndarray
    previous_control: np.ndarray
    terrain: FootprintTerrain
    snapshot: ReceivedSnapshot
    intent: UpstreamIntent | None = None
    feedback: PlannerFeedback = field(default_factory=PlannerFeedback)

    def __post_init__(self):
        if (type(self.spec) is not UpstreamSpec or type(self.env_id) is not int or self.env_id<0
                or type(self.agent_id) is not int or not 0<=self.agent_id<4 or type(self.episode) is not int or self.episode<0):
            raise ValueError("Observation identity/profile mismatch")
        if (not np.isfinite([self.now,self.episode_seconds]).all() or self.now<0 or self.episode_seconds<=0
                or type(self.terrain) not in (FootprintTerrain,GridTerrain) or self.terrain.profile_fingerprint!=self.spec.fingerprint
                or type(self.snapshot) is not ReceivedSnapshot):
            raise ValueError("Observation requires current local terrain, legal messages and a finite task clock")
        for name,shape in [('ego_pose',(3,)),('actual_velocity',(2,)),('previous_control',(2,))]:
            object.__setattr__(self,name,_owned(getattr(self,name),shape))
        validate_actuator_state(self.previous_control,self.actual_velocity,self.spec)
        if (self.terrain.readonly_snapshot().observed_at>self.now+1e-10).any():raise ValueError("Actor cannot read future local terrain")
        if self.intent is not None and (type(self.intent) is not UpstreamIntent or self.intent.timestamp>self.now+1e-10):
            raise ValueError("Actor intent is not a current typed request")
        if type(self.feedback) is not PlannerFeedback or not isinstance(self.feedback.status,str) or not np.isfinite(self.feedback.blocked_seconds) or self.feedback.blocked_seconds<0:
            raise ValueError("Invalid public planner feedback")
        _status(self.feedback.status)
        object.__setattr__(self,'feedback',deepcopy(self.feedback))
        snapshot=self.snapshot
        if (snapshot.env_id,snapshot.receiver_id)!=(self.env_id,self.agent_id) or snapshot.timestamp is None or not math.isclose(snapshot.timestamp,self.now,abs_tol=1e-9):
            raise ValueError("Actor snapshot belongs to another receiver/time")
        ids=np.asarray(snapshot.sender_indices)
        if ids.shape!=(3,) or ids.dtype.kind not in 'iu' or len(snapshot.messages)>3 or len(snapshot.predictions)!=len(snapshot.messages):
            raise ValueError("Invalid Actor sender slots")
        changes={'sender_indices':_owned(ids,(3,),np.int64),'ages':_owned(snapshot.ages,(3,))}
        for name in ('state_valid','full_messages','goal_valid','trajectory_valid'):
            array=np.asarray(getattr(snapshot,name))
            if array.dtype!=bool:raise ValueError("Actor message masks must be boolean")
            changes[name]=_owned(array,(3,),bool)
        messages=[];seen=set()
        for slot,(message,prediction) in enumerate(zip(snapshot.messages,snapshot.predictions)):
            if (type(message) is not ReceivedMessage or message.version!=MESSAGE_V2
                    or ids[slot]!=message.sender_id or message.sender_id==self.agent_id or message.sender_id in seen
                    or prediction.sender_id!=message.sender_id or prediction.contract_error(self.spec,self.spec.fingerprint)
                    or message.episode!=self.episode or message.timestamp>self.now+1e-10 or message.received_at>self.now+1e-10):
                raise ValueError("Actor sender/version/time mismatch")
            seen.add(message.sender_id);age=self.now-message.timestamp
            valid=age<=self.spec.communication.state_validity+1e-10
            full=valid and message.full
            trajectory=full and message.trajectory is not None and self.now-message.trajectory.timestamp<=self.spec.communication.trajectory_validity+1e-10
            if (not math.isclose(age,snapshot.ages[slot],abs_tol=1e-9) or bool(snapshot.state_valid[slot])!=valid
                    or bool(snapshot.full_messages[slot])!=full or bool(snapshot.goal_valid[slot])!=(full and message.goal_xy is not None)
                    or bool(snapshot.trajectory_valid[slot])!=trajectory):
                raise ValueError("Actor message validity was changed after transport")
            messages.append(ReceivedMessage(**vars(message)))
        if ((ids[len(messages):]!=-1).any() or any(np.asarray(getattr(snapshot,name))[len(messages):].any()
                for name in ('state_valid','full_messages','goal_valid','trajectory_valid'))):
            raise ValueError("Missing Actor sender slots cannot contain payloads")
        changes['messages']=tuple(messages)
        object.__setattr__(self,'snapshot',replace(snapshot,**changes))


@dataclass(frozen=True)
class ActorObservation:
    values: np.ndarray
    terrain_known: np.ndarray
    env_id: int
    agent_id: int
    episode: int
    timestamp: float
    episode_seconds: float
    profile_fingerprint: str
    sender_indices: np.ndarray
    version: str = OBSERVATION_VERSION

    def __post_init__(self):
        dimension=523 if self.version==GRID_OBSERVATION_VERSION else ACTOR_DIM
        object.__setattr__(self,'values',_owned(self.values,(dimension,),np.float32))
        object.__setattr__(self,'terrain_known',_owned(self.terrain_known,(112,),bool))
        object.__setattr__(self,'sender_indices',_owned(self.sender_indices,(3,),np.int64))
        if self.version not in (OBSERVATION_VERSION,GRID_OBSERVATION_VERSION):raise ValueError("Unsupported Actor observation version")


class UpstreamObservationEncoder:
    def __init__(self,spec,*,spatial_scaling="linear",obstacle_input=None):
        if type(spec) is not UpstreamSpec:raise ValueError("Observation encoder requires a typed profile")
        if spatial_scaling not in ('linear', 'log1p'):raise ValueError('Unknown spatial normalization')
        self.spatial_scaling=spatial_scaling
        self.obstacle_input=obstacle_input
        self.channels=3 if obstacle_input is not None else 2
        self.actor_dim=523 if obstacle_input is not None else ACTOR_DIM
        self.critic_dim=1439 if obstacle_input is not None else CRITIC_DIM
        self.spec=spec;p=spec.perception;v=spec.vehicle
        length=math.ceil(v.length/2+v.clearance_margin);width=math.ceil(v.width/2+v.clearance_margin)
        patches=[(np.linspace(-length,length,7),np.linspace(-width,width,9)),
                 (np.linspace(-p.reverse/2,p.forward/2,3),np.linspace(-p.lateral/2,p.lateral/2,7)),
                 (np.linspace(-p.reverse,p.forward,4),np.linspace(-p.lateral,p.lateral,7))]
        self._points=np.vstack([np.stack(np.meshgrid(x,y,indexing='ij'),-1).reshape(-1,2) for x,y in patches])
        self._goal_points=np.stack(np.meshgrid(np.linspace(-spec.rho_max,spec.rho_max,5),
                                             np.linspace(-spec.rho_max,spec.rho_max,5),indexing='ij'),-1).reshape(-1,2)
        self._points.setflags(write=False);self._goal_points.setflags(write=False)
        def regions(x,y):
            xe=np.r_[x[0],(x[:-1]+x[1:])/2,x[-1]]
            ye=np.r_[y[0],(y[:-1]+y[1:])/2,y[-1]]
            return [(xe[i],xe[i+1],ye[j],ye[j+1]) for i in range(len(x)) for j in range(len(y))]
        self._regions=sum([regions(x,y) for x,y in patches],[])
        axis=np.linspace(-spec.rho_max,spec.rho_max,5)
        self._goal_regions=regions(axis,axis)

    @property
    def metadata(self):
        if self.obstacle_input is not None:
            return {'observation_version':GRID_OBSERVATION_VERSION,'profile_fingerprint':self.spec.fingerprint,
                    'actor_dim':self.actor_dim,'critic_dim':self.critic_dim,'terrain_channels':3,'terrain_points':112,
                    'spatial_scaling':self.spatial_scaling,'obstacle_input':self.obstacle_input.metadata,
                    'oracle_features':'nine_reserved_zeros','aggregation':'exact_cell_intersection_area_v1'}
        return {**({'spatial_scaling':self.spatial_scaling} if self.spatial_scaling!='linear' else {}),
                'observation_version':OBSERVATION_VERSION,'profile_fingerprint':self.spec.fingerprint,
                'actor_dim':ACTOR_DIM,'critic_dim':CRITIC_DIM,'terrain_channels':2,'terrain_points':112,
                'extra_dim':115,'task_clock_dim':1,'grid_version':'body_bidirectional_112_upstream_v1',
                'oracle_features':'nine_reserved_zeros','normalization_version':'physical_limits_upstream_v1',
                'terrain_risk_version':'max_traversability_slope_rms_v1'}

    def relative_position(self, value):
        if self.spatial_scaling == 'linear':
            return value / self.spec.communication.full_radius
        return np.sign(value) * np.log1p(np.abs(value) / self.spec.task.dmax) / np.log1p(self.spec.communication.full_radius / self.spec.task.dmax)

    def _terrain(self,item,points):
        if self.obstacle_input is not None:
            if type(item.terrain) is not GridTerrain:raise ValueError('Semantic encoder requires a private grid')
            regions=self._regions if points is self._points else self._goal_regions
            values=item.terrain.aggregate(item.ego_pose,regions,parking=points is self._goal_points)
            return values[:,0],values[:,1],values[:,2]
        coordinates=_rotate(points,item.ego_pose[2])+item.ego_pose[:2]
        features,known=item.terrain.lookup(coordinates)
        center,supported=item.terrain.lookup(item.ego_pose[:2])
        height=features[:,0]-(center[0,0] if supported[0] else 0.)
        # A finite traversability proxy alone does not certify a steep or
        # rough direct observation. Normalize the existing local attributes
        # against the same physical thresholds, without adding network slots.
        risk=np.maximum.reduce([1-features[:,4],
            (np.linalg.norm(features[:,1:3],axis=-1)/self.spec.terrain.driving_slope)**2,
            (features[:,3]/self.spec.terrain.roughness_rms)**2]).clip(0.,1.)
        return np.where(known,height/self.spec.vehicle.height,0.),np.where(known,risk,1.),known

    def actor(self,item):
        if type(item) is not ObservationInput or item.spec.fingerprint!=self.spec.fingerprint:raise ValueError("Actor input/profile mismatch")
        item.__post_init__();spec=self.spec;pose=item.ego_pose;v,w=item.actual_velocity
        intent=item.intent;valid=intent is not None and item.now<=intent.expires_at+1e-10
        delta=np.zeros(2) if intent is None else _rotate(intent.goal_xy-pose[:2],-pose[2])/spec.rho_max
        yaw=0. if intent is None or intent.terminal_yaw is None else wrap(intent.terminal_yaw-pose[2])/math.pi
        cap=0. if intent is None or intent.speed_cap is None else intent.speed_cap/spec.vehicle.forward_limit
        ego=np.array([0.,0.,0.,1.,0.,v/spec.vehicle.forward_limit,0.,w/spec.vehicle.yaw_limit,
                      abs(v)/spec.vehicle.forward_limit,abs(w)/spec.vehicle.yaw_limit,*delta,cap,yaw,
                      float(valid and intent.hold)])
        neighbors=np.zeros((3,17));extra=np.zeros(115);snapshot=item.snapshot;comm=spec.communication
        for slot,message in enumerate(snapshot.messages):
            if snapshot.state_valid[slot]:
                angle=wrap(message.pose[2]-pose[2]);row=neighbors[slot]
                row[:2]=self.relative_position(_rotate(message.pose[:2]-pose[:2],-pose[2]))
                row[4:6]=[math.cos(angle),math.sin(angle)]
                if snapshot.full_messages[slot]:
                    row[2:4]=_rotate([message.actual_velocity[0],0.],angle)/spec.vehicle.forward_limit
                    if message.terrain_features is not None:
                        feature=message.terrain_features.copy();feature[1:3]=_rotate(feature[1:3],-pose[2])
                        feature[0]=0. # Sender-local datum; absolute DEM height is not a body-state feature.
                        feature[1:3]/=spec.terrain.driving_slope;feature[3]/=spec.terrain.roughness_rms
                        row[6:11]=feature
                    row[11]=1.
                    if message.goal_xy is not None:row[12:14]=self.relative_position(_rotate(message.goal_xy-pose[:2],-pose[2]))
                    row[16]=float(message.hold)
            values=extra[85+10*slot:95+10*slot]
            age=snapshot.ages[slot]
            values[:4]=[min(2.,age/comm.state_validity),float(snapshot.state_valid[slot]),2.,max(0.,1-age/comm.state_validity)]
            if snapshot.goal_valid[slot]:values[4:6]=self.relative_position(_rotate(message.goal_xy-pose[:2],-pose[2]))
            if snapshot.trajectory_valid[slot]:
                for j,dt in enumerate((.4,1.)):
                    point=np.array([np.interp(item.now+dt,message.trajectory.times,message.trajectory.states[:,axis]) for axis in range(2)])
                    values[6+2*j:8+2*j]=self.relative_position(_rotate(point-pose[:2],-pose[2]))
        height,risk,known=self._terrain(item,self._points)
        grids=np.stack([height,risk,known],-1) if self.obstacle_input is not None else np.stack([height,risk],-1)
        visible=snapshot.state_valid;distances=np.linalg.norm(neighbors[:,:2],axis=-1)
        count=int(visible.sum());aggregation=np.zeros(5)
        if count:aggregation[:]=[count/3,distances[visible].mean(),distances[visible].max(),
                                  neighbors[visible,11].mean(),(snapshot.ages[visible]/comm.state_validity).mean()]
        gh,gr,gk=self._terrain(item,self._goal_points)
        remaining=0. if intent is None or not valid else np.clip((intent.expires_at-item.now)/max(intent.expires_at-intent.timestamp,1e-12),0.,1.)
        extra[:10]=[*delta,float(intent.hold) if intent is not None else 0.,float(valid),remaining,
                    min(2.,item.feedback.blocked_seconds/spec.task.hold_duration),_status(item.feedback.status),
                    v/spec.vehicle.forward_limit,w/spec.vehicle.yaw_limit,gk.mean()]
        extra[10:85]=np.stack([gh,gr,gk],-1).reshape(-1)
        clock=np.clip(1-item.now/item.episode_seconds,0.,1.)
        values=np.concatenate([ego,neighbors.reshape(-1),grids.reshape(-1),aggregation,extra,[clock]])
        return ActorObservation(values,known>=1-1e-8,item.env_id,item.agent_id,item.episode,item.now,item.episode_seconds,
                                spec.fingerprint,snapshot.sender_indices,
                                version=GRID_OBSERVATION_VERSION if self.obstacle_input is not None else OBSERVATION_VERSION)

    def critic(self,poses_world,actual_velocity,observations,hold_counts,times,episode_seconds):
        poses=np.asarray(poses_world,float);speed=np.asarray(actual_velocity,float)
        if poses.ndim!=3 or poses.shape[1:]!=(4,3) or speed.shape!=(*poses.shape[:2],2):raise ValueError("Critic requires actual [E,4,3]/[E,4,2] state")
        envs=len(poses);counts=np.asarray(hold_counts);now=_owned(times,(envs,));deadlines=np.broadcast_to(np.asarray(episode_seconds,float),(envs,))
        if (not np.isfinite(poses).all() or not np.isfinite(speed).all() or counts.shape!=(envs,) or counts.dtype.kind not in 'iu'
                or (counts<0).any() or (counts>round(self.spec.task.hold_duration/self.spec.grid.dt)).any()
                or not np.isfinite(deadlines).all() or (deadlines<=0).any() or (now<0).any()
                or len(observations)!=envs or any(len(row)!=4 for row in observations)):
            raise ValueError("Invalid centralized Critic facts")
        actor=np.empty((envs,4,self.actor_dim),np.float32)
        for e,row in enumerate(observations):
            for a,observation in enumerate(row):
                if (type(observation) is not ActorObservation or observation.profile_fingerprint!=self.spec.fingerprint
                        or (observation.env_id,observation.agent_id)!=(e,a) or not math.isclose(observation.timestamp,now[e],abs_tol=1e-9)
                        or not math.isclose(observation.episode_seconds,deadlines[e],abs_tol=1e-9)):
                    raise ValueError("Critic local observation identity/time mismatch")
                actor[e,a]=observation.values
        xy=poses[...,:2];yaw=poses[...,2];v=speed[...,0];w=speed[...,1]
        body=np.stack([xy[...,0]/(self.spec.world_width/2),xy[...,1]/(self.spec.world_height/2),np.zeros_like(v),
                       np.cos(yaw),np.sin(yaw),v*np.cos(yaw)/self.spec.vehicle.forward_limit,
                       v*np.sin(yaw)/self.spec.vehicle.forward_limit,w/self.spec.vehicle.yaw_limit],-1)
        pair=np.linalg.norm(xy[:,:,None]-xy[:,None],axis=-1);center=xy.mean(1);mask=np.triu(np.ones((4,4),bool),1)
        team=np.stack([pair.max((1,2))/self.spec.task.dmax,((xy-center[:,None])**2).sum(-1).mean(-1)/self.spec.task.dispersion,
                       center[:,0]/(self.spec.world_width/2),center[:,1]/(self.spec.world_height/2),np.zeros(envs),
                       pair[:,mask].mean(-1)/self.spec.task.dmax,abs(v).mean(-1)/self.spec.vehicle.forward_limit,
                       counts/(self.spec.task.hold_duration/self.spec.grid.dt)],-1)
        if self.spatial_scaling == 'log1p':
            team[:,[0,1,5]]=np.log1p(team[:,[0,1,5]])/np.log(2.)
        end=66+112*self.channels
        grids=actor[:,:,66:end].reshape(envs,4,112,self.channels)
        if self.obstacle_input is not None:
            mask=grids[...,2]>0;count=np.maximum(mask.sum((1,2)),1)
            obstacle=np.where(mask,grids[...,0],0);ground=np.where(mask,grids[...,1],0)
            terrain=np.stack([obstacle.sum((1,2))/count,obstacle.max((1,2)),ground.sum((1,2))/count,
                              ground.max((1,2)),grids[...,2].mean((1,2))],-1)
        else:
            height=grids[...,0];risk=grids[...,1]
            terrain=np.stack([abs(height).mean((1,2)),height.clip(min=0).max((1,2)),(-height).clip(min=0).max((1,2)),
                              risk.mean((1,2)),risk.max((1,2))],-1)
        # Only centralized Critic receives actual team facts/hold count. Oracle
        # dimensions stay explicit reserved zeros; no Oracle API is accepted.
        values=np.concatenate([body.reshape(envs,32),team,terrain,np.zeros((envs,9)),grids.reshape(envs,4*112*self.channels),
                               actor[:,:,end+5:end+15].reshape(envs,40),np.clip(1-now/deadlines,0.,1.)[:,None]],-1)
        return _owned(values,(envs,self.critic_dim),np.float32)

    def actor_tensor(self,inputs,*,device='cpu'):
        observations=[self.actor(item) for item in inputs]
        array=np.stack([o.values for o in observations]) if observations else np.empty((0,self.actor_dim),np.float32)
        return torch.tensor(array,dtype=torch.float32,device=device)
