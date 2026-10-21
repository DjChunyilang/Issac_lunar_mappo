"""S03 legal distance transport, sender slots and bounded peer predictions."""

from copy import deepcopy
from dataclasses import dataclass
import math

import numpy as np

from .contracts import TimedTrajectory, wrap
from .upstream_execution import control_error, limit_control_numpy
from .upstream_footprint import FootprintNeighbor
from .upstream_sensing import world_to_odometry
from .upstream_spec import UpstreamSpec, digest

MESSAGE_VERSION = "sender_message_upstream_v1"
MESSAGE_V2 = "sender_message_upstream_v2"
COMMUNICATION_VERSION = "distance_cache_upstream_v1"
BOUND_VERSION = "enforced_motion_bound_upstream_v1"


def owned(value, shape):
    result = np.asarray(value, float).copy()
    if result.shape != shape or not np.isfinite(result).all():raise ValueError("Invalid communication array")
    result.setflags(write=False)
    return result


def validate_actuator_state(control, velocity, spec):
    """Check the same observed actuator/gain contract on delivery and restore."""
    if control_error(control, control, spec):
        raise ValueError("Illegal sender actuator state")
    moving = abs(control) > 1e-10
    ratios = velocity[moving] / control[moving]
    if (((~moving) & (abs(velocity) > 1e-8)).any()
            or (len(ratios) and ((ratios < .25-1e-7).any()
                or (ratios > 1+1e-7).any() or np.ptp(ratios) > 1e-7))):
        raise ValueError("Sender actual velocity contradicts applied control")


@dataclass(frozen=True)
class MotionBound:
    env_id: int
    sender_id: int
    episode: int
    speed: float
    starts_at: float
    expires_at: float
    profile_fingerprint: str
    version: str = BOUND_VERSION

    def __post_init__(self):
        if any(type(x) is not int or x < 0 for x in (self.env_id, self.sender_id, self.episode)) or self.sender_id >= 4:
            raise ValueError("Invalid motion-bound identity")
        if (not np.isfinite([self.speed, self.starts_at, self.expires_at]).all() or self.speed < 0
                or self.starts_at < 0 or self.expires_at < self.starts_at or self.version != BOUND_VERSION):
            raise ValueError("Invalid motion-bound interval/value")

    @property
    def token(self):return digest(vars(self))


class MotionBoundLedger:
    """Only an execution owner enforcing every active pledge may issue bounds.

    A newer higher bound never cancels an older unexpired lower bound. The
    environment must use limit() for actual commands while certificates exist.
    """
    def __init__(self, spec, num_envs):
        if type(spec) is not UpstreamSpec or type(num_envs) is not int or num_envs < 1:
            raise ValueError("Motion ledger needs a typed specification and positive environment count")
        self.spec, self.num_envs = spec, num_envs
        self.episodes = np.zeros(num_envs, dtype=np.int64)
        self._bounds = {}

    def pledge(self, env_id, sender_id, speed, now, expires_at, previous_control):
        if type(env_id) is not int or not 0 <= env_id < self.num_envs:raise ValueError("Motion-bound environment outside ledger")
        bound = MotionBound(env_id, sender_id, int(self.episodes[env_id]), float(speed), float(now), float(expires_at), self.spec.fingerprint)
        previous = owned(previous_control, (2,))
        if bound.speed > self.spec.communication.absolute_velocity_bound or abs(previous[0]) > bound.speed + 1e-10 or control_error(previous, previous, self.spec):
            raise ValueError("Cannot promise a lower bound than the actual actuator state")
        self._bounds = {key: value for key, value in self._bounds.items()
                        if value.env_id != env_id or value.expires_at >= now}
        self._bounds[bound.token] = bound
        return bound

    def verified(self, bound):
        return (type(bound) is MotionBound and 0 <= bound.env_id < self.num_envs
                and self._bounds.get(bound.token) == bound
                and bound.profile_fingerprint == self.spec.fingerprint
                and bound.episode == self.episodes[bound.env_id])

    def limit(self, env_id, sender_id, desired, previous, now):
        if (type(env_id) is not int or type(sender_id) is not int or not 0 <= env_id < self.num_envs
                or not 0 <= sender_id < 4 or not math.isfinite(now) or now < 0):
            raise ValueError("Invalid bound-enforced execution identity/time")
        cap=self.speed_cap(env_id,sender_id,now)
        previous = owned(previous, (2,));desired = owned(desired, (2,)).copy()
        if abs(previous[0]) > cap + 1e-10:raise ValueError("Existing motion promise was violated")
        desired[0] = np.clip(desired[0], -min(cap, self.spec.vehicle.reverse_limit), min(cap, self.spec.vehicle.forward_limit))
        applied = limit_control_numpy(desired, previous, self.spec)
        if abs(applied[0]) > cap + 1e-10:applied = limit_control_numpy([0., 0.], previous, self.spec)
        if abs(applied[0]) > cap + 1e-10:raise ValueError("Cannot enforce the active motion promise")
        return applied

    def speed_cap(self,env_id,sender_id,now):
        if (type(env_id) is not int or type(sender_id) is not int or not 0<=env_id<self.num_envs
                or not 0<=sender_id<4 or not math.isfinite(now) or now<0):raise ValueError('Invalid execution bound identity/time')
        active = [b.speed for b in self._bounds.values() if b.env_id == env_id and b.sender_id == sender_id
                  and b.episode == self.episodes[env_id] and b.starts_at <= now <= b.expires_at]
        return min(active, default=self.spec.communication.absolute_velocity_bound)

    def reset(self, env_id):
        if type(env_id) is not int or not 0 <= env_id < self.num_envs:raise ValueError("Invalid bound ledger reset")
        self.episodes[env_id] += 1
        self._bounds = {key: b for key, b in self._bounds.items() if b.env_id != env_id}

    def state_dict(self):
        return deepcopy({"version": BOUND_VERSION, "profile_fingerprint": self.spec.fingerprint,
                         "num_envs": self.num_envs, "episodes": self.episodes, "bounds": list(self._bounds.values())})

    def load_state_dict(self, state):
        if not isinstance(state, dict) or state.get("version") != BOUND_VERSION or state.get("profile_fingerprint") != self.spec.fingerprint or state.get("num_envs") != self.num_envs:
            raise ValueError("Motion ledger snapshot mismatch")
        episodes = np.asarray(state.get("episodes"))
        if episodes.shape != (self.num_envs,) or episodes.dtype.kind not in "iu" or (episodes < 0).any():raise ValueError("Invalid bound epochs")
        bounds = {}
        for bound in state.get("bounds", ()):
            if (type(bound) is not MotionBound or bound.profile_fingerprint != self.spec.fingerprint
                    or not 0 <= bound.env_id < self.num_envs or bound.episode != episodes[bound.env_id]
                    or bound.speed > self.spec.communication.absolute_velocity_bound or bound.token in bounds):
                raise ValueError("Invalid restored motion certificate")
            bounds[bound.token] = bound
        self.episodes, self._bounds = episodes.copy(), deepcopy(bounds)


@dataclass(frozen=True)
class RoverBroadcast:
    sender_id: int
    episode: int
    timestamp: float
    pose: np.ndarray
    actual_velocity: np.ndarray
    applied_control: np.ndarray
    hold: bool = False
    goal_xy: np.ndarray | None = None
    site_center: np.ndarray | None = None
    layout_yaw: float | None = None
    trajectory: TimedTrajectory | None = None
    motion_bound: MotionBound | None = None
    version: str = MESSAGE_VERSION
    terrain_features: np.ndarray | None = None

    def __post_init__(self):
        if type(self.sender_id) is not int or not 0 <= self.sender_id < 4 or type(self.episode) is not int or self.episode < 0:
            raise ValueError("Invalid broadcast sender/epoch")
        if self.version not in {MESSAGE_VERSION,MESSAGE_V2} or not math.isfinite(self.timestamp) or self.timestamp < 0 or type(self.hold) is not bool:
            raise ValueError("Invalid broadcast version/time/HOLD")
        if self.terrain_features is not None:
            if self.version != MESSAGE_V2:raise ValueError("Terrain payload requires explicit sender_message_upstream_v2")
            features=owned(self.terrain_features,(5,))
            if features[3]<0 or not 0<=features[4]<=1:raise ValueError("Invalid sender-local terrain attributes")
            object.__setattr__(self,'terrain_features',features)
        for name, shape in (("pose", (3,)), ("actual_velocity", (2,)), ("applied_control", (2,))):
            object.__setattr__(self, name, owned(getattr(self, name), shape))
        for name in ("goal_xy", "site_center"):
            if getattr(self, name) is not None:object.__setattr__(self, name, owned(getattr(self, name), (2,)))
        if self.layout_yaw is not None and not math.isfinite(self.layout_yaw):raise ValueError("Invalid layout yaw")
        if (self.site_center is None) != (self.layout_yaw is None):raise ValueError("Site center and yaw must be paired")
        if self.motion_bound is not None and (type(self.motion_bound) is not MotionBound or self.motion_bound.sender_id != self.sender_id or self.motion_bound.episode != self.episode):
            raise ValueError("Broadcast motion bound belongs to another sender")
        if self.trajectory is not None:
            trajectory = deepcopy(self.trajectory)
            if type(trajectory) is not TimedTrajectory:raise ValueError("Expected an execution trajectory")
            for name, shape in (("states", (41,3)), ("controls", (40,2)), ("velocities", (40,2)), ("times", (41,))):
                setattr(trajectory, name, owned(getattr(trajectory, name), shape))
            if (not math.isfinite(trajectory.timestamp) or trajectory.timestamp < 0
                    or not np.allclose(trajectory.times, trajectory.timestamp + np.arange(41)*.2, atol=1e-10, rtol=0)):
                raise ValueError("Invalid broadcast execution grid")
            index = round((self.timestamp - trajectory.timestamp)/.2)
            if not 0 <= index <= 40 or not math.isclose(self.timestamp, trajectory.times[index], abs_tol=1e-9):
                raise ValueError("Broadcast timestamp outside execution prediction")
            if not np.allclose(trajectory.states[index], self.pose, atol=1e-7, rtol=1e-6):
                raise ValueError("Broadcast pose differs from execution prediction")
            if index and not np.allclose(trajectory.velocities[index-1], self.actual_velocity, atol=1e-7, rtol=1e-6):
                raise ValueError("Broadcast actual velocity differs from execution prediction")
            object.__setattr__(self, "trajectory", trajectory)


@dataclass(frozen=True)
class ReceivedMessage:
    sender_id: int
    episode: int
    timestamp: float
    received_at: float
    full: bool
    pose: np.ndarray
    actual_velocity: np.ndarray
    applied_control: np.ndarray
    hold: bool
    goal_xy: np.ndarray | None
    site_center: np.ndarray | None
    layout_yaw: float | None
    trajectory: TimedTrajectory | None
    motion_bound: MotionBound | None
    version: str = MESSAGE_VERSION
    terrain_features: np.ndarray | None = None

    def __post_init__(self):
        if type(self.full) is not bool or not math.isfinite(self.received_at) or self.received_at < self.timestamp:
            raise ValueError("Invalid packet delivery time/type")
        if not self.full and (self.hold or any(x is not None for x in (self.goal_xy,self.site_center,self.layout_yaw,self.trajectory,self.motion_bound,self.terrain_features))):
            raise ValueError("Sparse packet cannot carry FULL-only intent payloads")
        validated = RoverBroadcast(self.sender_id,self.episode,self.timestamp,self.pose,self.actual_velocity,
            self.applied_control,self.hold,self.goal_xy,self.site_center,self.layout_yaw,self.trajectory,self.motion_bound,
            version=self.version,terrain_features=self.terrain_features)
        for name in ('pose','actual_velocity','applied_control','goal_xy','site_center','trajectory','terrain_features'):
            object.__setattr__(self,name,getattr(validated,name))


@dataclass(frozen=True)
class ReceivedSnapshot:
    sender_indices: np.ndarray
    state_valid: np.ndarray
    full_messages: np.ndarray
    ages: np.ndarray
    goal_valid: np.ndarray
    trajectory_valid: np.ndarray
    messages: tuple
    predictions: tuple
    prediction_metadata: tuple
    env_id: int | None = None
    receiver_id: int | None = None
    timestamp: float | None = None


class DistanceCommunicationCache:
    """Transport truth is used only to route packets, never to fill snapshots."""
    def __init__(self, spec, odometry_frames, *, ledger=None, message_version=MESSAGE_VERSION):
        frames = np.asarray(odometry_frames)
        if type(spec) is not UpstreamSpec or frames.ndim != 3 or frames.shape[1:] != (4,3) or len(frames) < 1:
            raise ValueError("Communication needs a typed specification and [E,4,3] fixed frames")
        self.spec, self.num_envs = spec, len(frames)
        if message_version not in {MESSAGE_VERSION,MESSAGE_V2}:raise ValueError("Unsupported communication message version")
        self._message_version=message_version
        self._frames = owned(frames, frames.shape)
        if ledger is not None and (ledger.spec.fingerprint != spec.fingerprint or ledger.num_envs != len(frames)):
            raise ValueError("Communication and execution ledger disagree")
        self._ledger = ledger
        self.episodes = np.zeros(len(frames), dtype=np.int64)
        self._messages = {}
        self._last_sent = np.full((len(frames),4,4), -np.inf)
        self._clock = np.zeros(len(frames))

    @property
    def metadata(self):
        metadata={"version": COMMUNICATION_VERSION, "profile_fingerprint": self.spec.fingerprint,
                "num_envs": self.num_envs, "verified_motion_ledger": self._ledger is not None}
        if self._message_version == MESSAGE_V2:
            metadata.update(version='distance_cache_upstream_v2',message_version=MESSAGE_V2,
                            slot_order_version='distance_1e-8_sender_v2')
        return metadata

    def _transform(self, env, receiver, sender, points):
        source, target = self._frames[env,sender], self._frames[env,receiver]
        c,s = np.cos(source[2]),np.sin(source[2])
        world = np.asarray(points) @ np.array([[c,s],[-s,c]]) + source[:2]
        return world_to_odometry(world, target)

    def transmit(self, broadcasts, routing_positions_world, now):
        position = owned(routing_positions_world, (self.num_envs,4,2));now = owned(now, (self.num_envs,))
        if (now < self._clock - 1e-10).any():raise ValueError("Communication time moved backwards")
        if len(broadcasts) != self.num_envs or any(len(row)!=4 for row in broadcasts):raise ValueError("Broadcast batch shape mismatch")
        for env,row in enumerate(broadcasts):
            for sender,packet in enumerate(row):
                if packet is None:continue
                if type(packet) is not RoverBroadcast or packet.sender_id != sender or packet.timestamp > now[env]+1e-10 or packet.version!=self._message_version:
                    raise ValueError("Invalid queued broadcast identity/time")
                validate_actuator_state(packet.applied_control,packet.actual_velocity,self.spec)
                bound=packet.motion_bound
                if bound is not None and bound.env_id != env:raise ValueError("Motion bound belongs to another environment")
                if self._ledger is not None and bound is not None and self._ledger.verified(bound) and abs(packet.applied_control[0])>bound.speed+1e-10:
                    raise ValueError("Certified sender violated its bound")
        messages, sent = dict(self._messages), self._last_sent.copy()
        count=0
        for env,row in enumerate(broadcasts):
            for sender,packet in enumerate(row):
                if packet is None or packet.episode != self.episodes[env]:continue
                for receiver in range(4):
                    if receiver==sender:continue
                    distance=np.linalg.norm(position[env,receiver]-position[env,sender])
                    if distance>self.spec.communication.sparse_radius:continue
                    full=bool(distance<=self.spec.communication.full_radius)
                    period=self.spec.communication.full_period if full else self.spec.communication.sparse_period
                    old=messages.get((env,receiver,sender))
                    if now[env]-sent[env,receiver,sender]<period-1e-10 or (old is not None and packet.timestamp<=old.timestamp):continue
                    offset=self._frames[env,sender,2]-self._frames[env,receiver,2]
                    pose=np.r_[self._transform(env,receiver,sender,packet.pose[:2][None])[0],wrap(packet.pose[2]+offset)]
                    trajectory=None
                    terrain_features=None
                    if full and packet.terrain_features is not None:
                        terrain_features=packet.terrain_features.copy();c,s=math.cos(offset),math.sin(offset)
                        terrain_features[1:3]=np.array([[c,-s],[s,c]])@terrain_features[1:3]
                    if full and packet.trajectory is not None:
                        trajectory=deepcopy(packet.trajectory)
                        states=trajectory.states.copy();states[:,:2]=self._transform(env,receiver,sender,states[:,:2]);states[:,2]=wrap(states[:,2]+offset)
                        trajectory.states=owned(states,(41,3))
                    message=ReceivedMessage(sender,packet.episode,packet.timestamp,float(now[env]),full,owned(pose,(3,)),
                        packet.actual_velocity,packet.applied_control,packet.hold if full else False,
                        owned(self._transform(env,receiver,sender,packet.goal_xy[None])[0],(2,)) if full and packet.goal_xy is not None else None,
                        owned(self._transform(env,receiver,sender,packet.site_center[None])[0],(2,)) if full and packet.site_center is not None else None,
                        wrap(packet.layout_yaw+offset) if full and packet.layout_yaw is not None else None,trajectory,
                        packet.motion_bound if full else None,version=packet.version,terrain_features=terrain_features)
                    messages[env,receiver,sender]=message;sent[env,receiver,sender]=now[env];count+=1
        self._messages,self._last_sent,self._clock=messages,sent,now.copy()
        return count

    def snapshot(self, env_id, receiver_id, now, receiver_pose, *, prediction_times=None):
        if type(env_id) is not int or not 0<=env_id<self.num_envs or type(receiver_id) is not int or not 0<=receiver_id<4 or not math.isfinite(now) or now<0:
            raise ValueError("Invalid snapshot identity/time")
        receiver_pose=owned(receiver_pose,(3,))
        times=np.asarray(prediction_times if prediction_times is not None else now+np.arange(41)*.2,float)
        if times.ndim!=1 or len(times)<2 or not np.isfinite(times).all() or (np.diff(times)<=0).any() or times[0]<now-1e-10:
            raise ValueError("Invalid peer prediction time grid")
        messages=[m for (e,r,_),m in self._messages.items() if e==env_id and r==receiver_id]
        if self._message_version == MESSAGE_V2:
            # Frame transforms can perturb mathematically equal distances by
            # a few ulps. This affects slot identity, never hard geometry/radio
            # thresholds. The explicit v2 ordering uses 10 nm distance bins.
            messages.sort(key=lambda m:(round(float(np.linalg.norm(m.pose[:2]-receiver_pose[:2]))/1e-8),m.sender_id))
        else:
            messages.sort(key=lambda m:(np.linalg.norm(m.pose[:2]-receiver_pose[:2]),m.sender_id))
        ids=np.full(3,-1,dtype=np.int64);state=np.zeros(3,bool);full=np.zeros(3,bool);ages=np.zeros(3)
        goals=np.zeros(3,bool);trajectories=np.zeros(3,bool);predictions=[];details=[]
        for slot,message in enumerate(messages):
            if message.timestamp>now+1e-10 or message.received_at>now+1e-10:
                raise ValueError("Snapshot cannot read future packets or deliveries")
            ids[slot]=message.sender_id;ages[slot]=now-message.timestamp
            state[slot]=ages[slot]<=self.spec.communication.state_validity+1e-10
            full[slot]=state[slot] and message.full
            goals[slot]=full[slot] and message.goal_xy is not None
            trajectories[slot]=(full[slot] and message.trajectory is not None
                                and now-message.trajectory.timestamp<=self.spec.communication.trajectory_validity+1e-10)
            bound=self.spec.communication.absolute_velocity_bound;source='global_hard_limit'
            certificate=message.motion_bound
            if (certificate is not None and self._ledger is not None and self._ledger.verified(certificate)
                    and certificate.starts_at<=message.timestamp and certificate.expires_at>=times[-1]):
                bound=certificate.speed;source='execution_enforced_certificate'
            centers=np.tile(message.pose[:2],(len(times),1))
            if trajectories[slot]:
                trajectory=message.trajectory
                centers=np.column_stack([np.interp(times,trajectory.times,trajectory.states[:,axis]) for axis in range(2)])
            # The base reachable ball is centered at the last legal state.
            # Moving that center to a noncommitted nominal trajectory adds its
            # displacement once; age is never inflated again downstream.
            radii=2*self.spec.vehicle.effective_radius+bound*(times-message.timestamp)
            radii+=np.linalg.norm(centers-message.pose[:2],axis=-1)
            predictions.append(FootprintNeighbor.for_spec(message.sender_id,times,centers,radii,self.spec))
            v,w=abs(message.actual_velocity)
            stop=bool(full[slot] and message.hold and v<=self.spec.task.linear_stop and w<=self.spec.task.angular_stop
                      and v+self.spec.vehicle.physical_radius*w<=self.spec.task.boundary_stop)
            details.append({'sender_id':message.sender_id,'state_valid':bool(state[slot]),'state_age':ages[slot],
                            'trajectory_used':bool(trajectories[slot]),'speed_bound':bound,'bound_source':source,'stop_intent':stop})
        for array in (ids,state,full,ages,goals,trajectories):array.setflags(write=False)
        return ReceivedSnapshot(ids,state,full,ages,goals,trajectories,tuple(messages),tuple(predictions),tuple(details),
                                env_id=env_id,receiver_id=receiver_id,timestamp=float(now))

    def reset(self, env_id, odometry_frames=None):
        if type(env_id) is not int or not 0<=env_id<self.num_envs:raise ValueError("Invalid communication reset")
        frames=self._frames.copy()
        if odometry_frames is not None:frames[env_id]=owned(odometry_frames,(4,3))
        self._frames=owned(frames,frames.shape);self.episodes[env_id]+=1
        self._messages={key:value for key,value in self._messages.items() if key[0]!=env_id}
        self._last_sent[env_id]=-np.inf
        # Restored clocks are validated owned arrays; the live cache must
        # retain a writable clock for asynchronous episode reset.
        self._clock=self._clock.copy();self._clock[env_id]=0.

    def state_dict(self):
        return deepcopy({'contracts':self.metadata,'frames':self._frames,'episodes':self.episodes,
                         'messages':self._messages,'last_sent':self._last_sent,'clock':self._clock})

    def load_state_dict(self,state):
        if not isinstance(state,dict) or state.get('contracts')!=self.metadata:raise ValueError("Communication state identity mismatch")
        frames=owned(state.get('frames'),(self.num_envs,4,3));clock=owned(state.get('clock'),(self.num_envs,))
        episodes=np.asarray(state.get('episodes'));sent=np.asarray(state.get('last_sent'),float)
        if episodes.shape!=(self.num_envs,) or episodes.dtype.kind not in 'iu' or (episodes<0).any() or (clock<0).any():raise ValueError("Invalid communication epochs/clock")
        if sent.shape!=(self.num_envs,4,4) or np.isnan(sent).any() or np.isposinf(sent).any() or (sent>clock[:,None,None]+1e-10).any():raise ValueError("Invalid communication delivery times")
        messages=deepcopy(state.get('messages'))
        if not isinstance(messages,dict):raise ValueError("Invalid restored messages")
        for key,message in messages.items():
            if (type(key) is not tuple or len(key)!=3 or any(type(x) is not int for x in key)
                    or not 0<=key[0]<self.num_envs or not 0<=key[1]<4 or not 0<=key[2]<4 or key[1]==key[2]
                    or type(message) is not ReceivedMessage or message.sender_id!=key[2] or message.version!=self._message_version
                    or message.episode!=episodes[key[0]] or message.timestamp<0 or message.timestamp>message.received_at
                    or message.received_at>clock[key[0]]+1e-10 or message.received_at!=sent[key]):
                raise ValueError("Invalid restored sender packet")
            owned(message.pose,(3,));owned(message.actual_velocity,(2,));owned(message.applied_control,(2,))
            restored = ReceivedMessage(**vars(message))
            validate_actuator_state(restored.applied_control,restored.actual_velocity,self.spec)
            if restored.motion_bound is not None and restored.motion_bound.env_id != key[0]:
                raise ValueError("Restored motion bound belongs to another environment")
            messages[key] = restored
        self._frames,self._clock,self.episodes,self._last_sent,self._messages=frames,clock,episodes.copy(),sent.copy(),messages
