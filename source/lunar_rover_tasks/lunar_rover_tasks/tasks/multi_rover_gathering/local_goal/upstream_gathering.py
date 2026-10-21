"""Legal leader square layout and latched APPROACH/BRAKING/HOLD for S03.

No Actor, task truth, global terrain, environment owner or Oracle is accepted.
The leader is one square corner; its published worksite center is not its pose.
Temporary communication/priority waits are distinct from parking capture.
"""
from copy import deepcopy
from dataclasses import dataclass
import math

import numpy as np

from .contracts import wrap
from .upstream_communication import ReceivedSnapshot, validate_actuator_state
from .upstream_execution import brake_sequence, control_error, rollout_numpy, terrain_gain_numpy
from .upstream_footprint import FootprintTerrain
from .upstream_review_cache import PoseCachedReviewer
from .upstream_planning import UpstreamIntent
from .upstream_spec import UpstreamSpec

SCRIPT_VERSION="leader_square_upstream_v1"
PHASES={"APPROACH","BRAKING","HOLD"}


def _array(value,shape):
    result=np.asarray(value,float).copy()
    if result.shape!=shape or not np.isfinite(result).all():raise ValueError("Invalid legal script input")
    result.setflags(write=False)
    return result


def _rotate(points,yaw):
    c,s=math.cos(yaw),math.sin(yaw)
    return np.asarray(points)@np.array([[c,s],[-s,c]])


@dataclass(frozen=True)
class GatheringDecision:
    intent: UpstreamIntent
    phase: str
    reason: str
    slot_xy: np.ndarray | None
    site_center: np.ndarray | None
    layout_yaw: float | None
    stopping_endpoint: np.ndarray | None
    stop_prediction_verified: bool
    temporary_wait: bool


class LeaderSquareScript:
    state_version="leader_square_upstream_state_v1"

    def __init__(self,spec,env_id,agent_id):
        if type(spec) is not UpstreamSpec or type(env_id) is not int or env_id<0 or type(agent_id) is not int or not 0<=agent_id<4:
            raise ValueError("Leader script requires an explicit profile and vehicle identity")
        self.spec,self.env_id,self.agent_id=spec,env_id,agent_id
        self.reviewer=PoseCachedReviewer(spec)
        self.reset()

    @property
    def metadata(self):
        return {"script_version":SCRIPT_VERSION,"state_version":self.state_version,
                "profile_fingerprint":self.spec.fingerprint,"env_id":self.env_id,"agent_id":self.agent_id}

    def reset(self):
        self._phase="APPROACH";self._site=None;self._yaw=None;self._hold_xy=None
        self._last_time=None;self._epoch=None;self._layout_source_time=None

    def _stopped(self,velocity):
        v,w=abs(velocity)
        return (v<=self.spec.task.linear_stop+1e-10 and w<=self.spec.task.angular_stop+1e-10
                and v+self.spec.vehicle.physical_radius*w<=self.spec.task.boundary_stop+1e-10)

    def _slots(self):return _rotate(self.spec.task.slots,self._yaw)+self._site

    def _priority_release_radius(self):
        return self.spec.task.capture_radius

    def _decision(self,pose,now,reason,*,wait=False,endpoint=None,verified=False):
        hold=wait or self._phase in {"BRAKING","HOLD"} or self.agent_id==0
        slot=None if self._site is None else self._slots()[self.agent_id]
        if hold:
            goal=pose[:2] if wait or self._hold_xy is None else self._hold_xy
        else:
            delta=slot-pose[:2];distance=np.linalg.norm(delta)
            # This is the original polar decode: rho is capped at 4 m while
            # beta is relative to the actual current heading. It is a rolling
            # target, not a stopping request.
            rho=min(distance,self.spec.rho_max)
            beta=wrap(math.atan2(delta[1],delta[0])-pose[2]) if distance>1e-12 else 0.
            goal=pose[:2]+rho*np.array([math.cos(pose[2]+beta),math.sin(pose[2]+beta)])
        return GatheringDecision(UpstreamIntent(goal,bool(hold),now,now+self.spec.grid.request_validity),
            self._phase,reason,None if slot is None else _array(slot,(2,)),
            None if self._site is None else _array(self._site,(2,)),self._yaw,
            None if endpoint is None else _array(endpoint,(3,)),bool(verified),bool(wait))

    def decide(self,pose,actual_velocity,previous_control,terrain,snapshot,now,episode):
        pose=_array(pose,(3,));velocity=_array(actual_velocity,(2,));previous=_array(previous_control,(2,))
        if (type(terrain) is not FootprintTerrain or terrain.profile_fingerprint!=self.spec.fingerprint
                or type(snapshot) is not ReceivedSnapshot or type(episode) is not int or episode<0
                or not math.isfinite(now) or now<0 or (self._last_time is not None and now<self._last_time-1e-10)):
            raise ValueError("Script accepts only current legal local inputs and monotone time")
        if control_error(previous,previous,self.spec):raise ValueError("Script previous control is illegal")
        validate_actuator_state(previous,velocity,self.spec)
        if (terrain.snapshot().observed_at>now+1e-10).any():raise ValueError("Script cannot read future terrain")
        if self._epoch is not None and episode!=self._epoch:raise ValueError("Script episode changed without reset")
        if len(snapshot.messages)>3 or len(snapshot.messages)!=len(snapshot.predictions):raise ValueError("Invalid legal sender snapshot")
        for index,message in enumerate(snapshot.messages):
            if (message.sender_id!=snapshot.sender_indices[index] or message.sender_id==self.agent_id
                    or snapshot.predictions[index].sender_id!=message.sender_id
                    or message.episode!=episode or message.timestamp>now+1e-10 or message.received_at>now+1e-10):
                raise ValueError("Script message identity/time mismatch")
        self._epoch=episode;self._last_time=now
        if self.agent_id==0:
            if self._site is None:
                self._yaw=pose[2];self._site=pose[:2]-_rotate(self.spec.task.slots[0],self._yaw)
                self._hold_xy=pose[:2].copy();self._layout_source_time=now
            self._phase="HOLD" if self._stopped(velocity) else "BRAKING"
            return self._decision(pose,now,"leader_frozen_corner")
        leader=None
        for slot,message in enumerate(snapshot.messages):
            valid=(message.timestamp<=now+1e-10 and now-message.timestamp<=self.spec.communication.state_validity+1e-10)
            if message.sender_id==0 and valid and message.full and snapshot.full_messages[slot] and message.site_center is not None:
                leader=message;break
        if leader is None:return self._decision(pose,now,"missing_or_stale_leader_layout",wait=True)
        if self._site is None:
            self._site=leader.site_center.copy();self._yaw=leader.layout_yaw;self._layout_source_time=leader.timestamp
        elif not np.allclose(self._site,leader.site_center,atol=1e-10,rtol=0) or abs(wrap(self._yaw-leader.layout_yaw))>1e-10:
            return self._decision(pose,now,"changed_frozen_layout",wait=True)
        goal=self._slots()[self.agent_id];error=np.linalg.norm(pose[:2]-goal)
        stopped=self._stopped(velocity)
        parking=self.reviewer.pose(pose,terrain,parking=True)
        # Once braking is latched, error noise cannot turn MOVE back on before
        # actual stopping. This also lets an expired/missing leader request
        # continue to stop through the explicit temporary HOLD path.
        if self._phase=="BRAKING":
            if not stopped:return self._decision(pose,now,"latched_braking")
            self._phase="HOLD"
        if self._phase=="HOLD":
            if stopped and error>self.spec.task.exit_radius+1e-10:
                self._phase="APPROACH";self._hold_xy=None
            else:return self._decision(pose,now,"captured_hold")
        if stopped and error<=self.spec.task.capture_radius+1e-10 and parking.passed:
            self._phase="HOLD";self._hold_xy=pose[:2].copy()
            return self._decision(pose,now,"actual_pose_capture",endpoint=pose,verified=True)
        brake=brake_sequence(previous,self.spec)
        commands=np.vstack([brake,np.zeros((max(1,self.spec.grid.steps-len(brake)),2))])
        def gain(state):
            feature,known=terrain.lookup(state[:2])
            return terrain_gain_numpy(state,feature[0],self.spec) if known[0] else .25
        prediction=rollout_numpy(pose,commands,self.spec,now,gain=gain)
        endpoint=prediction.states[-1]
        # Geometry here is local terrain only. The planner/execution admission
        # still checks all sender-bound predictions before issuing a control.
        stop_report=self.reviewer.trajectory(prediction,previous,terrain,require_stop=True,parking_end=True)
        if np.linalg.norm(endpoint[:2]-goal)<=self.spec.task.capture_radius+1e-10 and stop_report.passed:
            self._phase="BRAKING";self._hold_xy=pose[:2].copy()
            return self._decision(pose,now,"verified_stopping_endpoint_capture",endpoint=endpoint,verified=True)
        # Lower-number moving vehicles receive priority. Being stopped is only
        # a completion signal when the legal message is at that sender's slot;
        # waiting far away cannot form a permanent 'already gathered' claim.
        for slot,message in enumerate(snapshot.messages):
            if not 0<message.sender_id<self.agent_id:continue
            valid=now-message.timestamp<=self.spec.communication.state_validity+1e-10
            at_slot=np.linalg.norm(message.pose[:2]-self._slots()[message.sender_id])<=self._priority_release_radius()+1e-10
            if not(valid and message.full and message.hold and self._stopped(message.actual_velocity) and at_slot):
                return self._decision(pose,now,"lower_sender_priority",wait=True,endpoint=endpoint)
        missing=set(range(1,self.agent_id))-{m.sender_id for m in snapshot.messages}
        if missing:return self._decision(pose,now,"missing_priority_sender",wait=True,endpoint=endpoint)
        return self._decision(pose,now,"approach_slot",endpoint=endpoint)

    def state_dict(self):
        return deepcopy({"contracts":self.metadata,"phase":self._phase,"site_center":self._site,
            "layout_yaw":self._yaw,"hold_xy":self._hold_xy,"last_time":self._last_time,
            "episode":self._epoch,"layout_source_time":self._layout_source_time})

    def load_state_dict(self,state):
        if not isinstance(state,dict) or state.get('contracts')!=self.metadata or state.get('phase') not in PHASES:
            raise ValueError("Leader script snapshot identity/phase mismatch")
        site=state.get('site_center');yaw=state.get('layout_yaw');hold=state.get('hold_xy')
        now=state.get('last_time');epoch=state.get('episode');source=state.get('layout_source_time')
        if (site is None)!=(yaw is None) or (site is None)!=(source is None):raise ValueError("Incomplete frozen layout")
        if site is not None:
            site=_array(site,(2,))
            if not math.isfinite(yaw) or not math.isfinite(source) or source<0 or now is None or source>now:
                raise ValueError("Invalid frozen layout time/yaw")
        if hold is not None:hold=_array(hold,(2,))
        if (now is None)!=(epoch is None) or (now is not None and (not math.isfinite(now) or now<0 or type(epoch) is not int or epoch<0)):
            raise ValueError("Invalid restored script time/epoch")
        if state['phase'] in {'BRAKING','HOLD'} and hold is None:raise ValueError("Latched stop requires its frozen request pose")
        self._site=None if site is None else site.copy();self._yaw=yaw;self._hold_xy=None if hold is None else hold.copy()
        self._last_time=now;self._epoch=epoch;self._layout_source_time=source;self._phase=state['phase']
