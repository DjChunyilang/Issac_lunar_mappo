"""Observed queue history for S03; future predictions still use local terrain.

This history is actuator/odometry feedback, never a source of future peer truth
or terrain. The legacy S02 held-control timing path remains independently
versioned. An information-time reserve must check both the motion that already
occurred and a conservative stop from the current activation state.
"""

from dataclasses import dataclass
import math

import numpy as np

from .contracts import wrap
from .upstream_execution import control_error, midpoint_numpy, stop_envelope
from .upstream_spec import UpstreamSpec

HELD_TIMING_VERSION = "held_previous_control_v1"
HISTORY_TIMING_VERSION = "actual_control_history_v1"


def _owned(value, shape):
    result = np.asarray(value, dtype=float).copy()
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError("Actual history requires finite arrays on its explicit grid")
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class ActualControlHistory:
    env_id: int
    agent_id: int
    profile_fingerprint: str
    times: np.ndarray
    poses: np.ndarray
    controls: np.ndarray
    actual_velocities: np.ndarray
    initial_previous_control: np.ndarray
    version: str = HISTORY_TIMING_VERSION

    def __post_init__(self):
        if (type(self.env_id) is not int or self.env_id < 0 or type(self.agent_id) is not int
                or not 0 <= self.agent_id < 4 or self.version != HISTORY_TIMING_VERSION):
            raise ValueError("Actual history identity/version mismatch")
        times = np.asarray(self.times)
        if times.ndim != 1 or not len(times):
            raise ValueError("Actual history requires an explicit nonempty time grid")
        n = len(times)-1
        for name, shape in (("times", (n+1,)), ("poses", (n+1,3)), ("controls", (n,2)),
                            ("actual_velocities", (n,2)), ("initial_previous_control", (2,))):
            object.__setattr__(self,name,_owned(getattr(self,name),shape))
        if self.times[0] < 0 or not np.allclose(np.diff(self.times),.2,atol=1e-10,rtol=0):
            raise ValueError("Actual history grid must use increasing 0.2 s steps")

    def validate(self, spec, env_id, agent_id):
        self.__post_init__()
        if (type(spec) is not UpstreamSpec or self.profile_fingerprint != spec.fingerprint
                or (self.env_id,self.agent_id) != (env_id,agent_id)):
            raise ValueError("Actual history belongs to a different vehicle/profile")
        previous=self.initial_previous_control
        if control_error(previous,previous,spec):
            raise ValueError("Illegal information-time actuator state")
        for i,(control,velocity) in enumerate(zip(self.controls,self.actual_velocities)):
            if control_error(control,previous,spec):
                raise ValueError("Actual history violates actuator restrictions")
            moving=abs(control)>1e-10
            ratio=velocity[moving]/control[moving]
            if ((abs(velocity[~moving])>1e-8).any()
                    or (len(ratio) and ((ratio<.25-1e-7).any() or (ratio>1+1e-7).any()
                                         or np.ptp(ratio)>1e-7))):
                raise ValueError("Actual history velocity contradicts its applied actuator state")
            # Feedback is already the actual terrain-scaled velocity. It is
            # checked as observed motion, not rescaled a second time using an
            # older local snapshot.
            predicted,_=midpoint_numpy(self.poses[i],velocity,spec)
            observed=self.poses[i+1]
            angular_error=abs(wrap(predicted[2]-observed[2]))
            angular_tolerance=1e-10+1e-8*abs(wrap(predicted[2]))
            if (not np.allclose(predicted[:2],observed[:2],atol=1e-10,rtol=1e-8)
                    or angular_error>angular_tolerance):
                raise ValueError("Actual history odometry contradicts observed midpoint motion")
            previous=control

    @property
    def previous_at_activation(self):
        return self.controls[-1] if len(self.controls) else self.initial_previous_control


def review_actual_history(history, item, reviewer):
    """Check known swept footprints and peer envelopes at observed past times.

    The motion model was validated from actual feedback. The same spatial and
    relative-displacement sampling bounds as the future full-footprint reviewer
    are used here, without inventing a terrain gain for the past.
    """
    poses=0
    workspace=None if item.workspace is None else np.asarray(item.workspace).reshape(2,2)
    for i in range(len(history.controls)):
        delta=history.poses[i+1]-history.poses[i];delta[2]=wrap(delta[2])
        displacement=np.linalg.norm(delta[:2])+reviewer.sweep_radius*abs(delta[2])
        relative=max((np.linalg.norm(peer.at(history.times[i+1])[0]-peer.at(history.times[i])[0])
                      +abs(peer.at(history.times[i+1])[1]-peer.at(history.times[i])[1])
                      for peer in item.neighbors),default=0.)
        intervals=max(8,math.ceil((displacement+relative)/item.spec.vehicle.sweep_spacing))
        allowance=(np.linalg.norm(delta[:2])+relative)/(2*intervals)
        for alpha in np.linspace(0,1,intervals+1):
            pose=history.poses[i]+alpha*delta;poses+=1
            report=reviewer.pose(pose,item.terrain)
            if not report.passed:return report.reason,poses
            if workspace is not None and ((pose[:2]-item.spec.vehicle.effective_radius<workspace[0]).any()
                    or (pose[:2]+item.spec.vehicle.effective_radius>workspace[1]).any()):
                return "workspace_clearance",poses
            when=history.times[i]+alpha*item.spec.grid.dt
            for peer in item.neighbors:
                center,radius=peer.at(when)
                if np.linalg.norm(pose[:2]-center)<radius+allowance-1e-8:
                    return "neighbor_swept_clearance",poses
    return "",poses


def review_history_reserve(item, first_control, reviewer, gain):
    history=item.actual_history
    detail={"information_time":history.times[0],"activation_time":item.now,
            "delay_steps":len(history.controls),"timing_version":HISTORY_TIMING_VERSION,
            "information_reserve_verified":False,
            "delay_control_changes":int(np.count_nonzero(np.any(abs(np.diff(
                np.vstack([history.initial_previous_control,history.controls]),axis=0))>1e-10,axis=1)))}
    future=stop_envelope(item.ego_pose,item.previous_control,first_control,item.spec,
                         information_time=item.now,latency=0.,gain=gain)
    detail["reserve_steps"]=len(history.controls)+len(future.trajectory.controls)
    detail["reserve_center_distance"]=float(np.linalg.norm(np.diff(history.poses[:,:2],axis=0),axis=1).sum()
                                            +future.conservative_center_distance)
    if detail["reserve_steps"]>item.spec.grid.steps:
        return "stop_reserve_exceeds_horizon",detail
    past_reason,poses=review_actual_history(history,item,reviewer)
    detail["history_checked_poses"]=poses
    if past_reason:return "actual_history:"+past_reason,detail
    workspace=None if item.workspace is None else np.asarray(item.workspace).reshape(2,2)
    report=reviewer.trajectory(future.trajectory,item.previous_control,item.terrain,neighbors=item.neighbors,
                              workspace=workspace,require_stop=True,conservative_reserve=True)
    detail["information_reserve_verified"]=report.passed
    return ("" if report.passed else "stop_reserve:"+report.reason),detail
