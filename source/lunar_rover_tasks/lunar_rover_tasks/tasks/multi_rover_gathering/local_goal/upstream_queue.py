"""Absolute-expiry intent queue with immutable information-time inputs."""
from copy import deepcopy
from dataclasses import dataclass,replace
import math

import numpy as np

from .upstream_footprint import FootprintTerrain
from .upstream_grid import GridTerrain
from .upstream_planning import UpstreamIntent,UpstreamPlannerInput
from .upstream_timing import ActualControlHistory

QUEUE_VERSION='absolute_expiry_queue_upstream_v1'
ACCEPTED_QUEUE_VERSION='accepted_request_lease_upstream_v2'


@dataclass
class PendingPlan:
    env_id: int
    agent_id: int
    episode: int
    information_time: float
    information_pose: np.ndarray
    information_control: np.ndarray
    information_velocity: np.ndarray
    intent: UpstreamIntent
    terrain: FootprintTerrain
    neighbors: tuple
    ready_time: float
    feedback: list
    primitive_id: int | None = None

    @classmethod
    def capture(cls,cfg,env_id,agent_id,episode,pose,velocity,previous,intent,terrain,neighbors,now,*,primitive_id=None):
        if cfg.planner_adapter=='primitive47_upstream_v1':
            if type(primitive_id) is not int or not 0<=primitive_id<47:raise ValueError('Native queue requires an explicit primitive ID')
        elif primitive_id is not None:raise ValueError('Primitive ID supplied to a continuous planner')
        if (type(intent) is not UpstreamIntent or not math.isclose(intent.timestamp,now,abs_tol=1e-9)
                or not math.isclose(intent.expires_at,now+cfg.spec.grid.request_validity,abs_tol=1e-9)):
            raise ValueError('Queued request must retain its original issue time and configured absolute expiry')
        ready=now+math.ceil(cfg.simulated_latency/cfg.spec.grid.dt-1e-12)*cfg.spec.grid.dt
        history=ActualControlHistory(env_id,agent_id,cfg.spec.fingerprint,[now],np.asarray(pose)[None],
                                     np.zeros((0,2)),np.zeros((0,2)),previous)
        history.validate(cfg.spec,env_id,agent_id)
        detached=terrain if type(terrain) is GridTerrain else terrain.with_observation_times(terrain._timestamps)
        # Use the protocol validator before accepting any queued state.
        checked=UpstreamPlannerInput(env_id,agent_id,pose,velocity,intent,detached,neighbors,previous,now,
                             cfg.spec.grid.times(now),None,cfg.spec)
        return cls(env_id,agent_id,episode,now,history.poses[0],history.initial_previous_control,checked.ego_velocity,
                   deepcopy(intent),detached,deepcopy(tuple(neighbors)),ready,[],primitive_id)

    def append(self,spec,pose,control,velocity,now):
        expected=self.information_time+(len(self.feedback)+1)*spec.grid.dt
        if not math.isclose(now,expected,abs_tol=1e-8) or now>self.ready_time+1e-9:
            raise ValueError('Queue feedback must advance one actual step without passing activation')
        data={'pose':np.asarray(pose,float).copy(),'control':np.asarray(control,float).copy(),
              'velocity':np.asarray(velocity,float).copy(),'time':float(now)}
        trial=self.feedback+[data]
        self.history(spec,feedback=trial).validate(spec,self.env_id,self.agent_id)
        self.feedback=trial

    def history(self,spec,*,feedback=None):
        rows=self.feedback if feedback is None else feedback
        return ActualControlHistory(self.env_id,self.agent_id,spec.fingerprint,
            self.information_time+np.arange(len(rows)+1)*spec.grid.dt,
            np.asarray([self.information_pose,*[r['pose'] for r in rows]]),
            np.asarray([r['control'] for r in rows]).reshape(-1,2),
            np.asarray([r['velocity'] for r in rows]).reshape(-1,2),self.information_control)

    def input(self,cfg,pose,velocity,previous,now,workspace):
        if not math.isclose(now,self.ready_time,abs_tol=1e-8):raise ValueError('Queued plan cannot activate early or late')
        intent=self.intent
        # A live delivery ticket may create a separate execution lease. The
        # sampled goal/constraints and original ticket remain unchanged. A
        # late delivery never creates a lease or revives the previous request.
        if cfg.environment_execution_version=='actual_loop_upstream_v2' and now<=intent.expires_at+1e-10:
            intent=replace(intent,timestamp=now,expires_at=now+cfg.spec.grid.request_validity)
        return UpstreamPlannerInput(self.env_id,self.agent_id,pose,velocity,intent,self.terrain,self.neighbors,
            previous,now,cfg.spec.grid.times(now),workspace,cfg.spec,information_pose=self.information_pose,
            information_time=self.information_time,actual_history=self.history(cfg.spec))

    def state_dict(self):
        return deepcopy({'version':QUEUE_VERSION,'env_id':self.env_id,'agent_id':self.agent_id,'episode':self.episode,
            'information_time':self.information_time,'information_pose':self.information_pose,
            'information_control':self.information_control,'information_velocity':self.information_velocity,
            'intent':self.intent,'terrain':self.terrain.snapshot(),
            'neighbors':self.neighbors,'ready_time':self.ready_time,'feedback':self.feedback,'primitive_id':self.primitive_id})

    @classmethod
    def restore(cls,cfg,state):
        if not isinstance(state,dict) or state.get('version')!=QUEUE_VERSION:raise ValueError('Queue state version mismatch')
        item=cls.capture(cfg,state['env_id'],state['agent_id'],state['episode'],state['information_pose'],
            state['information_velocity'],state['information_control'],state['intent'],
            (GridTerrain.restore(state['terrain'],cfg) if cfg.obstacle_input is not None
             else FootprintTerrain(state['terrain'],cfg.spec,query_version=cfg.terrain_query_version)),
            state['neighbors'],state['information_time'],primitive_id=state.get('primitive_id'))
        if not math.isclose(item.ready_time,state['ready_time'],abs_tol=1e-9):raise ValueError('Queue latency state mismatch')
        for row in state['feedback']:item.append(cfg.spec,row['pose'],row['control'],row['velocity'],row['time'])
        return item
