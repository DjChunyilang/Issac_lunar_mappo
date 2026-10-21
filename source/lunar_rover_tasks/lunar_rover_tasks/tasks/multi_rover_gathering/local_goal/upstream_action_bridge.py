"""Explicit T11 prepared-action -> real upstream environment execution bridge.

Probability remains defined on the original sampled payload. A2's normalized
square maps smoothly into the same 4 m disc as A0/A1. A3 retains the sampled
native index and explicitly binds both source and new-scale library identities.
"""
import math

import numpy as np

from .action_ablation import ActionContract
from .contracts import wrap
from .upstream_baselines import PRIMITIVE_LIBRARY_VERSION,primitive_fingerprint,primitive_library
from .upstream_planning import UpstreamIntent

BRIDGE_VERSION='t11_upstream_execution_bridge_v1'


class UpstreamActionBridge:
    def __init__(self,env,contract):
        if type(contract) is not ActionContract or contract.profile_fingerprint!=env.spec.fingerprint:
            raise ValueError('Action bridge requires the matching probability/profile contract')
        native=contract.representation=='A3'
        if native!=(env.cfg.planner_adapter=='primitive47_upstream_v1'):
            raise ValueError('Native action and planner execution protocol disagree')
        self.env,self.contract=env,contract

    @property
    def metadata(self):
        return {'version':BRIDGE_VERSION,'probability_contract':self.contract.metadata,'environment':self.env.metadata,
            'goal_mapping':'normalized_square_to_disc_v1' if self.contract.representation=='A2' else 'native_or_polar_v1',
            'native_library_version':PRIMITIVE_LIBRARY_VERSION if self.contract.representation=='A3' else None,
            'native_library_fingerprint':primitive_fingerprint(self.env.spec) if self.contract.representation=='A3' else None,
            'formal_training_ready':False}

    def decode(self,payload):
        payload.validate(self.contract,(self.env.e,4))
        values=payload.continuous.detach().cpu().numpy();holds=payload.hold.detach().cpu().numpy()
        ids=payload.primitive_id.detach().cpu().numpy();requests=[];native=self.contract.representation=='A3'
        entries=primitive_library(self.env.spec)['entries'] if native else None
        for e in range(self.env.e):
            row=[];now=float(self.env.core.times[e])
            for a in range(4):
                p=self.env.local_pose(e,a);yaw=None;speed=None;hold=bool(holds[e,a])
                if native:
                    entry=entries[int(ids[e,a])];local=np.asarray(entry['local_xy']);hold=int(ids[e,a])==0
                    yaw=wrap(p[2]+entry['planned_yaw_delta'])
                elif hold:local=np.zeros(2)
                elif self.contract.representation=='A2':
                    x,y=values[e,a,:2]
                    local=self.contract.rho_max*np.array([x*math.sqrt(max(0.,1-y*y/2)),y*math.sqrt(max(0.,1-x*x/2))])
                    yaw=wrap(p[2]+math.pi*values[e,a,2]);speed=self.contract.speed_cap_max*(values[e,a,3]+1)/2
                else:
                    rho,beta=values[e,a];local=rho*np.array([math.cos(beta),math.sin(beta)])
                c,s=math.cos(p[2]),math.sin(p[2]);goal=p[:2]+np.array([[c,-s],[s,c]])@local
                row.append(UpstreamIntent(goal,hold,now,now+self.contract.request_seconds,terminal_yaw=yaw,speed_cap=speed))
            requests.append(row)
        return requests,ids.copy() if native else None

    def step(self,payload,*,max_steps=None):
        requests,ids=self.decode(payload)
        self.env.enqueue(requests,primitive_ids=ids)
        return self.env.step_intents(None,max_steps=max_steps)
