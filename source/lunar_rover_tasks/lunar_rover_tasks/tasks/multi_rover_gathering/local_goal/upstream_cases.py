"""Frozen independent new-scale calibration cases from real sensor sampling.

Analytic backend identities are explicit. Only the sensing owner holds this
backend; exported planner inputs contain detached local point snapshots and
legal received predictions. These are not independent geographic DEM samples.
"""
from dataclasses import dataclass
from types import SimpleNamespace
import math

import numpy as np
import torch

from ..gathering_env_cfg import TerrainCfg
from .upstream_communication import MESSAGE_V2,DistanceCommunicationCache,MotionBoundLedger,RoverBroadcast
from .upstream_execution import midpoint_numpy,terrain_gain_numpy
from .upstream_planning import UpstreamIntent,UpstreamPlannerInput
from .upstream_sensing import DenseSensor,DenseTerrainMemory
from .upstream_timing import ActualControlHistory
from .upstream_spec import digest

CASE_VERSION='upstream_local_cases_v1'
FAMILIES=('flat','slope','crater','narrow','head_on','cross','hold','reverse','unknown')
CALIBRATION_SEEDS=(101,103,107,109)


class AnalyticTerrainBank:
    version='analytic_sensor_backend_v1'

    def __init__(self,family,seed):self.family,self.seed=family,seed

    def height(self,xy):
        x,y=xy[...,0],xy[...,1]
        if self.family=='slope':return x*math.tan(math.radians(3.5+(self.seed%4)*.3))
        if self.family=='crater':
            radius=((x-8.)**2+(y-3.)**2).sqrt()
            return -.35*torch.exp(-radius.square()/18.)+.06*torch.exp(-(radius-4.).square()/2.)
        if self.family=='narrow':return .25*(1-torch.exp(-(y/(2.8+(self.seed%3)*.2)).pow(8)))
        return torch.zeros_like(x)

    def query(self,xy,scene_indices):
        height=self.height(xy);dx=torch.zeros_like(xy);dy=torch.zeros_like(xy);dx[...,0]=.05;dy[...,1]=.05
        gx=(self.height(xy+dx)-self.height(xy-dx))/.1
        gy=(self.height(xy+dy)-self.height(xy-dy))/.1
        features=torch.stack((height,gx,gy,torch.zeros_like(height)),-1)
        valid=torch.ones_like(height,dtype=torch.bool)
        if self.family=='unknown':valid&=xy[...,0]<=6.+(self.seed%3)*.2
        return SimpleNamespace(features=features,metrics_valid=valid)


@dataclass(frozen=True)
class LocalCase:
    identity: str
    family: str
    seed: int
    solvability: str
    input: UpstreamPlannerInput
    map_identity: str
    version: str = CASE_VERSION

    @property
    def fingerprint(self):
        item=self.input;snapshot=item.terrain.snapshot()
        return digest({'version':self.version,'identity':self.identity,'seed':self.seed,'family':self.family,
            'profile':item.spec.fingerprint,'pose':item.ego_pose.tolist(),'velocity':item.ego_velocity.tolist(),
            'previous_control':item.previous_control.tolist(),'goal':item.intent.goal_xy.tolist(),
            'hold':item.intent.hold,'intent_times':[item.intent.timestamp,item.intent.expires_at],
            'terminal_yaw':item.intent.terminal_yaw,'speed_cap':item.intent.speed_cap,'stop_at_goal':item.intent.stop_at_goal,
            'clock':item.times.tolist(),'map_identity':self.map_identity,'workspace':None if item.workspace is None else np.asarray(item.workspace).tolist(),
            'information_pose':item.information_pose.tolist(),'information_time':item.information_time,
            'actual_history':{k:getattr(item.actual_history,k).tolist() for k in ('times','poses','controls','actual_velocities','initial_previous_control')},
            'terrain_xy':snapshot.xy.tolist(),'terrain_features':snapshot.features.tolist(),
            'terrain_times':snapshot.observed_at.tolist(),
            'terrain_support':snapshot.support_radius,
            'neighbors':[{'sender':p.sender_id,'times':p.times.tolist(),'positions':p.positions.tolist(),'radii':p.radii.tolist()} for p in item.neighbors]})


def make_case(cfg,family,seed,index=0):
    if family not in FAMILIES:raise ValueError('Unknown new-scale case family')
    if type(seed) is not int or seed<0 or type(index) is not int or index<0:raise ValueError('Invalid case seed/index')
    rng=np.random.default_rng(np.random.SeedSequence([seed,FAMILIES.index(family),index]))
    n=cfg.simulation.num_envs;poses=np.zeros((n,4,3));poses[:,:,:2]=[[0.,0.],[70.,0.],[0.,70.],[70.,70.]]
    speed=(0.,1.,1.5,25/9)[index%4] if family in {'flat','hold'} else 0.
    goal=np.array([(.8,2.,3.2,4.)[index%4],rng.uniform(-.15,.15)])
    if family=='reverse':goal=np.array([-min(2.,goal[0]),0.])
    if family in {'narrow','head_on','cross','unknown'}:goal=np.array([4.,0.])
    if family=='head_on':poses[:,1]=[12.,0.,math.pi]
    if family=='cross':poses[:,1]=[5.,-9.,math.pi/2]
    identity=f'analytic_{family}_seed{seed}_index{index}_{CASE_VERSION}'
    terrain_cfg=TerrainCfg(type='lunar_scene_pack',raster_risk_slope_scale=.6,raster_risk_rms_scale_m=.05)
    runtime=SimpleNamespace(scene_bank=AnalyticTerrainBank(family,seed),scene_indices=torch.arange(n))
    anchors=poses.copy();anchors[...,2]=0.
    frames=DenseSensor(cfg.spec,terrain_cfg,runtime=runtime,map_identities=(identity,)).sample(poses,anchors,[0.]*n)
    memory=DenseTerrainMemory(cfg.spec);memory.observe(frames[0][0]);view=memory.view()
    previous=np.array([speed,0.]);state=np.zeros(3);commands=[];velocities=[];history_poses=[state.copy()]
    delay=math.ceil(cfg.simulated_latency/cfg.spec.grid.dt-1e-12)
    for _ in range(delay):
        f,known=view.lookup(state[:2]);gain=terrain_gain_numpy(state,f[0],cfg.spec) if known[0] else .25
        state,velocity=midpoint_numpy(state,previous,cfg.spec,gain=gain)
        commands.append(previous.copy());velocities.append(velocity);history_poses.append(state.copy())
    now=delay*cfg.spec.grid.dt
    if not delay:
        f,known=view.lookup(state[:2]);velocity=previous*(terrain_gain_numpy(state,f[0],cfg.spec) if known[0] else .25)
    ledger=MotionBoundLedger(cfg.spec,n);radio=DistanceCommunicationCache(cfg.spec,anchors,ledger=ledger,message_version=MESSAGE_V2)
    packets=[]
    for e in range(n):
        row=[]
        for a in range(4):
            peer_speed=.3 if family=='head_on' and a==1 else .35 if family=='cross' and a==1 else 0.
            bound=ledger.pledge(e,a,peer_speed if a else speed,0.,now+cfg.spec.grid.horizon+2.,[peer_speed if a else speed,0.])
            row.append(RoverBroadcast(a,0,0.,np.array([0.,0.,poses[e,a,2]]),[peer_speed if a else speed,0.],
                [peer_speed if a else speed,0.],motion_bound=bound,version=MESSAGE_V2))
        packets.append(row)
    radio.transmit(packets,poses[:,:,:2],[0.]*n)
    predictions=radio.snapshot(0,0,0.,np.zeros(3),prediction_times=np.arange(delay+41)*.2).predictions
    intent=UpstreamIntent(np.zeros(2) if family=='hold' else goal,family=='hold',now,now+cfg.spec.grid.request_validity)
    history=ActualControlHistory(0,0,cfg.spec.fingerprint,np.arange(delay+1)*.2,np.asarray(history_poses),
        np.asarray(commands).reshape(-1,2),np.asarray(velocities).reshape(-1,2),previous)
    item=UpstreamPlannerInput(0,0,state,velocity,intent,view,predictions,previous,now,cfg.spec.grid.times(now),
        (-100.,-100.,100.,100.),cfg.spec,information_pose=np.zeros(3),information_time=0.,actual_history=history)
    solvability='witnessed_reachable' if family in {'flat','slope','hold','reverse'} else 'unknown_solvability'
    return LocalCase(f'{family}_{seed}_{index:03d}',family,seed,solvability,item,identity)


def calibration_cases(cfg):
    return [make_case(cfg,family,seed,index) for family in FAMILIES for index,seed in enumerate(CALIBRATION_SEEDS)]


def trajectory_metrics(item,trajectory):
    initial=np.linalg.norm(item.ego_pose[:2]-item.intent.goal_xy)
    errors=np.linalg.norm(trajectory.states[:,:2]-item.intent.goal_xy,axis=-1)
    change=np.diff(np.vstack((item.previous_control,trajectory.controls)),axis=0)
    features,known=item.terrain.lookup(trajectory.states[:,:2])
    risk=np.linalg.norm(features[:,1:3],axis=-1)/item.spec.terrain.driving_slope
    margins=[]
    for p in item.neighbors:
        for pose,t in zip(trajectory.states,trajectory.times):
            center,radius=p.at(t);margins.append(float(np.linalg.norm(pose[:2]-center)-radius))
    return {'goal_error':float(errors[-1]),'closest_goal_error':float(errors.min()),
        'rolling_progress':float(initial-errors.min()),'progress':float(initial-errors[-1]),'mean_goal_error':float(errors.mean()),
        'maximum_slope_ratio':float(risk[known].max()) if known.any() else None,'unknown_nodes':int((~known).sum()),
        'control_variation':float(np.sum((change/np.array([item.spec.vehicle.forward_limit,item.spec.vehicle.yaw_limit]))**2)),
        'minimum_neighbor_margin':min(margins) if margins else None,'reverse_fraction':float((trajectory.controls[:,0]<0).mean())}
