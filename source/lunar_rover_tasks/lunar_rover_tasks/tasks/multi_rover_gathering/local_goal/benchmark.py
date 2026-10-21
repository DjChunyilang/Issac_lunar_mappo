"""Frozen legal local-case schema and independent physical evaluation."""
from dataclasses import replace
import hashlib
import json
import numpy as np
from .contracts import DT, N, NeighborPrediction
from .memory import LocalTerrainMemory
from .protocol import LocalIntent, PlannerInput, TerrainView, IdentifiedNeighbor

VERSION = 'planner_cases_v1'
FAMILIES = ('flat','slope','crater','narrow','head_on','cross','hold','reverse','unknown')
VARIANTS = ('line','quintic','primitive47','nmpc','mppi512','mppi1024','mppi2048')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',',':'), allow_nan=False).encode()).hexdigest()


def make_case(family, index, seed=23):
    rng=np.random.default_rng(np.random.SeedSequence([seed,FAMILIES.index(family),index]))
    axis=np.arange(-2,2.001,.2)
    xy=np.stack(np.meshgrid(axis,axis),-1).reshape(-1,2)
    features=np.zeros((len(xy),5));features[:,4]=1
    goal=np.array([rng.uniform(.5,1.3),rng.uniform(-.2,.2)])
    pose=np.zeros(3);previous=np.zeros(2);neighbors=[]
    if family == 'slope':
        features[:,1]=rng.uniform(.15,.35);features[:,0]=xy[:,0]*features[:,1]
        features[:,4]=.85
    elif family == 'crater':
        center=np.array([.8,.4]);r=np.linalg.norm(xy-center,axis=1)
        height=-.15*np.exp(-r*r/.18)
        features[:,0]=height;features[:,1:3]=-height[:,None]*(xy-center)/.09
        features[:,4]=np.clip(1-np.linalg.norm(features[:,1:3],axis=1),.2,1)
    elif family == 'narrow':
        features[np.abs(xy[:,1])>rng.uniform(.25,.4),4]=.1
        goal[1]=0
    elif family in ('head_on','cross'):
        t=np.arange(16)*DT
        positions=(np.stack([1.2-.3*t,np.full(16,.05)],-1) if family == 'head_on'
                   else np.stack([np.full(16,.65),.9-.5*t],-1))
        neighbors=[{'sender_id':2,'positions':positions.tolist(),'radii':(.46+.08*t).tolist(),
                    'timestamp':0.,'expires_at':.6}]
        goal[1]=0
    elif family == 'hold':
        previous[0]=rng.choice([0.,.3,.6]);goal=np.zeros(2)
    elif family == 'reverse':
        goal[0]*=-1
    elif family == 'unknown':
        keep=xy[:,0] <= rng.uniform(.3,.7);xy,features=xy[keep],features[keep]
        goal[0]=1.2
    case={'schema_version':VERSION,'case_id':f'{family}_{index:03d}','family':family,
          'seed':seed,'pose':pose.tolist(),'velocity':previous.tolist(),'previous_control':previous.tolist(),
          'goal':goal.tolist(),'hold':family=='hold','now':0.,'timestamp':0.,'expires_at':1.,
          'terrain':{'xy':xy.tolist(),'features':features.tolist(),'observed_at':[0.]*len(xy),
                     'support_radius':.155,'resolution':.1},'neighbors':neighbors,
          'map_identity':f'analytic_{family}_v1','solvability':('witnessed_reachable' if family in
          ('flat','hold','reverse') else 'blocked_by_design' if family=='unknown' else 'unknown_solvability'),
          'witness':('straight_midpoint_or_stationary_stop' if family in ('flat','hold','reverse') else None)}
    case['sha256']=digest(case)
    return case


def to_input(case, *, identity=None):
    if case['schema_version'] != VERSION or digest({k:v for k,v in case.items() if k!='sha256'}) != case['sha256']:
        raise ValueError('Case version/fingerprint mismatch')
    t=case['terrain'];m=LocalTerrainMemory(max(4096,len(t['xy'])),t['resolution'],t['support_radius'])
    if identity is None:identity=(case.get('env_id',0),case.get('agent_id',0))
    m.observe(np.asarray(t['xy']).reshape(-1,2),np.asarray(t['features']).reshape(-1,5),case['now'])
    m._timestamps[:len(m.samples)]=np.asarray(t['observed_at'])[:len(m.samples)]
    pose=np.asarray(case['pose'],float)
    intent=(LocalIntent.for_hold(pose,case['now']) if case['hold'] else
            LocalIntent(np.asarray(case['goal']),False,case['timestamp'],case['expires_at']))
    peers=tuple(IdentifiedNeighbor(p['sender_id'],NeighborPrediction(np.asarray(p['positions']),
          np.asarray(p['radii']),p['timestamp'],p['expires_at'])) for p in case['neighbors'])
    return PlannerInput(*identity,pose,np.asarray(case['velocity']),intent,TerrainView(m).conservative(),
                        peers,np.asarray(case['previous_control']),case['now'],case['now']+np.arange(16)*DT)


def trajectory_metrics(item, trajectory):
    states,u=trajectory.states,trajectory.controls
    distances=np.linalg.norm(states[1:,:2]-item.intent.goal_xy,axis=1)
    changes=np.diff(np.vstack([item.previous_control,u]),axis=0)
    features,known,_=item.terrain.conservative().lookup(states[1:,:2])
    risk=(np.linalg.norm(features[:,1:3],axis=1)/.8)**2+((1-features[:,4])/.75)**2
    parts={'goal':float(2*np.sum(distances**2)+12*distances[-1]**2),
           'control':float(np.sum(u*u*np.array([.05,.03]))),
           'smooth':float(np.sum(changes**2*np.array([.5,.2]))),
           'reverse':float(.2*np.sum(np.minimum(u[:,0],0)**2)),
           'terrain':float(2*np.sum(np.where(known,risk,0))), 'unknown':float(20*np.sum(~known)),
           'neighbor':0.}
    minimum=None
    for neighbor in item.neighbors:
        p=neighbor.prediction
        d=np.linalg.norm(states[1:,:2]-p.positions[1:],axis=1)
        parts['neighbor']+=float(8*np.sum(np.maximum((p.radii[1:]+.2-d)/.2,0)**2))
        clearance=float(np.min(d-p.radii[1:]));minimum=clearance if minimum is None else min(minimum,clearance)
    return {'common_cost':sum(parts.values()),'cost_components':parts,'goal_error':float(distances[-1]),
            'progress':float(np.linalg.norm(item.ego_pose[:2]-item.intent.goal_xy)-distances[-1]),
            'minimum_predicted_clearance':minimum,'terrain_risk':float(np.sum(np.where(known,risk,0))),
            'maximum_slope':float(np.max(np.linalg.norm(features[:,1:3],axis=1))),
            'reverse_ratio':float(np.mean(u[:,0]<0)), 'maximum_control_change':np.max(abs(changes),axis=0).tolist()}


def shifted_input(item, control, slope_scale=.75, *, neighbor_messages=None):
    from .execution import dynamics
    features=item.terrain.lookup(item.ego_pose[:2])[0][0]
    state,velocity=dynamics(item.ego_pose,control,features,slope_scale)
    peers=[];now=item.now+DT
    for peer in item.neighbors:
        p=peer.prediction if neighbor_messages is None else neighbor_messages[peer.sender_id]
        age=max(0.,now-p.timestamp);future=np.arange(16)*DT
        if now<=p.expires_at:
            times=(p.timestamp if neighbor_messages is not None else item.now)+future
            positions=np.stack([np.interp(now+future,times,p.positions[:,axis]) for axis in range(2)],-1)
            stopped=np.linalg.norm(p.positions[-1]-p.positions[-2])<1e-5
            radii=.46+.08*(age+future)+(0. if stopped else 1.15)*np.maximum(0.,now+future-times[-1])
        else:
            # Same stale FULL rule as the sensor environment: last reported position,
            # maximum-speed uncertainty; no old goal trajectory or truth replenishment.
            positions=np.tile(p.positions[0],(16,1))
            radii=.46+1.15*(age+future)
        peers.append(IdentifiedNeighbor(peer.sender_id,NeighborPrediction(positions,radii,p.timestamp,p.expires_at)))
    intent=(LocalIntent.for_hold(state,now) if item.intent.hold else
            replace(item.intent,timestamp=now,expires_at=now+1))
    return replace(item,ego_pose=state,ego_velocity=velocity,previous_control=control,now=now,
                   times=item.times+DT,neighbors=tuple(peers),intent=intent)
