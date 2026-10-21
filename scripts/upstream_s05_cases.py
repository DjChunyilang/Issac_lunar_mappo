"""Frozen formal evaluation inputs, separate from seeds101/103/107/109."""
from dataclasses import dataclass,replace
import math

import numpy as np

from upstream_local_closure import initial_local_scene
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_cases import LocalCase,make_case,FAMILIES
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_spec import digest
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_planning import UpstreamIntent

FORMAL_CASE_VERSION='upstream_formal_cases_v1'
SEEDS=(23,31,47)


@dataclass(frozen=True)
class FormalCase(LocalCase):
    terrain_seed: int = 23
    initial_poses: tuple = ()
    regression: str | None = None

    @property
    def fingerprint(self):
        return digest({'local':super().fingerprint,'formal_version':FORMAL_CASE_VERSION,
            'terrain_seed':self.terrain_seed,'initial_poses':self.initial_poses,'regression':self.regression})


def formal_case(cfg,family,seed,index,regression=None):
    # Public seed plus scene index yields a reproducible scenario generator.
    # Analytic map identities explicitly record the derived terrain seed.
    derived=seed+1000*index;source=make_case(cfg,family,derived,index)
    rng=np.random.default_rng(np.random.SeedSequence([seed,index,7919]))
    goal=source.input.intent.goal_xy.copy()
    if family not in {'hold','reverse'}:goal=[rng.uniform(1.,4.),rng.uniform(-.4,.4)]
    if family=='reverse':goal=[-rng.uniform(.5,2.),rng.uniform(-.2,.2)]
    intent=replace(source.input.intent,goal_xy=np.asarray(goal))
    poses=initial_local_scene(source);peers=source.input.neighbors
    if family=='unknown':
        peers=tuple(replace(p,positions=p.positions+[-140.,0.]) if p.sender_id in {1,3} else p for p in peers)
    if family in {'head_on','cross'}:
        offset=np.array([rng.uniform(-1.5,1.5),rng.uniform(-1.,1.)])
        poses[0,1,:2]+=offset
        peers=tuple(replace(p,positions=p.positions+offset) if p.sender_id==1 else p for p in peers)
    if regression=='zero_distance_move':intent=replace(intent,goal_xy=source.input.ego_pose[:2],hold=False)
    if regression=='expiry_boundary':intent=replace(intent,expires_at=source.input.now)
    if regression=='expired_request':intent=replace(intent,timestamp=0.,expires_at=.2)
    item=replace(source.input,intent=intent,neighbors=peers)
    identity=f'{family}_seed{seed}_{index:03d}'+('' if regression is None else '_'+regression)
    return FormalCase(identity,family,seed,source.solvability,item,source.map_identity,
        terrain_seed=derived,initial_poses=tuple(tuple(float(v) for v in p) for p in poses[0]),regression=regression)


def formal_cases(cfg):
    result=[formal_case(cfg,family,SEEDS[i%3],i) for family in FAMILIES for i in range(110)]
    regressions=(('flat','zero_distance_move',0),('flat','expiry_boundary',1),('flat','expired_request',2),
        ('hold','full_speed_hold',3),('flat','full_speed_move',3),('reverse','reverse_boundary',3),
        ('unknown','unknown_full_footprint',0),('cross','segment_crossing',2),
        ('slope','parking_slope_margin',1),('crater','reconstructed_ripple_stall',1))
    for i,(family,label,index) in enumerate(regressions):
        result.append(formal_case(cfg,family,SEEDS[i%3],1000+index,regression=label))
    return result


def gathering_scene(cfg,family,seed):
    names=('compact','irregular','reverse','crossing','slope','npb')
    if family not in names:raise ValueError('Unknown gathering family')
    tier=names.index(family) if family in names[:3] else 1
    low,high=cfg.spec.task.spawn_tiers[tier];rng=np.random.default_rng(np.random.SeedSequence([seed,names.index(family)]))
    poses=np.zeros((cfg.simulation.num_envs,4,3))
    for e in range(len(poses)):
        accepted=[np.zeros(2)]
        for a in range(1,4):
            for attempt in range(10000):
                radius=rng.uniform(low,high);angle=rng.uniform(-math.pi,math.pi)
                xy=radius*np.array([math.cos(angle),math.sin(angle)])
                if min(np.linalg.norm(xy-p) for p in accepted)>=cfg.spec.task.initial_distance:
                    accepted.append(xy);break
            else:raise ValueError('Formal birth geometry could not be sampled')
            poses[e,a,:2]=xy
            delta=np.asarray(cfg.spec.task.slots[a])-np.asarray(cfg.spec.task.slots[0])-xy
            yaw=math.atan2(delta[1],delta[0])
            poses[e,a,2]=yaw if family in {'compact','irregular'} else (yaw+math.pi if family=='reverse' else rng.uniform(-math.pi,math.pi))
    return poses
