"""New-scale finite reward, per-base-step potential differences and returns."""
from copy import deepcopy
from dataclasses import asdict,dataclass

import numpy as np

from .upstream_spec import digest

REWARD_VERSION='finite_reward_upstream_v1'


@dataclass(frozen=True)
class UpstreamRewardSettings:
    gamma: float = .99
    lam: float = .95
    distance_progress: float = 9.5
    dispersion_progress: float = 6.
    distance_level: float = .13
    dispersion_level: float = .04
    flatness_progress: float = 2.
    flatness_excess: float = .02
    hold_potential: float = 12.
    energy_weight: float = .003
    distance_cost: float = .02
    turn_cost: float = .007
    terrain_weight: float = .1
    roughness_cost: float = .15
    traversability_cost: float = .15
    speed_loss_cost: float = .14
    height_change_cost: float = .09
    collision_cost: float = 100.
    clearance_cost: float = 2.4
    success_bonus: float = 130.
    failure_penalty: float = 55.
    timeout_penalty: float = 80.
    gather_scaling: str = 'linear'
    hazard_cost: float = 0.
    continuous_cost_scale: float = 1.

    def __post_init__(self):
        if self.gather_scaling not in ('linear', 'log1p'):
            raise ValueError('Unknown gather reward scaling')
        if any(not np.isfinite(v) or v<0 for k,v in asdict(self).items() if k != 'gather_scaling') or not 0<self.gamma<=1 or not 0<=self.lam<=1:
            raise ValueError('Invalid finite reward scale or discount')
        if isinstance(self.continuous_cost_scale,bool) or self.continuous_cost_scale<=0:
            raise ValueError('Continuous cost scale must be positive')

    @property
    def metadata(self):
        result = asdict(self)
        if self.gather_scaling == 'linear':
            result.pop('gather_scaling')
        if self.hazard_cost == 0.:
            result.pop('hazard_cost')
        if self.continuous_cost_scale == 1.:
            result.pop('continuous_cost_scale')
        return result


def configured_reward_settings(gamma, lam, config):
    """One reward definition shared by execution, prelaunch audit and trainer."""
    overrides = config.get('reward_overrides', {})
    if set(overrides) - {'terrain_weight', 'hazard_cost', 'failure_penalty', 'continuous_cost_scale'}:
        raise ValueError('Unsupported terrain reward override')
    return UpstreamRewardSettings(gamma=gamma, lam=lam,
        gather_scaling=config['reward_scaling'], **overrides)


def actual_facts(spec,poses,velocity,reports,hold_count):
    p=np.asarray(poses,float);v=np.asarray(velocity,float);count=np.asarray(hold_count)
    if (p.ndim!=3 or p.shape[1:]!=(4,3) or v.shape!=(*p.shape[:2],2) or not np.isfinite(p).all()
            or not np.isfinite(v).all() or count.shape!=(len(p),) or not np.isfinite(count).all()):
        raise ValueError('Reward facts require finite actual team state')
    distances=np.linalg.norm(p[:,:,None,:2]-p[:,None,:,:2],axis=-1)
    pair=np.triu(np.ones((4,4),bool),1);center=p[...,:2].mean(1)
    flat=[]
    for row in reports:
        costs=[]
        for r in row:
            if r.minimum_body_clearance is None:cost=10.
            else:cost=max(r.slope/spec.terrain.parking_slope,r.roughness_rms/spec.terrain.roughness_rms,
                r.maximum_step/spec.terrain.step_height,1+max(0.,-r.minimum_body_clearance)/spec.vehicle.ground_clearance)
            costs.append(min(10.,cost))
        flat.append(max(costs))
    return {'dmax':distances.max((1,2)),'dispersion':((p[...,:2]-center[:,None])**2).sum(-1).mean(-1),
            'minimum_distance':distances[:,pair].min(-1),'flatness_cost':np.asarray(flat),
            'hold_count':count.copy()}


class UpstreamReward:
    def __init__(self,spec,num_envs,settings=None,*,obstacle_input=None):
        self.spec=spec;self.num_envs=num_envs;self.settings=settings or UpstreamRewardSettings()
        self.obstacle_input=obstacle_input
        self.graded_craters=obstacle_input is not None and obstacle_input.graded_craters
        self.medium_crater_paid=np.zeros(num_envs) if self.graded_craters else None
        self.raw_return=np.zeros(num_envs);self.discounted_return=np.zeros(num_envs)
        self.steps=np.zeros(num_envs,np.int64);self.component_returns={}

    @property
    def metadata(self):return {**({'obstacle_input':self.obstacle_input.metadata,
                                 'obstacle_reward_version':('executed_crater_classes_v2' if self.graded_craters else 'executed_obstacle_penalty_v1')} if self.obstacle_input is not None else {}),
        **({'continuous_cost_scope':'legacy_nonpotential_excluding_crater_medium_v1'} if self.settings.continuous_cost_scale!=1. else {}),
        'version':REWARD_VERSION,'profile_fingerprint':self.spec.fingerprint,
        'settings':self.settings.metadata,'settings_fingerprint':digest(self.settings.metadata),
        'potential_scale':('distance_and_dispersion_threshold_units_v1' if self.settings.gather_scaling=='linear'
                           else 'log1p_distance_and_dispersion_excess_v1'),'num_envs':self.num_envs}

    def gather_excess(self, facts):
        distance = np.maximum(facts['dmax'] - self.spec.task.dmax, 0.) / self.spec.task.dmax
        dispersion = np.maximum(facts['dispersion'] - self.spec.task.dispersion, 0.) / self.spec.task.dispersion
        if self.settings.gather_scaling == 'log1p':
            distance, dispersion = np.log1p(distance), np.log1p(dispersion)
        return distance, dispersion

    def potentials(self,facts):
        s=self.settings;t=self.spec.task
        distance,dispersion=self.gather_excess(facts)
        flat=np.clip((2*t.dmax-facts['dmax'])/t.dmax,0.,1.)*np.maximum(facts['flatness_cost']-1.,0.)
        ratio=np.clip(facts['hold_count']/(t.hold_duration/self.spec.grid.dt),0.,1.)
        return {'gather':-s.distance_progress*distance-s.dispersion_progress*dispersion,
                'flatness':-s.flatness_progress*flat,'success_hold':-s.hold_potential*(1-ratio)}

    def advance(self,before,after,task,velocity,features,gains,height_change,*,obstacle_distances=None,medium_crater_distances=None):
        s=self.settings;valid=np.asarray(task['learning_valid'],bool);ended=np.asarray(task['ended_this_step'],bool)
        scale=s.continuous_cost_scale
        previous=self.potentials(before);current=self.potentials(after)
        terms={key:s.gamma*np.where(ended,0.,current[key])-previous[key] for key in previous}
        if s.gather_scaling=='linear':
            # Preserve the archived protocol's arithmetic order as well as its units.
            terms['gather']-=(s.distance_level*scale)*np.maximum(after['dmax']-self.spec.task.dmax,0.)/self.spec.task.dmax
            terms['gather']-=(s.dispersion_level*scale)*np.maximum(after['dispersion']-self.spec.task.dispersion,0.)/self.spec.task.dispersion
        else:
            distance,dispersion=self.gather_excess(after)
            terms['gather']-=(s.distance_level*scale)*distance
            terms['gather']-=(s.dispersion_level*scale)*dispersion
        terms['flatness']-=(s.flatness_excess*scale)*np.clip((2*self.spec.task.dmax-after['dmax'])/self.spec.task.dmax,0.,1.)*np.maximum(after['flatness_cost']-1.,0.)
        v=np.asarray(velocity,float);f=np.asarray(features,float);gain=np.asarray(gains,float);dh=np.asarray(height_change,float)
        if (valid.shape!=(self.num_envs,) or v.shape!=(self.num_envs,4,2) or f.shape!=(self.num_envs,4,5)
                or gain.shape!=(self.num_envs,4) or dh.shape!=gain.shape
                or any(not np.isfinite(x).all() for x in (v,f,gain,dh))):raise ValueError('Invalid executed reward inputs')
        terms['energy']=-(s.energy_weight*scale)*(s.distance_cost*abs(v[...,0]).mean(-1)*self.spec.grid.dt+s.turn_cost*abs(v[...,1]).mean(-1))
        terms['terrain']=-(s.terrain_weight*scale)*(s.roughness_cost*f[...,3]+s.traversability_cost*(1-f[...,4])
            +s.speed_loss_cost*(1-gain)+s.height_change_cost*abs(dh)).mean(-1)
        if s.hazard_cost:
            risk = (np.linalg.norm(f[...,1:3],axis=-1)/self.spec.terrain.driving_slope)**2
            risk += np.clip(f[...,3]/self.spec.terrain.roughness_rms,0,10)**2
            risk += ((1-f[...,4])/.75)**2
            risk += np.clip(abs(dh)/self.spec.terrain.step_height,0,10)**2
            terms['terrain'] -= (s.hazard_cost*scale)*risk.mean(-1)
        reasons=np.asarray(task['terminal_reason'])
        terms['safety']=-(s.clearance_cost*scale)*np.maximum(2*self.spec.vehicle.effective_radius-after['minimum_distance'],0.)
        terms['safety']-=s.collision_cost*(ended&(reasons=='collision'))
        terms['terminal']=s.success_bonus*(ended&(reasons=='success'))-s.failure_penalty*(ended&np.isin(reasons,['collision','out_of_bounds','terrain_failure']))
        terms['terminal']-=s.timeout_penalty*(ended&(reasons=='task_deadline'))
        if self.obstacle_input is not None:
            d=np.asarray(obstacle_distances,float)
            if d.shape!=(self.num_envs,4) or not np.isfinite(d).all():raise ValueError('Expected pre-step observed obstacle distances')
            data=self.obstacle_input.data;band=data['cost']['soft_band_m'];weights=data['reward']
            terms['obstacle_near']=-self.spec.grid.dt*weights['near_weight_per_s']*(np.clip(1-d/band,0,1)**2).mean(-1)
            terms['obstacle_overlap']=-self.spec.grid.dt*weights['overlap_weight_per_s']*np.clip(-d/band,0,1).mean(-1)
            if scale!=1.:
                terms['obstacle_near']*=scale
                terms['obstacle_overlap']*=scale
            if self.graded_craters:
                medium=np.asarray(medium_crater_distances,float)
                if medium.shape!=(self.num_envs,4) or not np.isfinite(medium).all():
                    raise ValueError('Expected pre-step observed medium-crater distances')
                classes=data['crater_classes']
                charge=self.spec.grid.dt*classes['medium_reward_per_s']*np.clip(-medium/band,0,1).mean(-1)
                charge=np.minimum(charge,np.maximum(0.,classes['medium_reward_cap']-self.medium_crater_paid))
                charge=np.where(valid,charge,0.)
                self.medium_crater_paid+=charge
                terms['crater_medium']=-charge
        terms={key:np.where(valid,value,0.) for key,value in terms.items()}
        total=sum(terms.values());weight=s.gamma**self.steps
        self.raw_return+=total;self.discounted_return+=weight*total
        for key,value in terms.items():self.component_returns.setdefault(key,np.zeros(self.num_envs))[:] +=value
        self.steps+=valid.astype(np.int64)
        return total,terms

    def reset(self,ids):
        self.raw_return[ids]=0.;self.discounted_return[ids]=0.;self.steps[ids]=0
        if self.graded_craters:self.medium_crater_paid[ids]=0.
        for values in self.component_returns.values():values[ids]=0.

    def state_dict(self):return deepcopy({**({'medium_crater_paid':self.medium_crater_paid} if self.graded_craters else {}),
        'contracts':self.metadata,'raw_return':self.raw_return,
        'discounted_return':self.discounted_return,'steps':self.steps,'component_returns':self.component_returns})

    def load_state_dict(self,state):
        if not isinstance(state,dict) or state.get('contracts')!=self.metadata:raise ValueError('Reward snapshot identity mismatch')
        raw=np.asarray(state['raw_return'],float);discounted=np.asarray(state['discounted_return'],float);steps=np.asarray(state['steps'])
        parts=deepcopy(state['component_returns'])
        if self.graded_craters:
            paid=np.asarray(state['medium_crater_paid'],float)
            cap=self.obstacle_input.data['crater_classes']['medium_reward_cap']
            if paid.shape!=raw.shape or not np.isfinite(paid).all() or (paid<0).any() or (paid>cap+1e-10).any():
                raise ValueError('Invalid medium-crater penalty counter')
        if (raw.shape!=(self.num_envs,) or discounted.shape!=raw.shape or steps.shape!=raw.shape or steps.dtype.kind not in 'iu'
                or (steps<0).any() or not np.isfinite(raw).all() or not np.isfinite(discounted).all()
                or not isinstance(parts,dict) or any(np.shape(v)!=raw.shape or not np.isfinite(v).all() for v in parts.values())):
            raise ValueError('Invalid reward counters/returns')
        self.raw_return,self.discounted_return,self.steps,self.component_returns=raw.copy(),discounted.copy(),steps.copy(),parts
        if self.graded_craters:self.medium_crater_paid=paid.copy()
