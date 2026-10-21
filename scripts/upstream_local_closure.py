"""Actual sensing/execution local closure shared by calibration and evaluation.

Goal coordinates are frozen scenario inputs. Controllers consume only their
own state, local memory and legal messages; no task/terrain truth enters them.
"""
import json
import math
from types import SimpleNamespace

import numpy as np
import torch

from _common import ROOT,cfg_from_experiment
from lunar_rover_tasks.tasks.multi_rover_gathering.gathering_env_cfg import TerrainCfg
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_cases import AnalyticTerrainBank
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_environment import UpstreamEnvironment
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_execution import brake_sequence,rollout_numpy,terrain_gain_numpy,control_error
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_planning import UpstreamIntent
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_review_cache import PoseCachedReviewer
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_spec import parse_upstream_experiment
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_ablation_learning import renew_legal_bounds


class LocalTargetScript:
    version='local_target_stopping_capture_upstream_v1'

    def __init__(self,spec,goal,hold=False):
        self.spec=spec;self.goal=np.asarray(goal,float).copy();self.phase='BRAKING' if hold else 'APPROACH';self.force_hold=hold
        self.hold_xy=None;self.reviewer=PoseCachedReviewer(spec)
        self.best_error=math.inf;self.progress_time=0.;self.recoveries=0

    def stopped(self,velocity):
        v,w=np.abs(velocity)
        return (v<=self.spec.task.linear_stop+1e-10 and w<=self.spec.task.angular_stop+1e-10
                and v+self.spec.vehicle.physical_radius*w<=self.spec.task.boundary_stop+1e-10)

    def decide(self,pose,velocity,previous,terrain,now):
        error=float(np.linalg.norm(pose[:2]-self.goal));stopped=self.stopped(velocity)
        if self.phase=='BRAKING' and stopped:self.phase='HOLD'
        if self.phase=='HOLD' and not self.force_hold and error>self.spec.task.exit_radius and stopped:
            self.phase='APPROACH';self.hold_xy=None
        if self.phase=='APPROACH':
            commands=brake_sequence(previous,self.spec)
            commands=np.vstack((commands,np.zeros((max(1,self.spec.grid.steps-len(commands)),2))))
            def gain(p):
                f,known=terrain.lookup(p[:2]);return terrain_gain_numpy(p,f[0],self.spec) if known[0] else .25
            prediction=rollout_numpy(pose,commands,self.spec,now,gain=gain)
            if np.linalg.norm(prediction.states[-1,:2]-self.goal)<=self.spec.task.capture_radius+1e-10:
                report=self.reviewer.trajectory(prediction,previous,terrain,require_stop=True,parking_end=True)
                if report.passed:self.phase='HOLD' if stopped else 'BRAKING';self.hold_xy=pose[:2].copy()
        hold=self.phase!='APPROACH'
        if hold:
            if self.hold_xy is None:self.hold_xy=pose[:2].copy()
            goal=self.hold_xy
        else:
            delta=self.goal-pose[:2];goal=pose[:2]+delta*min(1.,self.spec.rho_max/max(error,1e-12))
        return UpstreamIntent(goal,hold,now,now+self.spec.grid.request_validity)

    def recovery(self,env,e,a,now):
        error=float(np.linalg.norm(env.local_pose(e,a)[:2]-self.goal))
        if self.phase!='APPROACH':self.best_error=error;self.progress_time=now;return False
        if error<self.best_error-.01:self.best_error=error;self.progress_time=now
        elif now-self.progress_time>=5.-1e-8:
            env.pool.reset_agent(e,a);self.progress_time=now;self.best_error=error;self.recoveries+=1;return True
        return False

    def state_dict(self):
        return {'version':self.version,'profile':self.spec.fingerprint,'goal':self.goal.copy(),'force_hold':self.force_hold,
                'phase':self.phase,'hold_xy':self.hold_xy,
                'best_error':self.best_error,'progress_time':self.progress_time,'recoveries':self.recoveries}

    def load_state_dict(self,state):
        if (state.get('version')!=self.version or state.get('profile')!=self.spec.fingerprint
                or not np.array_equal(state.get('goal'),self.goal) or state.get('force_hold')!=self.force_hold
                or state.get('phase') not in {'APPROACH','BRAKING','HOLD'}):raise ValueError('Local controller restore mismatch')
        hold=state.get('hold_xy');error=state.get('best_error');now=state.get('progress_time');count=state.get('recoveries')
        if (hold is not None and (np.shape(hold)!=(2,) or not np.isfinite(hold).all())
                or error is None or math.isnan(error) or error<0 or now is None or not math.isfinite(now) or now<0
                or type(count) is not int or count<0):raise ValueError('Invalid local controller state')
        self.phase=state['phase'];self.hold_xy=None if hold is None else np.asarray(hold,float).copy()
        self.best_error=error;self.progress_time=now;self.recoveries=count


def runtime_config(base_path,*,name,seed,adapter=None,device=None,safety=False,samples=512,parameters=None,episode_seconds=60.,num_envs=1):
    # Fully parse the effective config; fingerprints bind overrides too.
    base=cfg_from_experiment(base_path);data=json.loads(base.configuration_json)
    data['experiment'].update(name=name,seed=seed,num_envs=num_envs,episode_length_s=episode_seconds)
    u=data['upstream_scale'];u.update(mppi_samples=samples,mppi_iterations=2)
    if adapter is not None:u['planner_adapter']=adapter
    if device is not None:u['planner_device']=device
    u['execution_safety_version']='footprint_discrete_cbf_upstream_v1' if safety else 'projection_disabled'
    if parameters is not None:u['mppi_parameters']=parameters
    if not u['planner_adapter'].startswith('mppi'):
        u['planner_device']='cpu';u.pop('mppi_parameters',None)
    return parse_upstream_experiment(data,ROOT),data


def initial_local_scene(case):
    if getattr(case,'initial_poses',()):return np.asarray([case.initial_poses],float)
    poses=np.zeros((1,4,3));poses[0,:,:2]=[[0.,0.],[70.,0.],[0.,70.],[70.,70.]]
    # Remote fixtures must themselves start on supported terrain. Otherwise an
    # unrelated remote invalidity would end the unknown-boundary ego diagnostic.
    if case.family=='unknown':poses[0,[1,3],0]=-70.
    if case.family=='head_on':poses[0,1]=[12.,0.,np.pi]
    if case.family=='cross':poses[0,1]=[5.,-9.,np.pi/2]
    return poses


def run_closure(cfg,case,*,max_steps=300,on_macro=None,cache_dir=None,resume_state=None):
    terrain_cfg=TerrainCfg(type='lunar_scene_pack',raster_risk_slope_scale=.6,raster_risk_rms_scale_m=.05)
    runtime=SimpleNamespace(scene_bank=AnalyticTerrainBank(case.family,getattr(case,'terrain_seed',case.seed)),scene_indices=torch.arange(1))
    poses=initial_local_scene(case)
    env=UpstreamEnvironment(cfg,poses,terrain_cfg=terrain_cfg,terrain_runtime=runtime,
                            map_identities=(case.map_identity,),cache_dir=cache_dir)
    # Initial velocity is an explicit scenario condition, restored through the
    # public physical lifecycle. The sensor/runtime still supplies observations.
    physical=env.core.state_dict();physical['previous_control'][0,0]=case.input.actual_history.initial_previous_control
    f,known=env.memory[0][0].view().lookup([0.,0.]);gain=terrain_gain_numpy([0.,0.,0.],f[0],cfg.spec) if known[0] else .25
    physical['actual_velocity'][0,0]=physical['previous_control'][0,0]*gain
    if case.family in {'head_on','cross'}:
        physical['previous_control'][0,1]=[.3 if case.family=='head_on' else .35,0.]
        physical['actual_velocity'][0,1]=physical['previous_control'][0,1]
    env.core.load_state_dict(physical)
    target=case.input.intent.goal_xy.copy();scripts=[LocalTargetScript(cfg.spec,target,case.family=='hold')]
    for a in range(1,4):scripts.append(LocalTargetScript(cfg.spec,[0.,0.],True))
    if case.family=='head_on':scripts[1]=LocalTargetScript(cfg.spec,[-4.,0.])
    if case.family=='cross':scripts[1]=LocalTargetScript(cfg.spec,[0.,20.])
    dwell=0;reference=None;violations=[];native_ids=[];records=[];steps=0;success=False
    if resume_state is not None:
        if resume_state.get('case_fingerprint')!=case.fingerprint:raise ValueError('Local closure case changed on resume')
        env.load_state_dict(resume_state['environment'])
        for script,saved in zip(scripts,resume_state['scripts']):script.load_state_dict(saved)
        dwell=resume_state['dwell'];reference=resume_state['reference'];steps=resume_state['steps']
        violations=list(resume_state['violations']);native_ids=list(resume_state['native_ids'])
        success=dwell>=round(cfg.spec.task.hold_duration/cfg.spec.grid.dt)
    while steps<max_steps and not env._ended[0] and not success:
        now=float(env.core.times[0]);renew_legal_bounds(env)
        requests=[[script.decide(env.local_pose(0,a),env.core.actual_velocity[0,a],env.core.previous_control[0,a],
                    env.memory[0][a].view(),now) for a,script in enumerate(scripts)]]
        scripts[0].recovery(env,0,0,now)
        ids=None
        if cfg.planner_adapter=='primitive47_upstream_v1':
            # Native selector is implemented by the common evaluation wrapper.
            ids=np.asarray([[choose_native_intent(env,a,requests[0][a]) for a in range(4)]],dtype=np.int64)
            native_ids.append(ids.copy())
        if ids is not None:env.enqueue(requests,primitive_ids=ids)
        transition=env.step_intents(None if ids is not None else requests,max_steps=min(5,max_steps-steps))
        for record in transition.records:
            if 'execution' not in record:continue
            execution=record['execution'];steps+=1
            if execution['geometry_overlap'][0]:violations.append({'step':steps,'reason':'geometry_overlap'})
            if execution['clearance_violation'][0]:violations.append({'step':steps,'reason':'clearance'})
            for a in range(4):
                error=control_error(execution['applied_control'][0,a],execution['previous_control'][0,a],cfg.spec)
                if error:violations.append({'step':steps,'reason':error,'agent_id':a})
            pose=execution['after'][0,0];velocity=execution['actual_velocity'][0,0]
            # This world/local frame coincides for the ego in frozen cases.
            near=case.family=='hold' or np.linalg.norm(pose[:2]-target)<=cfg.spec.task.capture_radius+1e-10
            parking=env.reviewer.pose(pose,env.memory[0][0].view(),parking=True)
            if near and scripts[0].stopped(velocity) and parking.passed:
                if reference is None:reference=pose.copy()
                drift=np.linalg.norm(pose[:2]-reference[:2])
                yaw=abs(math.atan2(math.sin(pose[2]-reference[2]),math.cos(pose[2]-reference[2])))
                if drift<=cfg.spec.task.position_drift+1e-10 and yaw<=cfg.spec.task.heading_drift+1e-10:dwell+=1
                else:dwell=0;reference=pose.copy()
            else:dwell=0;reference=None
            records.append(record)
            if dwell>=round(cfg.spec.task.hold_duration/cfg.spec.grid.dt):success=True
        if on_macro:on_macro(steps,env,transition,scripts,dwell,{'case_fingerprint':case.fingerprint,
            'environment':env.state_dict(),'scripts':[s.state_dict() for s in scripts],'steps':steps,
            'dwell':dwell,'reference':reference,'violations':violations,'native_ids':native_ids})
        if success:break
    return {'case_id':case.identity,'family':case.family,'seed':case.seed,'case_fingerprint':case.fingerprint,
        'completed':success,'dwell_steps':dwell,'steps':steps,'simulation_seconds':float(env.core.times[0]),
        'goal_error':float(np.linalg.norm(env.local_pose(0,0)[:2]-target)),'violations':violations,
        'terminal_reason':None if not env._ended[0] else str(env.last_task['terminal_reason'][0]),
        'recovery_count':scripts[0].recoveries,'native_ids':[x.tolist() for x in native_ids],
        'metadata':env.metadata,'records':records,'state':env.state_dict()}


def native_preview_config(cfg):
    """Information-time enumeration has no queued latency or future feedback.

    Execution retains the original configuration and queues the selected ID.
    This preview is explicitly identified in its effective configuration.
    """
    data=json.loads(cfg.configuration_json)
    data['experiment']['name']+='_native_information_preview'
    data['upstream_scale']['simulated_latency_s']=0.
    return parse_upstream_experiment(data,ROOT)


def choose_native_intent(env,a,intent):
    if intent.hold:return 0
    # Full native enumeration at information time; the chosen raw ID is then
    # queued unchanged. This cost includes all47 candidates and selection.
    from benchmark_upstream_planners import native_plan
    from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_factory import create_upstream_adapter
    from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_planning import UpstreamPlannerInput
    from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_timing import ActualControlHistory
    now=float(env.core.times[0]);p=env.local_pose(0,a);previous=env.core.previous_control[0,a]
    history=ActualControlHistory(0,a,env.spec.fingerprint,[now],[p],np.empty((0,2)),np.empty((0,2)),previous)
    item=UpstreamPlannerInput(0,a,p,env.core.actual_velocity[0,a],intent,env.memory[0][a].view(),
        env.radio.snapshot(0,a,now,p).predictions,previous,now,env.spec.grid.times(now),env.workspace(0,a),env.spec,
        information_pose=p,information_time=now,actual_history=history)
    _,selection=native_plan(create_upstream_adapter(native_preview_config(env.cfg)),item)
    return selection['primitive_id']
