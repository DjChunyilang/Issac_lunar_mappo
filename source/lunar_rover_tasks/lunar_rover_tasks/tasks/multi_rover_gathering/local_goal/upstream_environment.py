"""S03 diagnostic environment with real sensing, queues, task and macro reward.

No learner or optimizer is constructed. Safety projection is not yet connected;
every issued control still undergoes public joint limits and footprint stopping
admission. Absolute intent expiry is preserved during the injected queue delay.
"""
from copy import deepcopy
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from threading import local
import math

import numpy as np
import torch

from .contracts import PlannerFeedback,wrap
from .upstream_communication import MESSAGE_V2,DistanceCommunicationCache,MotionBoundLedger,RoverBroadcast
from .upstream_execution import terrain_gain_numpy,stop_envelope
from .upstream_factory import UpstreamPlannerPool
from .upstream_footprint import AdmissionDecision,admit_control
from .upstream_gathering import LeaderSquareScript
from .upstream_observation import ActorObservation,ObservationInput,UpstreamObservationEncoder,OBSERVATION_VERSION
from .upstream_proxy import UpstreamActuatorProxy
from .upstream_queue import PendingPlan,QUEUE_VERSION,ACCEPTED_QUEUE_VERSION
from .upstream_reward import UpstreamReward,actual_facts
from .upstream_review_cache import PoseCachedReviewer
from .upstream_sensing import DenseSensor,DenseTerrainMemory,world_to_odometry
from .upstream_spec import UpstreamRuntimeConfig
from .upstream_task import UpstreamTask
from .upstream_planning import UpstreamIntent
from .upstream_grid import GridMemory
from .upstream_obstacle_spec import OBSERVATION_VERSION as GRID_OBSERVATION_VERSION

ENVIRONMENT_VERSION='actual_loop_upstream_v1'
ENVIRONMENT_V2='actual_loop_upstream_v2'
REASON_CODE={'none':0,'success':1,'collision':2,'out_of_bounds':3,'task_deadline':4,'environment_invalid':5,'terrain_failure':6}


@lru_cache(maxsize=4)
def _cpu_executor(workers):
    # Shared across environment restoration; no executor enters checkpoints.
    return ThreadPoolExecutor(max_workers=workers, thread_name_prefix='rover-cpu')


def _cpu_call(job):
    function, *arguments = job
    return function(*arguments)


_grid_reviewers = local()


def _grid_reviewer(spec, acceleration_version):
    """Thread-owned disposable witness cache; never enters environment state."""
    key=(spec.fingerprint,acceleration_version)
    saved=getattr(_grid_reviewers,'reviewer',None)
    if saved is None or saved[0]!=key:
        from .upstream_flat_review import create_reviewer
        saved=(key,create_reviewer(spec,acceleration_version));_grid_reviewers.reviewer=saved
    return saved[1]


@dataclass
class UpstreamMacroTransition:
    reward: torch.Tensor
    duration: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    terminal_actor: torch.Tensor
    terminal_critic: torch.Tensor
    actor: torch.Tensor
    critic: torch.Tensor
    episodes: list
    records: list
    terminal_reason: torch.Tensor
    bootstrap_allowed: torch.Tensor
    learning_valid: torch.Tensor
    reward_components: dict
    version: str = 'upstream_macro_transition_v1'


class UpstreamEnvironment:
    cpu_workers = 6
    state_version='upstream_environment_state_v1'

    def __init__(self,cfg,initial_poses,*,terrain_cfg=None,terrain_runtime=None,
                 map_identities=('flat_proxy',),planner_pool=None,reward_settings=None,cache_dir=None,initial_control=None,curriculum=None):
        if (type(cfg) is not UpstreamRuntimeConfig or cfg.environment_execution_version not in {ENVIRONMENT_VERSION,ENVIRONMENT_V2}
                or cfg.observation_execution_version not in (OBSERVATION_VERSION,GRID_OBSERVATION_VERSION) or cfg.timing_version!='actual_control_history_v1'
                or cfg.clock_mode!='offline_simulated_latency' or not cfg.valid_execution_role):
            raise ValueError('Environment requires explicit S03 observation/history/runtime role versions')
        self.curriculum=curriculum
        self.cfg=cfg;self.spec=cfg.spec;self.e=cfg.simulation.num_envs;self.a=4
        self.graded_craters=cfg.obstacle_input is not None and cfg.obstacle_input.graded_craters
        self._initial=np.asarray(initial_poses,float).copy()
        self.core=UpstreamActuatorProxy(cfg,self._initial);self.anchors=self._initial.copy();self.anchors[...,2]=0.
        self._initial_control=None if initial_control is None else np.asarray(initial_control,float).copy()
        if self._initial_control is not None:
            if self._initial_control.shape!=(self.e,4,2) or not np.isfinite(self._initial_control).all():
                raise ValueError('Initial control must contain finite linear/angular velocities for every rover')
            self.core.previous_control[:]=self._initial_control
            self.core.actual_velocity[:]=self._initial_control
        self._terrain_cfg=deepcopy(terrain_cfg);self._terrain_runtime=terrain_runtime
        self.sensor=DenseSensor(self.spec,terrain_cfg,runtime=terrain_runtime,map_identities=map_identities)
        self.map_identities=tuple(map_identities)
        self.obstacle_atlases=None
        if cfg.obstacle_input is not None:
            from ....terrain_data.obstacle_atlas import ObstacleAtlas
            from ....terrain_data.scene_pack import file_hash
            if terrain_runtime is None or terrain_runtime.scene_bank is None:
                raise ValueError('Obstacle input requires explicit derived lunar atlases')
            self.obstacle_atlases=[ObstacleAtlas(cfg.obstacle_input.atlas_root/file_hash(scene.path)/'obstacle.json',
                cfg.obstacle_input,scene) for scene in terrain_runtime.scene_bank.scenes]
            physical={'driving_slope':self.spec.terrain.driving_slope,'parking_slope':self.spec.terrain.parking_slope,
                      'roughness_rms':self.spec.terrain.roughness_rms,'step_height':self.spec.terrain.step_height}
            if any(atlas.meta['vehicle_terrain']!=physical for atlas in self.obstacle_atlases):
                raise ValueError('Derived ground costs disagree with physical terrain thresholds')
            if self.graded_craters:
                ground=cfg.obstacle_input.data['ground']
                if (ground['slope_score_scale']!=terrain_cfg.raster_risk_slope_scale
                        or ground['rms_score_scale_m']!=terrain_cfg.raster_risk_rms_scale_m):
                    raise ValueError('Derived driving surface disagrees with sensor score scales')
        self.memory=[[(GridMemory(self.spec,cfg.obstacle_input) if cfg.obstacle_input is not None
                       else DenseTerrainMemory(self.spec)) for _ in range(4)] for _ in range(self.e)]
        script_type=LeaderSquareScript
        if cfg.leader_execution_version=="leader_square_visibility_upstream_v2":
            from .upstream_visibility import LeaderSquareVisibilityScript
            script_type=LeaderSquareVisibilityScript
        elif cfg.leader_execution_version=="leader_square_visibility_upstream_v3":
            from .upstream_visibility import LeaderSquareVisibilityHysteresisScript
            script_type=LeaderSquareVisibilityHysteresisScript
        elif cfg.leader_execution_version=="leader_square_visibility_upstream_v4":
            from .upstream_visibility import LeaderSquareClearanceScript
            script_type=LeaderSquareClearanceScript
        self.scripts=[[script_type(self.spec,e,a) for a in range(4)] for e in range(self.e)]
        self.ledger=MotionBoundLedger(self.spec,self.e)
        self.radio=DistanceCommunicationCache(self.spec,self.anchors,ledger=self.ledger,message_version=MESSAGE_V2)
        self.pool=planner_pool or UpstreamPlannerPool(cfg,cache_dir=cache_dir)
        self._pool_injected=planner_pool is not None;self.cache_dir=cache_dir
        from .upstream_flat_review import create_reviewer
        self.reviewer=create_reviewer(self.spec,cfg.review_acceleration_version);self.encoder=UpstreamObservationEncoder(self.spec,spatial_scaling='linear' if curriculum is None else curriculum.config['observation_scaling'],obstacle_input=cfg.obstacle_input)
        self._physical_views=[[None]*4 for _ in range(self.e)]
        self._driving_views=[[None]*4 for _ in range(self.e)]
        self.physical_query_points=0
        if cfg.obstacle_input is not None:self.sensor.prepare_birth(self.reviewer.points)
        self.safety=None;self.execution_predictions=[[None]*4 for _ in range(self.e)]
        if cfg.execution_safety_version!='projection_disabled':
            from .upstream_safety import UpstreamSafetyProjector
            self.safety=[[UpstreamSafetyProjector(self.spec,reviewer=create_reviewer(self.spec,cfg.review_acceleration_version)) for _ in range(4)] for _ in range(self.e)]
        self.task=UpstreamTask(self.spec,self.e,cfg.simulation.episode_length_s if curriculum is None else curriculum.deadlines,
            terrain_failure_terminal=self.graded_craters or (curriculum is not None and curriculum.config.get('terrain_failure_semantics')=='known_terrain_terminal_v1'))
        self.reward_tracker=UpstreamReward(self.spec,self.e,reward_settings,obstacle_input=cfg.obstacle_input)
        self.pending={};self.intents=[[None]*4 for _ in range(self.e)];self.plans=[[None]*4 for _ in range(self.e)]
        self.feedback=[[PlannerFeedback() for _ in range(4)] for _ in range(self.e)]
        self.bounds=[[None]*4 for _ in range(self.e)];self._ended=np.zeros(self.e,bool)
        self._valid=np.ones(self.e,bool);self.last_task=None;self.last_terms={};self._records=[];self._episodes=[]
        self._pose_reports=[[None]*4 for _ in range(self.e)]
        self._center_features=[[None]*4 for _ in range(self.e)]
        self._refresh(np.ones(self.e,bool))
        for e in range(self.e):self._initialize_leader(e)
        self._publish()

    @property
    def metadata(self):
        return {**({'curriculum':self.curriculum.metadata} if self.curriculum is not None else {}),
            **({'initial_control':self._initial_control.tolist()} if self._initial_control is not None else {}),
            'contracts':self.cfg.metadata,'environment_version':self.cfg.environment_execution_version,'state_version':self.state_version,
            'map_identities':self.map_identities,'sensor_version':self.sensor.version,
            **({'obstacle_atlases':[a.identity for a in self.obstacle_atlases],
                'physical_measurement':('graded_craters_driving_original_parking_v2' if self.graded_craters else 'instantaneous_original_grid_footprint_subset_v1'),
                'sensor_version':'incremental_fixed_semantic_grid_v1'} if self.obstacle_atlases is not None else {}),
            'queue_version':ACCEPTED_QUEUE_VERSION if self.cfg.environment_execution_version==ENVIRONMENT_V2 else QUEUE_VERSION,
            'reward':self.reward_tracker.metadata,'observation':self.encoder.metadata,
            'fixed_frame_version':'translated_world_axes_upstream_v1',
            'terrain_type':'flat_proxy' if self._terrain_cfg is None else self._terrain_cfg.type,
            'safety_projection':self.cfg.execution_safety_version,'training_eligible':self.cfg.learning_runtime}

    def local_pose(self,e,a):
        p=self.core.poses[e,a];frame=self.anchors[e,a]
        return np.r_[world_to_odometry(p[:2][None],frame)[0],wrap(p[2]-frame[2])]

    def workspace(self,e,a):
        # Fixed odometry frames retain world-axis orientation. This preserves
        # the configured world rectangle even for arbitrary initial rover headings.
        frame=self.anchors[e,a]
        return tuple(np.r_[[-self.spec.world_width/2,-self.spec.world_height/2]-frame[:2],
                            [self.spec.world_width/2,self.spec.world_height/2]-frame[:2]])

    def _cpu_map(self,jobs):
        if not self.cpu_workers:return [_cpu_call(job) for job in jobs]
        return list(_cpu_executor(self.cpu_workers).map(_cpu_call,jobs))

    def _refresh(self,active):
        if self.obstacle_atlases is not None:
            jobs=[];identities=[]
            for e in np.flatnonzero(active):
                for a in range(4):
                    # Sensor counters and scene access retain their owner-thread
                    # ordering. Jobs own distinct memories and physical views.
                    identities.append((e,a));jobs.append((self._refresh_grid_rover,e,a,self._physical_view(e,a)))
            for (e,a),(report,center) in zip(identities,self._cpu_map(jobs),strict=True):
                self._pose_reports[e][a]=report;self._center_features[e][a]=center
            return
        frames=self.sensor.sample(self.core.poses,self.anchors,self.core.times)
        self._cpu_map([(self.memory[e][a].observe,frames[e][a])
            for e in range(self.e) if active[e] for a in range(4)])

    def _refresh_grid_rover(self,e,a,physical):
        pose=self.local_pose(e,a);now=float(self.core.times[e])
        index=int(self._terrain_runtime.scene_indices[e])
        view=self.memory[e][a].update(self.core.poses[e,a],self.anchors[e,a],now,self.obstacle_atlases[index])
        reviewer=_grid_reviewer(self.spec,self.cfg.review_acceleration_version)
        driving=self._physical_view(e,a,driving=True) if self.graded_craters else physical
        return ((view,pose.tobytes(),reviewer.pose(pose,physical,parking=True)),
                (view,pose.tobytes(),driving.lookup(pose[:2])))

    def _observe_and_predict(self,e,a,frame,pose,now,peers):
        # Each job exclusively owns one rover's memory and safety projector.
        self.memory[e][a].observe(frame)
        prediction=None
        if self.safety is not None:
            from .upstream_safety import SafetyContext
            context=SafetyContext(pose,self.core.previous_control[e,a],self.memory[e][a].view(),peers,
                now,np.asarray(self.workspace(e,a)).reshape(2,2),self.ledger.speed_cap(e,a,now))
            prediction=self.safety[e][a].execution_prediction(context,self.core.previous_control[e,a])
        view=self.memory[e][a].view()
        report=(view,pose.tobytes(),self.reviewer.pose(pose,view,parking=True))
        center=(view,pose.tobytes(),view.lookup(pose[:2]))
        return prediction,report,center

    def _refresh_and_predict(self,active):
        if self.obstacle_atlases is not None:
            self._refresh(active)
            return
        if self.safety is None and self.cfg.predictive_footprint_checks:
            self._refresh(active)
            return
        frames=self.sensor.sample(self.core.poses,self.anchors,self.core.times)
        jobs=[];identities=[]
        for e in range(self.e):
            if not active[e]:continue
            for a in range(4):
                pose=self.local_pose(e,a);now=float(self.core.times[e])
                # Freeze messages before any publish, as in the two-barrier path.
                peers=self.radio.snapshot(e,a,now,pose).predictions if self.safety is not None else ()
                identities.append((e,a))
                jobs.append((self._observe_and_predict,e,a,frames[e][a],pose,now,peers))
        for (e,a),prediction in zip(identities,self._cpu_map(jobs),strict=True):
            self.execution_predictions[e][a],self._pose_reports[e][a],self._center_features[e][a]=prediction

    def _center_lookup(self,e,a,pose):
        view=self.memory[e][a].view();saved=self._center_features[e][a]
        if saved is None or saved[0] is not view or saved[1]!=pose.tobytes():
            source=self._physical_view(e,a,driving=self.graded_craters) if self.obstacle_atlases is not None else view
            saved=(view,pose.tobytes(),source.lookup(pose[:2]));self._center_features[e][a]=saved
        return saved[2]

    def _physical_view(self,e,a,*,driving=False):
        from .upstream_footprint import FootprintTerrain
        cache=self._driving_views if driving else self._physical_views
        pose=self.core.poses[e,a];saved=cache[e][a]
        key=(pose.tobytes(),self.anchors[e,a].tobytes())
        if saved is None or saved[0]!=key:
            atlas=None
            if driving:
                atlas=self.obstacle_atlases[int(self._terrain_runtime.scene_indices[e])]
            xy,features=self.sensor._birth_arrays(pose,scene_index=e,driving_atlas=atlas)
            # Preserve the original two translations and their rounding order.
            xy=xy+pose[:2]-self.anchors[e,a,:2]
            terrain=FootprintTerrain._from_owned_arrays(xy,features,
                np.full(len(xy),float(self.core.times[e])),self.spec)
            saved=(key,terrain);cache[e][a]=saved
            self.physical_query_points+=len(self.sensor._birth_query)
        return saved[1]

    def _reports(self,*,parking=True):
        from dataclasses import replace
        if self.graded_craters and not parking:
            result=[]
            vehicle=self.spec.vehicle
            for e in range(self.e):
                atlas=self.obstacle_atlases[int(self._terrain_runtime.scene_indices[e])]
                invaded=atlas.exact_large_crater_overlap(self.core.poses[e],vehicle.length,vehicle.width)
                row=[]
                for a in range(4):
                    report=self.reviewer.pose(self.local_pose(e,a),self._physical_view(e,a,driving=True),parking=False)
                    if invaded[a] and report.reason not in {'unknown_footprint','degenerate_contact_plane'}:
                        report=replace(report,passed=False,reason='large_crater')
                    row.append(report)
                result.append(row)
            return result
        result=[]
        for e in range(self.e):
            row=[]
            for a in range(4):
                pose=self.local_pose(e,a);view=self.memory[e][a].view();saved=self._pose_reports[e][a]
                if saved is None or saved[0] is not view or saved[1]!=pose.tobytes():
                    source=self._physical_view(e,a) if self.obstacle_atlases is not None else view
                    saved=(view,pose.tobytes(),self.reviewer.pose(pose,source,parking=True));self._pose_reports[e][a]=saved
                report=saved[2]
                if not parking and report.reason=='parking_slope':
                    limits=self.spec.terrain
                    if report.slope>limits.driving_slope+1e-8:reason='driving_slope'
                    elif report.roughness_rms>limits.roughness_rms+1e-8:reason='contact_roughness'
                    elif report.minimum_body_clearance < -1e-8:reason='body_clearance'
                    elif report.maximum_step>limits.step_height+1e-8:reason='terrain_step'
                    else:reason=''
                    report=replace(report,passed=not reason,reason=reason)
                row.append(report)
            result.append(row)
        return result

    def _initialize_leader(self,e):
        now=float(self.core.times[e]);p=self.local_pose(e,0)
        self.scripts[e][0].decide(p,self.core.actual_velocity[e,0],self.core.previous_control[e,0],
            self._physical_view(e,0) if self.obstacle_atlases is not None else self.memory[e][0].view(),self.radio.snapshot(e,0,now,p),now,int(self.core.episodes[e]))

    def _publish(self):
        packets=[]
        for e in range(self.e):
            row=[];now=float(self.core.times[e])
            for a in range(4):
                p=self.local_pose(e,a);intent=self.intents[e][a];valid=intent is not None and now<=intent.expires_at+1e-10
                layout=self.scripts[e][a].state_dict();feature,known=self._center_lookup(e,a,p)
                row.append(RoverBroadcast(a,int(self.core.episodes[e]),now,p,self.core.actual_velocity[e,a],
                    self.core.previous_control[e,a],bool(valid and intent.hold),goal_xy=intent.goal_xy if valid else None,
                    site_center=layout['site_center'] if a==0 else None,layout_yaw=layout['layout_yaw'] if a==0 else None,
                    motion_bound=self.bounds[e][a],version=MESSAGE_V2,terrain_features=feature[0] if known[0] else None))
                if self.execution_predictions[e][a] is not None:
                    from dataclasses import replace
                    row[-1]=replace(row[-1],trajectory=self.execution_predictions[e][a])
            packets.append(row)
        self.radio.transmit(packets,self.core.poses[...,:2],self.core.times)

    def observe(self):
        observations=[]
        for e in range(self.e):
            row=[];now=float(self.core.times[e])
            for a in range(4):
                p=self.local_pose(e,a)
                row.append(self.encoder.actor(ObservationInput(self.spec,e,a,int(self.core.episodes[e]),now,
                    self.task.deadlines[e],p,self.core.actual_velocity[e,a],self.core.previous_control[e,a],
                    self.memory[e][a].view(),self.radio.snapshot(e,a,now,p),self.intents[e][a],self.feedback[e][a])))
            observations.append(row)
        actor=np.stack([[o.values for o in row] for row in observations])
        count=np.zeros(self.e,np.int64) if self.last_task is None else self.last_task['hold_count']
        critic=self.encoder.critic(self.core.poses,self.core.actual_velocity,observations,count,self.core.times,
                                   self.task.deadlines)
        actor[~self._valid]=0.;critic=critic.copy();critic[~self._valid]=0.
        return torch.tensor(actor),torch.tensor(critic)

    def pledge_speed(self,e,a,speed,until):
        bound=self.ledger.pledge(e,a,speed,float(self.core.times[e]),until,self.core.previous_control[e,a])
        self.bounds[e][a]=bound
        # Publication occurs at the next actual timestamp; duplicates do not
        # refresh an already-delivered message or silently replace its payload.
        return bound

    def enqueue(self,intents,*,primitive_ids=None):
        if len(intents)!=self.e or any(len(row)!=4 for row in intents):raise ValueError('Expected one intent per environment/vehicle')
        if self.cfg.planner_adapter=='primitive47_upstream_v1':
            primitive_ids=np.asarray(primitive_ids)
            if primitive_ids.shape!=(self.e,4) or primitive_ids.dtype.kind not in 'iu' or ((primitive_ids<0)|(primitive_ids>=47)).any():
                raise ValueError('Native environment requires original integer primitive IDs')
        elif primitive_ids is not None:raise ValueError('Native IDs supplied to a continuous environment')
        trial={}
        for e,row in enumerate(intents):
            if self._ended[e]:continue
            now=float(self.core.times[e]);delay=math.ceil(self.cfg.simulated_latency/self.spec.grid.dt-1e-12)
            for a,intent in enumerate(row):
                if (e,a) in self.pending:raise ValueError('Cannot replace a request still queued for activation')
                p=self.local_pose(e,a);times=now+np.arange(self.spec.grid.steps+delay+1)*self.spec.grid.dt
                snapshot=self.radio.snapshot(e,a,now,p,prediction_times=times)
                trial[e,a]=PendingPlan.capture(self.cfg,e,a,int(self.core.episodes[e]),p,self.core.actual_velocity[e,a],
                    self.core.previous_control[e,a],intent,self.memory[e][a].view(),snapshot.predictions,now,
                    primitive_id=None if primitive_ids is None else int(primitive_ids[e,a]))
        self.pending.update(trial)

    def script_requests(self):
        result=[]
        for e in range(self.e):
            row=[];now=float(self.core.times[e])
            for a in range(4):
                p=self.local_pose(e,a)
                row.append(self.scripts[e][a].decide(p,self.core.actual_velocity[e,a],self.core.previous_control[e,a],
                    self.memory[e][a].view(),self.radio.snapshot(e,a,now,p),now,int(self.core.episodes[e])).intent)
            result.append(row)
        return result

    def polar_requests(self,rho,beta,hold):
        rho=np.asarray(rho,float);beta=np.asarray(beta,float);hold=np.asarray(hold)
        if (rho.shape!=(self.e,4) or beta.shape!=rho.shape or hold.shape!=rho.shape or hold.dtype!=bool
                or not np.isfinite(rho).all() or not np.isfinite(beta).all() or (rho<0).any() or (rho>self.spec.rho_max).any()
                or (abs(beta)>math.pi).any()):raise ValueError('Invalid original polar/HOLD action payload')
        result=[]
        for e in range(self.e):
            row=[];now=float(self.core.times[e])
            for a in range(4):
                p=self.local_pose(e,a);goal=p[:2] if hold[e,a] else p[:2]+rho[e,a]*np.array([math.cos(p[2]+beta[e,a]),math.sin(p[2]+beta[e,a])])
                row.append(UpstreamIntent(goal,bool(hold[e,a]),now,now+self.spec.grid.request_validity))
            result.append(row)
        return result

    def _activate(self):
        keys=[key for key,item in sorted(self.pending.items()) if math.isclose(item.ready_time,float(self.core.times[key[0]]),abs_tol=1e-8)]
        if not keys:return
        inputs=[self.pending[e,a].input(self.cfg,self.local_pose(e,a),self.core.actual_velocity[e,a],
            self.core.previous_control[e,a],float(self.core.times[e]),self.workspace(e,a)) for e,a in keys]
        if self.cfg.planner_adapter=='primitive47_upstream_v1':
            from .upstream_baselines import UpstreamPrimitiveInput
            payloads=[UpstreamPrimitiveInput.for_context(item,self.pending[key].primitive_id) for key,item in zip(keys,inputs)]
        else:payloads=inputs
        outputs=self.pool.plan_batch(payloads)
        for (e,a),item,output in zip(keys,inputs,outputs):
            ticket=self.pending[e,a].intent
            self._records.append({'event':'request_activation','env_id':e,'agent_id':a,
                'original_intent':ticket,'active_intent':item.intent,'accepted':item.now<=ticket.expires_at+1e-10,
                'primitive_id':self.pending[e,a].primitive_id})
            if self.obstacle_atlases is not None:
                self._records[-1].update(planner_cost_parts=output.diagnostics.get('selected_cost_parts',{}),
                    planner_grid_upload_bytes=output.diagnostics.get('grid_upload_bytes',0))
            self.intents[e][a]=item.intent;self.plans[e][a]=output;self.feedback[e][a]=output.feedback
            del self.pending[e,a]

    def _nominal(self,e,a):
        intent=self.intents[e][a];output=self.plans[e][a];now=float(self.core.times[e])
        if intent is None or output is None or now>intent.expires_at+1e-10:return np.zeros(2),'expired_or_missing_intent'
        if intent.hold or not output.valid:return np.zeros(2),'hold_or_emergency_brake'
        trajectory=output.trajectory;index=round((now-trajectory.timestamp)/self.spec.grid.dt)
        if not 0<=index<len(trajectory.controls):return np.zeros(2),'exhausted_plan'
        target=trajectory.states[index];pose=self.local_pose(e,a);error=target[:2]-pose[:2]
        c,s=math.cos(pose[2]),math.sin(pose[2]);forward=c*error[0]+s*error[1];lateral=-s*error[0]+c*error[1]
        command=trajectory.controls[index].copy();command[0]+=.8*forward
        command[1]+=1.2*wrap(target[2]-pose[2])+2.*command[0]*lateral
        return command,'body_error_feedforward_upstream_v1'

    def _project_control(self,item):
        e,a,p,view,nominal,source,limited,peers=item
        safety_record={}
        if not self.cfg.predictive_footprint_checks:
            return AdmissionDecision(limited.copy(),False,'cost_reward_only','predictive_review_disabled',
                self.ledger.speed_cap(e,a,float(self.core.times[e])),0.,None,0),safety_record
        if self.safety is None:
            decision=admit_control(limited,self.core.previous_control[e,a],p,view,self.spec,information_time=float(self.core.times[e]),
                latency=0.,neighbors=peers,workspace=np.asarray(self.workspace(e,a)).reshape(2,2),reviewer=self.reviewer,
                hold=source in {'hold_or_emergency_brake','expired_or_missing_intent'})
        else:
            from .upstream_safety import SafetyContext
            decision=self.safety[e][a].project(SafetyContext(p,self.core.previous_control[e,a],view,peers,
                float(self.core.times[e]),np.asarray(self.workspace(e,a)).reshape(2,2),
                self.ledger.speed_cap(e,a,float(self.core.times[e])),
                source in {'hold_or_emergency_brake','expired_or_missing_intent'}),limited)
            safety_record={'qp_status':decision.qp_status,'residual':decision.residual,
                'constraint_sources':decision.constraint_sources,'qp_iterations':decision.iterations,
                'safety_seconds':decision.seconds,'safety_correction':decision.control-limited}
        return decision,safety_record

    def tick(self):
        active=~self._ended;self._activate()
        obstacle_before=([[m.view() for m in row] for row in self.memory] if self.obstacle_atlases is not None else None)
        if obstacle_before is not None:
            previous_reads=sum(m.stats['atlas_cells_read'] for row in self.memory for m in row)
            previous_physical=self.physical_query_points
        previous_reports=self._reports(parking=not self.graded_craters);count=np.zeros(self.e,np.int64) if self.last_task is None else self.last_task['hold_count']
        before=actual_facts(self.spec,self.core.poses,self.core.actual_velocity,previous_reports,count)
        command=np.zeros((self.e,4,2));gains=np.ones((self.e,4));features=np.zeros((self.e,4,5));decisions=[];height_before=np.zeros((self.e,4));prepared=[]
        for e in range(self.e):
            for a in range(4):
                p=self.local_pose(e,a);view=self.memory[e][a].view();f,known=self._center_lookup(e,a,p)
                features[e,a]=f[0];height_before[e,a]=f[0,0];gains[e,a]=terrain_gain_numpy(p,f[0],self.spec) if known[0] else .25
                if not active[e]:continue
                nominal,source=self._nominal(e,a);limited=self.ledger.limit(e,a,nominal,self.core.previous_control[e,a],float(self.core.times[e]))
                peers=self.radio.snapshot(e,a,float(self.core.times[e]),p).predictions
                prepared.append((e,a,p,view,nominal,source,limited,peers))
        jobs=[(self._project_control,item) for item in prepared]
        results=self._cpu_map(jobs) if self.safety is not None else [_cpu_call(job) for job in jobs]
        for (e,a,p,view,nominal,source,limited,peers),(decision,safety_record) in zip(prepared,results,strict=True):
            command[e,a]=self.ledger.limit(e,a,decision.control,self.core.previous_control[e,a],float(self.core.times[e]))
            verified=decision.verified;reason=decision.reason
            if self.cfg.predictive_footprint_checks and not np.allclose(command[e,a],decision.control,atol=1e-10,rtol=0):
                def gain(pose):
                    feature,supported=view.lookup(pose[:2])
                    return terrain_gain_numpy(pose,feature[0],self.spec) if supported[0] else .25
                reserve=stop_envelope(p,self.core.previous_control[e,a],command[e,a],self.spec,
                    information_time=float(self.core.times[e]),latency=0.,gain=gain)
                report=self.reviewer.trajectory(reserve.trajectory,self.core.previous_control[e,a],view,
                    neighbors=peers,workspace=np.asarray(self.workspace(e,a)).reshape(2,2),require_stop=True)
                verified=report.passed and reserve.fits_horizon;reason='' if verified else 'motion_bound:'+report.reason
            decisions.append({'env_id':e,'agent_id':a,'nominal_control':nominal.copy(),'issued_control':command[e,a].copy(),
                'tracking_source':source,'execution_verified':verified,'execution_status':decision.status,
                'execution_reason':reason,'safety_projection':self.cfg.execution_safety_version,**safety_record})
        execution=self.core.step(command,gains=gains,active_envs=active)
        self._refresh_and_predict(active)
        self._publish()
        reports=self._reports();driving=self._reports(parking=False)
        valid=np.asarray([[r.reason not in {'unknown_footprint','degenerate_contact_plane','impassable_footprint','body_clearance','terrain_step','driving_slope','contact_roughness'}
                           for r in row] for row in driving]).all(-1)
        terrain_failed = None
        if self.task.terrain_failure_terminal:
            # Known physical hazards are finite task outcomes, like collision.
            # Missing support/degenerate geometry remain unusable learning data.
            physical = {'impassable_footprint','body_clearance','terrain_step','driving_slope','contact_roughness','large_crater'}
            terrain_failed = np.asarray([[r.reason in physical for r in row] for row in driving]).any(-1)
            valid = np.asarray([[r.reason not in {'unknown_footprint','degenerate_contact_plane'}
                                 for r in row] for row in driving]).all(-1)
        self._valid&=valid
        self.last_task=self.task.advance(self.core.poses,self.core.actual_velocity,reports,self.core.times,
                                        active_envs=active,environment_valid=self._valid,terrain_failed=terrain_failed)
        after=actual_facts(self.spec,self.core.poses,self.core.actual_velocity,driving if self.graded_craters else reports,self.last_task['hold_count'])
        height_after=np.array([[self._center_lookup(e,a,self.local_pose(e,a))[0][0,0] for a,m in enumerate(row)] for e,row in enumerate(self.memory)])
        obstacle_distances=obstacle_known=None
        medium_crater_distances=None
        if obstacle_before is not None:
            measured=[[obstacle_before[e][a].obstacle_distance(self.local_pose(e,a)) for a in range(4)] for e in range(self.e)]
            obstacle_distances=np.asarray([[item[0][0] for item in row] for row in measured])
            obstacle_known=np.asarray([[item[1][0] for item in row] for row in measured])
            if self.graded_craters:
                medium_crater_distances=np.asarray([[obstacle_before[e][a].obstacle_distance(
                    self.local_pose(e,a),kind='crater_medium')[0][0] for a in range(4)] for e in range(self.e)])
        reward,terms=self.reward_tracker.advance(before,after,self.last_task,self.core.actual_velocity,features,gains,height_after-height_before,
                                                obstacle_distances=obstacle_distances,medium_crater_distances=medium_crater_distances)
        for (e,a),item in list(self.pending.items()):
            if active[e]:item.append(self.spec,self.local_pose(e,a),self.core.previous_control[e,a],self.core.actual_velocity[e,a],float(self.core.times[e]))
        self._ended|=self.last_task['terminated'];self.last_terms=terms
        for e in np.flatnonzero(self.last_task['ended_this_step']):
            for key in [key for key in self.pending if key[0]==e]:del self.pending[key]
            self._episodes.append({**({'curriculum_level':int(self.curriculum.episode_levels[e])} if self.curriculum is not None else {}),'env_id':int(e),'episode':int(self.core.episodes[e]),'terminal_reason':self.last_task['terminal_reason'][e],
                'raw_return':float(self.reward_tracker.raw_return[e]),'discounted_return':float(self.reward_tracker.discounted_return[e]),
                'reward_components':{k:float(v[e]) for k,v in self.reward_tracker.component_returns.items()},
                **({'terrain_failure_reasons':[r.reason for r in driving[e]]} if self.graded_craters else {}),
                'steps':int(self.reward_tracker.steps[e]),'hold_count':int(self.last_task['hold_count'][e])})
        record={'execution':execution,'decisions':decisions,'task':deepcopy(self.last_task),'reward':reward.copy(),'reward_components':deepcopy(terms)}
        if obstacle_before is not None:
            record.update(obstacle_distance=obstacle_distances,obstacle_footprint_known=obstacle_known,
                perception={'atlas_cells_read':sum(m.stats['atlas_cells_read'] for row in self.memory for m in row)-previous_reads,
                            'physical_query_points':self.physical_query_points-previous_physical})
        if self.graded_craters:
            record.update(terrain_failure_reasons=[[r.reason for r in row] for row in driving],
                          medium_crater_distance=medium_crater_distances)
        self._records.append(record)
        return reward,terms,record

    def step_intents(self,intents,*,max_steps=None):
        facts=self._advance_intents(intents,max_steps=max_steps)
        return self._transition_from_facts(facts,*self.observe())

    def _advance_intents(self,intents,*,max_steps=None):
        """Advance one planner slice; the decision boundary owns observation encoding."""
        steps=round(self.spec.grid.high_level_period/self.spec.grid.dt) if max_steps is None else max_steps
        if type(steps) is not int or steps<1 or steps>round(self.spec.grid.high_level_period/self.spec.grid.dt):raise ValueError('Invalid macro slice length')
        if intents is not None:self.enqueue(intents)
        reward=np.zeros(self.e);parts={};duration=np.zeros(self.e,np.int64);learning=np.zeros(self.e,bool)
        start_records=len(self._records);start_episodes=len(self._episodes)
        for _ in range(steps):
            active=~self._ended
            if not active.any():break
            r,terms,_=self.tick();weight=self.reward_tracker.settings.gamma**duration
            reward+=weight*r;duration+=active;learning|=active
            for name,value in terms.items():parts.setdefault(name,np.zeros(self.e))[:]+=weight*value
        self._activate()
        learning&=self._valid&(duration>0);reward=np.where(learning,reward,0.)
        reason=np.zeros(self.e,np.int64) if self.last_task is None else np.asarray([REASON_CODE[x] for x in self.last_task['terminal_reason']],np.int64)
        return {'reward':torch.tensor(reward,dtype=torch.float32),'duration':torch.tensor(duration),
            'terminated':torch.tensor(self._ended.copy()),'truncated':torch.zeros(self.e,dtype=torch.bool),
            'episodes':deepcopy(self._episodes[start_episodes:]),'records':deepcopy(self._records[start_records:]),
            'terminal_reason':torch.tensor(reason),'bootstrap_allowed':torch.tensor(~self._ended&learning),
            'learning_valid':torch.tensor(learning),
            'reward_components':{k:torch.tensor(np.where(learning,v,0.),dtype=torch.float32) for k,v in parts.items()}}

    def _transition_from_facts(self,facts,actor,critic):
        return UpstreamMacroTransition(**facts,terminal_actor=actor.clone(),terminal_critic=critic.clone(),
                                       actor=actor,critic=critic)

    def reset(self,ids,poses=None):
        ids=np.asarray(ids)
        if ids.ndim!=1 or ids.dtype.kind not in 'iu' or len(set(ids.tolist()))!=len(ids) or (ids<0).any() or (ids>=self.e).any():raise ValueError('Invalid asynchronous reset IDs')
        if self.curriculum is not None and poses is None:
            poses=self.curriculum.sample(ids)
            self.task.deadlines[ids]=self.curriculum.deadlines[ids]
        fresh=self._initial[ids] if poses is None else np.asarray(poses,float)
        self.core.reset(ids,fresh);self.anchors[ids]=fresh;self.anchors[ids,:,2]=0.;self.task.reset(ids);self.reward_tracker.reset(ids)
        if self._initial_control is not None:
            self.core.previous_control[ids]=self._initial_control[ids]
            self.core.actual_velocity[ids]=self._initial_control[ids]
        for e in ids.tolist():
            self.pool.reset(e);self.ledger.reset(e);self.radio.reset(e,odometry_frames=self.anchors[e])
            for a in range(4):self.memory[e][a].reset();self.scripts[e][a].reset();self.intents[e][a]=None;self.plans[e][a]=None;self.feedback[e][a]=PlannerFeedback();self.bounds[e][a]=None
            for a in range(4):
                self.execution_predictions[e][a]=None
                self._physical_views[e][a]=None
                self._driving_views[e][a]=None
                if self.safety is not None:self.safety[e][a].reset()
            for key in [key for key in self.pending if key[0]==e]:del self.pending[key]
        self._ended[ids]=False;self._valid[ids]=True
        active=np.zeros(self.e,bool);active[ids]=True;self._refresh(active)
        for e in ids.tolist():self._initialize_leader(e)
        if self.last_task is not None:
            self.last_task['hold_count'][ids]=0;self.last_task['terminated'][ids]=False;self.last_task['ended_this_step'][ids]=False
            self.last_task['terminal_reason'][ids]='none'
        self._publish()

    def discard_delivered_history(self):
        """Collector has consumed independent copies; history is not sim state."""
        self._records.clear()
        self._episodes.clear()

    def state_dict(self):
        return deepcopy({**({'curriculum':self.curriculum.state_dict()} if self.curriculum is not None else {}),
            'metadata':self.metadata,'core':self.core.state_dict(),'anchors':self.anchors,'task':self.task.state_dict(),
            'reward':self.reward_tracker.state_dict(),'planners':self.pool.state_dict(),'ledger':self.ledger.state_dict(),'radio':self.radio.state_dict(),
            'memory':[[m.state_dict() for m in row] for row in self.memory],'scripts':[[s.state_dict() for s in row] for row in self.scripts],
            'intents':self.intents,'plans':self.plans,'feedback':self.feedback,'bounds':self.bounds,
            'pending':[item.state_dict() for _,item in sorted(self.pending.items())],'ended':self._ended,'valid':self._valid,
            'safety':None if self.safety is None else [[p.state_dict() for p in row] for row in self.safety],
            'execution_predictions':self.execution_predictions,
            'last_task':self.last_task,'last_terms':self.last_terms})

    def load_state_dict(self,state):
        if not isinstance(state,dict) or state.get('metadata')!=self.metadata:raise ValueError('Environment snapshot identity mismatch')
        if self._pool_injected:raise ValueError('Injected test pools do not define production restore')
        trial=UpstreamEnvironment(self.cfg,self._initial,map_identities=self.map_identities,
            terrain_cfg=self._terrain_cfg,terrain_runtime=self._terrain_runtime,
            reward_settings=self.reward_tracker.settings,cache_dir=self.cache_dir,initial_control=self._initial_control,curriculum=deepcopy(self.curriculum))
        if trial.curriculum is not None:
            trial.curriculum.load_state_dict(state['curriculum'])
            trial.task.deadlines[:]=trial.curriculum.deadlines
        trial.core.load_state_dict(state['core']);trial.anchors=np.asarray(state['anchors'],float).copy()
        trial._physical_views=[[None]*4 for _ in range(self.e)]
        trial._driving_views=[[None]*4 for _ in range(self.e)]
        trial._pose_reports=[[None]*4 for _ in range(self.e)];trial._center_features=[[None]*4 for _ in range(self.e)]
        if trial.anchors.shape!=(self.e,4,3) or not np.isfinite(trial.anchors).all():raise ValueError('Invalid fixed frames')
        if (abs(trial.anchors[...,2])>1e-10).any():raise ValueError('Snapshot changed fixed odometry orientation')
        trial.task.load_state_dict(state['task']);trial.reward_tracker.load_state_dict(state['reward']);trial.pool.load_state_dict(state['planners'])
        task_state=trial.task.state_dict()
        if (not np.allclose(task_state['last_time'],trial.core.times,atol=1e-8,rtol=0)
                or not np.array_equal(np.asarray(state['ended']),task_state['terminal_reason']!='none')):
            raise ValueError('Environment task/time/end state disagree')
        trial.ledger.load_state_dict(state['ledger']);trial.radio=DistanceCommunicationCache(self.spec,trial.anchors,ledger=trial.ledger,message_version=MESSAGE_V2)
        trial.radio.load_state_dict(state['radio'])
        for e in range(self.e):
            for a in range(4):trial.memory[e][a].load_state_dict(state['memory'][e][a]);trial.scripts[e][a].load_state_dict(state['scripts'][e][a])
        if trial.obstacle_atlases is not None:
            for e in range(self.e):
                for a in range(4):
                    memory=trial.memory[e][a]
                    if not np.array_equal(memory.anchor,trial.anchors[e,a]) or memory.view().timestamp>trial.core.times[e]+1e-10:
                        raise ValueError('Restored grid episode frame/time mismatch')
        ended=np.asarray(state['ended']);valid=np.asarray(state['valid'])
        if ended.shape!=(self.e,) or valid.shape!=ended.shape or ended.dtype!=bool or valid.dtype!=bool:raise ValueError('Invalid environment masks')
        pending={}
        for raw in state['pending']:
            item=PendingPlan.restore(self.cfg,raw);key=(item.env_id,item.agent_id)
            if key in pending or not 0<=item.env_id<self.e or ended[item.env_id] or item.episode!=trial.core.episodes[item.env_id]:raise ValueError('Invalid restored queued identity')
            if not math.isclose(item.information_time+len(item.feedback)*self.spec.grid.dt,trial.core.times[item.env_id],abs_tol=1e-8):raise ValueError('Restored queue does not join current time')
            pending[key]=item
        for name in ('intents','plans','feedback','bounds'):
            data=deepcopy(state[name])
            if len(data)!=self.e or any(len(row)!=4 for row in data):raise ValueError('Environment vehicle state shape mismatch')
            setattr(trial,name,data)
        trial.execution_predictions=deepcopy(state.get('execution_predictions',[[None]*4 for _ in range(self.e)]))
        if trial.safety is not None:
            for e in range(self.e):
                for a in range(4):trial.safety[e][a].load_state_dict(state['safety'][e][a])
        elif state.get('safety') is not None:raise ValueError('Safety state supplied to disabled projection')
        trial.pending=pending;trial._ended=ended.copy();trial._valid=valid.copy();trial.last_task=deepcopy(state['last_task']);trial.last_terms=deepcopy(state['last_terms'])
        trial.observe() # Validate before replacing any live component.
        self.__dict__.update(trial.__dict__)
