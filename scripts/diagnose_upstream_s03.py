#!/usr/bin/env python3
"""Frozen S03 component evidence; not the complete migrated environment gate.

Dense sensing, private memory, legal distance messages, enforced speed pledges,
actual queue feedback, real MPPI and actual parking are exercised together.
Initial states are a square fixture, not the S05 10--50 m birth distribution.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch

from _common import ROOT, cfg_from_experiment
from diagnose_upstream_planners import archive_inputs,serial,TEST_FILES as S02_FILES
from planner_evidence import EvidenceRun,file_sha256,json_write,source_snapshot
from upstream_evidence import require_contract_tests
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_communication import MESSAGE_VERSION,MESSAGE_V2,DistanceCommunicationCache,MotionBoundLedger,RoverBroadcast
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.contracts import PlannerFeedback
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_factory import UpstreamPlannerPool
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_gathering import LeaderSquareScript
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_planning import UpstreamIntent,UpstreamPlannerInput
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_proxy import UpstreamActuatorProxy
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_review_cache import PoseCachedReviewer
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_sensing import DenseSensor,DenseTerrainMemory,world_to_odometry
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_task import UpstreamTask
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_timing import ActualControlHistory

TEST_FILES=('tests/test_upstream_sensing.py','tests/test_upstream_task.py','tests/test_upstream_communication.py',
    'tests/test_upstream_timing.py','tests/test_upstream_gathering.py','tests/test_upstream_review_cache.py',
    'tests/test_upstream_mppi_precision.py',
    'tests/test_upstream_observation.py',
    'tests/test_lunar_training_maps.py','tests/test_terrain_observation_validation.py')
DRIVER_FILES=('tests/test_upstream_s03_diagnostics.py',)
RELATED_FILES=(*S02_FILES,'tests/test_p0_learning_contracts.py','tests/test_polar_hold.py',
    'tests/test_local_goal_nmpc.py','tests/test_planner_protocol.py','tests/test_planner_baselines.py',
    'tests/test_gathering_failure_repairs.py','tests/test_stage5_diagnostics.py','tests/test_observation.py',
    'tests/test_reward.py','tests/test_termination.py','tests/test_tiered_communication.py',
    'tests/test_action_ablation.py','tests/test_t11_action_diagnostics.py')
SETTINGS={'stage':'S03_components_with_legal_communication_actual_history_v1','torch_threads':1,
    'initial_layout':'already_gathered_square_fixture','full_runtime_environment_connected':False,
    'training_interactions':0,'optimizer_updates':0}


def create_run(path,config,*,resume=False,contracts_from=None):
    cfg=cfg_from_experiment(config)
    if (cfg.simulation.num_envs!=2 or cfg.timing_version!='actual_control_history_v1'
            or cfg.review_acceleration_version!='pose_scope_cache_v1' or cfg.planner_adapter not in {'mppi_upstream_v1','mppi_upstream_v2'}
            or cfg.simulated_latency!=.5 or not cfg.training_paused):
        raise ValueError('S03 component diagnostic requires two environments, explicit history/cache and offline 0.5 s MPPI config')
    settings=dict(SETTINGS)
    if contracts_from is not None:settings['contracts_reference']=str(Path(contracts_from).resolve())
    run=EvidenceRun(path,config,resume=resume,settings=settings);run.resolved(cfg)
    return cfg,run


def prepare(run,cfg,*,contracts_from=None):
    archive_inputs(run,cfg)
    contracts=DRIVER_FILES if contracts_from is not None else (*TEST_FILES,*DRIVER_FILES)
    for name,files in [('contracts',contracts),('related',RELATED_FILES)]:
        target=run.path/f'metrics/{name}.source.json'
        if not target.exists():json_write(target,{'source_snapshot':run.identity['source_snapshot'],
            'test_sources':{f:file_sha256(ROOT/f) for f in files}})


def component_radio(cfg,anchors,ledger):
    """The same explicit protocol factory is used for cold start and restore."""
    version=MESSAGE_V2 if cfg.observation_execution_version!='not_connected' else MESSAGE_VERSION
    return DistanceCommunicationCache(cfg.spec,anchors,ledger=ledger,message_version=version)


def validate_reuse_sources(current,previous):
    if not previous or set(current)!=set(previous):raise ValueError('Prior test source coverage differs')
    changed=sorted(name for name in current if current[name]!=previous[name])
    if any(name!='scripts/diagnose_upstream_s03.py' for name in changed):
        raise ValueError('Actual production/test dependency changed; execute a new full contract suite')
    return changed


def prior_contract_evidence(run,path):
    """Reuse executed JUnit only for a repair confined to this diagnostic driver.

    Reports retain the original paths/run and do not pretend these tests ran in
    the new run. Every other producing Python file, archived byte, test hash,
    configuration, dependency and setting must match. Driver contracts execute
    separately in the new run and the actual component loop is rerun.
    """
    path=Path(path).resolve();manifest=json.loads((path/'run_manifest.json').read_text())
    identity=manifest['identity']
    if path==run.path.resolve():raise ValueError('Prior evidence cannot reference the current run')
    changed=validate_reuse_sources(run.identity['source_snapshot'],identity['source_snapshot'])
    current_settings={k:v for k,v in run.identity['settings'].items() if k!='contracts_reference'}
    prior_settings={k:v for k,v in identity['settings'].items() if k!='contracts_reference'}
    if (identity['resolved_config']!=run.identity['resolved_config'] or identity['dependencies']!=run.identity['dependencies']
            or current_settings!=prior_settings):raise ValueError('Prior test configuration/dependency/settings mismatch')
    for name,sha in identity['source_snapshot'].items():
        if file_sha256(path/'config/source_snapshot'/name)!=sha:raise ValueError('Prior source archive mismatch')
    reference=SimpleNamespace(path=path,identity=identity)
    tests=require_contract_tests(path/'metrics/contracts.xml',reference,TEST_FILES)
    related=require_contract_tests(path/'metrics/related.xml',reference,RELATED_FILES)
    for report in (tests,related):report.update(reused_from_run=str(path),executed_in_current_run=False,
        reused_dependency_audit='identical_except_current_diagnostic_driver',changed_source_paths=changed)
    return tests,related


def component_loop(run,cfg):
    spec=cfg.spec;torch.set_num_threads(1)
    initial=np.zeros((2,4,3));initial[...,:2]=spec.task.slots;initial[1,:,0]+=20.
    anchors=initial.copy();core=UpstreamActuatorProxy(cfg,initial);task=UpstreamTask(spec,2)
    sensor=DenseSensor(spec,None,map_identities=('flat_proxy_s03_component_fixture',))
    memory=[[DenseTerrainMemory(spec) for _ in range(4)] for _ in range(2)]
    scripts=[[LeaderSquareScript(spec,e,a) for a in range(4)] for e in range(2)]
    observation_connected=cfg.observation_execution_version!='not_connected'
    message_version=MESSAGE_V2 if observation_connected else MESSAGE_VERSION
    ledger=MotionBoundLedger(spec,2);radio=component_radio(cfg,anchors,ledger)
    reviewer=PoseCachedReviewer(spec);pool=UpstreamPlannerPool(cfg);trace=[];last_task=None
    observation_intents=[[None]*4 for _ in range(2)]
    observation_feedback=[[PlannerFeedback() for _ in range(4)] for _ in range(2)]
    observation_records=[]
    if observation_connected:
        from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_observation import ObservationInput,UpstreamObservationEncoder
        from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.learning import RecurrentPolicy
        encoder=UpstreamObservationEncoder(spec);torch.manual_seed(cfg.seed)
        backbone=RecurrentPolicy(991,observation_version='local_intent_clock_v3',action_version='polar_hold_v2',rho_max=spec.rho_max).eval()
        backbone.requires_grad_(False)
    bounds=[[ledger.pledge(e,a,.05,0.,30.,[0.,0.]) for a in range(4)] for e in range(2)]

    def local_pose(e,a,proxy=core):
        p=proxy.poses[e,a];frame=anchors[e,a]
        return np.r_[world_to_odometry(p[:2][None],frame)[0],p[2]-frame[2]]

    def sense(active=None):
        active=np.ones(2,bool) if active is None else active
        frames=sensor.sample(core.poses,anchors,core.times)
        for e in range(2):
            for a in range(4):
                if active[e]:memory[e][a].observe(frames[e][a])
        return frames

    def publish():
        row=[]
        for e in range(2):
            messages=[]
            for a in range(4):
                state=scripts[e][a].state_dict();now=float(core.times[e]);p=local_pose(e,a)
                intent=observation_intents[e][a]
                intent_valid=intent is not None and now<=intent.expires_at+1e-10
                feature,known=memory[e][a].view().lookup(p[:2])
                messages.append(RoverBroadcast(a,int(core.episodes[e]),now,p,core.actual_velocity[e,a],
                    core.previous_control[e,a],bool(intent_valid and intent.hold),
                    goal_xy=intent.goal_xy if intent_valid else None,
                    site_center=state['site_center'] if a==0 else None,
                    layout_yaw=state['layout_yaw'] if a==0 else None,motion_bound=bounds[e][a],version=message_version,
                    terrain_features=feature[0] if observation_connected and known[0] else None))
            row.append(messages)
        radio.transmit(row,core.poses[...,:2],core.times)

    def capture_observation(phase):
        if not observation_connected:return
        observations=[]
        for e in range(2):
            row=[];now=float(core.times[e])
            for a in range(4):
                p=local_pose(e,a)
                row.append(encoder.actor(ObservationInput(spec,e,a,int(core.episodes[e]),now,
                    cfg.simulation.episode_length_s,p,core.actual_velocity[e,a],core.previous_control[e,a],
                    memory[e][a].view(),radio.snapshot(e,a,now,p),observation_intents[e][a],observation_feedback[e][a])))
            observations.append(row)
        actor=np.stack([[o.values for o in row] for row in observations])
        counts=np.zeros(2,np.int64) if last_task is None else last_task['hold_count']
        critic=encoder.critic(core.poses,core.actual_velocity,observations,counts,core.times,cfg.simulation.episode_length_s)
        with torch.no_grad():
            encoded=backbone.encode(torch.tensor(actor));_,hidden=backbone.distribution(torch.tensor(actor),torch.zeros((2,4,128)))
            value=backbone.critic(torch.tensor(critic))
        if not (torch.isfinite(encoded).all() and torch.isfinite(hidden).all() and torch.isfinite(value).all()):
            raise AssertionError('Actual private observations cannot be consumed by the shared backbone')
        record=serial({'phase':phase,'actor':actor,'critic':critic,'metadata':encoder.metadata,
            'times':core.times,'sender_indices':[[o.sender_indices for o in row] for row in observations],
            'model_hidden_mode':'fresh_zero_hidden_forward_only','sampled_policy_actions':0,'optimizer_updates':0})
        observation_records.append(record)
        json_write(run.path/f'samples/observation_{len(observation_records):04d}.json',record)

    def actual_step(desired,phase):
        nonlocal last_task
        active=np.ones(2,bool) if last_task is None else ~last_task['terminated']
        applied=np.empty((2,4,2))
        for e in range(2):
            for a in range(4):applied[e,a]=ledger.limit(e,a,desired[e,a],core.previous_control[e,a],float(core.times[e]))
        execution=core.step(applied,active_envs=active)
        np.testing.assert_allclose(execution['applied_control'],applied,atol=1e-10,rtol=1e-8)
        if execution['geometry_overlap'].any() or execution['clearance_violation'].any():raise AssertionError('Actual envelope/clearance violation')
        frames=sense(active);publish()
        reports=[[reviewer.pose(local_pose(e,a),memory[e][a].view(),parking=True) for a in range(4)] for e in range(2)]
        last_task=task.advance(core.poses,core.actual_velocity,reports,core.times,active_envs=active)
        trace.append(serial({'phase':phase,'execution':execution,'task':last_task,
            'communication':[[radio.snapshot(e,a,float(core.times[e]),local_pose(e,a)).prediction_metadata
                              for a in range(4)] for e in range(2)]}))
        if len(trace)%5==0:json_write(run.path/f'samples/actual_trace_{len(trace):04d}.json',trace[-5:])
        if len(trace)%5==0 or last_task['ended_this_step'].any():capture_observation(phase)
        return execution,frames

    def freeze_information(intents):
        info={'times':core.times.copy(),'poses':[[local_pose(e,a) for a in range(4)] for e in range(2)],
              'previous':core.previous_control.copy(),'terrain':[[m.view() for m in row] for row in memory],
              'intents':intents,'feedback':[],'neighbors':[]}
        for e in range(2):
            row=[];now=float(core.times[e]);grid=now+np.arange(44)*.2
            for a in range(4):row.append(radio.snapshot(e,a,now,local_pose(e,a),prediction_times=grid).predictions)
            info['neighbors'].append(row)
        return info

    def queue_step(info,desired,phase):
        execution,frames=actual_step(desired,phase);info['feedback'].append(execution)
        return frames

    def activate(info):
        inputs=[]
        for e in range(2):
            for a in range(4):
                controls=np.stack([f['applied_control'][e,a] for f in info['feedback']])
                velocities=np.stack([f['actual_velocity'][e,a] for f in info['feedback']])
                poses=np.stack([info['poses'][e][a],*[np.r_[world_to_odometry(f['after'][e,a,:2][None],anchors[e,a])[0],
                                                            f['after'][e,a,2]-anchors[e,a,2]] for f in info['feedback']]])
                history=ActualControlHistory(e,a,spec.fingerprint,float(info['times'][e])+np.arange(len(controls)+1)*.2,
                    poses,controls,velocities,info['previous'][e,a])
                now=float(core.times[e]);low=np.array([-100.,-100.])-anchors[e,a,:2];high=np.array([100.,100.])-anchors[e,a,:2]
                inputs.append(UpstreamPlannerInput(e,a,local_pose(e,a),core.actual_velocity[e,a],info['intents'][e][a],
                    info['terrain'][e][a],info['neighbors'][e][a],core.previous_control[e,a],now,now+np.arange(41)*.2,
                    tuple(np.r_[low,high]),spec,information_pose=info['poses'][e][a],information_time=float(info['times'][e]),actual_history=history))
        outputs=pool.plan_batch(inputs)
        for e in range(2):
            for a in range(4):
                observation_intents[e][a]=info['intents'][e][a]
                observation_feedback[e][a]=outputs[4*e+a].feedback
        return inputs,outputs

    frames=sense();publish()
    capture_observation('initial_private_sensor_frame')
    for e in range(2):
        now=float(core.times[e]);p=local_pose(e,0)
        scripts[e][0].decide(p,[0.,0.],[0.,0.],memory[e][0].view(),radio.snapshot(e,0,now,p),now,0)
    # Fresh delivery of the leader layout uses a later publication time; a
    # duplicate time-zero packet cannot refresh payloads/ages.
    intents=[[UpstreamIntent([.3,0.],False,0.,1.,speed_cap=.05) for _ in range(4)] for _ in range(2)]
    info=freeze_information(intents)
    for _ in range(3):queue_step(info,np.zeros((2,4,2)),'initial_zero_control_latency')
    inputs,outputs=activate(info)
    first_valid=sum(o.valid for o in outputs)
    sample_export=pool.export_diagnostics()
    run.snapshot('move_raw_samples_final_iteration.pkl',sample_export)
    if len(sample_export['planners'])!=8 or any(r['samples'] is None or r['samples']['raw_controls'].shape!=(cfg.mppi_samples,40,2)
                                              for r in sample_export['planners']):
        raise AssertionError('MPPI sampling evidence does not cover the eight requested instances')
    raw=pool.state_dict() # public state; no solver objects
    run.snapshot('move_planner_state.pkl',raw);run.snapshot('initial_dense_frames.pkl',frames)
    json_write(run.path/'samples/move_outputs.json',serial([vars(o) | {'trajectory':vars(o.trajectory),'feedback':vars(o.feedback)} for o in outputs]))
    start=core.poses.copy()
    for step in range(3):actual_step(np.stack([o.trajectory.controls[step] for o in outputs]).reshape(2,4,2),'active_move')
    progressed=np.linalg.norm(core.poses[...,:2]-start[...,:2],axis=-1)
    if not np.any(progressed>1e-5):raise AssertionError('Actual MPPI MOVE produced no measurable progress')
    publish();intents=[]
    for e in range(2):
        row=[];now=float(core.times[e])
        for a in range(4):
            p=local_pose(e,a);decision=scripts[e][a].decide(p,core.actual_velocity[e,a],core.previous_control[e,a],
                memory[e][a].view(),radio.snapshot(e,a,now,p),now,0)
            if not decision.intent.hold:raise AssertionError('Small actual displacement should be captured by legal script')
            row.append(decision.intent)
        intents.append(row)
    hold_info=freeze_information(intents)
    for _ in range(3):queue_step(hold_info,np.zeros((2,4,2)),'expired_move_braking_during_queue')
    _,hold_outputs=activate(hold_info)
    if not all(o.valid and o.feedback.status=='hold' for o in hold_outputs):raise AssertionError('Legal moving HOLD review failed')
    json_write(run.path/'samples/hold_outputs.json',serial([{'trajectory':vars(o.trajectory),'feedback':vars(o.feedback),
        'valid':o.valid,'diagnostics':o.diagnostics} for o in hold_outputs]))
    snapshot={'proxy':core.state_dict(),'task':task.state_dict(),'pool':pool.state_dict(),
        'memory':[[m.state_dict() for m in row] for row in memory],
        'scripts':[[s.state_dict() for s in row] for row in scripts],
        'ledger':ledger.state_dict(),'radio':radio.state_dict()}
    run.snapshot('public_components_state.pkl',snapshot)
    restored_core=UpstreamActuatorProxy(cfg,initial);restored_core.load_state_dict(snapshot['proxy'])
    restored_task=UpstreamTask(spec,2);restored_task.load_state_dict(snapshot['task'])
    restored_ledger=MotionBoundLedger(spec,2);restored_ledger.load_state_dict(snapshot['ledger'])
    restored_radio=component_radio(cfg,anchors,restored_ledger);restored_radio.load_state_dict(snapshot['radio'])
    restored_pool=UpstreamPlannerPool(cfg);restored_pool.load_state_dict(snapshot['pool'])
    for e in range(2):
        for a in range(4):
            restored_script=LeaderSquareScript(spec,e,a);restored_script.load_state_dict(snapshot['scripts'][e][a])
            restored_memory=DenseTerrainMemory(spec);restored_memory.load_state_dict(snapshot['memory'][e][a])
            np.testing.assert_array_equal(restored_memory.view().snapshot().xy,memory[e][a].view().snapshot().xy)
            original=radio.snapshot(e,a,float(core.times[e]),local_pose(e,a))
            replay=restored_radio.snapshot(e,a,float(core.times[e]),local_pose(e,a))
            np.testing.assert_array_equal(original.sender_indices,replay.sender_indices)
            for p,q in zip(original.predictions,replay.predictions):np.testing.assert_array_equal(p.radii,q.radii)
    replay_steps=0
    for step in range(40):
        if last_task['terminated'].all():break
        target=np.stack([o.trajectory.controls[step] for o in hold_outputs]).reshape(2,4,2)
        if (core.times>intents[0][0].expires_at+1e-10).any():target=np.zeros_like(target)
        execution,_=actual_step(target,'hold_or_expired_hold')
        restored_core.step(target,active_envs=execution['active_envs'])
        reports=[[reviewer.pose(np.r_[world_to_odometry(restored_core.poses[e,a,:2][None],anchors[e,a])[0],
                                      restored_core.poses[e,a,2]-anchors[e,a,2]],memory[e][a].view(),parking=True)
                  for a in range(4)] for e in range(2)]
        replay=restored_task.advance(restored_core.poses,restored_core.actual_velocity,reports,restored_core.times,
                                    active_envs=execution['active_envs'])
        for name in ('poses','previous_control','actual_velocity','times'):
            np.testing.assert_array_equal(getattr(core,name),getattr(restored_core,name))
        for name in last_task:np.testing.assert_array_equal(last_task[name],replay[name])
        replay_steps+=1
    if last_task['terminal_reason'].tolist()!=['success','success']:raise AssertionError('Actual five-second parking did not complete')
    other=deepcopy(radio.state_dict());core.reset([0],initial[:1]);task.reset([0]);pool.reset(0);ledger.reset(0);radio.reset(0)
    for a in range(4):memory[0][a].reset();scripts[0][a].reset()
    np.testing.assert_array_equal(radio.state_dict()['last_sent'][1],other['last_sent'][1])
    if radio.snapshot(0,0,0.,[0.,0.,0.]).messages:raise AssertionError('Reset retained stale messages')
    json_write(run.path/'samples/actual_component_trace.json',trace)
    return {'actual_batch_size':8,'actual_mppi_sampling_batch_size':8,'samples':cfg.mppi_samples,
        'iterations':cfg.mppi_iterations,'move_valid_count':first_valid,'actual_move_distances':progressed.tolist(),
        'sampled_calls_recorded':8,'recorded_iterations_per_call':1,'total_iterations_per_call':cfg.mppi_iterations,
        'steps':len(trace),'parking_hold_count':last_task['hold_count'].tolist(),
        'terminal_reason':last_task['terminal_reason'].tolist(),'public_restore_steps_identical':replay_steps,
        'communication_restore_identical':True,'asynchronous_reset_isolated':True,
        'actual_motion_promises_enforced':True,'information_timing_version':cfg.timing_version,
        'initial_layout':'already_gathered_square_fixture','full_birth_approach_executed':False,
        'safety_projection_connected':False,'reward_actor_learning_connected':False,
        'actor_observation_connected':observation_connected,'observation_frames':len(observation_records),
        'shared_backbone_forward_pass':observation_connected,'recurrent_rollout_connected':False,
        's03_full_acceptance_pass':False,'gate_b_pass':False,'t11_pass':False}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default='configs/experiment/upstream_scale_s03.yaml')
    p.add_argument('--contracts-from',help='Reuse already-executed suites only when this diagnostic driver is the sole source change')
    p.add_argument('--run-dir',required=True);p.add_argument('--prepare-only',action='store_true');p.add_argument('--resume',action='store_true')
    args=p.parse_args();cfg,run=create_run(args.run_dir,args.config,resume=args.resume,contracts_from=args.contracts_from)
    prepare(run,cfg,contracts_from=args.contracts_from)
    if args.prepare_only:print(run.path,flush=True);return
    if args.contracts_from:
        tests,related=prior_contract_evidence(run,args.contracts_from)
        driver=require_contract_tests(run.path/'metrics/contracts.xml',run,DRIVER_FILES)
    else:
        tests=require_contract_tests(run.path/'metrics/contracts.xml',run,(*TEST_FILES,*DRIVER_FILES))
        related=require_contract_tests(run.path/'metrics/related.xml',run,RELATED_FILES);driver=None
    result=component_loop(run,cfg)
    if source_snapshot()!=run.identity['source_snapshot']:raise ValueError('Source changed during S03 producer; preserve evidence and start a new run')
    json_write(run.path/'metrics/component_summary.json',serial({'contract_tests':tests,'related_tests':related,'driver_tests':driver,
        'components':result,'producer_sha256':file_sha256(__file__),'training_interactions':0,'optimizer_updates':0}))
    print('S03_COMPONENTS_PASS',serial(result),flush=True)


if __name__=='__main__':main()
