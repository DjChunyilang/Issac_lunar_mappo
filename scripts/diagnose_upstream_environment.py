#!/usr/bin/env python3
"""Actual sensor/communication/execution/task loop; no training or gate shortcut."""
import argparse
from collections import Counter
from dataclasses import asdict,is_dataclass
import hashlib
import json
from pathlib import Path
import pickle
import time
import traceback

import numpy as np
import torch

from _common import ROOT,cfg_from_experiment
from diagnose_upstream_planners import archive_inputs,serial
from planner_evidence import EvidenceRun,json_write
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_environment import UpstreamEnvironment


def event_data(value):
    if is_dataclass(value):return event_data(asdict(value))
    if isinstance(value,dict):return {str(k):event_data(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [event_data(v) for v in value]
    return serial(value)


def initial_scene(cfg,scenario):
    poses=np.zeros((cfg.simulation.num_envs,4,3))
    if scenario=='parked':poses[...,:2]=cfg.spec.task.slots
    else:
        for e in range(len(poses)):
            spacing=12.+2*e
            poses[e,:,:2]=[[0.,0.],[0.,spacing],[spacing,0.],[spacing,spacing]]
            poses[e,:,2]=[0.,-np.pi/2,np.pi,-3*np.pi/4]
    return poses


def script_with_bounds(env):
    requests=env.script_requests()
    if env.cfg.motion_lease_version=='braking_clearance_lease_v3':
        from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_speed_lease import renew_braking_leases
        renew_braking_leases(env,requests)
        return requests
    for e,row in enumerate(requests):
        if env._ended[e]:continue
        now=float(env.core.times[e])
        for a,intent in enumerate(row):
            # A HOLD certificate is backed by an enforceable control lease,
            # never inferred from stop_intent. Active old low pledges persist.
            p=env.local_pose(e,a);snapshot=env.radio.snapshot(e,a,now,p)
            gaps=[np.linalg.norm(p[:2]-message.pose[:2])-2*env.spec.vehicle.effective_radius
                  for i,message in enumerate(snapshot.messages) if snapshot.state_valid[i]]
            clearance_cap=.8*max(0.,min(gaps,default=np.inf))/(env.spec.grid.horizon+env.spec.grid.request_validity+env.cfg.simulated_latency)
            requested=0. if intent.hold else min(env.spec.vehicle.forward_limit,clearance_cap)
            cap=max(abs(env.core.previous_control[e,a,0]),requested)
            env.pledge_speed(e,a,float(cap),now+env.spec.grid.horizon+env.cfg.simulated_latency+2*env.spec.grid.high_level_period)
    return requests


def recover_stagnation(env,requests,progress):
    events=[]
    for e,row in enumerate(requests):
        if env._ended[e]:continue
        now=float(env.core.times[e])
        for a,intent in enumerate(row):
            key=(e,a)
            if intent.hold:progress.pop(key,None);continue
            layout=env.scripts[e][a].state_dict()
            c,s=np.cos(layout['layout_yaw']),np.sin(layout['layout_yaw'])
            slot=np.asarray(layout['site_center'])+np.array([[c,-s],[s,c]])@np.asarray(env.spec.task.slots[a])
            error=float(np.linalg.norm(env.local_pose(e,a)[:2]-slot))
            prior=progress.get(key)
            if prior is None or error<prior['error']-.01:
                progress[key]={'time':now,'error':error,'resets':0 if prior is None else prior['resets']}
            elif now-prior['time']>=5.-1e-8:
                env.pool.reset_agent(e,a)
                progress[key]={'time':now,'error':error,'resets':prior['resets']+1}
                events.append({'event':'stagnation_cold_start','env_id':e,'agent_id':a,'now':now,
                    'goal_error':error,'resets':progress[key]['resets'],'original_action_unchanged':True})
    return events


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/experiment/upstream_scale_s03_environment_v2.yaml')
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--scenario',choices=('parked','spawn_flat'),default='spawn_flat')
    parser.add_argument('--max-steps',type=int,default=None)
    parser.add_argument('--restore-state',help='Explicit trusted local diagnostic snapshot; starts a new run with its own source identity')
    parser.add_argument('--cold-start',action='store_true',help='Explicitly reset planner warm states after restoring actual environment data')
    parser.add_argument('--recover-stagnation',action='store_true',help='Explore bounded per-agent cold recovery after 5 s without 1 cm target progress')
    args=parser.parse_args();torch.set_num_threads(1)
    if args.cold_start and not args.restore_state:parser.error('--cold-start requires an explicit restored diagnostic state')
    cfg=cfg_from_experiment(args.config)
    maximum=cfg.simulation.max_episode_steps if args.max_steps is None else args.max_steps
    if maximum<1 or maximum>cfg.simulation.max_episode_steps:raise ValueError('Invalid diagnostic step budget')
    settings={'stage':'S03_actual_environment_exploration','scenario':args.scenario,
              'max_steps':maximum,'torch_threads':1,'bound_policy':'script_enforced_clearance_lease_v2',
              'formal_training':False,'safety_projection':cfg.execution_safety_version,
              'restore_state':str(Path(args.restore_state).resolve()) if args.restore_state else None,
              'warm_state_policy':'explicit_cold_start_after_restore_v1' if args.cold_start else 'preserve_public_warm_state',
              'recovery':'five_second_progress_cold_start_v1' if args.recover_stagnation else 'disabled'}
    run=EvidenceRun(ROOT/'outputs/runs/upstream_scale_s03'/args.run_id,args.config,settings=settings)
    run.resolved(cfg);archive_inputs(run,cfg)
    poses=initial_scene(cfg,args.scenario);json_write(run.path/'config/scene.json',{'initial_poses':poses.tolist(),'map_id':'flat_proxy_actual_sensor'})
    env=UpstreamEnvironment(cfg,poses,map_identities=('flat_proxy_actual_sensor',))
    if args.restore_state:
        path=Path(args.restore_state).resolve()
        if not path.is_relative_to((ROOT/'outputs/runs').resolve()):raise ValueError('Only explicit local diagnostic snapshots are accepted')
        data=path.read_bytes();checksum=hashlib.sha256(data).hexdigest()
        if json.loads(path.with_suffix(path.suffix+'.json').read_text())['sha256']!=checksum:raise ValueError('Diagnostic snapshot checksum mismatch')
        env.load_state_dict(pickle.loads(data))
        json_write(run.path/'config/restored_state.json',{'path':str(path),'sha256':checksum,
            'restored_simulation_seconds':env.core.times.tolist(),'scope':'cross-source exploratory continuation, not a frozen S05 gate'})
        if args.cold_start:
            env.pool.reset()
            json_write(run.path/'config/cold_start.json',{'scope':'planner public reset only; actual state, memory, script and leases preserved',
                'simulation_seconds':env.core.times.tolist(),'action_payload_rewritten':False})
    initial_times=env.core.times.copy()
    elapsed=0;episodes=[];statuses=Counter();rejected=Counter();times=[];recorded_samples=0;progress={};recovery_events=[]
    start=time.perf_counter()
    try:
        with (run.path/'metrics/trace.jsonl').open('x') as trace:
            while elapsed<maximum and not env._ended.all():
                requests=script_with_bounds(env)
                recovery=recover_stagnation(env,requests,progress) if args.recover_stagnation else []
                recovery_events.extend(recovery);tick=time.perf_counter()
                transition=env.step_intents(requests,max_steps=min(5,maximum-elapsed));times.append(time.perf_counter()-tick)
                elapsed+=int(transition.duration.max());episodes.extend(transition.episodes)
                for record in transition.records:
                    for decision in record.get('decisions',[]):statuses[decision['execution_status']]+=1
                for row in env.plans:
                    for output in row:
                        if output is not None and not output.valid:rejected[output.fallback_reason]+=1
                record={'step':elapsed,'simulation_time':env.core.times,'poses':env.core.poses,
                        'actual_velocity':env.core.actual_velocity,'macro_reward':transition.reward,
                        'macro_duration':transition.duration,'terminal_reason':transition.terminal_reason,
                        'bootstrap_allowed':transition.bootstrap_allowed,'learning_valid':transition.learning_valid,
                        'records':transition.records,'plans':env.plans,'wall_seconds':times[-1],'recovery':recovery}
                trace.write(json.dumps(event_data(record),ensure_ascii=False,allow_nan=False)+'\n');trace.flush()
                run.snapshot(f'environment_step_{elapsed:04d}.pkl',env.state_dict())
                if args.recover_stagnation:run.snapshot(f'recovery_step_{elapsed:04d}.pkl',progress)
                if elapsed<=5 or elapsed%25==0:
                    diagnostics=env.pool.export_diagnostics()
                    if diagnostics:
                        run.snapshot(f'samples_step_{elapsed:04d}.pkl',diagnostics);recorded_samples+=1
                if elapsed%25==0 or env._ended.any():
                    print(json.dumps({'steps':elapsed,'time':env.core.times.tolist(),'dmax':env.last_task['dmax'].tolist(),
                        'hold':env.last_task['hold_count'].tolist(),'ended':env._ended.tolist(),'seconds':round(time.perf_counter()-start,2)}),flush=True)
        summary={'scope':'actual_environment_exploration','scenario':args.scenario,'steps':elapsed,
                 'simulation_seconds':env.core.times.tolist(),'wall_seconds':time.perf_counter()-start,
                 'restored_simulation_seconds':initial_times.tolist(),
                 'macro_p50_seconds':float(np.median(times)),'macro_p95_seconds':float(np.quantile(times,.95)),
                 'episodes':episodes,'initial_poses':poses.tolist(),'final_poses':env.core.poses.tolist(),
                 'execution_status_counts':dict(statuses),'planner_rejection_counts':dict(rejected),
                 'all_episodes_finished':bool(env._ended.all()),'task_success_count':sum(e['terminal_reason']=='success' for e in episodes),
                 'early_diagnostic_cut':bool(not env._ended.all()),'sample_events':recorded_samples,
                 'recovery_events':recovery_events,
                 'mppi_samples':cfg.mppi_samples,'mppi_iterations':cfg.mppi_iterations,
                 'source_snapshot':run.identity['source_snapshot'],'environment_metadata':env.metadata,
                 's03_accepted':False,'s05_accepted':False,'formal_training_ready':False,'optimizer_updates':0}
        json_write(run.path/'metrics/environment_summary.json',summary)
        print(json.dumps({k:summary[k] for k in ('steps','wall_seconds','task_success_count','early_diagnostic_cut','planner_rejection_counts')},ensure_ascii=False),flush=True)
    except Exception:
        failure=traceback.format_exc();(run.path/'logs/failure.log').write_text(failure)
        raise


if __name__=='__main__':main()
