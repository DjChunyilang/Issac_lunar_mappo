#!/usr/bin/env python3
"""Bounded public-state NMPC experiment; never counts as a full S05 episode."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np
import torch

from _common import ROOT
from diagnose_upstream_environment import event_data,script_with_bounds,recover_stagnation
from diagnose_upstream_planners import archive_inputs
from planner_evidence import file_sha256,json_write,source_snapshot
from upstream_run_state import restore_state,save_state
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_environment import UpstreamEnvironment
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_spec import parse_upstream_experiment
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_execution import control_error
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_speed_lease import braking_clearance_limit


def slot_errors(env):
    result=[]
    for e in range(env.e):
        row=[]
        for a in range(4):
            state=env.scripts[e][a].state_dict();yaw=state['layout_yaw'];c,s=np.cos(yaw),np.sin(yaw)
            goal=np.asarray(state['site_center'])+np.array([[c,-s],[s,c]])@np.asarray(env.spec.task.slots[a])
            row.append(float(np.linalg.norm(env.local_pose(e,a)[:2]-goal)))
        result.append(row)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-run',required=True);p.add_argument('--snapshot',required=True)
    p.add_argument('--scene-config',required=True);p.add_argument('--run-id',required=True)
    p.add_argument('--mode',choices=('baseline','lease_cap','braking_lease','braking_lease_cap'),required=True)
    p.add_argument('--macro-steps',type=int,default=15)
    args=p.parse_args();torch.set_num_threads(1)
    source=Path(args.source_run).resolve()
    if not source.is_relative_to((ROOT/'outputs/runs').resolve()):raise ValueError('Expected an owned source run')
    snapshot=source/'state'/args.snapshot;scene=json.loads((source/'config'/args.scene_config).read_text())
    cfg=parse_upstream_experiment(scene['effective_configuration'],ROOT)
    path=ROOT/'outputs/runs/upstream_scale_t11'/args.run_id
    if path.exists():raise ValueError('Diagnostic output already exists')
    for name in ('config','metrics','logs','state'):(path/name).mkdir(parents=True)
    identity={'source_snapshot':source_snapshot(),'dependencies':{'torch':torch.__version__},
              'source_state':str(snapshot),'source_state_sha256':file_sha256(snapshot),'contracts':cfg.metadata,
              'mode':args.mode,'scope':'bounded diagnostic request constraint; no full episode claim',
              'macro_steps':args.macro_steps,'formal_training':False}
    json_write(path/'run_manifest.json',identity)
    run=SimpleNamespace(path=path,identity=identity,planner_device=cfg.planner_device)
    archive_inputs(run,cfg);json_write(path/'config/effective.json',scene)
    saved=restore_state(snapshot)
    env=UpstreamEnvironment(cfg,np.asarray(scene['initial_poses']),map_identities=('flat_proxy_actual_sensor',),cache_dir=source/'state/solver_cache')
    env.load_state_dict(saved['environment']);progress=saved['progress'];initial=slot_errors(env)
    rows=[];violations=[];episodes=[];start=time.perf_counter()
    with (path/'metrics/trace.jsonl').open('x') as stream:
        for step in range(args.macro_steps):
            if env._ended.all():break
            if args.mode.startswith('braking_lease'):
                requests=env.script_requests()
                for e,row in enumerate(requests):
                    if env._ended[e]:continue
                    now=float(env.core.times[e])
                    for a,intent in enumerate(row):
                        pose=env.local_pose(e,a);messages=env.radio.snapshot(e,a,now,pose)
                        gaps=[np.linalg.norm(pose[:2]-message.pose[:2])-2*env.spec.vehicle.effective_radius
                              for i,message in enumerate(messages.messages) if messages.state_valid[i]]
                        proposed=0. if intent.hold else braking_clearance_limit(env.spec,min(gaps,default=np.inf),cfg.simulated_latency)
                        cap=max(abs(env.core.previous_control[e,a,0]),proposed)
                        env.pledge_speed(e,a,cap,now+env.spec.grid.horizon+cfg.simulated_latency+2*env.spec.grid.high_level_period)
            else:requests=script_with_bounds(env)
            # The experiment changes explicit non-learning script constraints.
            # It neither modifies a sampled policy action nor cancels a lease.
            if args.mode in {'lease_cap','braking_lease_cap'}:
                requests=[[replace(intent,speed_cap=min(env.ledger.speed_cap(e,a,float(env.core.times[e])),
                    cfg.spec.vehicle.forward_limit if intent.speed_cap is None else intent.speed_cap))
                    if not intent.hold else intent for a,intent in enumerate(row)] for e,row in enumerate(requests)]
            recoveries=recover_stagnation(env,requests,progress)
            transition=env.step_intents(requests);episodes.extend(transition.episodes)
            for record in transition.records:
                if 'execution' not in record:continue
                execution=record['execution']
                for e in range(env.e):
                    if not execution['active_envs'][e]:continue
                    if execution['geometry_overlap'][e]:violations.append('geometry_overlap')
                    if execution['clearance_violation'][e]:violations.append('clearance')
                    for a in range(4):
                        reason=control_error(execution['applied_control'][e,a],execution['previous_control'][e,a],cfg.spec)
                        if reason:violations.append(reason)
            summary={'macro':step+1,'times':env.core.times.tolist(),'slot_errors':slot_errors(env),
                     'controls':env.core.previous_control.tolist(),'recoveries':recoveries}
            rows.append(summary);stream.write(json.dumps(event_data({**summary,'records':transition.records}),allow_nan=False)+'\n');stream.flush()
            print(json.dumps(summary),flush=True)
    save_state(run,'final',{'environment':env.state_dict(),'progress':progress})
    result={'mode':args.mode,'initial_errors':initial,'final_errors':slot_errors(env),'rows':rows,
            'episodes':episodes,'violations':violations,'wall_seconds':time.perf_counter()-start,
            'complete_episode_acceptance':False,'formal_training':False}
    json_write(path/'metrics/summary.json',event_data(result))


if __name__=='__main__':main()
