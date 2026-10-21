#!/usr/bin/env python3
"""Full S05 frozen seven-setting benchmark; append-only, resumable phases."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import resource
import time

import numpy as np
import torch

from _common import ROOT,cfg_from_experiment
from diagnose_upstream_environment import event_data
from diagnose_upstream_planners import archive_inputs
from planner_evidence import EvidenceRun,json_write
from upstream_local_closure import runtime_config,run_closure
from upstream_s05_cases import formal_cases,FORMAL_CASE_VERSION
from upstream_run_state import save_state,restore_state
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_baselines import UpstreamPrimitiveInput
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_cases import FAMILIES,trajectory_metrics
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_factory import create_upstream_adapter,plan_batch as plan_upstream_batch
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_footprint import admit_control
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_review_cache import PoseCachedReviewer
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_safety import UpstreamSafetyProjector,SafetyContext
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_flat_review import create_reviewer

VARIANTS=(('line','line_upstream_v1',512),('quintic','quintic_upstream_v1',512),
    ('primitive47','primitive47_upstream_v1',512),('nmpc','nmpc_upstream_v2',512),
    ('mppi512','mppi_upstream_v4',512),('mppi1024','mppi_upstream_v4',1024),('mppi2048','mppi_upstream_v4',2048))
EXPECTED={'plan':7000,'repeat':630,'local':252,'batch':14}


def evaluation_score(metrics):
    return (metrics['mean_goal_error']+2*metrics['goal_error']
        +.2*(metrics['maximum_slope_ratio'] or 0.)+.01*metrics['control_variation'])


def native_plan(adapter,item):
    candidates=[];outputs=[];start=time.perf_counter()
    for primitive_id in range(47):
        adapter.reset();output=adapter.plan(UpstreamPrimitiveInput.for_context(item,primitive_id))
        metrics=trajectory_metrics(item,output.trajectory);outputs.append(output)
        candidates.append({'primitive_id':primitive_id,'valid':output.valid,'fallback_reason':output.fallback_reason,
            'metrics':metrics,'score':evaluation_score(metrics)})
    selected=min(range(47),key=lambda i:(not outputs[i].valid,candidates[i]['score'],i))
    return outputs[selected],{'primitive_id':selected,'candidates':candidates,'enumeration_seconds':time.perf_counter()-start,
        'semantics':'47 native candidates; public physical selection, not learned categorical performance'}


def plan(adapter,item):
    if adapter.adapter_id=='primitive47_upstream_v1':return native_plan(adapter,item)
    return adapter.plan(item),{}


def replay_check(adapter, item):
    """Replay cold and warm snapshots through the public reset/load lifecycle.

    A cold snapshot has no vehicle identity. Reset before loading it into an
    adapter that has since bound a vehicle; retain the cross-vehicle guard.
    """
    before = adapter.state_dict()
    first, _ = plan(adapter, item)
    warm = adapter.state_dict()
    adapter.reset()
    adapter.load_state_dict(before)
    second, _ = plan(adapter, item)
    adapter.reset()
    adapter.load_state_dict(warm)
    warm_first, _ = plan(adapter, item)
    adapter.reset()
    adapter.load_state_dict(warm)
    warm_second, _ = plan(adapter, item)

    def compare(left, right):
        names = ('states', 'controls', 'velocities', 'times')
        error = max(float(np.max(np.abs(getattr(left.trajectory, n)
                                       - getattr(right.trajectory, n)))) for n in names)
        same = (all(np.allclose(getattr(left.trajectory, n), getattr(right.trajectory, n),
                                atol=1e-5, rtol=1e-5) for n in names)
                and left.valid == right.valid and left.fallback_reason == right.fallback_reason)
        return {'equal': bool(same), 'maximum_numeric_error': error}

    cold_result = compare(first, second)
    warm_result = compare(warm_first, warm_second)
    return {'restore_equal': cold_result['equal'] and warm_result['equal'],
            'maximum_state_error': float(np.max(np.abs(first.trajectory.states-second.trajectory.states))),
            'cold_restore': cold_result, 'warm_restore': warm_result}, {
                'before': before, 'warm': warm, 'first': first, 'second': second,
                'warm_first': warm_first, 'warm_second': warm_second}


def execution_pair(item,output,*,acceleration='pose_scope_cache_v1'):
    # One native output feeds both switches; no extra random planner call.
    nominal=output.trajectory.controls[0] if output.valid and not item.intent.hold and item.now<=item.intent.expires_at else np.zeros(2)
    context=SafetyContext(item.ego_pose,item.previous_control,item.terrain,item.neighbors,item.now,
        np.asarray(item.workspace).reshape(2,2),item.intent.speed_cap,bool(item.intent.hold or item.now>item.intent.expires_at or not output.valid))
    reviewer=create_reviewer(item.spec,acceleration)
    tick=time.perf_counter();disabled=admit_control(nominal,item.previous_control,item.ego_pose,item.terrain,item.spec,
        information_time=item.now,latency=0.,neighbors=item.neighbors,workspace=context.workspace,
        hold=context.hold,reviewer=reviewer);disabled_seconds=time.perf_counter()-tick
    projected=UpstreamSafetyProjector(item.spec,reviewer=reviewer).project(context,nominal)
    return event_data({'disabled':{'control':disabled.control,'verified':disabled.verified,'status':disabled.status,
        'reason':disabled.reason,'seconds':disabled_seconds},'enabled':projected})


def record(run,phase,key,value,state=None):
    target=run.path/'metrics'/phase/(key+'.json')
    json_write(target,event_data(value))
    if state is not None:run.snapshot(phase+'_'+key+'.pkl',state)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--run-id',required=True)
    p.add_argument('--phase',choices=('prepare','plan','repeat','local','batch','all','report'),default='all')
    p.add_argument('--resume',action='store_true');p.add_argument('--start',type=int,default=0);p.add_argument('--stop',type=int,default=1000)
    p.add_argument('--variants',nargs='+',choices=[v[0] for v in VARIANTS],default=[v[0] for v in VARIANTS])
    args=p.parse_args();torch.set_num_threads(1);base=cfg_from_experiment(args.config)
    if base.planner_adapter!='mppi_upstream_v4':raise ValueError('S05 requires frozen post-calibration implementation')
    if not 0<=args.start<args.stop<=1000:raise ValueError('Invalid case shard')
    run=EvidenceRun(ROOT/'outputs/runs/upstream_scale_s05'/args.run_id,args.config,resume=args.resume,
        settings={'stage':'upstream_full_gate_b_v1','case_version':FORMAL_CASE_VERSION,'expected':EXPECTED,
            'variants':[list(v) for v in VARIANTS],'torch_threads':1,'mppi_wall_deadline':None,'nmpc_deadline':10.,'formal_training':False})
    archive_inputs(run,base);run.resolved(base)
    manifest=run.path/'config/cases.json'
    if args.phase=='prepare':
        cases=formal_cases(base);run.snapshot('cases.pkl',cases)
        json_write(manifest,{'count':1000,'case_version':FORMAL_CASE_VERSION,'cases':[
            {'id':c.identity,'fingerprint':c.fingerprint,'family':c.family,'seed':c.seed,
             'map_identity':c.map_identity,'solvability':c.solvability,'regression':c.regression} for c in cases]})
        print('Prepared1000 new-scale cases',flush=True);return
    cases=run.restore('cases.pkl');frozen=json.loads(manifest.read_text())['cases']
    if len(cases)!=1000 or any(c.fingerprint!=r['fingerprint'] for c,r in zip(cases,frozen)):raise ValueError('Case fingerprint mismatch')
    variants=[v for v in VARIANTS if v[0] in args.variants]
    if args.phase in {'plan','all'}:
        for index in range(args.start,args.stop):
            case=cases[index]
            for name,adapter_id,k in variants:
                key=f'{index:04d}_{name}';target=run.path/'metrics/plan'/(key+'.json')
                if target.exists():continue
                cfg,data=runtime_config(args.config,name='s05_'+name,seed=case.seed,adapter=adapter_id,samples=k)
                tick=time.perf_counter();adapter=create_upstream_adapter(cfg,cache_dir=run.path/'state/solver_cache');construction=time.perf_counter()-tick
                # Wall deadline belongs to solver runtime, not physical time.
                tick=time.perf_counter();output,selection=plan(adapter,case.input);planning=time.perf_counter()-tick
                pair=execution_pair(case.input,output,acceleration=cfg.review_acceleration_version)
                row={'case_index':index,'case_id':case.identity,'case_fingerprint':case.fingerprint,'variant':name,
                    'configuration':data,'contracts':cfg.metadata,'valid':output.valid,'fallback_reason':output.fallback_reason,
                    'feedback':output.feedback,'metrics':trajectory_metrics(case.input,output.trajectory),
                    'construction_seconds':construction,'planning_seconds':planning,'execution':pair,'selection':selection,
                    'memory_peak_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
                record(run,'plan',key,row,{'output':output,'samples':getattr(adapter,'last_samples',None),
                    'diagnostics':getattr(adapter,'last_diagnostics',None)})
            if (index+1)%25==0:
                shard=run.path/'metrics'/f'plan_shard_{index//25:02d}.json'
                completed=[str(x.relative_to(run.path)) for x in sorted((run.path/'metrics/plan').glob('*.json'))
                     if index-24<=int(x.name[:4])<=index]
                if not shard.exists() and len(completed)==175:json_write(shard,{'first':index-24,'last':index,'completed':completed})
                print('Planning cases completed through',index+1,flush=True)
    if args.phase in {'repeat','all'}:
        chosen=[cases[i] for i in [0,1,2,110,111,112,220,221,222,330,331,332,440,441,442,550,551,552,
            660,661,662,770,771,772,880,881,882,990,993,999]]
        for index,case in enumerate(chosen):
            for seed in (23,31,47):
                for name,adapter_id,k in variants:
                    key=f'{index:02d}_{seed}_{name}'
                    if (run.path/'metrics/repeat'/(key+'.json')).exists():continue
                    cfg,_=runtime_config(args.config,name='s05_'+name,seed=seed,adapter=adapter_id,samples=k)
                    adapter=create_upstream_adapter(cfg,cache_dir=run.path/'state/solver_cache')
                    comparison,state=replay_check(adapter,case.input)
                    record(run,'repeat',key,{'case_id':case.identity,'variant':name,'noise_seed':seed,
                        **comparison}, state)
    if args.phase in {'local','all'}:
        chosen=[cases[j*110+i] for j in range(9) for i in range(2)]
        for index,case in enumerate(chosen):
            for name,adapter_id,k in variants:
                for safety in (False,True):
                    key=f'{index:02d}_{name}_safe{int(safety)}'
                    if (run.path/'metrics/local'/(key+'.json')).exists():continue
                    cfg,_=runtime_config(args.config,name='s05_local_'+name,seed=case.seed,adapter=adapter_id,samples=k,safety=safety)
                    snapshots=sorted((run.path/'state').glob(f'local_{key}_step_*.pkl.gz'))
                    resume_state=restore_state(snapshots[-1]) if snapshots else None
                    attempt=len(list((run.path/'metrics').glob(f'local_{key}_trace_*.jsonl')))
                    stream=(run.path/'metrics'/f'local_{key}_trace_{attempt:03d}.jsonl').open('x')
                    def save(steps,env,transition,scripts,dwell,context):
                        stream.write(json.dumps(event_data({'steps':steps,'records':transition.records,'plans':env.plans,'dwell':dwell}),allow_nan=False)+'\n');stream.flush()
                        save_state(run,f'local_{key}_step_{steps:04d}',context)
                    try:row=run_closure(cfg,case,on_macro=save,cache_dir=run.path/'state/solver_cache',resume_state=resume_state)
                    finally:stream.close()
                    row.pop('records');state=row.pop('state');record(run,'local',key,{**row,'variant':name,'safety':safety},state)
    if args.phase in {'batch','all'}:
        for name,adapter_id,k in variants:
            for b in (4,16):
                key=f'{name}_b{b}'
                if (run.path/'metrics/batch'/(key+'.json')).exists():continue
                cfg,_=runtime_config(args.config,name='s05_batch_'+name,seed=23,adapter=adapter_id,samples=k,num_envs=b)
                items=[replace(cases[i].input,env_id=i,actual_history=replace(cases[i].input.actual_history,env_id=i)) for i in range(b)]
                adapters=[create_upstream_adapter(cfg,cache_dir=run.path/'state/solver_cache') for _ in items]
                native=None
                if name=='primitive47':
                    native=[]
                    for adapter,item in zip(adapters,items):
                        _,selection=native_plan(adapter,item);native.append(selection['primitive_id']);adapter.reset()
                    payloads=[UpstreamPrimitiveInput.for_context(item,i) for item,i in zip(items,native)]
                else:payloads=items
                tick=time.perf_counter();outputs=plan_upstream_batch(adapters,payloads);elapsed=time.perf_counter()-tick
                record(run,'batch',key,{'variant':name,'batch_size':b,'seconds':elapsed,'outputs':len(outputs),
                    'valid':sum(o.valid for o in outputs),'native_ids':native},outputs)
    if args.phase in {'report','all'}:
        rows={phase:[json.loads(x.read_text()) for x in sorted((run.path/'metrics'/phase).glob('*.json'))] for phase in EXPECTED}
        counts={key:len(value) for key,value in rows.items()};complete=counts==EXPECTED
        local_simple=[r for r in rows['local'] if r['family']=='flat' and r['variant'] in {'nmpc','mppi512'}]
        violations=sum(len(r['violations']) for r in rows['local'])
        passed=complete and all(r['restore_equal'] for r in rows['repeat']) and len(local_simple)==8 and all(r['completed'] and not r['violations'] for r in local_simple) and violations==0
        report={'case_count':1000,'counts':counts,'expected':EXPECTED,'matrix_complete':complete,
            'restore_passed':all(r['restore_equal'] for r in rows['repeat']),'contract_violations':violations,
            'simple_closure':[{k:r[k] for k in ('case_id','variant','safety','completed','goal_error')} for r in local_simple],
            'device':base.planner_device,'cpu_gate_b':passed if base.planner_device=='cpu' else False,
            'gate_b_functional':passed,'formal_training_ready':False}
        for variant,_,_ in VARIANTS:
            plans=[r for r in rows['plan'] if r['variant']==variant]
            if plans:report[variant]={'calls':len(plans),'valid_fraction':float(np.mean([r['valid'] for r in plans])),
                'planning_p50_p95_p99':np.quantile([r['planning_seconds'] for r in plans],[.5,.95,.99]).tolist()}
        json_write(run.path/'metrics'/f'coverage_{sum(counts.values()):05d}.json',report)
        print(json.dumps(report),flush=True)


if __name__=='__main__':main()
