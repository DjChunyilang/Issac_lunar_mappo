#!/usr/bin/env python3
"""Nine-family new-scale calibration closure using actual private sensors."""
import argparse
import json
from pathlib import Path
import time

import torch

from _common import ROOT,cfg_from_experiment
from diagnose_upstream_environment import event_data
from diagnose_upstream_planners import archive_inputs
from planner_evidence import EvidenceRun,json_write,file_sha256
from upstream_local_closure import runtime_config,run_closure
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_cases import make_case,FAMILIES


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/experiment/upstream_scale_s04_calibration_cuda_v4.yaml')
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--calibration-summary',type=Path)
    parser.add_argument('--families',nargs='+',choices=FAMILIES,default=list(FAMILIES))
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();torch.set_num_threads(1);base=cfg_from_experiment(args.config)
    parameters=None
    if args.calibration_summary:
        summary=json.loads(args.calibration_summary.read_text())
        if summary.get('case_count')!=36 or not summary.get('selected'):raise ValueError('Incomplete independent calibration')
        parameters=summary['selected']
    settings={'stage':'nine_family_actual_sensor_calibration_closure_v1','families':args.families,
        'seed':101,'maximum_steps':300,'samples':512,'iterations':2,'safety':False,'torch_threads':1,
        'calibration':None if args.calibration_summary is None else {'path':str(args.calibration_summary.resolve()),
            'sha256':file_sha256(args.calibration_summary)},'parameters':parameters,'formal_training':False}
    run=EvidenceRun(ROOT/'outputs/runs/upstream_scale_s04'/args.run_id,args.config,resume=args.resume,settings=settings)
    archive_inputs(run,base);run.resolved(base);results=[]
    for family in args.families:
        target=run.path/'metrics'/f'{family}.json'
        if target.exists():results.append(json.loads(target.read_text()));continue
        cfg,data=runtime_config(args.config,name=f's04_local_{family}_seed101',seed=101,parameters=parameters)
        case=make_case(cfg,family,101,3 if family=='hold' else 0)
        json_write(run.path/'config'/f'{family}.json',{'effective_config':data,'case_fingerprint':case.fingerprint,
            'map_identity':case.map_identity,'solvability':case.solvability})
        with (run.path/'metrics'/f'{family}_trace.jsonl').open('x') as stream:
            def save(steps,env,transition,scripts,dwell,context):
                stream.write(json.dumps(event_data({'steps':steps,'poses':env.core.poses,'time':env.core.times,
                    'records':transition.records,'plans':env.plans,'dwell':dwell}),allow_nan=False)+'\n');stream.flush()
                run.snapshot(f'{family}_step_{steps:04d}.pkl',context)
                if steps==5 or steps%25==0:run.snapshot(f'{family}_samples_{steps:04d}.pkl',env.pool.export_diagnostics())
            start=time.perf_counter();row=run_closure(cfg,case,on_macro=save,cache_dir=run.path/'state/solver_cache')
        row.pop('records');row.pop('state');row['wall_seconds']=time.perf_counter()-start
        row['solvability']=case.solvability;json_write(target,event_data(row));results.append(row)
        print(json.dumps({k:row[k] for k in ('family','completed','goal_error','steps','wall_seconds','terminal_reason')}),flush=True)
    required={'flat','slope','hold','reverse'}
    complete=set(args.families)==set(FAMILIES) and len(results)==9
    mandatory=complete and all(r['completed'] and not r['violations'] for r in results if r['family'] in required)
    json_write(run.path/'metrics/closure_summary.json',{'cases':results,'nine_families_complete':complete,
        'required_reachable_families':sorted(required),'mandatory_closure_passed':mandatory,
        'contract_violation_count':sum(len(r['violations']) for r in results),'calibration_source':settings['calibration'],
        's04_accepted':bool(args.calibration_summary is not None and mandatory and not any(r['violations'] for r in results)),
        'formal_training_ready':False})


if __name__=='__main__':main()
