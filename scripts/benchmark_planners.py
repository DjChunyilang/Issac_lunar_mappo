#!/usr/bin/env python3
"""T09 full fixed-case benchmark; no training, immutable shards and explicit resume."""
import argparse
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import resource
import time
import numpy as np
import torch
import xml.etree.ElementTree as ET
import hashlib
import os
from _common import cfg_from_experiment
from planner_evidence import (
    EvidenceRun,json_write,planner_device,planner_metadata,source_snapshot,
    cuda_acceptance_evidence,CUDA_EVIDENCE_VERSION,
)
from diagnose_planners import output_row
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.benchmark import (
    FAMILIES,VARIANTS,make_case,to_input,trajectory_metrics,shifted_input,digest,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.registry import create_adapters,ADAPTERS
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.baselines import PrimitiveReplayInput
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.protocol import execution_passthrough,LocalIntent,plan_batch
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.safety import SafetyInput,SafetyProjector,decision_record
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.contracts import contract_metadata,DT
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.planner import build_solver

ADAPTER={'line':'deterministic_line_v1','quintic':'deterministic_quintic_v1',
         'primitive47':'primitive47_adapter_v1','nmpc':'nmpc_polar_v1',
         **{f'mppi{k}':'mppi_terrain_neighbor_v1' for k in (512,1024,2048)}}


class Runner:
    def __init__(self,cfg,variant,cache):
        self.suite_device=planner_device(cfg)
        self.cfg=deepcopy(cfg);c=self.cfg.local_goal_contract
        configured=c.planner_adapter
        c.planner_adapter=(configured if variant.startswith('mppi') and configured.startswith('mppi_')
                           else ADAPTER[variant])
        c.planner_revision=ADAPTERS[c.planner_adapter][0]
        if variant.startswith('mppi'): c.mppi_samples=int(variant[4:])
        else:c.planner_device='cpu'
        self.variant=variant;self.cache=cache;self.built=False
        self.meta=contract_metadata(self.cfg)

    def synchronize(self):
        if self.variant.startswith('mppi') and planner_device(self.cfg)=='cuda':
            torch.cuda.synchronize(torch.device(self.cfg.local_goal_contract.planner_device))

    def device_evidence(self):
        if self.suite_device!='cuda':return {}
        return {'evidence_protocol':CUDA_EVIDENCE_VERSION,'suite_planner_device':'cuda',
                'planner_device':planner_device(self.cfg),'contracts':contract_metadata(self.cfg),
                'timing_scope':'CUDA synchronized host call including export and final host review'}

    def factory(self,path,slope,*,build):
        result=build_solver(path,slope,build=not self.built)
        self.built=True
        return result

    def new(self,seed=23):
        self.cfg.local_goal_contract.planner_seed=seed
        return create_adapters(1,self.cache,self.cfg.terrain.slope_speed_scale,
                              self.cfg.local_goal_contract,solver_factory=self.factory)[0]

    def plan(self,planner,item):
        if hasattr(planner,'bind'):planner.bind(item)
        if self.variant!='primitive47': return planner.plan(item),None
        saved=planner.state_dict();best=None;beststate=None;rows=[]
        for pid in range(47):
            planner.load_state_dict(saved)
            out=planner.plan(PrimitiveReplayInput(item,pid))
            metric=trajectory_metrics(item,out.trajectory)
            rows.append({'primitive_id':pid,'valid':out.valid,'cost':metric['common_cost'],
                         'reason':out.fallback_reason})
            eligible=out.valid and (not item.intent.hold or pid==0)
            if eligible and (best is None or (metric['common_cost'],pid)<best[:2]):
                best=(metric['common_cost'],pid,out);beststate=planner.state_dict()
        if best is None:
            planner.load_state_dict(saved);out=planner.plan(PrimitiveReplayInput(item,0))
        else:
            planner.load_state_dict(beststate);out=best[2]
        return out,{'candidates':rows,'chosen_id':0 if best is None else best[1]}


def frozen_cases(run=None):
    cases=[make_case(f,i) for f in FAMILIES for i in range(110)]
    for i in range(10):
        case=make_case('slope' if i<4 else FAMILIES[(i-4)%len(FAMILIES)],200+i)
        case['case_id']=f'regression_{i:02d}'
        case['provenance']={'type':'reconstructed' if i<4 else 'boundary_reference',
            'description':'T07 inherited slope stall family; observations reconstructed, not raw replay' if i<4 else 'fixed boundary reference',
            'source':'docs/experiments/t07_mppi_terrain_neighbor_validation_2026-10-10.md'}
        if i<4:
            case['goal']=[.425424 if i==0 else .299407,0]
        case['sha256']=digest({k:v for k,v in case.items() if k!='sha256'})
        cases.append(case)
    if run is not None:
        from reconstruct_t07_cases import reconstruct
        source=Path('outputs/runs/t07_mppi_terrain_neighbor/acceptance_seed23_cpu_20261010_01/metrics')
        rebuilt=reconstruct(source/'sensor_loop_seed23_retry1.json',run.path/'config/t07_reconstruction_long.json',
                            [(0,(0,0)),(1,(1,0)),(2,(1,3))])
        short=json.loads((source/'sensor_short_loop_seed23.json').read_text())
        failed=np.argwhere(np.asarray(short['final_errors'])>.2)
        if len(failed)!=1:raise ValueError('Expected one recorded short-goal failure')
        rebuilt+=reconstruct(source/'sensor_short_loop_seed23.json',run.path/'config/t07_reconstruction_short.json',
                             [(3,tuple(int(x) for x in failed[0]))])
        cases[990:994]=rebuilt
    return cases


def noise_artifact(planner,path):
    value=getattr(planner,'last_samples',None)
    if value is not None:
        if path.exists():
            with np.load(path) as existing:
                if not all(k in existing and np.array_equal(existing[k],v.detach().cpu().numpy()) for k,v in value.items()):
                    raise ValueError('Existing sample fingerprint differs on resume')
            return
        path.parent.mkdir(parents=True,exist_ok=True)
        import tempfile, os
        with tempfile.NamedTemporaryFile('wb',dir=path.parent,delete=False) as stream:
            np.savez_compressed(stream,**{k:v.detach().cpu().numpy() for k,v in value.items()})
            temporary=Path(stream.name)
        try:os.link(temporary,path)
        finally:temporary.unlink()


def single_case(runner,case,*,seed=23,save_samples=None):
    item=to_input(case)
    begin=time.perf_counter();planner=runner.new(seed);constructed=time.perf_counter()
    if hasattr(planner,'bind'):planner.bind(item)
    before=planner.state_dict()
    runner.synchronize()
    begin_plan=time.perf_counter();out,native=runner.plan(planner,item)
    runner.synchronize();planned=time.perf_counter()
    projector=SafetyProjector(runner.cfg.terrain.slope_speed_scale)
    nominal=execution_passthrough(item.ego_pose,out)
    safe=projector.project(SafetyInput(item),nominal.nominal_control)
    reviewed=time.perf_counter()
    if save_samples is not None: noise_artifact(planner,save_samples)
    row={'case_id':case['case_id'],'family':case['family'],'variant':runner.variant,'seed':seed,
         'solvability':case['solvability'],'contracts':contract_metadata(runner.cfg),'input_sha256':case['sha256'],
         'native':output_row(out),'metrics':trajectory_metrics(item,out.trajectory),
         'safety_off':decision_record(nominal),'safety_on':decision_record(safe),'primitive_selection':native,
         'timings':{'construction_s':constructed-begin,'planning_s':planned-begin_plan,
                    'safety_s':reviewed-planned,'end_to_end_s':reviewed-begin_plan},
         'rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
         'diagnostics':getattr(planner,'last_diagnostics',{}),**runner.device_evidence()}
    # Public cold-state restore must reproduce the same outcome, not merely an RNG hash.
    return row,planner,before,item


def local_loop(runner,case,safety_enabled):
    item=to_input(case);goal=item.intent.goal_xy.copy();planner=runner.new(case['seed'])
    messages={p.sender_id: p.prediction for p in item.neighbors}
    projector=SafetyProjector(runner.cfg.terrain.slope_speed_scale);rows=[];dwell=0;collision=False;limit=False
    for step in range(100):
        if np.linalg.norm(item.ego_pose[:2]-goal)<=.2:
            item=replace(item,intent=LocalIntent.for_hold(item.ego_pose,item.now))
        out,native=runner.plan(planner,item);decision=execution_passthrough(item.ego_pose,out)
        if safety_enabled: decision=projector.project(SafetyInput(item),decision.nominal_control)
        u=decision.applied_control
        next_item=shifted_input(item,u,runner.cfg.terrain.slope_speed_scale,neighbor_messages=messages)
        for peer in item.neighbors:
            # Evaluation alone advances the scripted physical peer. It does not fill a receiver's expired message.
            source=messages[peer.sender_id]
            times=case['now']+np.arange(16)*DT
            at=next_item.now
            def actual_position(time_at):
                position=np.array([np.interp(time_at,times,source.positions[:,axis]) for axis in range(2)])
                return position+max(0.,time_at-times[-1])*(source.positions[-1]-source.positions[-2])/DT
            before=item.ego_pose[:2]-actual_position(item.now)
            after=next_item.ego_pose[:2]-actual_position(at)
            delta=after-before
            closest=np.clip(-before@delta/max(delta@delta,1e-12),0.,1.)
            if np.linalg.norm(before+closest*delta)<.28:collision=True
        from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.execution import limit_control
        limit |= bool(np.max(abs(limit_control(u,item.previous_control)-u))>1e-4)
        error=float(np.linalg.norm(next_item.ego_pose[:2]-goal))
        parked=abs(next_item.ego_velocity[0])<=.05 and abs(next_item.ego_velocity[1])<=.1
        dwell=dwell+1 if error<=.2 and parked else 0
        rows.append({'step':step,'error':error,'dwell':dwell,'native':output_row(out),
                     'execution':decision_record(decision),'metrics':trajectory_metrics(item,out.trajectory),
                     'primitive_selection':native})
        item=next_item
        if collision or dwell>=25:break
    return {'case_id':case['case_id'],'family':case['family'],'variant':runner.variant,
        'safety_enabled':safety_enabled,'success':dwell>=25 and not collision and not limit,
        'collision':collision,'limit_violation':limit,'final_error':rows[-1]['error'],
        'steps':len(rows),'reason':'success' if dwell>=25 else 'collision' if collision else 'local_deadline',
        'rows':rows,**runner.device_evidence()}


def batch_panel(runner, cases, size):
    started=time.perf_counter()
    inputs=[to_input(cases[i],identity=(i//4,i%4)) for i in range(size)]
    packed=time.perf_counter()
    planners=[runner.new(cases[i]['seed']) for i in range(size)]
    runner.synchronize();constructed=time.perf_counter()
    if runner.variant=='primitive47':
        outputs=[runner.plan(p,item)[0] for p,item in zip(planners,inputs)]
    else:
        outputs=plan_batch(planners,inputs)
    runner.synchronize();planned=time.perf_counter()
    projectors=[SafetyProjector(runner.cfg.terrain.slope_speed_scale) for _ in inputs]
    decisions=[p.project(SafetyInput(item),execution_passthrough(item.ego_pose,out).nominal_control)
               for p,item,out in zip(projectors,inputs,outputs)]
    ended=time.perf_counter()
    return {'variant':runner.variant,'batch_size':size,'torch_threads':torch.get_num_threads(),
        'scheduler':'tensor_batch' if runner.variant.startswith('mppi') else 'sequential_instances',
        'timings':{'packing_s':packed-started,'construction_s':constructed-packed,
                   'planning_s':planned-constructed,'safety_s':ended-planned,'end_to_end_s':ended-started},
        'throughput_plans_s':size/(planned-constructed),
        'rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        'outputs':[output_row(out) for out in outputs],
        'executions':[decision_record(d) for d in decisions],**runner.device_evidence()}


def regression_evidence(path,*,device='cpu'):
    if path is None:return {'passed':False,'reason':'missing_required_junit'}
    path=Path(path);tests=list(ET.parse(path).iter('testcase'))
    failed=[t.attrib['name'] for t in tests if t.find('failure') is not None or t.find('error') is not None]
    skipped=[t.attrib['name'] for t in tests if t.find('skipped') is not None]
    unexpected_skip=skipped if device=='cuda' else [name for name in skipped if 'cuda' not in name.lower()]
    real=[t for t in tests if 'real_acados' in t.attrib['name']]
    baseline=Path('outputs/runs/t07_mppi_terrain_neighbor/acceptance_seed23_cpu_20261010_01/metrics/contracts_acceptance_compat_final.xml')
    expected={(t.attrib['classname'],t.attrib['name']) for t in ET.parse(baseline).iter('testcase') if 'real_acados' in t.attrib['name']}
    passed_real={(t.attrib['classname'],t.attrib['name']) for t in real if all(t.find(tag) is None for tag in ('failure','error','skipped'))}
    # JUnit provenance is bound to the exact production source used by this run.
    from planner_evidence import source_snapshot
    source_file=path.with_suffix('.source.json')
    source_matches=source_file.exists() and json.loads(source_file.read_text())==source_snapshot()
    return {'passed':len(tests)>=600 and not failed and not unexpected_skip and len(expected)==18 and
                    expected<=passed_real and source_matches,
        'junit':str(path.resolve()),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'tests':len(tests),'failures':failed,'skipped':skipped,'real_acados_passed':len(passed_real),
        'expected_real_acados':len(expected),'source_matches':source_matches}


def aggregate(run, junit=None, *, cuda_junit=None, cuda_acceptance=None):
    cases=json.loads((run.path/'config/cases.json').read_text())['cases']
    expected_ids={c['case_id'] for c in cases}
    shards=list((run.path/'metrics').glob('single_*_*.json'))
    results=[r for p in shards for r in json.loads(p.read_text())['rows']]
    variants={}
    for variant in VARIANTS:
        rows=[r for r in results if r['variant']==variant]
        times=[r['timings']['planning_s'] for r in rows]
        variants[variant]={'coverage':len(rows),'unique_cases':len({r['case_id'] for r in rows}),
            'valid_rate':sum(r['native']['valid'] for r in rows)/max(1,len(rows)),
            'p50_p95_p99_s':np.percentile(times,[50,95,99]).tolist() if times else None,
            'throughput_plans_s':len(times)/sum(times) if times else None,
            'rss_peak_mib':max((r['rss_mib'] for r in rows),default=None),
            'safety_interventions':sum(r['safety_on']['safety_status']!='unchanged' for r in rows),
            'unverified_brakes':sum(r['safety_on']['safety_status']=='unverified_brake' for r in rows),
            'mean_goal_progress_m':float(np.mean([r['metrics']['progress'] for r in rows])) if rows else None,
            'fallbacks':{reason:sum(r['native']['reason']==reason for r in rows)
                         for reason in sorted(set(r['native']['reason'] for r in rows),key=str)}}
    loops=[json.loads(p.read_text()) for p in (run.path/'metrics').glob('loop_*.json')]
    repeats=[json.loads(p.read_text()) for p in (run.path/'metrics').glob('repeat_*.json')]
    panels=[json.loads(p.read_text()) for p in (run.path/'metrics').glob('panel_*.json')]
    simple=[r for r in loops if r['variant'] in ('nmpc','mppi512') and r['family']=='flat' and r['safety_enabled']]
    simple_pass=len(simple)==4 and all(r['success'] for r in simple)
    expected_loops={(v,c['case_id'],enabled) for v in VARIANTS for c in
                    [cases[110*i+j] for i in range(9) for j in range(2)] for enabled in (False,True)}
    loop_ids={(r['variant'],r['case_id'],r['safety_enabled']) for r in loops}
    panel_cases=[cases[110*i+j] for i in range(9) for j in range(3)]+cases[990:993]
    expected_repeats={(v,c['case_id'],seed) for v in VARIANTS for c in panel_cases for seed in (23,31,47)}
    repeat_ids={(r['row']['variant'],r['row']['case_id'],r['row']['seed']) for r in repeats}
    complete=all(len([r for r in results if r['variant']==v])==1000 and
                 {r['case_id'] for r in results if r['variant']==v}==expected_ids for v in VARIANTS)
    complete &= len(loops)==252 and loop_ids==expected_loops and len(repeats)==630 and repeat_ids==expected_repeats
    complete &= {(r['variant'],r['batch_size']) for r in panels}=={(v,b) for v in VARIANTS for b in (4,16)}
    repeat_pass=bool(repeats) and all(r['passed'] for r in repeats)
    cfg=getattr(run,'cfg',None)
    device=planner_device(cfg) if cfg is not None else 'cpu'
    regression=regression_evidence(junit,device=device)
    value={'variants':variants,'single_records':len(results),'local_episodes':len(loops),
        'repeat_records':len(repeats),'batch_panels':len(panels),
        'required_counts':{'single':7000,'local':252,'repeat':630,'panel':14},
        'coverage_complete':bool(complete),'simple_loops_pass':simple_pass,'simple_loops':simple,
        'numeric_repeat_pass':repeat_pass,'regression_evidence':regression,
        'gate_b_cpu':bool(device=='cpu' and complete and simple_pass and repeat_pass and regression['passed']),
        'gate_note':'Complete frozen coverage, repeatability, simple closed loops and fingerprinted necessary regressions required',
        'cuda_validation':'deferred'}
    if device=='cuda':
        expected_meta={v:Runner(cfg,v,run.path/'solver_cache'/v).device_evidence() for v in VARIANTS}
        device_records=results+[r['row'] for r in repeats]+loops+panels
        records_match=all(all(row.get(key)==expected_meta[row['variant']][key]
                             for key in ('evidence_protocol','suite_planner_device','planner_device'))
                          and row.get('contracts',{}).get('mppi')==expected_meta[row['variant']]['contracts'].get('mppi')
                          for row in device_records)
        source_matches=run.identity['source_snapshot']==source_snapshot()
        complete=bool(complete and len(panels)==14 and len(expected_ids)==1000 and records_match)
        acceptance=cuda_acceptance_evidence(cuda_acceptance,cfg,cuda_junit)
        passed=bool(complete and simple_pass and repeat_pass and regression['passed']
                    and acceptance['passed'] and source_matches)
        value.update(evidence_protocol=CUDA_EVIDENCE_VERSION,planner_device='cuda',
                     planner_metadata=planner_metadata(cfg),coverage_complete=complete,
                     device_records_match=records_match,production_source_matches=source_matches,
                     cuda_acceptance_evidence=acceptance,gate_b_cuda=passed,
                     cuda_validation='passed' if passed else 'incomplete',
                     numeric_contract_pass=acceptance.get('numeric_evidence',{}).get('passed',False),
                     runtime_acceptance_pass=acceptance['passed'],full_task_validation='not_run',
                     formal_training='paused',optimizer_updates=0,training_interactions=0)
    return value


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/experiment/t09_planner_benchmark.yaml')
    p.add_argument('--run-dir',required=True,type=Path);p.add_argument('--resume',action='store_true')
    p.add_argument('--variants',nargs='+',choices=VARIANTS,default=list(VARIANTS))
    p.add_argument('--phase',choices=['single','repeat','loop','panel','all','aggregate'],default='all')
    p.add_argument('--start',type=int,default=0);p.add_argument('--stop',type=int,default=1000)
    p.add_argument('--regression-junit',type=Path);p.add_argument('--no-summary',action='store_true')
    p.add_argument('--cuda-junit',type=Path);p.add_argument('--cuda-acceptance',type=Path)
    args=p.parse_args();torch.set_num_threads(1)
    if args.start%25 or args.stop%25:raise ValueError('Shard boundaries must be multiples of 25')
    if not 0<=args.start<args.stop<=1000:raise ValueError('Case interval outside frozen suite')
    run=EvidenceRun(args.run_dir,args.config,resume=args.resume,settings={'suite':'planner_cases_v1','count':1000})
    cfg=cfg_from_experiment(args.config);run.resolved(cfg)
    suite=run.path/'config/cases.json'
    if not suite.exists():json_write(suite,{'version':'planner_cases_v1','cases':frozen_cases(run)})
    cases=json.loads(suite.read_text())['cases']
    phases=['single','repeat','loop','panel'] if args.phase=='all' else [args.phase]
    for variant in args.variants:
        runner=Runner(cfg,variant,run.path/'solver_cache'/variant)
        warm=run.path/f'metrics/warmup_{variant}.json'
        if args.phase!='aggregate':
            runner.synchronize();started=time.perf_counter();warm_output=runner.plan(runner.new(23),to_input(cases[0]))[0]
            runner.synchronize()
            warm_row={'variant':variant,'seconds':time.perf_counter()-started,
                      'valid':warm_output.valid,'threads':torch.get_num_threads(),'affinity':sorted(os.sched_getaffinity(0))}
            if not warm.exists():json_write(warm,warm_row)
        if 'single' in phases:
            for start in range(args.start,args.stop,25):
                path=run.path/f'metrics/single_{variant}_{start:04d}.json'
                if path.exists():
                    recorded=json.loads(path.read_text())['rows']
                    if [r['case_id'] for r in recorded]!=[c['case_id'] for c in cases[start:start+25]] or any(r['variant']!=variant for r in recorded):
                        raise ValueError('Incomplete or incompatible existing shard')
                    continue
                rows=[]
                for case in cases[start:min(start+25,args.stop)]:
                    result_path=run.path/f'metrics/cases/{variant}_{case["case_id"]}.json'
                    if result_path.exists():row=json.loads(result_path.read_text())
                    else:
                        row,_,_,_=single_case(runner,case,save_samples=run.path/f'samples/{variant}_{case["case_id"]}.npz')
                        json_write(result_path,row)
                    rows.append(row)
                json_write(path,{'rows':rows})
                print(json.dumps({'phase':'single','variant':variant,'completed_through':start+len(rows)}),flush=True)
        if 'repeat' in phases:
            panel=[cases[110*i+j] for i in range(9) for j in range(3)]+cases[990:993]
            for case in panel:
                for seed in (23,31,47):
                    path=run.path/f'metrics/repeat_{variant}_{case["case_id"]}_{seed}.json'
                    if path.exists():continue
                    row,planner,before,item=single_case(runner,case,seed=seed)
                    planner.load_state_dict(before);again,_=runner.plan(planner,item)
                    passed=(again.valid==row['native']['valid'] and again.fallback_reason==row['native']['reason']
                            and np.allclose(again.trajectory.controls,row['native']['controls'],atol=1e-5,rtol=1e-5))
                    json_write(path,{'passed':bool(passed),'row':row,
                        'maximum_error':float(np.max(abs(again.trajectory.controls-np.array(row['native']['controls']))))})
        if 'loop' in phases:
            for case in [cases[110*i+j] for i in range(9) for j in range(2)]:
                for enabled in (False,True):
                    path=run.path/f'metrics/loop_{variant}_{case["case_id"]}_{int(enabled)}.json'
                    if not path.exists():
                        json_write(path,local_loop(runner,case,enabled))
                        print(json.dumps({'phase':'loop','variant':variant,'case':case['case_id'],'safety':enabled}),flush=True)
        if 'panel' in phases:
            for size in (4,16):
                path=run.path/f'metrics/panel_{variant}_{size}.json'
                if not path.exists():json_write(path,batch_panel(runner,cases,size))
    if args.no_summary:return
    summary=aggregate(run,args.regression_junit,cuda_junit=args.cuda_junit,cuda_acceptance=args.cuda_acceptance)
    path=run.path/f'metrics/summary_{len(list((run.path/"metrics").glob("summary_*.json"))):04d}.json'
    json_write(path,summary);print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
