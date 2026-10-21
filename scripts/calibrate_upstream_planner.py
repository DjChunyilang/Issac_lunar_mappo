#!/usr/bin/env python3
"""Independent 36-case sequential calibration; no formal training or gate shortcut."""
import argparse
from dataclasses import asdict,replace
import json
import time

import numpy as np
import torch

from _common import ROOT,cfg_from_experiment
from diagnose_upstream_planners import archive_inputs
from planner_evidence import EvidenceRun,json_write
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_cases import calibration_cases,trajectory_metrics,CALIBRATION_SEEDS
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_factory import create_upstream_adapter
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_mppi_calibration import UpstreamMPPISettingsV3

GROUPS={'goal_scale':(.5,1.,2.),'terrain_scale':(.5,1.,2.),'neighbor_scale':(.5,1.,2.),
        'smooth_scale':(.5,1.,2.),'noise_v':(.125,.25,.375),'noise_w':(.05,.10,.15),'temperature':(.5,1.,2.)}


def quality(rows):
    # Independent physical metrics, never internal planner total_cost.
    return (sum(not r['valid'] for r in rows),
        sum(r['metrics']['closest_goal_error'] for r in rows if r['family']!='hold'),
        sum(r['metrics']['maximum_slope_ratio'] or 0. for r in rows),
        sum(r['metrics']['control_variation'] for r in rows),sum(r['seconds'] for r in rows))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/experiment/upstream_scale_s04_target_probe_cuda_v4.yaml')
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();torch.set_num_threads(1);cfg=cfg_from_experiment(args.config)
    if cfg.planner_adapter!='mppi_upstream_v4' or json.loads(cfg.mppi_parameter_overrides):
        raise ValueError('Calibration starts from independent v4 defaults; exploratory selected parameters are not initial defaults')
    run=EvidenceRun(ROOT/'outputs/runs/upstream_scale_s04'/args.run_id,args.config,resume=args.resume,
        settings={'stage':'independent_36_case_sequential_calibration_v1','seeds':list(CALIBRATION_SEEDS),'groups':{k:list(v) for k,v in GROUPS.items()},
                  'torch_threads':1,'samples':512,'iterations':2,'formal_training':False})
    archive_inputs(run,cfg);run.resolved(cfg)
    if args.prepare:
        cases=calibration_cases(cfg)
        run.snapshot('cases.pkl',cases)
        json_write(run.path/'config/cases.json',{'seeds':CALIBRATION_SEEDS,'count':36,
            'cases':[{'id':c.identity,'family':c.family,'seed':c.seed,'fingerprint':c.fingerprint,'map_identity':c.map_identity,
                      'solvability':c.solvability} for c in cases]})
        print('Prepared 36 independent legal-sensor cases',flush=True);return
    cases=run.restore('cases.pkl');identities=json.loads((run.path/'config/cases.json').read_text())['cases']
    if len(cases)!=36 or any(c.fingerprint!=r['fingerprint'] for c,r in zip(cases,identities)):raise ValueError('Frozen calibration input changed')
    base=UpstreamMPPISettingsV3(samples=512,iterations=2,device=cfg.planner_device)
    selected={};executed={};start=time.perf_counter();configs=0
    for group,values in GROUPS.items():
        candidates=[]
        for value in values:
            settings=replace(base,**selected,**{group:value});identity=json.dumps(asdict(settings),sort_keys=True)
            if identity in executed:rows=executed[identity]
            else:
                configs+=1;rows=[]
                for index,case in enumerate(cases):
                    target=run.path/f'metrics/config_{configs:02d}_case_{index:02d}.json'
                    if target.exists():
                        record=json.loads(target.read_text())
                        if record['settings']!=asdict(settings) or record['case_fingerprint']!=case.fingerprint:raise ValueError('Calibration resume identity changed')
                        rows.append(record);continue
                    adapter=create_upstream_adapter(cfg,mppi_settings=settings)
                    generator=torch.Generator().manual_seed(case.seed*1000+index)
                    standard=torch.randn((2,1,512,40,2),generator=generator)
                    noise=standard*torch.tensor([settings.noise_v,settings.noise_w])
                    tick=time.perf_counter();output=adapter.batch_kernel.plan_batch([adapter],[case.input],noise=noise)[0]
                    record={'case_id':case.identity,'case_fingerprint':case.fingerprint,'family':case.family,
                        'settings':asdict(settings),'valid':output.valid,'fallback_reason':output.fallback_reason,
                        'metrics':trajectory_metrics(case.input,output.trajectory),'seconds':time.perf_counter()-tick,
                        'selected_source':output.diagnostics.get('selected_source')}
                    json_write(target,record);rows.append(record)
                    run.snapshot(f'config_{configs:02d}_case_{index:02d}.pkl',{'output':output,'raw_standard_noise_seed':case.seed*1000+index,
                        'diagnostics':adapter.last_diagnostics,'samples':adapter.last_samples})
                executed[identity]=rows
            candidates.append((quality(rows),value,asdict(settings)))
            print(json.dumps({'group':group,'value':value,'score':quality(rows),'elapsed':time.perf_counter()-start}),flush=True)
        best=min(candidates,key=lambda x:x[0]);selected[group]=best[1]
        target=run.path/f'metrics/selected_{group}.json'
        if not target.exists():json_write(target,{'selected':selected,'candidates':candidates,'scope':'single-planning calibration; closure still required'})
    json_write(run.path/'metrics/calibration_summary.json',{'selected':selected,'executed_configurations':configs,
        'case_count':36,'calls':configs*36,'seconds':time.perf_counter()-start,'scope':'single-planning calibration',
        'nine_family_closure_passed':False,'s04_accepted':False,'formal_training_ready':False})


if __name__=='__main__':main()
