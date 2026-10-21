#!/usr/bin/env python3
"""S05 new-scale144 actual no-learning episodes; frozen source and paired state."""
import argparse
from collections import Counter
import json
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np
import torch

from _common import ROOT,cfg_from_experiment
from diagnose_upstream_environment import event_data,script_with_bounds,recover_stagnation
from diagnose_upstream_planners import archive_inputs
from evaluate_gathering_no_learning import map_sources
from planner_evidence import EvidenceRun,json_write
from upstream_local_closure import runtime_config
from upstream_s05_cases import gathering_scene
from upstream_run_state import save_state,restore_state
from lunar_rover_tasks.tasks.multi_rover_gathering.gathering_env_cfg import TerrainCfg
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_cases import AnalyticTerrainBank
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_environment import UpstreamEnvironment
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_execution import control_error

FAMILIES=('compact','irregular','reverse','crossing','slope','npb')


def terrain_for(family,seed,n):
    if family not in {'slope','npb'}:return None,None,('flat_proxy_actual_sensor',)
    cfg=TerrainCfg(type='lunar_scene_pack',raster_risk_slope_scale=.6,raster_risk_rms_scale_m=.05)
    if family=='slope':
        return cfg,SimpleNamespace(scene_bank=AnalyticTerrainBank('slope',seed),scene_indices=torch.arange(n)),(f'analytic_slope_seed{seed}',)
    from lunar_rover_tasks.terrain_data.batched import SceneBank
    scene=ROOT/'outputs/runs/lunar_terrain_validation/nasa_sfd_v2_20260924/scenes/real_base/scene.json'
    cfg.scene_manifests=[str(scene)];cfg.raster_device='cpu'
    bank=SceneBank(cfg.scene_manifests,device='cpu',max_bytes=1536*1024**2)
    indices=torch.zeros(n,dtype=torch.long)
    bank.prefetch(indices)
    return cfg,SimpleNamespace(scene_bank=bank,scene_indices=indices),('npb_real_base_same_200m_development_window',)


def execute_pair(run,family,seeds,variant,safety,duration):
    tag=f'{family}_seeds'+('_'.join(map(str,seeds)))+f'_{variant}_safe{int(safety)}_t{duration}'
    paths=[run.path/'metrics/episodes'/f'{family}_{seed}_{variant}_safe{int(safety)}_t{duration}.json' for seed in seeds]
    if all(p.exists() for p in paths):return
    cfg,data=runtime_config(run.config,name='s05_gather_'+tag,seed=seeds[0],num_envs=len(seeds),safety=safety,
        episode_seconds=float(duration),adapter=('nmpc_upstream_v1' if variant=='nmpc_v1' else 'nmpc_upstream_v2')
        if variant in {'nmpc','nmpc_v1','nmpc_v2'} else 'mppi_upstream_v4')
    poses=np.stack([gathering_scene(runtime_config(run.config,name='scene',seed=seed)[0],family,seed)[0] for seed in seeds])
    config_path=run.path/'config'/(tag+'.json')
    scene_record={'effective_configuration':data,'initial_poses':poses.tolist(),'layout_seeds':list(seeds),
        'planner_seed':seeds[0],'birth_tier':0 if family=='compact' else 2 if family=='reverse' else 1,
        'initial_distance':cfg.spec.task.initial_distance,'declared_forward_limit':25/9,
        'high_level':cfg.leader_execution_version,'enforced_bounds':('braking_clearance_lease_v3'
            if cfg.motion_lease_version=='braking_clearance_lease_v3' else 'script_enforced_clearance_lease_v2'),
        'recovery':'five_second_progress_cold_start_v1'}
    if not config_path.exists():json_write(config_path,scene_record)
    elif json.loads(config_path.read_text())!=scene_record:raise ValueError('Frozen gathering scene/config changed')
    terrain,runtime,identities=terrain_for(family,seeds[0],len(seeds))
    env=UpstreamEnvironment(cfg,poses,terrain_cfg=terrain,terrain_runtime=runtime,map_identities=identities,cache_dir=run.path/'state/solver_cache')
    initial_reports=env._reports(parking=False)
    report_path=run.path/'config'/(tag+'_initial_support.json')
    if not report_path.exists():json_write(report_path,event_data({'reports':initial_reports,
        'minimum_distances':[float(np.min(np.linalg.norm(p[:,None,:2]-p[None,:,:2],axis=-1)+np.eye(4)*1000)) for p in poses],
        'scope':'preplanning legal sensor support; invalid births are explicitly recorded, not filtered by planner success'}))
    macro=0;progress={};counts=[Counter() for _ in seeds];violations=[[] for _ in seeds];samples=0;wall=0.;episodes=[];last_failure=()
    snapshots=sorted(p for p in (run.path/'state').glob(tag+'_step_*.pkl.gz') if p.with_suffix('.gz.json').exists())
    if snapshots:
        state=restore_state(snapshots[-1]);env.load_state_dict(state['environment']);macro=state['macro'];progress=state['progress']
        counts=state['counts'];violations=state['violations'];samples=state['samples'];wall=state['wall_seconds'];episodes=state['episodes']
        last_failure=state.get('last_failure',())
    attempt=len(list((run.path/'metrics').glob(tag+'_trace_*.jsonl')))
    start=time.perf_counter()
    with (run.path/'metrics'/f'{tag}_trace_{attempt:03d}.jsonl').open('x') as stream:
        while not env._ended.all():
            requests=script_with_bounds(env);recoveries=recover_stagnation(env,requests,progress)
            transition=env.step_intents(requests);macro+=1;episodes.extend(transition.episodes)
            for record in transition.records:
                if 'execution' not in record:continue
                execution=record['execution']
                for e in range(env.e):
                    if not execution['active_envs'][e]:continue
                    if execution['geometry_overlap'][e]:violations[e].append('geometry_overlap')
                    if execution['clearance_violation'][e]:violations[e].append('clearance')
                    for a in range(4):
                        reason=control_error(execution['applied_control'][e,a],execution['previous_control'][e,a],cfg.spec)
                        if reason:violations[e].append(reason)
                for decision in record['decisions']:
                    counts[decision['env_id']]['execution:'+decision['execution_status']]+=1
                    if not decision['execution_verified']:counts[decision['env_id']]['unverified_execution']+=1
            for e,row in enumerate(env.plans):
                for output in row:
                    if output is not None and not output.valid:counts[e]['planner:'+output.fallback_reason]+=1
            stream.write(json.dumps(event_data({'macro':macro,'time':env.core.times,'poses':env.core.poses,
                'records':transition.records,'plans':env.plans,'recoveries':recoveries}),allow_nan=False)+'\n');stream.flush()
            failure=tuple((e,a,output.fallback_reason) for e,row in enumerate(env.plans) for a,output in enumerate(row)
                if output is not None and not output.valid)
            if macro==1 or macro%5==0 or failure!=last_failure and failure:
                diagnostics=env.pool.export_diagnostics()
                if diagnostics:save_state(run,tag+f'_samples_{macro:04d}',diagnostics);samples+=1
            last_failure=failure
            save_state(run,tag+f'_step_{macro:04d}',{'environment':env.state_dict(),'macro':macro,'progress':progress,
                'counts':counts,'violations':violations,'samples':samples,'episodes':episodes,'last_failure':last_failure,'wall_seconds':wall+time.perf_counter()-start})
            if macro%25==0:print(tag,macro,env.core.times.tolist(),env.last_task['dmax'].tolist(),env._ended.tolist(),flush=True)
    for e,path in enumerate(paths):
        if path.exists():continue
        episode=next(x for x in episodes if x['env_id']==e)
        category=episode['terminal_reason']
        if category=='task_deadline':
            if counts[e]['unverified_execution']>20:category='safety_or_stopping_reserve_rejection'
            elif any(k.startswith('planner:') and v>20 for k,v in counts[e].items()):category='planning_rejection'
        json_write(path,event_data({'family':family,'seed':seeds[e],'variant':variant,'safety':safety,'deadline':duration,
            'episode':episode,'successful':episode['terminal_reason']=='success','failure_category':category,
            'simulation_seconds':env.core.times[e],'violations':violations[e],'counts':dict(counts[e]),
            'sample_events_batch':samples,'macro_count':macro,'wall_seconds_batch':wall+time.perf_counter()-start,
            'configuration_fingerprint':cfg.metadata['configuration_fingerprint'],'metadata':env.metadata}))
        print(path.name,episode['terminal_reason'],float(env.core.times[e]),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--run-id',required=True)
    p.add_argument('--resume',action='store_true');p.add_argument('--families',nargs='+',choices=FAMILIES,default=list(FAMILIES))
    p.add_argument('--variants',nargs='+',choices=('nmpc','mppi512'),default=['nmpc','mppi512'])
    p.add_argument('--report-only',action='store_true')
    args=p.parse_args();torch.set_num_threads(1);cfg=cfg_from_experiment(args.config)
    sources=map_sources();run=EvidenceRun(ROOT/'outputs/runs/upstream_scale_s05'/args.run_id,args.config,resume=args.resume,
        settings={'stage':'upstream_full_gathering144_v1','families':list(FAMILIES),'seeds':[23,31,47],
            'variants':['nmpc','mppi512'],'safety':[False,True],'deadlines':[180,360],
            'samples':512,'iterations':2,'torch_threads':1,'map_sources':sources,'formal_training':False})
    archive_inputs(run,cfg);run.resolved(cfg)
    if not args.report_only:
        # Mandatory simple long episodes first; failed cases remain in matrix.
        for family in args.families:
            for duration in (360,180):
                for safety in (True,False):
                    for variant in args.variants:
                        for seeds in ((23,31),(47,)):execute_pair(run,family,seeds,variant,safety,duration)
    rows=[json.loads(p.read_text()) for p in sorted((run.path/'metrics/episodes').glob('*.json'))]
    simple=[r for r in rows if r['family'] in {'compact','irregular','reverse'} and r['safety'] and r['deadline']==360]
    expected={(f,s,v,a,d) for f in FAMILIES for s in (23,31,47) for v in ('nmpc','mppi512') for a in (False,True) for d in (180,360)}
    actual={(r['family'],r['seed'],r['variant'],r['safety'],r['deadline']) for r in rows}
    complete=actual==expected and len(rows)==144
    passed=complete and len(simple)==18 and all(r['successful'] and not r['violations'] for r in simple)
    json_write(run.path/'metrics'/f'coverage_{len(rows):03d}.json',{'count':len(rows),'expected':144,'matrix_complete':complete,
        'simple_gate_18_passed':len(simple)==18 and all(r['successful'] and not r['violations'] for r in simple),
        'successful':sum(r['successful'] for r in rows),'failure_categories':dict(Counter(r['failure_category'] for r in rows)),
        'violations':sum(len(r['violations']) for r in rows),'s05_gathering_passed':passed,'formal_training_ready':False})


if __name__=='__main__':main()
