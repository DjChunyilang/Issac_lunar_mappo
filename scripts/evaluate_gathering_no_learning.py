#!/usr/bin/env python3
"""T10 full 144-episode K512 CPU/CUDA matrix, strict device gates and resumable state."""
import argparse
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import time
from collections import Counter
import hashlib
import resource
import numpy as np
import torch
from _common import cfg_from_experiment,load_yaml
from planner_evidence import (
    EvidenceRun,json_write,planner_device,planner_metadata,cuda_acceptance_evidence,
    CUDA_EVIDENCE_VERSION,file_sha256,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.gathering_script import LeaderSlotPolicy
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.environment import LocalGoalEnvironment
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.contracts import DT
from lunar_rover_tasks.tasks.multi_rover_gathering import terrain_features as terrain
from benchmark_planners import noise_artifact

FAMILIES=('flat_compact','flat_irregular','flat_reverse','crossing','slope','npb')


def map_sources(config_path='configs/terrain/npb_interface_diagnostic.yaml',
                scene_path='outputs/runs/lunar_terrain_validation/nasa_sfd_v2_20260924/scenes/real_base/scene.json'):
    # Runtime arrays and the family-specific config are not Python source files.
    config=Path(config_path)
    scene=Path(scene_path).resolve()
    sources={str(config):hashlib.sha256(config.read_bytes()).hexdigest(),
             str(scene):hashlib.sha256(scene.read_bytes()).hexdigest()}
    for record in json.loads(scene.read_text())['files'].values():
        path=scene.parent/record['file']
        actual=hashlib.sha256(path.read_bytes()).hexdigest()
        if actual!=record['sha256']:raise ValueError(f'NPB runtime array fingerprint mismatch: {path}')
        sources[str(path)]=actual
    return sources


def scene_cases():
    cases=[]
    for family in FAMILIES:
        for seed in (23,31,47):
            rng=np.random.default_rng(seed)
            yaw=float(rng.uniform(-np.pi,np.pi));angles=yaw+np.arange(3)*2*np.pi/3
            if family=='crossing':angles=np.roll(angles,1)
            radius=(rng.uniform(1.3,1.7,3) if family=='flat_compact' else rng.uniform(2.4,3.4,3))
            if family=='flat_irregular':angles+=rng.uniform(-.2,.2,3)
            positions=np.zeros((4,3));positions[1:,:2]=radius[:,None]*np.stack([np.cos(angles),np.sin(angles)],-1)
            yaws=np.r_[yaw,angles+np.pi if family!='flat_reverse' else angles]
            cases.append({'case_id':f'{family}_seed{seed}','family':family,'seed':seed,
                'positions':positions.tolist(),'yaws':yaws.tolist(),'terrain_phase':0.,
                'map_identity':'npb_real_base_development' if family=='npb' else 'analytic_slope_v1' if family=='slope' else 'flat_proxy',
                'layout':'explicit diagnostic poses; not adapted spawn tiers'})
    return cases


def apply_scenes(env,cases):
    for e,case in enumerate(cases):
        env.core.positions[e]=torch.tensor(case['positions'])
        env.core.yaws[e]=torch.tensor(case['yaws'])
        env.core.velocities_xy[e]=0;env.core.angular_velocities[e]=0
        env.core.terrain_runtime.phase[e]=case['terrain_phase']
    features=terrain.query_terrain_features(env.core.positions[...,:2],env.core.cfg.terrain,env.core.terrain_runtime)
    env.core.positions[...,2]=features[...,0]
    env.reset(np.arange(env.e),reset_core=False)


def config_for_family(config,family,variant,safety,duration):
    cfg=cfg_from_experiment(config);cfg.simulation.num_envs=2
    cfg.simulation.episode_length_s=float(duration)
    c=cfg.local_goal_contract
    configured=c.planner_adapter
    c.planner_adapter=(c.diagnostic_nmpc_adapter if variant=='nmpc' else configured
                       if configured.startswith('mppi_') else 'mppi_terrain_neighbor_v1')
    from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.registry import ADAPTERS
    c.planner_revision=ADAPTERS[c.planner_adapter][0]
    if variant=='nmpc':c.planner_device='cpu'
    c.mppi_samples=512;c.safety_projection='discrete_cbf_qp_v1' if safety else 'disabled'
    cfg.terrain.type='lunar_crater_proxy' if family=='slope' else 'flat_proxy'
    cfg.terrain.amplitude=.09 if family=='slope' else 0.
    cfg.terrain.crater_count=0;cfg.terrain.randomize_per_reset=False
    if family=='npb':
        raw=load_yaml('configs/terrain/npb_interface_diagnostic.yaml')['terrain']
        for key,value in raw.items():setattr(cfg.terrain,key,value)
        scene=Path('outputs/runs/lunar_terrain_validation/nasa_sfd_v2_20260924/scenes/real_base/scene.json').resolve()
        if not scene.exists():raise FileNotFoundError('Required accepted NPB scene is missing')
        cfg.terrain.scene_manifests=[str(scene)]
        cfg.terrain.raster_device='cpu';cfg.observation.schema_version='ego_v12_lunar_multiscale'
    return cfg


def failure_category(episode,counts):
    # Transient startup misses must not mask a whole episode of solver rejection.
    if counts['proxy_swept_collision'] or episode.get('collision',False):return 'collision'
    if counts['execution_limit_violation']:return 'execution_limit'
    if episode['success']:return 'success'
    if episode['terminal_reason']!='task_deadline':return episode['terminal_reason']
    if not episode.get('flatness_ok',True):return 'terminal_terrain'
    decisions=max(1,sum(counts[key] for key in
        ('missing_or_expired_leader','priority_yield','move_to_slot','slot_hold')))
    calls=max(1,4*episode['duration_low_steps'])
    if counts['missing_or_expired_leader']/decisions>=.5:return 'communication_missing'
    if counts['priority_yield']/decisions>=.5:return 'high_level_wait'
    if counts['local_unknown']/calls>=.25:return 'local_unknown'
    if counts['safety_infeasible']/calls>=.25:return 'safety_infeasible'
    rejected=sum(value for key,value in counts.items() if key.startswith('planner:') and
                 key not in ('planner:hold_request','planner:expired_request'))
    if rejected/calls>=.25:return 'planning_rejection'
    return 'task_deadline'


def publish_episode(path,result):
    if path.exists():
        original=json.loads(path.read_text())
        stable=lambda row:{key:value for key,value in row.items() if key!='seconds_batch'}
        if stable(original)!=stable(result):raise ValueError(f'Episode replay mismatch: {path}')
    else:json_write(path,result)


def load_gate_b(path,cfg,*,cuda_junit=None,cuda_acceptance=None):
    """Keep the original sole consumer exception, then require device evidence."""
    path=Path(path);gate=json.loads(path.read_text());device=planner_device(cfg)
    flag='gate_b_cuda' if device=='cuda' else 'gate_b_cpu'
    if not gate.get(flag):raise ValueError(f'{device.upper()} Gate B has not passed; formal T10 remains gated')
    from planner_evidence import source_snapshot
    manifest=path.parent.parent/'run_manifest.json'
    producer=json.loads(manifest.read_text())['identity']['source_snapshot'];current=source_snapshot()
    changed={name for name in set(producer)|set(current) if producer.get(name)!=current.get(name)}
    consumer='scripts/evaluate_gathering_no_learning.py'
    if changed-{consumer}:raise ValueError('Gate B planning/evaluation dependencies differ from current implementation')
    gate={**gate,'t10_dependency_check':{'matched':True,'consumer_changed':consumer in changed,
                                      'consumer_sha256':current[consumer]}}
    if device=='cuda':
        if cuda_junit is None or cuda_acceptance is None:
            raise ValueError('Formal CUDA T10 requires --cuda-junit and --cuda-acceptance')
        acceptance=cuda_acceptance_evidence(cuda_acceptance,cfg,cuda_junit,reference_snapshot=producer)
        recorded=gate.get('cuda_acceptance_evidence',{})
        matches=(gate.get('evidence_protocol')==CUDA_EVIDENCE_VERSION
                 and gate.get('planner_device')=='cuda' and gate.get('gate_b_cpu') is False
                 and gate.get('planner_metadata')==planner_metadata(cfg)
                 and gate.get('coverage_complete') is True and gate.get('numeric_repeat_pass') is True
                 and gate.get('simple_loops_pass') is True and gate.get('device_records_match') is True
                 and gate.get('production_source_matches') is True
                 and gate.get('regression_evidence',{}).get('passed') is True
                 and gate.get('regression_evidence',{}).get('skipped')==[]
                 and gate.get('regression_evidence',{}).get('tests',0)>=600
                 and gate.get('regression_evidence',{}).get('expected_real_acados')==18
                 and gate.get('regression_evidence',{}).get('real_acados_passed',0)>=18
                 and all(gate.get(name)==count for name,count in
                         (('single_records',7000),('local_episodes',252),('repeat_records',630),('batch_panels',14)))
                 and acceptance['passed'] and recorded.get('passed') is True
                 and recorded.get('path')==acceptance.get('path')
                 and recorded.get('sha256')==acceptance.get('sha256')
                 and recorded.get('numeric_evidence',{}).get('sha256')==acceptance.get('numeric_evidence',{}).get('sha256')
                 and recorded.get('numeric_evidence',{}).get('source_sha256')==acceptance.get('numeric_evidence',{}).get('source_sha256'))
        if not matches:raise ValueError('CUDA Gate B/acceptance device, precision, numerics or provenance mismatch')
        gate['t10_cuda_acceptance']=acceptance
    return gate


def execute_pair(run,scenes,variant,safety,duration,*,resume=True,max_macros=None):
    family=scenes[0]['family'];tag=f'{family}_{scenes[0]["seed"]}_{scenes[-1]["seed"]}_{variant}_s{int(safety)}_t{duration}'
    cfg=config_for_family(run.config,family,variant,safety,duration)
    result_paths=[run.path/f'metrics/episode_{s["case_id"]}_{variant}_s{int(safety)}_t{duration}.json' for s in scenes]
    if all(p.exists() for p in result_paths):return
    json_path=run.path/f'config/{tag}.json'
    effective=json.loads(json.dumps(asdict(cfg),allow_nan=False))
    if json_path.exists():
        if json.loads(json_path.read_text())!=effective:raise ValueError('Scenario effective config mismatch on resume')
    else:json_write(json_path,effective)
    if family=='npb':
        sources={path:hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in cfg.terrain.scene_manifests}
        source_path=run.path/f'config/{tag}_map_sources.json'
        if source_path.exists():
            if json.loads(source_path.read_text())!=sources:raise ValueError('NPB scene fingerprint mismatch on resume')
        else:json_write(source_path,sources)
    construction=time.perf_counter()
    env=LocalGoalEnvironment(cfg,run.path/'solver_cache'/tag)
    setup_attempt=len(list((run.path/'metrics').glob(f'setup_{tag}_*.json')))
    json_write(run.path/f'metrics/setup_{tag}_{setup_attempt:04d}.json',
               {'construction_s':time.perf_counter()-construction,
                'rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                'solver_cache':str(run.path/'solver_cache'/tag),'scenario_seeds':[s['seed'] for s in scenes],
                'planner_seed':cfg.local_goal_contract.planner_seed,
                'environment_seed':cfg.seed,'layout_seed_scope':'scenario seed controls initial poses'})
    from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.gathering_script import create_script
    policy=create_script(cfg);active=torch.tensor([True,scenes[0]['case_id']!=scenes[1]['case_id']]);counts=[Counter(),Counter()]
    macro=0;started=time.perf_counter();saved_seconds=0.
    snapshots=sorted(p for p in (run.path/'state').glob(f'{tag}_*.pkl') if p.with_suffix('.pkl.json').exists())
    try:
        if snapshots and resume:
            state=run.restore(snapshots[-1].name)
            env.load_state_dict(state['environment']);policy.load_state_dict(state['policy'])
            active=state['active'];counts=[Counter(v) for v in state['counts']]
            macro=state['macro'];saved_seconds=state['seconds']
        else:apply_scenes(env,scenes)
        sample_reasons=state.get('sample_reasons',{}) if snapshots and resume else {}
        sampled=set(state.get('sampled',[])) if snapshots and resume else set()
        while active.any():
            attempt=sum(1 for _ in (run.path/'metrics').glob(f'trace_{tag}_{macro:04d}_*.json'))
            action,diagnostics=policy.action(env.script_observations(),(2,4))
            for row in diagnostics:
                if active[row['env_id']]:counts[row['env_id']][row['reason']]+=1
            previous_commands=env.commands.copy();previous_positions=env.core.positions.numpy().copy()
            try:
                sample_paths=[]
                def sample_step(record):
                    low_step=macro*5+len(sample_paths)
                    artifacts=[]
                    for row in record.get('planner_execution',[]):
                        i=row['env_id']*4+row['agent_id']
                        reason=row.get('fallback_reason') or row.get('safety',{}).get('reason')
                        failure=reason if reason and reason not in ('hold_request','expired_request') and not str(reason).startswith('hold') else None
                        new_failure=failure is not None and failure!=sample_reasons.get(i)
                        sample_reasons[i]=failure
                        available=getattr(env.planners[i],'last_samples',None) is not None
                        if available and (i not in sampled or low_step%25==0 or new_failure):
                            sampled.add(i)
                            path=run.path/f'samples/{tag}_{low_step:04d}_{i}_a{attempt}.npz'
                            noise_artifact(env.planners[i],path)
                            if path.exists():artifacts.append(str(path.relative_to(run.path)))
                    sample_paths.append(artifacts)
                transition=env.step(action,max_low_steps=5,enabled=active,diagnostic_callback=sample_step)
            except (ValueError,RuntimeError,FloatingPointError) as error:
                json_write(run.path/f'metrics/environment_failure_{tag}_{macro:04d}_a{attempt}.json',
                    {'category':'environment_failure','exception':repr(error),'macro':macro,'active':active.tolist()})
                raise
            from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.execution import limit_control
            for record in transition.records:
                applied=np.asarray(record['commands'])
                current_positions=np.asarray(record['positions'])
                for e in range(2):
                    if not record['active'][e]:continue
                    for first in range(4):
                        for second in range(first+1,4):
                            begin=previous_positions[e,first,:2]-previous_positions[e,second,:2]
                            end=current_positions[e,first,:2]-current_positions[e,second,:2]
                            delta=end-begin;fraction=np.clip(-begin@delta/max(delta@delta,1e-12),0,1)
                            if np.linalg.norm(begin+fraction*delta)<.28:counts[e]['proxy_swept_collision']+=1
                    for a in range(4):
                        if np.max(abs(limit_control(applied[e,a],previous_commands[e,a])-applied[e,a]))>1e-4:
                            counts[e]['execution_limit_violation']+=1
                previous_commands=applied
                previous_positions=current_positions
                for row in record.get('planner_execution',[]):
                    e=row['env_id']
                    if row['fallback_reason']:counts[e][f'planner:{row["fallback_reason"]}']+=1
                    if row.get('safety',{}).get('safety_status')=='unverified_brake':counts[e]['unverified_brake']+=1
                    reason=row.get('safety',{}).get('reason','')
                    if 'qp:' in reason or 'qp_exception' in reason:counts[e]['safety_infeasible']+=1
                    if 'unknown' in reason or 'terrain_support' in reason:counts[e]['local_unknown']+=1
            trace=run.path/f'metrics/trace_{tag}_{macro:04d}_a{attempt}.json'
            json_write(trace,{'script':diagnostics,'records':transition.records,'episodes':transition.episodes,
                             'sample_artifacts':sample_paths,'attempt':attempt,'simulated_low_steps':len(transition.records)})
            for episode in transition.episodes:
                e=episode['env'];result={'scenario':scenes[e],'variant':variant,'samples':512 if variant=='mppi512' else None,
                    'safety_enabled':safety,'duration_s':duration,'episode':episode,'failure_counts':dict(counts[e]),
                    'category':failure_category(episode,counts[e]),'contracts':env.contracts,
                    'seconds_batch':saved_seconds+time.perf_counter()-started,'observation_source':'actual_sensor_transport',
                    'execution_limit_violations':counts[e]['execution_limit_violation'],
                    'proxy_swept_collisions':counts[e]['proxy_swept_collision'],
                    'training_interactions':0,'optimizer_updates':0}
                if getattr(run,'planner_device','cpu')=='cuda':
                    result.update(evidence_protocol=CUDA_EVIDENCE_VERSION,suite_planner_device='cuda',
                                  planner_device=planner_device(cfg),planner_metadata=planner_metadata(cfg)
                                  if variant=='mppi512' else None)
                publish_episode(result_paths[e],result)
                active[e]=False;policy.reset([e])
            macro+=1
            run.snapshot(f'{tag}_{macro:04d}.pkl',{'environment':env.state_dict(),'policy':policy.state_dict(),
                'active':active,'counts':[dict(c) for c in counts],'macro':macro,
                'seconds':saved_seconds+time.perf_counter()-started,'sample_reasons':sample_reasons,'sampled':sorted(sampled)})
            print(json.dumps({'phase':'T10','batch':tag,'macro':macro,'active':active.tolist()}),flush=True)
            if max_macros is not None and macro>=max_macros:break
    finally:env.close()


def summary(run):
    results=[json.loads(p.read_text()) for p in (run.path/'metrics').glob('episode_*.json')]
    expected={(case['case_id'],variant,safety,duration) for case in scene_cases()
              for variant in ('nmpc','mppi512') for safety in (False,True) for duration in (96,180)}
    actual=[(r['scenario']['case_id'],r['variant'],r['safety_enabled'],r['duration_s']) for r in results]
    frozen={case['case_id']:case for case in scene_cases()}
    metadata_matches=all(r['scenario']==frozen.get(r['scenario']['case_id']) and
                         (r['variant']!='mppi512' or r['samples']==512) for r in results)
    complete=len(results)==len(expected) and set(actual)==expected and metadata_matches
    simple=[r for r in results if r['scenario']['family'].startswith('flat_') and r['safety_enabled'] and r['duration_s']==180]
    allsimple=len(simple)==18 and all(r['episode']['success'] and not r['episode']['collision'] and r['execution_limit_violations']==0 and r['proxy_swept_collisions']==0 for r in simple)
    value={'required_episodes':144,'completed_episodes':len(results),'simple_required':18,
        'simple_completed':len(simple),'simple_passed':sum(r['episode']['success'] and not r['episode']['collision'] and
                            r['execution_limit_violations']==0 and r['proxy_swept_collisions']==0 for r in simple),
        'coverage_complete':complete,'matrix_metadata_matches':metadata_matches,
        'missing_matrix_entries':sorted(expected-set(actual)),'unexpected_matrix_entries':sorted(set(actual)-expected),
        't10_functional_pass':complete and allsimple,
        'failure_categories':dict(Counter(failure_category(r['episode'],Counter(r.get('failure_counts',{}))) for r in results)),
        'recorded_failure_categories':dict(Counter(r['category'] for r in results)),
        'failure_classifier_version':'symptom_frequency_v2',
        'raw_sample_snapshots':len(list((run.path/'samples').glob('*.npz'))),
        'execution_trace_files':sum(p.stem.rsplit('_',1)[-1].startswith('a') for p in (run.path/'metrics').glob('trace_*.json')),
        'cuda_validation':'deferred','formal_training':'paused','physx':'not_validated'}
    if getattr(run,'planner_device','cpu')=='cuda':
        metadata=planner_metadata(run.cfg)
        records_match=all(r.get('evidence_protocol')==CUDA_EVIDENCE_VERSION
                          and r.get('suite_planner_device')=='cuda'
                          and r.get('planner_device')==('cuda' if r['variant']=='mppi512' else 'cpu')
                          and (r['variant']!='mppi512' or r.get('planner_metadata')==metadata)
                          and r.get('observation_source')=='actual_sensor_transport'
                          and r.get('training_interactions')==0 and r.get('optimizer_updates')==0
                          for r in results)
        evidence_path=run.path/'config/gate_b.json';gate=None;gate_error=None
        if evidence_path.exists():
            evidence=json.loads(evidence_path.read_text())
            numeric=evidence['result'].get('cuda_acceptance_evidence',{})
            try:
                if evidence['sha256']!=file_sha256(evidence['path']):raise ValueError('Gate B file checksum changed')
                gate=load_gate_b(evidence['path'],run.cfg,
                                cuda_junit=numeric.get('numeric_evidence',{}).get('junit'),
                                cuda_acceptance=numeric.get('path'))
            except (OSError,ValueError,KeyError) as error:gate_error=str(error)
        else:gate_error='missing_formal_cuda_gate_b'
        pilot=run.identity.get('settings',{}).get('pilot',False)
        passed=bool(complete and records_match and allsimple and gate is not None and not pilot)
        value.update(evidence_protocol=CUDA_EVIDENCE_VERSION,planner_device='cuda',planner_metadata=metadata,
                     device_records_match=records_match,coverage_complete=bool(complete and records_match),
                     numeric_contract_pass=gate is not None,gate_b_cuda_pass=gate is not None,
                     gate_error=gate_error,pilot=pilot,t10_functional_pass=passed,t10_cuda_functional_pass=passed,
                     full_task_validation='passed' if passed else 'pilot' if pilot else 'incomplete',
                     cuda_validation='passed' if passed else 'incomplete')
    return value


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/experiment/t10_gathering_no_learning.yaml')
    p.add_argument('--run-dir',required=True,type=Path);p.add_argument('--resume',action='store_true')
    p.add_argument('--family',choices=FAMILIES);p.add_argument('--variant',choices=['nmpc','mppi512'])
    p.add_argument('--safety',choices=['on','off']);p.add_argument('--duration',type=int,choices=[96,180])
    p.add_argument('--gate-b',type=Path);p.add_argument('--max-macros',type=int);p.add_argument('--aggregate',action='store_true')
    p.add_argument('--cuda-junit',type=Path);p.add_argument('--cuda-acceptance',type=Path)
    args=p.parse_args();torch.set_num_threads(1)
    cfg=cfg_from_experiment(args.config);device=planner_device(cfg)
    gate=None
    if args.max_macros is not None and args.max_macros<=0:raise ValueError('Pilot budget must be positive')
    if not args.aggregate and args.max_macros is None:
        if args.gate_b is None:raise ValueError('Formal T10 requires --gate-b; use --max-macros only for a bounded wiring pilot')
        gate=load_gate_b(args.gate_b,cfg,cuda_junit=args.cuda_junit,cuda_acceptance=args.cuda_acceptance)
    settings={'matrix':'gathering_scenes_v1','samples':512,'episodes':144,
        'pilot':args.max_macros is not None,'consumer_version':'t10_evidence_driver_v3','map_sources':map_sources()}
    if device=='cuda':
        settings['consumer_version']='t10_evidence_driver_cuda_v1'
    run=EvidenceRun(args.run_dir,args.config,resume=args.resume,settings=settings)
    if gate is not None and not (run.path/'config/gate_b.json').exists():
        json_write(run.path/'config/gate_b.json',{'path':str(args.gate_b.resolve()),'sha256':hashlib.sha256(args.gate_b.read_bytes()).hexdigest(),'result':gate})
    elif gate is not None:
        recorded=json.loads((run.path/'config/gate_b.json').read_text())
        if recorded['path']!=str(args.gate_b.resolve()) or recorded['sha256']!=file_sha256(args.gate_b):
            raise ValueError('Formal Gate B evidence changed on resume')
    cases_path=run.path/'config/scenes.json'
    if not cases_path.exists():json_write(cases_path,scene_cases())
    cases=json.loads(cases_path.read_text())
    if cases!=scene_cases():raise ValueError('Frozen scenario list mismatch')
    if not args.aggregate:
        for family in ([args.family] if args.family else FAMILIES):
            group=[s for s in cases if s['family']==family]
            # Pad the odd scenario by a disabled duplicate, rather than count it twice.
            for variant in ([args.variant] if args.variant else ('nmpc','mppi512')):
                for enabled in ([args.safety=='on'] if args.safety else (False,True)):
                    for duration in ([args.duration] if args.duration else (96,180)):
                        execute_pair(run,group[:2],variant,enabled,duration,max_macros=args.max_macros)
                        execute_pair(run,[group[2],group[2]],variant,enabled,duration,max_macros=args.max_macros)
    value=summary(run)
    while True:
        index=len(list((run.path/'metrics').glob('summary_*.json')))
        try:
            json_write(run.path/f'metrics/summary_{index:04d}.json',value)
            break
        except FileExistsError:
            continue  # Concurrent disjoint workers retain separate immutable summaries.
    print(json.dumps(value),flush=True)

if __name__=='__main__':main()
