#!/usr/bin/env python3
"""Read owned session evidence and publish an offline, non-admitting report."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np
import torch

from _common import ROOT, cfg_from_experiment
from planner_evidence import file_sha256, json_write
from upstream_run_state import restore_state


def array(value):
    return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)


def stats(values):
    a=np.asarray(values,dtype=float).reshape(-1);finite=a[np.isfinite(a)]
    return {'count':len(a),'nonfinite':int(len(a)-len(finite)),
            'mean':float(finite.mean()) if len(finite) else None,
            'p50_p95_p99':np.quantile(finite,[.5,.95,.99]).tolist() if len(finite) else None,
            'min':float(finite.min()) if len(finite) else None,'max':float(finite.max()) if len(finite) else None}


def summarize_interactions(saved, spec):
    valid=array(saved['learning_valid']).astype(bool);duration=array(saved['duration'])
    if valid.shape!=duration.shape or len(saved['samples'])!=len(valid):raise ValueError('Interaction shape mismatch')
    counts=Counter();hold=[];continuous=[];logs=[];primitives=Counter()
    for sample,mask in zip(saved['samples'],valid):
        h=array(sample.action.hold)[mask];hold.extend(h.reshape(-1).tolist())
        # HOLD has no continuous policy contribution; do not include its zero padding.
        c=array(sample.action.continuous)[mask];continuous.extend(c[~h].tolist())
        logs.extend(array(sample.log_prob)[mask].reshape(-1).tolist())
        primitives.update(int(x) for x in array(sample.action.primitive_id)[mask].reshape(-1) if x>=0)
    requested={};reward=Counter();discounted=Counter();steps=Counter();status=Counter();reasons=Counter();source=Counter();safety=[];speed=[];timeline=[]
    for record in saved['execution']:
        if record.get('event')=='request_activation':
            counts['activation_events']+=1;counts['accepted_requests']+=bool(record['accepted'])
            outcome=record.get('planner_outcome')
            if outcome is not None:
                counts['planner_outcome_records']+=1;counts['planner_invalid']+=not outcome['valid']
                reasons['planner:'+str(outcome['fallback_reason'] or 'none')]+=1
            intent=record.get('active_intent')
            if intent is not None:requested[record['env_id'],record['agent_id']]=intent
            continue
        if 'execution' not in record:continue
        execution=record['execution'];active=array(execution['active_envs']).astype(bool)
        velocity=array(execution['actual_velocity']);times=array(execution['activation_time'])
        counts['active_team_bottom_steps']+=int(active.sum())
        for key in ('geometry_overlap','clearance_violation'):
            counts[key+'_team_steps']+=int(array(execution[key])[active].sum())
        for key,value in record.get('reward_components',{}).items():
            reward[key]+=float(array(value)[active].sum())
            discounted[key]+=sum(.99**steps[int(e)]*float(array(value)[e]) for e in np.flatnonzero(active))
        counts['maximum_team_hold_count']=max(counts['maximum_team_hold_count'],int(np.max(record.get('task',{}).get('hold_count',[0]))))
        for e in np.flatnonzero(active):steps[int(e)]+=1
        for decision in record['decisions']:
            e,a=decision['env_id'],decision['agent_id']
            if not active[e]:continue
            counts['agent_execution_steps']+=1;status[decision['execution_status']]+=1
            reasons[decision['execution_reason'] or 'none']+=1;source[decision['tracking_source']]+=1
            counts['unverified_steps']+=not decision['execution_verified']
            nominal=array(decision['nominal_control']);issued=array(decision['issued_control'])
            counts['nominal_changed_steps']+=bool(np.max(np.abs(nominal-issued))>1e-6)
            if 'safety_correction' in decision:
                counts['qp_modified_steps']+=bool(np.max(np.abs(array(decision['safety_correction'])))>1e-6)
            if 'safety_seconds' in decision:safety.append(decision['safety_seconds'])
            v,w=np.abs(velocity[e,a]);speed.append(v)
            stopped=(v<=spec.task.linear_stop and w<=spec.task.angular_stop
                     and v+spec.vehicle.physical_radius*w<=spec.task.boundary_stop)
            intent=requested.get((e,a));kind='unknown_or_expired'
            if intent is not None and times[e]<=intent.expires_at+1e-9:kind='hold' if intent.hold else 'move'
            counts[kind+'_execution_steps']+=1;counts[kind+'_stopped_steps']+=bool(stopped)
            timeline.append({'time':float(times[e]),'env':e,'agent':a,'intent':kind,'v':float(velocity[e,a,0]),
                'omega':float(velocity[e,a,1]),'nominal_v':float(nominal[0]),'issued_v':float(issued[0]),
                'verified':bool(decision['execution_verified']),'reason':decision['execution_reason']})
        ended=array(record.get('task',{}).get('ended_this_step',np.zeros(len(active),bool)))
        for e in np.flatnonzero(ended):
            steps[int(e)]=0
            for a in range(4):requested.pop((int(e),a),None)
    width=len(continuous[0]) if continuous else 0
    action={'valid_agent_samples':len(hold),'hold_fraction':float(np.mean(hold)) if hold else None,
            'continuous_move_only':[stats(np.asarray(continuous)[:,i]) for i in range(width)],
            'log_prob':stats(logs),'primitive_counts':dict(primitives)}
    learning=saved.get('learning')
    learning_stats=None if learning is None else {k:stats(array(v)[valid]) for k,v in learning.items() if k in ('reward','value','next_value')}
    return {'valid_team_transitions':int(valid.sum()),'invalid_team_transitions':int((~valid).sum()),
        'duration_bottom_steps':stats(duration[valid]),'actions':action,'execution_counts':dict(counts),
        'execution_status':dict(status),'execution_reasons':dict(reasons),'tracking_source':dict(source),
        'raw_reward_component_sums':dict(reward),'discounted_partial_component_sums':dict(discounted),'actual_abs_v':stats(speed),
        'recorded_safety_seconds':{'sum':float(sum(safety)),**stats(safety)},'learning_values':learning_stats,
        'missing_fields':[] if learning is not None else ['reward/value/next_value/GAE input not saved by original producer'],
        'scope':'sample metrics use learning-valid masks; execution metrics cover active physical steps; HOLD is effective unexpired intent, not newest sampled action; raw/discounted reward sums cover saved partial batches (discount origin resets at batch start), not completed-episode returns; absent planner outcome records are unknown, not zero rejects'},timeline


def owned(path):
    p=Path(path).resolve()
    if not p.is_relative_to((ROOT/'outputs/runs').resolve()):raise ValueError('Expected an owned run')
    return p


def analyze_session(path):
    path=owned(path);manifest=json.loads((path/'run_manifest.json').read_text())
    cfg=cfg_from_experiment(path/'config/effective_configuration.json')
    files=[path/'run_manifest.json',path/'config/effective_configuration.json'];batches=[];timelines=[]
    for p in sorted((path/'state').glob('interactions_*.pkl.gz')):
        row,timeline=summarize_interactions(restore_state(p),cfg.spec);row['file']=p.name
        batches.append(row);timelines.extend({**v,'batch':p.stem} for v in timeline)
        files.extend([p,p.with_suffix('.gz.json')])
    metrics={}
    for kind in ('update','evaluation','session_summary','timing'):
        paths=sorted((path/'metrics').glob(kind+'_*.json'));files.extend(paths)
        metrics[kind]=[json.loads(p.read_text()) for p in paths]
    # A summary is emitted once per invocation; update.wall_seconds is cumulative within it.
    wall=sum(r['wall_seconds'] for r in metrics['session_summary'])
    transitions=sum(r['valid_team_transitions'] for r in batches)
    safety=sum(r['recorded_safety_seconds']['sum'] for r in batches)
    return {'source_run':str(path),'identity':{k:v for k,v in manifest['identity'].items() if k!='source_snapshot'},
        'inputs':{str(p.relative_to(ROOT)):file_sha256(p) for p in files},'batches':batches,'metrics':metrics,
        'throughput':{'session_wall_seconds':wall,'valid_team_transitions':transitions,
            'team_transitions_per_second':transitions/wall if wall else None,
            'recorded_safety_seconds':safety,'recorded_safety_fraction_of_session':safety/wall if wall else None,
            'note':'Safety timer excludes post-execution prediction and other review calls. Missing phases cannot be inferred from residual time; CPU diagnostic throughput is not a GPU estimate.'}},timelines


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--sessions',nargs='+',required=True)
    parser.add_argument('--run-id',required=True);args=parser.parse_args()
    target=owned(ROOT/'outputs/runs/upstream_scale_t11'/args.run_id)
    target.mkdir(parents=True,exist_ok=False)
    for name in ('metrics','config','figures'):(target/name).mkdir()
    results=[]
    for path in args.sessions:
        report,timeline=analyze_session(path);results.append(report)
        with (target/'metrics'/f'{Path(path).name}_timeline.jsonl').open('x') as f:
            for row in timeline:f.write(json.dumps(row,allow_nan=False)+'\n')
    json_write(target/'metrics/offline_analysis.json',{'version':'t11_offline_analysis_v1','sessions':results,
        'formal_training':False,'optimizer_updates':0,'acceptance_granted':False})
    json_write(target/'run_manifest.json',{'kind':'read_only_analysis','script_sha256':file_sha256(__file__),
        'inputs':{p:h for r in results for p,h in r['inputs'].items()},'formal_training':False})
    (target/'config/analyze_upstream_t11.py').write_bytes(Path(__file__).read_bytes())
    for r in results:print(Path(r['source_run']).name,r['throughput'])


if __name__=='__main__':main()
