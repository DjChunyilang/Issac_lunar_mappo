#!/usr/bin/env python3
"""Standalone plots and frozen failure timelines from existing owned evidence."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from _common import ROOT,cfg_from_experiment
from planner_evidence import json_write,file_sha256
from upstream_run_state import restore_state


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--analysis-run',required=True)
    p.add_argument('--profiles',nargs='+',required=True);p.add_argument('--failure-runs',nargs='+',required=True);p.add_argument('--run-id',required=True)
    a=p.parse_args();target=ROOT/'outputs/runs/upstream_scale_t11'/a.run_id;target.mkdir(parents=True,exist_ok=False)
    for name in ('config','metrics','figures'):(target/name).mkdir()
    inputs={}
    def read(path):
        path=Path(path).resolve()
        if not path.is_relative_to((ROOT/'outputs/runs').resolve()):raise ValueError('Expected owned report input')
        inputs[str(path.relative_to(ROOT))]=file_sha256(path);return json.loads(path.read_text())
    report=read(Path(a.analysis_run)/'metrics/offline_analysis.json');session=next(s for s in report['sessions'] if s['identity']['representation']=='A1')
    batch=session['batches'][0];counts=batch['execution_counts'];fig,axes=plt.subplots(1,3,figsize=(13,3.8))
    axes[0].bar(['HOLD','MOVE'],[batch['actions']['hold_fraction'],1-batch['actions']['hold_fraction']]);axes[0].set_title('Valid sampled actions (A1, n=512)')
    axes[1].bar(['HOLD stopped','MOVE stopped','Nominal changed'],[counts['hold_stopped_steps']/counts['hold_execution_steps'],counts['move_stopped_steps']/counts['move_execution_steps'],counts['nominal_changed_steps']/counts['agent_execution_steps']]);axes[1].tick_params(axis='x',rotation=20);axes[1].set_title('Physical execution fractions')
    reward=batch['raw_reward_component_sums'];axes[2].bar(reward.keys(),reward.values());axes[2].tick_params(axis='x',rotation=45);axes[2].set_title('Partial raw reward; not task success')
    fig.tight_layout();fig.savefig(target/'figures/a1_existing_batch.png',dpi=150);fig.savefig(target/'figures/a1_existing_batch.pdf');plt.close(fig)
    profiles=[read(Path(r)/'metrics/profile.json') for r in a.profiles]
    fig,ax=plt.subplots(figsize=(7,4));labels=[Path(r).name.replace('training_cpu_spans_','') for r in a.profiles]
    bottom=np.zeros(len(profiles))
    for key in ('collect_s','ppo_update_s','interaction_save_s'):
        values=[r[key]/r['valid_team_transitions'] for r in profiles];ax.bar(labels,values,bottom=bottom,label=key);bottom+=values
    ax.set_ylabel('Seconds / valid team transition');ax.set_title('CPU K64 fresh-state diagnostic; concurrent load');ax.legend();fig.tight_layout();fig.savefig(target/'figures/cpu_phase_cost.png',dpi=150);plt.close(fig)
    spec=cfg_from_experiment('configs/experiment/upstream_scale_s04_braking_lease_cuda_v5.yaml').spec
    failures=[];fig,axes=plt.subplots(2,1,figsize=(10,7),sharex=True)
    for run_name in a.failure_runs:
        run=Path(run_name).resolve()
        if not run.is_relative_to((ROOT/'outputs/runs').resolve()):raise ValueError('Expected owned failure run')
        snapshots=sorted(x for x in (run/'state').glob('*nmpc*step_*.pkl.gz') if x.with_suffix('.gz.json').is_file())
        snapshot=snapshots[-1];saved=restore_state(snapshot);env=saved['environment'];inputs[str(snapshot.relative_to(ROOT))]=file_sha256(snapshot)
        assert env['metadata']['contracts']['profile_fingerprint']==spec.fingerprint
        goals=np.zeros((2,4,2))
        for e in range(2):
            for agent in range(4):
                s=env['scripts'][e][agent];c,t=np.cos(s['layout_yaw']),np.sin(s['layout_yaw'])
                goals[e,agent]=env['anchors'][e,agent,:2]+s['site_center']+np.array([[c,-t],[t,c]])@np.asarray(spec.task.slots[agent])
        timeline={};symptoms=Counter()
        for trace in sorted((run/'metrics').glob('*nmpc*trace_*.jsonl')):
            raw=trace.read_bytes();complete=raw[:raw.rfind(b'\n')+1]
            archive=target/'config'/f'{run.name}_{trace.name}';archive.write_bytes(complete)
            inputs[str(trace.relative_to(ROOT))]={'complete_prefix_bytes':len(complete),'prefix_sha256':hashlib.sha256(complete).hexdigest()}
            for line in complete.splitlines():
                row=json.loads(line)
                if row['macro']<=saved['macro']:timeline[row['macro']]=row
        compact=[]
        for macro,row in sorted(timeline.items()):
            errors=np.linalg.norm(np.asarray(row['poses'])[...,:2]-goals,axis=-1)
            compact.append({'macro':macro,'times':row['time'],'slot_errors':errors.tolist()})
            for record in row['records']:
                for d in record.get('decisions',[]):symptoms[d['execution_reason'] or 'none']+=1
        for e in range(2):
            for agent in (2,3):axes[e].plot([r['times'][e] for r in compact],[r['slot_errors'][e][agent] for r in compact],label=f'{run.name.split("_")[0]} agent {agent}')
            axes[e].set_ylabel(f'Seed {23 if e==0 else 31}: slot error (m)');axes[e].legend()
        failures.append({'run':str(run),'last_saved_macro':saved['macro'],'timeline':compact,'execution_symptom_counts':dict(symptoms),
            'episodes':saved['episodes'],'violations':saved['violations'],'scope':'Symptoms are not causal labels; current run may be incomplete. Frozen complete trace prefixes stop at a checksum-protected public state.'})
    axes[-1].set_xlabel('Simulation seconds');fig.tight_layout();fig.savefig(target/'figures/nmpc_progress_comparison.png',dpi=150);fig.savefig(target/'figures/nmpc_progress_comparison.pdf');plt.close(fig)
    json_write(target/'metrics/failure_timelines.json',{'cases':failures,'formal_training':False})
    json_write(target/'run_manifest.json',{'kind':'offline_illustration','script_sha256':file_sha256(__file__),'inputs':inputs,'formal_training':False})
    print(target)


if __name__=='__main__':main()
