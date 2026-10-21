#!/usr/bin/env python3
"""Sequential user-authorized A1/A0/A2/A3 signal runs; never escalate100k."""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

from _common import ROOT
from planner_evidence import json_write


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--suite-id',required=True)
    parser.add_argument('--readiness',required=True)
    parser.add_argument('--authorize-signal100k',action='store_true')
    parser.add_argument('--sampling-protocol',choices=('legacy2','owners48_v1','owners24_v1','owners16_v1'),default='legacy2')
    args=parser.parse_args()
    if not args.authorize_signal100k:parser.error('Explicit signal authorization required')
    base=ROOT/'outputs/runs/upstream_scale_t11';suite=base/args.suite_id
    with (suite/'logs/suite.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for representation in ('A1','A0','A2','A3'):
            run_id=args.suite_id+'_'+representation.lower();run=base/run_id
            summaries=sorted((run/'metrics').glob('session_summary_*.json'))
            if summaries and json.loads(summaries[-1].read_text()).get('signal_budget_complete') is True:
                continue
            command=[sys.executable,'scripts/run_upstream_t11.py','--config',args.config,'--run-id',run_id,
                '--representation',representation,'--mode','signal','--authorize-signal100k',
                '--readiness',args.readiness,'--seed','23','--budget','100000','--samples','512',
                '--num-envs','2','--rollout-steps','128']
            if args.sampling_protocol!='legacy2':
                owners,environments={'owners48_v1':('6','48'),'owners24_v1':('6','24'),'owners16_v1':('4','16')}[args.sampling_protocol]
                command[command.index('--num-envs')+1]=environments
                command+=['--env-owners',owners,'--owner-cpu-workers','2','--sampling-protocol',args.sampling_protocol]
            if (run/'run_manifest.json').exists():
                states=sorted(p for p in (run/'checkpoints').glob('update_*.pt')
                    if p.stem.removeprefix('update_').isdigit() and p.with_suffix('.json').exists())
                if not states:raise RuntimeError('Existing signal session has no committed update; preserve it and investigate')
                command+=['--resume-state',str(states[-1])]
            stamp=time.time_ns();log=suite/'logs'/f'{representation}_{stamp}.log'
            with log.open('x') as stream:
                process=subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
                json_write(suite/'metrics'/f'launch_{representation}_{stamp}.json',
                    {'pid':process.pid,'command':command,'log':str(log),'created_unix':time.time()})
                print('START',representation,'PID',process.pid,flush=True)
                code=process.wait()
            json_write(suite/'metrics'/f'exit_{representation}_{stamp}.json',{'returncode':code,'finished_unix':time.time()})
            if code:raise RuntimeError(f'{representation} failed; later candidates remain queued; see {log}')
        json_write(suite/'metrics'/f'suite_complete_{time.time_ns()}.json',
            {'order':['A1','A0','A2','A3'],'budget_each':100000,'seed':23,'gate_c_passed':False,
             'system_acceptance_complete':False})


if __name__=='__main__':main()
