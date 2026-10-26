#!/usr/bin/env python3
"""Bounded old/new sampling comparison with identical fixed policy distributions.

No optimizer or learner checkpoint is constructed. Physics counts and real-time
TensorBoard are mandatory. Action heads are fixed explicitly for this benchmark;
full encoders/GRU and normal rollout/MPPI/execution still run.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
import time

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from _common import ROOT
from upstream_curriculum_experiment import load_curriculum,timing_for,initialize_fresh_policy
from upstream_decision_experiment import parameter_digest
from upstream_process_rollout import ProcessUpstreamRollout


def write(path,data):
    Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2,default=lambda x:x.tolist() if hasattr(x,'tolist') else str(x))+'\n')


def archive(path):
    files=subprocess.check_output(['rg','--files','source','scripts','configs','tests'],cwd=ROOT,text=True).splitlines()
    hashes={}
    with tarfile.open(path/'source.tar.gz','w:gz') as tar:
        for name in files:
            p=ROOT/name;hashes[name]=hashlib.sha256(p.read_bytes()).hexdigest();tar.add(p,arcname=name)
    write(path/'source_sha256.json',hashes)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run-id',required=True)
    p.add_argument('--owners',type=int,default=1)
    p.add_argument('--envs-per-owner',type=int,default=4)
    p.add_argument('--decisions',type=int,default=2)
    p.add_argument('--budget',type=int,default=800)
    p.add_argument('--arms',nargs='+',choices=['baseline','grid'],default=['baseline','grid'])
    a=p.parse_args();reservation=len(a.arms)*a.owners*a.envs_per_owner*a.decisions*25
    if min(a.owners,a.envs_per_owner,a.decisions)<1 or reservation>a.budget:
        raise ValueError('Probe reservation exceeds explicit physical step budget')
    root=ROOT/'outputs/runs/local_obstacle_input'/a.run_id;root.mkdir(parents=True,exist_ok=False)
    (root/'config').mkdir();archive(root/'config')
    manifest={'role':'bounded_sampling_probe','optimizer_updates':0,'base_step_budget':a.budget,
              'reserved_base_steps':reservation,'axis':'actual_base_environment_steps','args':vars(a),
              'policy':'zero mean/hold weights; fresh quarter std and p_hold=.35; identical per-interval torch RNG',
              'tensorboard':{'required':True,'url':'http://127.0.0.1:6007','directory':str(root)},'status':'running'}
    write(root/'run_manifest.json',manifest);total=0;results={}
    torch.set_num_threads(2)
    for arm in a.arms:
        path=root/arm;path.mkdir();(path/'metrics').mkdir();(path/'state').mkdir()
        writer=SummaryWriter(str(path/'tensorboard'),flush_secs=1)
        writer.add_text('run/configuration',json.dumps(manifest,ensure_ascii=False),0)
        writer.add_scalar('Progress/base_steps',0,0);writer.flush()
        config=load_curriculum('configs/training/upstream_1km_far_'+('soft_hazard' if arm=='baseline' else 'obstacle_input')+'_v1.yaml')
        base='configs/experiment/upstream_1km_'+('soft_hazard' if arm=='baseline' else 'obstacle_input')+'_v1.yaml'
        c=None;start=time.perf_counter();intervals=[];consumed=0
        try:
            c=ProcessUpstreamRollout(base,'A1',23,owners=a.owners,envs_per_owner=a.envs_per_owner,
                samples=512,episode_seconds=1500.,mode='diagnostic',cpu_workers=2,
                learning_time=timing_for(config),curriculum_config=config)
            initialize_fresh_policy(c.policy,config)
            with torch.no_grad():
                c.policy.mean.weight.zero_();c.policy.mean.bias.zero_();c.policy.hold_logit.weight.zero_()
            original=parameter_digest(c.policy)
            initial=c._request('perception_stats',[None]*a.owners)
            initialization=time.perf_counter()-start
            torch.save({'policy':c.policy.state_dict(),'metadata':c.metadata},path/'state/initial_policy.pt')
            write(path/'metrics/initialization.json',{'seconds':initialization,'metadata':c.metadata,'perception':initial})
            writer.add_scalar('Perf/initialization_s',initialization,0);writer.flush()
            for index in range(a.decisions):
                # Same distribution and raw policy samples across old/new input semantics.
                torch.manual_seed(1000+index)
                t=time.perf_counter();batch,episodes,records=c.collect(1);elapsed=time.perf_counter()-t
                steps=int(batch['duration'].sum());total+=steps;consumed+=steps
                if total>a.budget:raise AssertionError('Physical step accounting exceeded budget')
                stats=c._request('perception_stats',[None]*a.owners)
                data={'interval':index,'base_steps':steps,'cumulative_base_steps':consumed,'seconds':elapsed,'steps_s':steps/elapsed,
                      'valid_transitions':int(batch['learning_valid'].sum()),'episodes':episodes,'perception':stats,
                      'reward_mean':float(batch['reward'].mean()),'actions':batch['samples'][0].action.continuous,
                      'hold':batch['samples'][0].action.hold}
                write(path/f'metrics/interval_{index:03d}.json',data)
                torch.save({'batch':batch,'episodes':episodes,'records':records},path/f'state/interval_{index:03d}.pt')
                writer.add_scalar('Progress/base_steps',consumed,consumed);writer.add_scalar('Perf/steps_s',steps/elapsed,consumed)
                writer.add_scalar('Perf/interval_s',elapsed,consumed);writer.add_scalar('Progress/optimizer_updates',0,consumed)
                counts={}
                for record in records:
                    if 'obstacle_distance' in record:
                        d=np.asarray(record['obstacle_distance']);writer.add_scalar('Obstacle/min_distance',float(d.min()),consumed)
                    for name,value in record.get('reward_components',{}).items():counts.setdefault(name,[]).extend(np.asarray(value).ravel().tolist())
                for key,values in counts.items():writer.add_scalar('Reward/'+key,float(np.mean(values)),consumed)
                for key in ('obstacle_near','obstacle_overlap','terrain','goal','unknown'):
                    values=[r['planner_cost_parts'][key] for r in records if key in r.get('planner_cost_parts',{})]
                    if values:writer.add_scalar('MPPI/'+key,float(np.mean(values)),consumed)
                writer.flush();intervals.append(data)
                print(json.dumps({'arm':arm,'interval':index,'steps':steps,'seconds':elapsed,'steps_s':steps/elapsed}),flush=True)
            if parameter_digest(c.policy)!=original:raise AssertionError('Probe unexpectedly changed policy weights')
            result={'base_steps':consumed,'optimizer_updates':0,'initialization_s':initialization,
                    'sampling_s':sum(i['seconds'] for i in intervals),'intervals':intervals,'policy_unchanged':True}
            result['steps_s']=consumed/result['sampling_s'];write(path/'metrics/summary.json',result);results[arm]=result
        except BaseException as error:
            write(path/'metrics/failure.json',{'error':repr(error),'completed_steps':consumed,
                'uncommitted_step_upper_bound':a.owners*a.envs_per_owner*25,'total_budget':a.budget})
            manifest.update(status='failed',completed_base_steps=total,
                            uncommitted_step_upper_bound=a.owners*a.envs_per_owner*25)
            write(root/'run_manifest.json',manifest)
            raise
        finally:
            if c is not None:c.close()
            writer.close();del c;gc.collect()
    manifest.update(status='complete',actual_base_steps=total)
    write(root/'run_manifest.json',manifest)
    write(root/'summary.json',{'base_steps':total,'optimizer_updates':0,'arms':results,
          'grid_speedup':results['grid']['steps_s']/results['baseline']['steps_s'] if len(results)==2 else None})


if __name__=='__main__':main()
