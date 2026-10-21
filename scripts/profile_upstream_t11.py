#!/usr/bin/env python3
"""Bounded real CPU collection/PPO profile; no formal training/checkpoint."""
import argparse
import cProfile
from dataclasses import asdict
import pstats
import time
from collections import defaultdict
from contextlib import contextmanager
from functools import wraps

import torch

from _common import ROOT, cfg_from_experiment
from diagnose_upstream_planners import archive_inputs
from planner_evidence import EvidenceRun,json_write
from run_upstream_t11 import make_collector
from upstream_run_state import save_state
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_training import PreparedUpstreamTrainer,TrainingOptions


@contextmanager
def measured_calls(collector):
    """Explicit nested host timers; exclusive columns are disjoint."""
    from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_footprint import FootprintTerrain,FootprintReviewer
    from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_safety import UpstreamSafetyProjector as SafetyProjector
    targets=[(type(collector),'collect'),(type(collector.env),'tick'),(type(collector.env),'_refresh'),
        (type(collector.env),'_reports'),(type(collector.env),'observe'),(type(collector.env.pool),'plan_batch'),
        (type(collector.env.sensor),'sample'),(type(collector.env.memory[0][0]),'observe'),
        (FootprintTerrain,'lookup'),(FootprintReviewer,'pose'),(FootprintReviewer,'trajectory'),
        (SafetyProjector,'project'),(SafetyProjector,'execution_prediction')]
    stack=[];rows=defaultdict(lambda:{'calls':0,'inclusive_s':0.,'exclusive_s':0.});original=[]
    def wrapper(fn,key):
        @wraps(fn)
        def call(*args,**kwargs):
            start=time.perf_counter();frame=[0.];stack.append(frame)
            try:return fn(*args,**kwargs)
            finally:
                elapsed=time.perf_counter()-start;stack.pop()
                if stack:stack[-1][0]+=elapsed
                row=rows[key];row['calls']+=1;row['inclusive_s']+=elapsed;row['exclusive_s']+=elapsed-frame[0]
        return call
    try:
        for cls,name in targets:
            fn=getattr(cls,name);original.append((cls,name,fn));setattr(cls,name,wrapper(fn,cls.__name__+'.'+name))
        yield rows
    finally:
        for cls,name,fn in reversed(original):setattr(cls,name,fn)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--run-id',required=True)
    p.add_argument('--num-envs',type=int,choices=(1,2),default=1);p.add_argument('--macros',type=int,choices=(1,2,4),default=2)
    p.add_argument('--samples',type=int,choices=(16,64,512),default=64)
    p.add_argument('--timer',choices=('spans','cprofile'),default='spans');args=p.parse_args();torch.set_num_threads(1)
    if cfg_from_experiment(args.config).planner_device!='cpu':raise ValueError('This cProfile panel requires CPU; CUDA needs synchronized device timing')
    run=EvidenceRun(ROOT/'outputs/runs/upstream_scale_t11'/args.run_id,args.config,settings={'stage':'bounded_cpu_profile_v1',
        'seed':23,'num_envs':args.num_envs,'macros':args.macros,'samples':args.samples,'torch_threads':1,'timer':args.timer,'formal_training':False})
    archive_inputs(run,run.cfg);clock=time.perf_counter()
    collector=make_collector(args.config,'A1',23,samples=args.samples,num_envs=args.num_envs)
    setup=time.perf_counter()-clock;trainer=PreparedUpstreamTrainer(collector,TrainingOptions(rollout_macro_steps=args.macros))
    rows=[];spans={};clock=time.perf_counter()
    if args.timer=='cprofile':
        profiler=cProfile.Profile();profiler.enable()
        batch,episodes,records=trainer.collect();profiler.disable()
        profiler.dump_stats(str(run.path/'metrics/collection.prof'));statistics=pstats.Stats(profiler)
        rows=[{'file':key[0],'line':key[1],'function':key[2],'primitive_calls':v[0],'calls':v[1],
               'self_seconds':v[2],'cumulative_seconds':v[3]} for key,v in statistics.stats.items()]
    else:
        with measured_calls(collector) as spans:batch,episodes,records=trainer.collect()
    collect=time.perf_counter()-clock
    clock=time.perf_counter();learning=trainer.diagnostic_update(batch);update=time.perf_counter()-clock
    clock=time.perf_counter();save_state(run,'interactions_000001',{'samples':batch['samples'],'execution':records,
        'duration':batch['duration'].cpu(),'learning_valid':batch['learning_valid'].cpu(),
        'learning':{k:batch[k].cpu() for k in ('reward','value','next_value','terminated','truncated','bootstrap_allowed','terminal_reason')}})
    saving=time.perf_counter()-clock
    json_write(run.path/'metrics/profile.json',{'setup_s':setup,'collect_s':collect,'ppo_update_s':update,'interaction_save_s':saving,
        'valid_team_transitions':int(batch['learning_valid'].sum()),'options':asdict(trainer.options),'learning':learning,'episodes':episodes,
        'top_cumulative':sorted(rows,key=lambda r:r['cumulative_seconds'],reverse=True)[:60],
        'top_self':sorted(rows,key=lambda r:r['self_seconds'],reverse=True)[:60],
        'profile_total_self_s':sum(r['self_seconds'] for r in rows),'spans':dict(spans),
        'cprofile_cumulative_consistent':None if not rows else all(r['cumulative_seconds']<=collect*1.01 for r in rows),
        'formal_training':False,'training_checkpoint_created':False,
        'scope':'fresh CPU environment, bounded diagnostic; profiler overhead and concurrent workers apply; cumulative times overlap and must not be added'})
    print({'run':str(run.path),'collect_s':collect,'ppo_update_s':update,'valid_team_transitions':collector.transitions},flush=True)


if __name__=='__main__':main()
