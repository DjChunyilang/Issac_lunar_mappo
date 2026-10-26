#!/usr/bin/env python3
"""Fixed-pose frontend comparison. No environment/optimizer advancement."""
import argparse
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from _common import cfg_from_experiment
from lunar_rover_tasks.terrain_data.batched import SceneBank
from lunar_rover_tasks.terrain_data.obstacle_atlas import ObstacleAtlas
from lunar_rover_tasks.terrain_data.scene_pack import file_hash
from lunar_rover_tasks.tasks.multi_rover_gathering.gathering_env_cfg import TerrainCfg
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_grid import GridMemory
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_sensing import DenseSensor,DenseTerrainMemory
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_footprint import FootprintReviewer


def main():
    p=argparse.ArgumentParser();p.add_argument('--initialization',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    cfg=cfg_from_experiment('configs/experiment/upstream_1km_obstacle_input_v1.yaml')
    initial=json.loads(a.initialization.read_text());poses0=np.array(initial['perception'][0]['poses'])[0:1]
    anchors=poses0.copy();anchors[...,2]=0
    scene='outputs/runs/lunar_terrain_validation/npb_1km_fixed_20261021_01/scenes/train_0/scene.json'
    bank=SceneBank([scene],storage='mmap');bank.prefetch([0]);runtime=SimpleNamespace(scene_bank=bank,scene_indices=torch.zeros(1,dtype=torch.long))
    terrain=TerrainCfg(type='lunar_scene_pack',raster_risk_slope_scale=.6,raster_risk_rms_scale_m=.05)
    atlas=ObstacleAtlas(cfg.obstacle_input.atlas_root/file_hash(scene)/'obstacle.json',cfg.obstacle_input,bank.scenes[0])
    dense=DenseSensor(cfg.spec,terrain,runtime=runtime);physical=DenseSensor(cfg.spec,terrain,runtime=runtime)
    reviewer=FootprintReviewer(cfg.spec,version='footprint_review_v2');physical.prepare_birth(reviewer.points)
    old=[DenseTerrainMemory(cfg.spec) for _ in range(4)];new=[GridMemory(cfg.spec,cfg.obstacle_input) for _ in range(4)]
    writer=SummaryWriter(str(a.output/'tensorboard'),flush_secs=1)
    writer.add_text('run/units','x axis=fixed sensor frame; 0 environment steps, 0 optimizer steps; alternating old/new order',0);writer.flush()
    rows=[];physical_cache=[None]*4
    for i in range(16):
        # Include stationary repeats, then translated and rotated observations.
        delta=max(0,i-3)*.2;pose=poses0.copy()
        pose[...,:2]+=delta*np.stack([np.cos(poses0[...,2]),np.sin(poses0[...,2])],-1)
        pose[...,2]+=max(0,i-6)*.012
        times={'frame':i};reports={}
        def old_frame():
            t=time.perf_counter();frames=dense.sample(pose,anchors,[i*.2]);times['dense_query_s']=time.perf_counter()-t
            t=time.perf_counter()
            for agent in range(4):old[agent].observe(frames[0][agent])
            times['dense_memory_s']=time.perf_counter()-t;t=time.perf_counter();out=[]
            for agent in range(4):
                local=np.r_[pose[0,agent,:2]-anchors[0,agent,:2],pose[0,agent,2]]
                out.append(vars(reviewer.pose(local,old[agent].view(),parking=True)))
            times['dense_review_s']=time.perf_counter()-t;reports['dense']=out
        def new_frame():
            t=time.perf_counter()
            for agent in range(4):new[agent].update(pose[0,agent],anchors[0,agent],i*.2,atlas)
            times['grid_update_s']=time.perf_counter()-t;t=time.perf_counter();out=[]
            for agent in range(4):
                key=pose[0,agent].tobytes()
                if physical_cache[agent] is None or physical_cache[agent][0]!=key:
                    physical_cache[agent]=(key,physical.birth_terrain(pose[0,agent],scene_index=0))
                out.append(vars(reviewer.pose(np.r_[0.,0.,pose[0,agent,2]],physical_cache[agent][1],parking=True)))
            times['grid_physical_s']=time.perf_counter()-t;reports['grid']=out
        for call in ((old_frame,new_frame) if i%2==0 else (new_frame,old_frame)):call()
        times['dense_total_s']=sum(times[k] for k in ('dense_query_s','dense_memory_s','dense_review_s'))
        times['grid_total_s']=times['grid_update_s']+times['grid_physical_s']
        for key,value in times.items():
            if key!='frame':writer.add_scalar('Frontend/'+key,value,i)
        writer.flush();rows.append({'timing':times,'reports':reports,'new_cells_cumulative':sum(m.stats['new_cells'] for m in new)})
    writer.close()
    total=lambda key:sum(r['timing'][key] for r in rows)
    result={'base_steps':0,'optimizer_updates':0,'frames':len(rows),'rovers':4,
            'dense_query_points':dense.queried_points,'grid_physical_query_points':physical.queried_points,
            'grid_atlas_cells_read':sum(m.stats['atlas_cells_read'] for m in new),
            'dense_s':total('dense_total_s'),'grid_s':total('grid_total_s'),
            'frontend_speedup':total('dense_total_s')/total('grid_total_s'),
            'source_scene_sha256':file_hash(scene),'atlas_sha256':atlas.identity,'rows':rows,
            'grid_memory_stats':[m.stats for m in new]}
    (a.output/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','grid_memory_stats')}),flush=True)


if __name__=='__main__':main()
