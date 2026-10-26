#!/usr/bin/env python3
"""Derive immutable labels/coarse metrics from existing scenes, without resampling terrain."""
import argparse
from _common import cfg_from_experiment
from lunar_rover_tasks.terrain_data.scene_pack import ScenePack, file_hash
from lunar_rover_tasks.terrain_data.obstacle_atlas import build_obstacle_atlas


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',default='configs/experiment/upstream_1km_obstacle_input_v1.yaml')
    p.add_argument('--scene',action='append',required=True)
    args=p.parse_args();cfg=cfg_from_experiment(args.config);settings=cfg.obstacle_input
    for path in args.scene:
        scene=ScenePack(path)
        output=settings.atlas_root/file_hash(scene.path)
        print(build_obstacle_atlas(scene,output,settings,cfg.spec),flush=True)


if __name__=='__main__':main()
