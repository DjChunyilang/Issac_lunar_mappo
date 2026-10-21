#!/usr/bin/env python3
"""Freeze four actual-head training definitions without running formal training."""
import argparse
from dataclasses import asdict,replace
import json

from _common import ROOT,cfg_from_experiment
from diagnose_upstream_planners import archive_inputs
from planner_evidence import EvidenceRun,json_write,file_sha256
from upstream_local_closure import runtime_config
from upstream_s05_cases import gathering_scene
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_training import TrainingOptions,TRAINER_VERSION
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_baselines import primitive_fingerprint


def independent_scenes(cfg):
    # Match evaluate(): one independent environment for each family/seed.
    one=replace(cfg,simulation=replace(cfg.simulation,num_envs=1))
    return [{'family':family,'seed':seed,'num_envs':1,
        'initial_poses':gathering_scene(one,family,seed).tolist(),
        'map_scope':'same NPB development window' if family=='npb' else 'analytic/flat proxy'}
        for family in ('compact','irregular','reverse','crossing','slope','npb') for seed in (151,157,163)]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--run-id',required=True)
    p.add_argument('--readiness',help='Owned post-evaluation readiness audit; absent means not ready')
    args=p.parse_args();cfg=cfg_from_experiment(args.config);options=TrainingOptions()
    run=EvidenceRun(ROOT/'outputs/runs/upstream_scale_t11'/args.run_id,args.config,
        settings={'stage':'formal_training_definitions_prepared_v1','representations':['A0','A1','A2','A3'],
            'options':asdict(options),'budgets':[100000,500000,2000000],'seeds':[23,31,47],'formal_training':False})
    archive_inputs(run,cfg);run.resolved(cfg)
    definitions=[]
    for representation in ('A0','A1','A2','A3'):
        native=representation=='A3'
        effective,data=runtime_config(args.config,name=f'upstream_t11_{representation.lower()}',seed=23,
            adapter='primitive47_upstream_v1' if native else cfg.planner_adapter,device='cpu' if native else cfg.planner_device,
            episode_seconds=360.,safety=True,num_envs=2)
        row={'representation':representation,'runtime_configuration':data,'runtime_contracts':effective.metadata,
            'actor_dim':411,'critic_dim':991,'gru_dim':128,'trainer_version':TRAINER_VERSION,'training_options':asdict(options),
            'budget_definition':'valid team macro transitions; count agent decisions separately',
            'stages':[{'transitions':100000,'seeds':[23],'exit':'signal check before escalation'},
                      {'transitions':500000,'seeds':[23,31,47],'exit':'multiseed positive evidence'},
                      {'transitions':2000000,'seeds':[23,31,47],'exit':'representation retained after lower stage'}],
            'shared_backbone_initialization':'identical backbone perseed; separate necessary output heads',
            'native_execution_library':primitive_fingerprint(cfg.spec) if native else None,
            'source_probability_payload':'original sampled values/ID unchanged by decoding or execution',
            'independent_eval_seeds':[151,157,163],'training_eligible':False,'pause_released':False}
        json_write(run.path/'config'/f'{representation.lower()}.json',row);definitions.append(row)
    scenes=independent_scenes(cfg)
    json_write(run.path/'config/independent_evaluation_scenes.json',scenes)
    readiness=None if args.readiness is None else {'path':args.readiness,'sha256':file_sha256(args.readiness),
        'value':json.loads(open(args.readiness).read())}
    json_write(run.path/'metrics/preparation_summary.json',{'representations':list('A'+str(i) for i in range(4)),
        'actual_runtime_definitions':len(definitions),'independent_scene_definitions':len(scenes),
        'readiness':readiness,'formal_training_ready':False,'training_interactions':0,'optimizer_updates':0,
        'checkpoint_created':False,'pause_released':False,
        'remaining':'S04 closure, complete S05 Gate B and144 gathering matrix, dependency-bound acceptance audit'})


if __name__=='__main__':main()
