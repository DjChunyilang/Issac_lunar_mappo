#!/usr/bin/env python3
"""Four real recurrent action heads and bounded preparation updates, no training budget."""
import argparse
import json
import time

import numpy as np
import torch

from _common import ROOT,cfg_from_experiment
from diagnose_upstream_environment import initial_scene,event_data
from diagnose_upstream_planners import archive_inputs
from planner_evidence import EvidenceRun,json_write
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.action_ablation import build_ablation_policies,replay_action_sequence
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_environment import UpstreamEnvironment
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_ablation_learning import UpstreamAblationRollout,diagnostic_ablation_update
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_spec import parse_upstream_experiment


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--config',default='configs/experiment/upstream_scale_s04_target_probe_cuda_v4.yaml')
    args=parser.parse_args();torch.set_num_threads(1);base=cfg_from_experiment(args.config)
    run=EvidenceRun(ROOT/'outputs/runs/upstream_scale_t11'/args.run_id,args.config,
        settings={'stage':'four_actual_recurrent_heads_preparation_v1','seeds':[23,31,47],
                  'macro_steps':2,'mppi_samples':64,'formal_training':False,'maximum_updates':12})
    archive_inputs(run,base);run.resolved(base);panels=[]
    for seed in (23,31,47):
        torch.manual_seed(seed);policies=build_ablation_policies(991,base.spec)
        for representation,policy in policies.items():
            data=json.loads(base.configuration_json)
            data['experiment'].update(seed=seed,episode_length_s=2.,name=f'actual_{representation.lower()}_seed{seed}_preparation')
            data['upstream_scale'].update(mppi_samples=64,
                planner_adapter='primitive47_upstream_v1' if representation=='A3' else base.planner_adapter,
                planner_device='cpu' if representation=='A3' else base.planner_device)
            if representation=='A3':data['upstream_scale'].pop('mppi_parameters',None)
            cfg=parse_upstream_experiment(data,ROOT)
            env=UpstreamEnvironment(cfg,initial_scene(cfg,'spawn_flat'))
            collector=UpstreamAblationRollout(env,policy);start=time.perf_counter()
            batch,episodes,records=collector.collect(2)
            replay=replay_action_sequence(policy,batch['obs'],batch['hidden'][0],batch['samples'],
                batch['learning_valid'],batch['terminated'])
            ratio_error=float((replay['ratio'][batch['learning_valid']]-1).abs().max())
            if ratio_error>1e-5:raise ValueError('Actual recurrent probability replay mismatch')
            result=diagnostic_ablation_update(collector,torch.optim.Adam(policy.parameters(),lr=1e-4),batch)
            name=f'{representation.lower()}_seed{seed}'
            run.snapshot(name+'_actual_batch.pkl',{'batch':batch,'episodes':episodes,'records':records,
                'diagnostic_policy_state':collector.state_dict()})
            row={'representation':representation,'seed':seed,'resolved_contracts':collector.metadata,
                'actual_macro_transitions':collector.transitions,'actual_agent_decisions':len(batch['samples'])*env.e*4,
                'ratio_error':ratio_error,'update':result,'episodes':episodes,'seconds':time.perf_counter()-start,
                'mppi_samples':64,'scope':'two actual macro steps; not100k learning signal or task success'}
            json_write(run.path/'metrics'/f'{name}.json',event_data(row));panels.append(row)
            print(name,ratio_error,result,flush=True)
    json_write(run.path/'metrics/preparation_summary.json',event_data({'panels':panels,
        'actual_macro_transitions':sum(p['actual_macro_transitions'] for p in panels),
        'agent_decisions':sum(p['actual_agent_decisions'] for p in panels),
        'diagnostic_optimizer_updates':sum(p['update']['optimizer_updates'] for p in panels),
        'formal_training_interactions':0,'formal_training_ready':False,'t11_accepted':False,'gate_c_passed':False}))


if __name__=='__main__':main()
