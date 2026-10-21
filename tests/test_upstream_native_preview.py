"""Native information-time selection preserves the actual delayed execution."""
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_upstream_t11 import make_collector
from upstream_local_closure import choose_native_intent,native_preview_config
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_planning import UpstreamIntent


def test_all47_information_preview_then_actual_delayed_native_execution():
    torch.set_num_threads(1)
    env=make_collector('configs/experiment/upstream_scale_s05_benchmark_cpu_review_v2.yaml',
                       'A3',23,samples=16,num_envs=1,episode_seconds=2.).env
    preview=native_preview_config(env.cfg)
    assert preview.simulated_latency==0. and env.cfg.simulated_latency==.5
    assert preview.metadata['configuration_fingerprint']!=env.cfg.metadata['configuration_fingerprint']
    requests=[[UpstreamIntent(env.local_pose(0,a)[:2],True,0.,1.) for a in range(4)]]
    requests[0][0]=UpstreamIntent(env.local_pose(0,0)[:2]+[2.,0.],False,0.,1.)
    selected=choose_native_intent(env,0,requests[0][0])
    assert isinstance(selected,int) and 0<=selected<47
    env.enqueue(requests,primitive_ids=np.array([[selected,0,0,0]],np.int64))
    result=env.step_intents(None)
    activation=next(x for x in result.records if x.get('event')=='request_activation' and x['agent_id']==0)
    assert activation['primitive_id']==selected and activation['accepted']
    assert abs(activation['active_intent'].timestamp-.6)<1e-9
    assert activation['original_intent'].timestamp==0.
