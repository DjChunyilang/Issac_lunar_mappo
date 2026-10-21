"""Original T11 payloads retain probability and use explicit physical decoding."""
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from _common import cfg_from_experiment
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.action_ablation import ActionContract,ActionPayload
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.baselines import PRIMITIVE_VERSION,primitive_fingerprint
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_action_bridge import UpstreamActionBridge
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_environment import UpstreamEnvironment


def scene(native=False):
    path='configs/experiment/upstream_scale_s03_environment_native_v2.yaml' if native else 'configs/experiment/upstream_scale_s03_environment_v2.yaml'
    cfg=cfg_from_experiment(path);cfg=replace(cfg,mppi_samples=32)
    p=np.zeros((2,4,3));p[...,:2]=cfg.spec.task.slots
    return UpstreamEnvironment(cfg,p)


@pytest.mark.parametrize('representation',['A0','A1','A2'])
def test_continuous_zero_move_and_square_corner_decoding_preserves_raw_payload(representation):
    env=scene();contract=ActionContract.from_spec(representation,env.spec);bridge=UpstreamActionBridge(env,contract)
    continuous=torch.zeros((2,4,contract.continuous_dim));hold=torch.zeros((2,4),dtype=torch.bool)
    if representation=='A2':continuous[...,0:2]=1.;continuous[...,3]=-1.
    if representation=='A1':hold[0,0]=True
    payload=ActionPayload(continuous,hold,torch.full((2,4),-1,dtype=torch.int64),contract.version)
    original=continuous.clone();requests,ids=bridge.decode(payload)
    assert ids is None and torch.equal(payload.continuous,original)
    for e in range(2):
        for a in range(4):
            intent=requests[e][a]
            assert intent.hold==bool(hold[e,a])
            assert np.linalg.norm(intent.goal_xy-env.local_pose(e,a)[:2])<=4.+1e-12
            if representation=='A2':assert intent.speed_cap==0. and not intent.hold
    if representation=='A2':assert np.linalg.norm(requests[0][0].goal_xy)==pytest.approx(4.)


def test_native_ids_and_explicit_new_library_reach_actual_environment_queue():
    env=scene(native=True);contract=ActionContract.from_spec('A3',env.spec);bridge=UpstreamActionBridge(env,contract)
    ids=torch.tensor([[0,40,41,42],[43,44,45,46]],dtype=torch.int64)
    payload=ActionPayload(torch.empty((2,4,0)),torch.zeros((2,4),dtype=torch.bool),ids,contract.version,
                          PRIMITIVE_VERSION,primitive_fingerprint())
    transition=bridge.step(payload,max_steps=3)
    assert torch.equal(payload.primitive_id,ids) and not env.pending
    events=[r for r in transition.records if r.get('event')=='request_activation']
    assert [r['primitive_id'] for r in events]==ids.flatten().tolist()
    assert bridge.metadata['native_library_fingerprint']!=payload.primitive_fingerprint
    for e in range(2):
        for a in range(4):assert env.plans[e][a].diagnostics['primitive_id']==ids[e,a]
