"""Prepared learner preserves actual packets and cannot silently start training."""
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from upstream_local_closure import runtime_config
from upstream_s05_cases import formal_case,gathering_scene
from upstream_run_state import save_state,restore_state
from types import SimpleNamespace
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_environment import UpstreamEnvironment
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.action_ablation import build_ablation_policies
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_ablation_learning import UpstreamAblationRollout
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_training import PreparedUpstreamTrainer,TrainingOptions


def test_actual_prepared_rollout_guard_and_atomic_public_state():
    torch.set_num_threads(1);torch.manual_seed(23)
    cfg,_=runtime_config('configs/experiment/upstream_scale_s04_selected_cuda_v4.yaml',name='training_preparation_contract',seed=23,
        samples=16,device='cpu',episode_seconds=2.,safety=True)
    poses=gathering_scene(cfg,'compact',23)
    collector=UpstreamAblationRollout(UpstreamEnvironment(cfg,poses),build_ablation_policies(991,cfg.spec)['A1'])
    trainer=PreparedUpstreamTrainer(collector,TrainingOptions(rollout_macro_steps=2))
    batch,_,_=trainer.collect()
    assert len(batch['samples'])==2 and batch['duration'].tolist()==[[5],[5]]
    originals=[s.action.continuous.clone() for s in batch['samples']]
    with pytest.raises(ValueError,match='paused'):trainer.update(batch,readiness={})
    assert trainer.optimizer is None and trainer.updates==0
    assert all(torch.equal(x,s.action.continuous) for x,s in zip(originals,batch['samples']))
    state=trainer.state_dict();bad={**state,'updates':-1}
    with pytest.raises(ValueError):trainer.load_state_dict(bad)
    assert torch.equal(state['collector']['hidden'],trainer.collector.hidden.cpu())


def test_formal_cases_bind_peer_variation_expiry_and_birth_geometry():
    torch.set_num_threads(1)
    cfg,_=runtime_config('configs/experiment/upstream_scale_s04_selected_cuda_v4.yaml',name='case_contract',seed=23)
    a=formal_case(cfg,'cross',23,0);b=formal_case(cfg,'cross',23,3)
    assert a.fingerprint!=b.fingerprint
    assert not np.array_equal(a.input.neighbors[0].positions,b.input.neighbors[0].positions)
    expired=formal_case(cfg,'flat',23,1002,'expired_request')
    assert expired.input.intent.expires_at<expired.input.now
    for family,tier in (('compact',0),('irregular',1),('reverse',2)):
        scene=gathering_scene(cfg,family,23)[0];r=np.linalg.norm(scene[1:,:2],axis=-1)
        assert (r>=cfg.spec.task.spawn_tiers[tier][0]).all() and (r<=cfg.spec.task.spawn_tiers[tier][1]).all()
        distances=np.linalg.norm(scene[:,None,:2]-scene[None,:,:2],axis=-1)+np.eye(4)*1000
        assert distances.min()>=cfg.spec.task.initial_distance


def test_compressed_run_state_is_lossless_and_rejects_damage(tmp_path):
    (tmp_path/'state').mkdir();run=SimpleNamespace(path=tmp_path)
    state={'clock':np.array([0.,1.]),'sender':(0,3),'tensor':torch.arange(10)}
    target=save_state(run,'snapshot',state);loaded=restore_state(target)
    np.testing.assert_array_equal(loaded['clock'],state['clock']);assert torch.equal(loaded['tensor'],state['tensor'])
    target.write_bytes(target.read_bytes()+b'bad')
    with pytest.raises(ValueError,match='checksum'):restore_state(target)
