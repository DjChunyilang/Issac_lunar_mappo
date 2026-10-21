"""Real four-head recurrent preparation, finite masks and unchanged samples."""
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from _common import cfg_from_experiment
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.action_ablation import build_ablation_policies,replay_action_sequence
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_environment import UpstreamEnvironment
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_ablation_learning import UpstreamAblationRollout,diagnostic_ablation_update


def scene(representation):
    cfg=cfg_from_experiment('configs/experiment/upstream_scale_s04_target_probe_cuda_v4.yaml')
    cfg=replace(cfg,planner_device='cpu',mppi_samples=16,
        simulation=replace(cfg.simulation,episode_length_s=2.),
        planner_adapter='primitive47_upstream_v1' if representation=='A3' else 'mppi_upstream_v4')
    poses=np.zeros((2,4,3));poses[:,:,:2]=[[0.,0.],[0.,12.],[12.,0.],[12.,12.]]
    poses[0,1,:2]=[0.,5.] # Designed first-step collision; not a legal birth benchmark.
    return cfg,poses


@pytest.mark.parametrize('representation',['A0','A1','A2','A3'])
def test_actual_four_head_sampling_replay_gru_cuts_restore_and_one_update(representation):
    torch.set_num_threads(1);torch.manual_seed(23)
    cfg,poses=scene(representation);policy=build_ablation_policies(991,cfg.spec)[representation]
    collector=UpstreamAblationRollout(UpstreamEnvironment(cfg,poses),policy)
    batch,episodes,_=collector.collect(1)
    assert batch['duration'].tolist()==[[1,5]]
    assert batch['terminal_reason'].tolist()==[[2,0]]
    assert batch['bootstrap_allowed'].tolist()==[[False,True]]
    assert not collector.hidden[0].any() and collector.hidden[1].abs().sum()>0
    replay=replay_action_sequence(policy,batch['obs'],batch['hidden'][0],batch['samples'],
                                 batch['learning_valid'],batch['terminated'])
    torch.testing.assert_close(replay['ratio'],torch.ones_like(replay['ratio']),atol=1e-5,rtol=1e-5)
    saved=collector.state_dict();tail=collector.hidden[1].clone()
    following,ends,_=collector.collect(1)
    assert torch.equal(following['hidden'][0,1],tail)
    assert following['terminal_reason'].tolist()==[[2,4]]
    assert not following['bootstrap_allowed'].any()
    fresh_policy=build_ablation_policies(991,cfg.spec)[representation]
    restored=UpstreamAblationRollout(UpstreamEnvironment(cfg,poses),fresh_policy);restored.load_state_dict(saved)
    again,_,_=restored.collect(1)
    assert torch.equal(following['reward'],again['reward'])
    torch.testing.assert_close(following['samples'][0].log_prob,again['samples'][0].log_prob,atol=1e-6,rtol=1e-6)
    original=again['samples'][0].action.continuous.clone();ids=again['samples'][0].action.primitive_id.clone()
    head=restored.policy.primitive_logits if representation=='A3' else restored.policy.mean
    before=head.weight.detach().clone()
    result=diagnostic_ablation_update(restored,torch.optim.Adam(restored.policy.parameters(),lr=1e-4),again)
    assert all(np.isfinite(v) for v in result.values()) and result['optimizer_updates']==1
    assert not torch.equal(before,head.weight)
    assert torch.equal(original,again['samples'][0].action.continuous)
    assert torch.equal(ids,again['samples'][0].action.primitive_id)
    assert all(e['terminal_reason'] in {'collision','task_deadline'} for e in episodes+ends)
