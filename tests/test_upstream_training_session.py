"""Execute the prepared PPO path and protect its distinct diagnostic lifecycle."""
from pathlib import Path
import subprocess
import sys

import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_upstream_t11 import make_collector, readiness
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_training import PreparedUpstreamTrainer, TrainingOptions


def test_actual_complete_update_and_optimizer_recovery(tmp_path):
    torch.set_num_threads(1)
    base='configs/experiment/upstream_scale_s05_benchmark_cpu_review_v2.yaml'
    collector=make_collector(base,'A1',23,samples=16,num_envs=1,episode_seconds=2.)
    trainer=PreparedUpstreamTrainer(collector,TrainingOptions(rollout_macro_steps=2))
    batch,episodes,records=trainer.collect()
    assert episodes and collector.env._ended.all()
    activations=[r for r in records if r.get('event')=='request_activation']
    assert activations and all('planner_outcome' in r for r in activations)
    values=[s.action.continuous.clone() for s in batch['samples']]
    initial=[p.detach().clone() for p in collector.policy.parameters()]
    result=trainer.diagnostic_update(batch)
    assert result['updates']==5 and result['formal_training_interactions']==0
    assert result['diagnostics']['version']=='t11_learning_diagnostics_v1'
    assert result['epochs'][0]['ratio_mean']==pytest.approx(1.,abs=1e-6)
    assert result['epochs'][0]['approx_kl']==pytest.approx(0.,abs=1e-6)
    assert all(0<=epoch['clip_fraction']<=1 and epoch['ratio_min']>0 for epoch in result['epochs'])
    assert any(not torch.equal(a,b) for a,b in zip(initial,collector.policy.parameters()))
    assert all(torch.equal(a,s.action.continuous) for a,s in zip(values,batch['samples']))
    state=trainer.state_dict()
    replay=PreparedUpstreamTrainer(make_collector(base,'A1',23,samples=16,num_envs=1,episode_seconds=2.),trainer.options)
    replay.load_state_dict(state)
    assert replay.updates==5 and replay.diagnostic_batches==1 and replay.optimizer.state
    assert all(torch.equal(a,b) for a,b in zip(collector.policy.parameters(),replay.collector.policy.parameters()))
    # An ordinary chunk starts a new episode only because the saved one ended.
    next_batch,_,_=replay.collect(1)
    assert torch.count_nonzero(next_batch['hidden'][0])==0
    replay.diagnostic_update(next_batch)
    with pytest.raises(ValueError,match='limited'):replay.diagnostic_update(next_batch)
    with pytest.raises(ValueError,match='diagnostic learner'):
        replay.update(next_batch,training_authorized=True,
                      readiness={'formal_training_ready':True,'profile_fingerprint':collector.env.spec.fingerprint})
    with pytest.raises(ValueError,match='Missing'):readiness(None,collector.env.cfg)


def test_cli_rejects_formal_training_before_run_creation():
    root=Path(__file__).resolve().parents[1]
    result=subprocess.run([sys.executable,'scripts/run_upstream_t11.py','--config',
        'configs/experiment/upstream_scale_s05_benchmark_cpu_review_v2.yaml','--run-id',
        'forbidden_training_entry_contract','--representation','A1','--mode','train'],
        cwd=root,text=True,capture_output=True)
    assert result.returncode!=0 and 'remains paused' in result.stderr
    assert not (root/'outputs/runs/upstream_scale_t11/forbidden_training_entry_contract').exists()
