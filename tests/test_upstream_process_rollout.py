"""Persistent owners consume one central policy batch and resume at macro boundaries."""
from pathlib import Path
from dataclasses import asdict,is_dataclass
import sys
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from upstream_process_rollout import ProcessUpstreamRollout
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_skrl import SkrlUpstreamTrainer
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_training import TrainingOptions


def test_parallel_owner_rollout_matches_serial_and_updates():
    torch.set_num_threads(1)
    collector=ProcessUpstreamRollout('configs/experiment/upstream_scale_s05_benchmark_cpu_review_v2.yaml','A1',23,
        owners=2,envs_per_owner=1,samples=8,episode_seconds=2.,cpu_workers=1)
    try:
        trainer=SkrlUpstreamTrainer(collector,TrainingOptions(rollout_macro_steps=2,epochs=1))
        state=collector.state_dict()
        collector.serial_dispatch=True;expected,episodes,_=trainer.collect()
        collector.load_state_dict(state);collector.serial_dispatch=False;actual,ends,_=trainer.collect()
        for key,value in expected.items():
            if torch.is_tensor(value):assert torch.equal(value,actual[key]),key
        assert episodes==ends
        for a,b in zip(expected['samples'],actual['samples']):
            assert torch.equal(a.log_prob,b.log_prob)
            assert torch.equal(a.action.continuous,b.action.continuous)
            assert torch.equal(a.action.hold,b.action.hold)
        result=trainer.diagnostic_update(actual)
        assert result['updates']==1 and result['valid_transitions']==4
        assert collector.transitions==4
    finally:collector.close()
    assert not collector.processes


def test_learner_import_resets_cohort_preserves_adam_and_continues():
    from run_upstream_t11 import make_collector
    torch.set_num_threads(1)
    base='configs/experiment/upstream_scale_s05_benchmark_cpu_review_v2.yaml'
    source=make_collector(base,'A1',23,samples=8,num_envs=1,mode='signal')
    options=TrainingOptions(rollout_macro_steps=2,epochs=1)
    old=SkrlUpstreamTrainer(source,options)
    audit={'formal_training_ready':True,'profile_fingerprint':source.env.spec.fingerprint}
    batch,_,_=old.collect();old.update(batch,readiness=audit,training_authorized=True)
    saved=old.state_dict()
    cohort=ProcessUpstreamRollout(base,'A1',23,owners=2,envs_per_owner=1,samples=8,cpu_workers=1,mode='signal')
    try:
        trainer=SkrlUpstreamTrainer(cohort,options)
        trainer.import_learner_state(saved)
        assert cohort.hidden.shape==(2,4,128) and not torch.count_nonzero(cohort.hidden)
        assert cohort.transitions==2 and trainer.updates==trainer.agent.epoch_updates==1
        for key,value in source.policy.state_dict().items():assert torch.equal(value,cohort.policy.state_dict()[key])
        for key,slots in saved['optimizer']['state'].items():
            for name,value in slots.items():assert torch.equal(value,trainer.optimizer.state_dict()['state'][key][name])
        boundary=trainer.state_dict()
        trainer.load_state_dict(boundary)
        batch,_,_=trainer.collect()
        result=trainer.update(batch,readiness=audit,training_authorized=True)
        assert result['valid_transitions']==4 and cohort.transitions==6 and trainer.updates==2
    finally:cohort.close()
