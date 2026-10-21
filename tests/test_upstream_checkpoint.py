"""OOM regression: shard publication, recovery, bounded history and storage."""
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from upstream_checkpoint import save_checkpoint, read_checkpoint, LEARNER_FORMAT
from upstream_process_rollout import ProcessUpstreamRollout
from upstream_run_state import save_state, restore_state
from run_upstream_t11 import make_collector
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_skrl import SkrlUpstreamTrainer
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_training import TrainingOptions

BASE='configs/experiment/upstream_scale_s05_benchmark_cpu_review_v2.yaml'


def test_shards_resume_exact_next_update_and_reject_corruption(tmp_path):
    torch.set_num_threads(1)
    collector=ProcessUpstreamRollout(BASE,'A1',23,owners=2,envs_per_owner=1,
        samples=8,episode_seconds=2.,cpu_workers=1)
    try:
        trainer=SkrlUpstreamTrainer(collector,TrainingOptions(rollout_macro_steps=2,epochs=1))
        batch,_,_=trainer.collect();trainer.diagnostic_update(batch)
        trainer.release_rollout()
        assert not trainer.agent.memories['team'].tensors
        target=tmp_path/'update_000001.pt'
        # Saving must never use the all-owner state RPC.
        original=collector._request
        def request(command,values):
            assert command!='state'
            return original(command,values)
        collector._request=request
        save_checkpoint(trainer,{'test':True},1,target,scope='diagnostic')
        checkpoint=read_checkpoint(target)
        assert 'owners' not in checkpoint['trainer']['collector']
        assert len(checkpoint['owner_shards'])==2
        expected,ends,_=trainer.collect()
        result=trainer.diagnostic_update(expected)
        parameters={k:v.clone() for k,v in collector.policy.state_dict().items()}
        trainer.load_state_dict(checkpoint['trainer'])
        actual,episodes,_=trainer.collect()
        assert episodes==ends
        for name,value in expected.items():
            if torch.is_tensor(value):assert torch.equal(value,actual[name]),name
        assert trainer.diagnostic_update(actual)==result
        for name,value in collector.policy.state_dict().items():assert torch.equal(value,parameters[name]),name
        learner=target.with_name('update_000001.learner.pt')
        with pytest.raises(ValueError,match='Learner-only'):read_checkpoint(learner)
        assert read_checkpoint(learner,learner_only=True)['checkpoint_format']==LEARNER_FORMAT
        shard=tmp_path/checkpoint['owner_shards'][0]['file']
        with shard.open('ab') as stream:stream.write(b'corrupt')
        with pytest.raises(ValueError,match='checksum'):read_checkpoint(target)
    finally:collector.close()


def test_failed_environment_save_keeps_learner_only_recovery(tmp_path,monkeypatch):
    torch.set_num_threads(1)
    collector=make_collector(BASE,'A1',23,samples=8,num_envs=1,mode='signal')
    trainer=SkrlUpstreamTrainer(collector,TrainingOptions(rollout_macro_steps=2,epochs=1))
    batch,_,_=trainer.collect()
    trainer.update(batch,readiness={'formal_training_ready':True,
        'profile_fingerprint':collector.env.spec.fingerprint},training_authorized=True)
    def fail(directory):raise OSError('injected shard write failure')
    monkeypatch.setattr(collector,'save_owner_shards',fail,raising=False)
    target=tmp_path/'update_000001.pt'
    with pytest.raises(OSError,match='injected'):save_checkpoint(trainer,{},1,target,scope='signal100k')
    assert not target.exists() and not target.with_suffix('.json').exists()
    recovery=read_checkpoint(tmp_path/'update_000001.learner.pt',learner_only=True)
    new=make_collector(BASE,'A1',23,samples=8,num_envs=1,mode='signal')
    learner=SkrlUpstreamTrainer(new,trainer.options)
    learner.import_learner_state(recovery['trainer'])
    assert new.transitions==2 and learner.updates==1
    for name,value in collector.policy.state_dict().items():assert torch.equal(value,new.policy.state_dict()[name])
    assert len(learner.optimizer.state)>0
    assert not new.hidden.count_nonzero()


def test_consumed_history_is_bounded_without_changing_rollout():
    torch.set_num_threads(1)
    collector=make_collector(BASE,'A1',23,samples=8,num_envs=1)
    initial=collector.state_dict()
    drain=collector.env.discard_delivered_history
    collector.env.discard_delivered_history=lambda:None
    expected,ends,records=collector.collect(4)
    assert collector.env._records
    collector.env.discard_delivered_history=drain
    collector.load_state_dict(initial)
    actual,episodes,trace=collector.collect(4)
    assert not collector.env._records and not collector.env._episodes
    assert len(trace)==len(records) and episodes==ends
    for name,value in expected.items():
        if torch.is_tensor(value):assert torch.equal(value,actual[name]),name
    for a,b in zip(actual['samples'],expected['samples']):assert torch.equal(a.log_prob,b.log_prob)


def test_streamed_interactions_keep_codec_and_are_append_only(tmp_path):
    (tmp_path/'state').mkdir()
    value={'array':np.arange(10000,dtype=np.float64),'tensor':torch.ones(8,4)}
    target=save_state(SimpleNamespace(path=tmp_path),'interactions',value)
    actual=restore_state(target)
    assert np.array_equal(value['array'],actual['array'])
    assert torch.equal(value['tensor'],actual['tensor'])
    with pytest.raises(FileExistsError):save_state(SimpleNamespace(path=tmp_path),'interactions',value)


def test_suite_ignores_newer_learner_only_backup(tmp_path,monkeypatch):
    import run_upstream_t11_signal_suite as suite
    base=tmp_path/'outputs/runs/upstream_scale_t11'
    run=base/'test_a1'
    (base/'test/logs').mkdir(parents=True)
    (run/'checkpoints').mkdir(parents=True)
    (run/'run_manifest.json').write_text('{}')
    for name in ('update_000014','update_000015.learner'):
        (run/'checkpoints'/f'{name}.pt').touch()
        (run/'checkpoints'/f'{name}.json').write_text('{}')
    commands=[]
    def launch(command,**kwargs):
        commands.append(command)
        return SimpleNamespace(pid=123,wait=lambda:1)
    monkeypatch.setattr(suite,'ROOT',tmp_path)
    monkeypatch.setattr(suite.subprocess,'Popen',launch)
    monkeypatch.setattr(sys,'argv',['suite','--config','base','--suite-id','test',
        '--readiness','evidence','--authorize-signal100k','--sampling-protocol','owners24_v1'])
    with pytest.raises(RuntimeError,match='failed'):suite.main()
    assert len(commands)==1
    command=commands[0]
    assert command[command.index('--num-envs')+1]=='24'
    assert command[command.index('--resume-state')+1].endswith('update_000014.pt')
