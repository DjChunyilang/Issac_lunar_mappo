"""100k permission does not assert full-system readiness or authorize500k."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import audit_upstream_t11_readiness as audit
import run_upstream_t11 as entry
from _common import cfg_from_experiment
from planner_evidence import file_sha256


@pytest.fixture
def signal(tmp_path,monkeypatch):
    cfg=cfg_from_experiment('configs/experiment/upstream_scale_s04_braking_lease_cuda_v5.yaml')
    for module in (audit,entry):
        monkeypatch.setattr(module,'ROOT',tmp_path)
        monkeypatch.setattr(module,'source_snapshot',lambda:{'runtime.py':'frozen'})
    monkeypatch.setattr(entry,'cfg_from_experiment',lambda _:cfg)
    folder=tmp_path/'outputs/runs/signal';folder.mkdir(parents=True)
    junit=folder/'tests.xml';junit.write_text('<testsuite><testcase name="contract"/></testsuite>')
    validation=folder/'validation.json';validation.write_text(json.dumps({'failures':0,'skips':0,
        'junit':[{'path':str(junit),'sha256':file_sha256(junit)}]}))
    result=audit.audit_signal_readiness(cfg,validation,authorized=True)
    path=folder/'readiness.json';path.write_text(json.dumps(result))
    return cfg,validation,path,result


def test_signal_admits_without_matrices_but_not_system(signal):
    cfg,_,path,result=signal
    assert result['signal_training_ready'] and not result['formal_training_ready']
    assert not result['system_acceptance_complete']
    assert entry.readiness(path,cfg,scope='signal100k',budget=100000,representation='A1',seed=23,base_config='base')==result
    with pytest.raises(ValueError,match='readiness'):entry.readiness(path,cfg)


@pytest.mark.parametrize('budget,seed',[(500000,23),(2000000,23),(100000,31)])
def test_signal_cannot_escalate_or_change_seed(signal,budget,seed):
    cfg,_,path,_=signal
    with pytest.raises(ValueError,match='scope'):
        entry.readiness(path,cfg,scope='signal100k',budget=budget,representation='A1',seed=seed,base_config='base')


def test_signal_needs_authorization_and_unchanged_evidence(signal):
    cfg,validation,path,_=signal
    assert not audit.audit_signal_readiness(cfg,validation)['signal_training_ready']
    validation.write_text('{}')
    with pytest.raises(ValueError,match='changed'):
        entry.readiness(path,cfg,scope='signal100k',budget=100000,representation='A1',seed=23,base_config='base')


def test_explicit_owner48_protocol_and_budget_tail():
    entry.validate_signal_sampling('owners48_v1',owners=6,num_envs=48,rollout_steps=128,cpu_workers=2)
    with pytest.raises(ValueError,match='mismatch'):
        entry.validate_signal_sampling('legacy2',owners=6,num_envs=48,rollout_steps=128,cpu_workers=2)
    for used in (3584,95616,99968,100000):
        length=entry.remaining_rollout_steps(100000,used,48,128)
        assert used+length*48<=100000
    assert entry.remaining_rollout_steps(100000,99968,48,128)==0
    assert entry.remaining_rollout_steps(100000,99840,48,128)==3


def test_explicit_owner24_memory_protocol():
    entry.validate_signal_sampling('owners24_v1',owners=6,num_envs=24,rollout_steps=128,cpu_workers=2)
    with pytest.raises(ValueError,match='mismatch'):
        entry.validate_signal_sampling('owners48_v1',owners=6,num_envs=24,rollout_steps=128,cpu_workers=2)
    for used in (3584,9728,99968,99992,100000):
        length=entry.remaining_rollout_steps(100000,used,24,128)
        assert used+length*24<=100000
    assert entry.remaining_rollout_steps(100000,99992,24,128)==0


def test_explicit_owner16_memory_protocol():
    entry.validate_signal_sampling('owners16_v1',owners=4,num_envs=16,rollout_steps=128,cpu_workers=2)
    with pytest.raises(ValueError,match='mismatch'):
        entry.validate_signal_sampling('owners16_v1',owners=6,num_envs=16,rollout_steps=128,cpu_workers=2)
    assert entry.remaining_rollout_steps(100000,99984,16,128)==1
