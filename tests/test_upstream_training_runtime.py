"""Learning runtime identity is separate from evidence and pause release."""
from copy import deepcopy
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_upstream_t11 import make_collector, session_configuration
from upstream_local_closure import runtime_config
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_spec import parse_upstream_experiment
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_training import PreparedUpstreamTrainer

BASE='configs/experiment/upstream_scale_s05_benchmark_cpu_review_v2.yaml'


def test_runtime_requires_explicit_matching_role_and_keeps_profile():
    cfg,data=runtime_config(BASE,name='runtime_role',seed=23,samples=16)
    assert session_configuration(cfg,'diagnostic') is cfg
    assert 'training_session_version' not in cfg.metadata
    learning=session_configuration(cfg,'train')
    assert learning.learning_runtime and not learning.metadata['training_paused']
    assert learning.spec.fingerprint==cfg.spec.fingerprint
    assert learning.metadata['configuration_fingerprint']!=cfg.metadata['configuration_fingerprint']
    with pytest.raises(ValueError,match='mismatch'):
        learning.validate_state_identity({'contracts':cfg.metadata})
    for field,value in [('training_paused',False),('training_session_version','upstream_t11_learning_v1')]:
        invalid=deepcopy(data)
        if field=='training_paused':invalid['experiment'][field]=value
        else:invalid['upstream_scale'][field]=value
        with pytest.raises(ValueError,match='pause'):
            parse_upstream_experiment(invalid,Path.cwd())


def test_actual_runtime_constructs_without_learning_and_rejects_role_mixing():
    torch.set_num_threads(1)
    diagnostic=make_collector(BASE,'A1',23,samples=16,num_envs=1,episode_seconds=2.)
    learning=make_collector(BASE,'A1',23,samples=16,num_envs=1,episode_seconds=2.,mode='train')
    assert not diagnostic.env.metadata['training_eligible']
    assert learning.env.metadata['training_eligible']
    assert learning.metadata['training_session_version']=='upstream_t11_learning_v1'
    assert 'formal_training_ready' not in learning.metadata
    assert learning.transitions==0 and diagnostic.transitions==0
    with pytest.raises(ValueError,match='paused'):
        PreparedUpstreamTrainer(diagnostic).update({},training_authorized=True,
            readiness={'formal_training_ready':True,'profile_fingerprint':diagnostic.env.spec.fingerprint})
    with pytest.raises(ValueError,match='diagnostic runtime'):
        PreparedUpstreamTrainer(learning).diagnostic_update({})
    with pytest.raises(ValueError):learning.load_state_dict(diagnostic.state_dict())
