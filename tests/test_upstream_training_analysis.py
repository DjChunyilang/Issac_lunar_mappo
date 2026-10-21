"""Reports preserve masks, distinguish physical stopping, and expose missing data."""
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from analyze_upstream_t11 import summarize_interactions, stats
from _common import cfg_from_experiment
from prepare_upstream_t11 import independent_scenes
from verify_upstream_transfer import verify


def test_masked_actions_and_active_execution_have_distinct_denominators():
    spec=cfg_from_experiment('configs/experiment/upstream_scale_s05_benchmark_cpu_review_v2.yaml').spec
    action=SimpleNamespace(hold=np.array([[True],[False]]),continuous=np.array([[[0.,0.]],[[99.,99.]]]),
        primitive_id=np.array([[-1],[-1]]))
    sample=SimpleNamespace(action=action,log_prob=np.array([[0.],[np.nan]]))
    intent=SimpleNamespace(hold=True,expires_at=1.)
    decision={'env_id':0,'agent_id':0,'execution_status':'verified_brake','execution_verified':True,
        'execution_reason':'hold_request','tracking_source':'hold_or_emergency_brake',
        'nominal_control':[0.,0.],'issued_control':[.04,0.],'safety_seconds':.2,'safety_correction':[0.,0.]}
    record={'execution':{'active_envs':[True,False],'actual_velocity':[[[.04,0.]],[[0.,0.]]],
        'activation_time':[.2,.2],'geometry_overlap':[False,True],'clearance_violation':[False,True]},
        'decisions':[decision],'reward_components':{'gather':[1.,999.]},'task':{'hold_count':[0,0]}}
    data={'learning_valid':np.array([[True,False]]),'duration':np.array([[5,0]]),'samples':[sample],
        'execution':[{'event':'request_activation','env_id':0,'agent_id':0,'accepted':True,'active_intent':intent},record]}
    result,_=summarize_interactions(data,spec)
    assert result['valid_team_transitions']==1 and result['actions']['valid_agent_samples']==1
    assert result['actions']['continuous_move_only']==[] and result['actions']['log_prob']['nonfinite']==0
    counts=result['execution_counts']
    assert counts['hold_execution_steps']==1 and counts['hold_stopped_steps']==0
    assert counts['nominal_changed_steps']==1 and counts['qp_modified_steps']==0
    assert counts['geometry_overlap_team_steps']==0 and result['raw_reward_component_sums']['gather']==1.
    assert result['learning_values'] is None and result['missing_fields']
    data['execution'][0]['active_intent'].expires_at=.1
    result,_=summarize_interactions(data,spec)
    assert result['execution_counts']['unknown_or_expired_execution_steps']==1


def test_missing_values_are_not_fabricated_zero_and_shape_error_is_rejected():
    assert stats([])['mean'] is None and stats([np.nan,1.])['nonfinite']==1
    spec=cfg_from_experiment('configs/experiment/upstream_scale_s05_benchmark_cpu_review_v2.yaml').spec
    with pytest.raises(ValueError,match='shape'):
        summarize_interactions({'learning_valid':np.ones((1,1),bool),'duration':np.ones((2,1)),'samples':[]},spec)


def test_independent_scene_definitions_match_single_environment_evaluator():
    cfg=cfg_from_experiment('configs/experiment/upstream_scale_s04_braking_lease_cuda_v5.yaml')
    rows=independent_scenes(cfg)
    assert len(rows)==18 and {r['seed'] for r in rows}=={151,157,163}
    assert all(np.shape(r['initial_poses'])==(1,4,3) for r in rows)


def test_transfer_checksum_and_path_boundary(tmp_path):
    import hashlib
    (tmp_path/'file').write_bytes(b'owned')
    manifest={'files':{'file':{'sha256':hashlib.sha256(b'owned').hexdigest()}}}
    assert verify(tmp_path,manifest)==[]
    (tmp_path/'file').write_bytes(b'changed');assert verify(tmp_path,manifest)[0]['reason']=='checksum'
    with pytest.raises(ValueError,match='escapes'):verify(tmp_path,{'files':{'../outside':{}}})
