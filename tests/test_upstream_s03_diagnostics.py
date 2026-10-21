"""Protocol construction/restore and strict evidence reuse after driver repair."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from _common import cfg_from_experiment
from diagnose_upstream_s03 import component_radio,validate_reuse_sources
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_communication import MotionBoundLedger,RoverBroadcast


@pytest.mark.parametrize('config,version',[
    ('upstream_scale_s03.yaml','sender_message_upstream_v1'),
    ('upstream_scale_s03_precision_v2.yaml','sender_message_upstream_v1'),
    ('upstream_scale_s03_observation_v1.yaml','sender_message_upstream_v2'),
])
def test_initial_and_restored_radio_use_same_explicit_protocol(config,version):
    cfg=cfg_from_experiment('configs/experiment/'+config);frames=np.zeros((2,4,3))
    ledger=MotionBoundLedger(cfg.spec,2);radio=component_radio(cfg,frames,ledger)
    positions=np.zeros((2,4,2));positions[:,:,0]=np.arange(4)*10.
    packets=[[RoverBroadcast(a,0,0.,np.r_[positions[e,a],0.],[0.,0.],[0.,0.],version=version) for a in range(4)] for e in range(2)]
    radio.transmit(packets,positions,[0.,0.]);snapshot=radio.state_dict()
    restored_ledger=MotionBoundLedger(cfg.spec,2);restored_ledger.load_state_dict(ledger.state_dict())
    restored=component_radio(cfg,frames,restored_ledger);restored.load_state_dict(snapshot)
    assert radio.metadata==restored.metadata
    left=radio.snapshot(0,0,0.,[0.,0.,0.]);right=restored.snapshot(0,0,0.,[0.,0.,0.])
    np.testing.assert_array_equal(left.sender_indices,right.sender_indices)
    assert [m.version for m in right.messages]==[version]*3


def test_prior_test_reuse_allows_only_driver_change_and_rejects_any_algorithm_delta():
    previous={'scripts/diagnose_upstream_s03.py':'before','source/planner.py':'same'}
    assert validate_reuse_sources(previous,previous)==[]
    current={**previous,'scripts/diagnose_upstream_s03.py':'after'}
    assert validate_reuse_sources(current,previous)==['scripts/diagnose_upstream_s03.py']
    with pytest.raises(ValueError,match='dependency changed'):
        validate_reuse_sources({**current,'source/planner.py':'changed'},previous)
    with pytest.raises(ValueError,match='coverage differs'):
        validate_reuse_sources({**current,'source/extra.py':'new'},previous)
    with pytest.raises(ValueError):validate_reuse_sources({}, {})
