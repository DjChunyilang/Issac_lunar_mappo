"""Lease proposals preserve public promises and actual execution limits."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from _common import cfg_from_experiment
from diagnose_upstream_environment import script_with_bounds
from run_upstream_t11 import make_collector
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_communication import MotionBoundLedger
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_execution import control_error
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_speed_lease import braking_clearance_limit,renew_braking_leases
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_spec import parse_upstream_experiment

BASE='configs/experiment/upstream_scale_s04_braking_lease_cuda_v5.yaml'


def test_physical_stopping_budget_and_monotonicity():
    spec=cfg_from_experiment(BASE).spec
    gaps=np.r_[0.,1e-12,.1,.65,1.,4.,10.,100.]
    speeds=np.array([braking_clearance_limit(spec,gap,.5) for gap in gaps])
    stopping=speeds*1.5+speeds**2/(2*spec.vehicle.linear_deceleration)
    assert (stopping<=.8*gaps+1e-10).all() and (np.diff(speeds)>=0).all()
    assert (speeds<=spec.vehicle.forward_limit).all() and speeds[0]==0.
    assert braking_clearance_limit(spec,np.inf,.5)==spec.vehicle.forward_limit
    assert braking_clearance_limit(spec,-.2,.5)==0.
    with pytest.raises(ValueError):braking_clearance_limit(spec,np.nan,.5)


def test_higher_bound_never_revokes_old_pledge_and_restore_preserves_it():
    spec=cfg_from_experiment(BASE).spec;ledger=MotionBoundLedger(spec,1)
    ledger.pledge(0,1,0.,0.,10.,[0.,0.])
    proposed=braking_clearance_limit(spec,.65,.5)
    ledger.pledge(0,1,proposed,1.,11.,[0.,0.])
    assert proposed>.2 and ledger.speed_cap(0,1,10.)==0.
    replay=MotionBoundLedger(spec,1);replay.load_state_dict(ledger.state_dict())
    for now in (5.,10.,10.2):assert ledger.speed_cap(0,1,now)==replay.speed_cap(0,1,now)
    assert replay.limit(0,1,[1.,0.],[0.,0.],10.)[0]==0.
    assert 0.<replay.limit(0,1,[1.,0.],[0.,0.],10.2)[0]<=proposed


def test_version_is_explicit_and_original_configuration_keeps_metadata():
    old=cfg_from_experiment('configs/experiment/upstream_scale_s04_clearance_recovery_cuda_v4.yaml')
    new=cfg_from_experiment(BASE)
    assert old.motion_lease_version=='clearance_horizon_lease_v2' and 'motion_lease_version' not in old.metadata
    assert new.metadata['motion_lease_version']=='braking_clearance_lease_v3'
    assert old.spec.fingerprint==new.spec.fingerprint
    with pytest.raises(ValueError):new.validate_state_identity({'contracts':old.metadata})
    data=json.loads(new.configuration_json);data['upstream_scale']['motion_lease_version']='unknown'
    with pytest.raises(ValueError,match='lease'):parse_upstream_experiment(data,Path.cwd())


def test_actual_sender_bound_execution_and_reset_isolation():
    torch.set_num_threads(1)
    env=make_collector(BASE,'A1',23,samples=16,num_envs=2,episode_seconds=2.).env
    original=deepcopy(env.ledger.state_dict())
    renew_braking_leases(env)
    positive=[b for b in env.ledger.state_dict()['bounds'] if b.speed>0]
    assert positive
    requests=script_with_bounds(env)
    result=env.step_intents(requests)
    for record in result.records:
        if 'execution' not in record:continue
        x=record['execution']
        for e in range(2):
            for a in range(4):
                assert not control_error(x['applied_control'][e,a],x['previous_control'][e,a],env.spec)
                assert abs(x['applied_control'][e,a,0])<=env.ledger.speed_cap(e,a,float(x['activation_time'][e]))+1e-8
    other=[b for b in env.ledger.state_dict()['bounds'] if b.env_id==1]
    env.reset([0])
    assert [b for b in env.ledger.state_dict()['bounds'] if b.env_id==1]==other
    assert original['episodes'][1]==env.ledger.episodes[1]
