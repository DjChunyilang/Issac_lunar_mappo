"""Real stationary-mode witness, compatible cold solve and strict recovery."""
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from _common import cfg_from_experiment
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_cases import make_case
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_execution import rollout_numpy
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_factory import create_upstream_adapter
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_nmpc import build_upstream_solver


@pytest.fixture(scope='module')
def setup(tmp_path_factory):
    torch.set_num_threads(1)
    cfg=cfg_from_experiment('configs/experiment/upstream_scale_s04_nmpc_recovery_v2.yaml')
    case=make_case(cfg,'flat',101,0)
    solver=build_upstream_solver(cfg.spec,tmp_path_factory.mktemp('nmpc_recovery'))
    return cfg,case.input,solver


def stationary(adapter,item):
    adapter._previous_trajectory=rollout_numpy(item.ego_pose,np.zeros((40,2)),item.spec,item.now)


def test_real_acados_zero_mode_witness_and_direct_goal_seed_recovery(setup):
    cfg,item,solver=setup
    old=create_upstream_adapter(replace(cfg,planner_adapter='nmpc_upstream_v1'),solver=solver)
    stationary(old,item);before=old.candidate_controls(item)[0].controls
    assert np.max(np.abs(before))<1e-7
    new=create_upstream_adapter(cfg,solver=solver);stationary(new,item)
    output=new.plan(item)
    assert output.valid and output.diagnostics['stationary_mode_reseed']
    assert output.trajectory.controls[0,0]>.001
    assert np.linalg.norm(output.trajectory.states[-1,:2]-item.intent.goal_xy)<np.linalg.norm(item.intent.goal_xy-item.ego_pose[:2])


def test_real_acados_nonstationary_cold_pair_and_public_restore(setup):
    cfg,item,solver=setup
    old=create_upstream_adapter(replace(cfg,planner_adapter='nmpc_upstream_v1'),solver=solver)
    new=create_upstream_adapter(cfg,solver=solver)
    a=old.plan(item);b=new.plan(item)
    np.testing.assert_allclose(a.trajectory.states,b.trajectory.states,atol=1e-7,rtol=1e-6)
    np.testing.assert_allclose(a.trajectory.controls,b.trajectory.controls,atol=1e-7,rtol=1e-6)
    state=new.state_dict();c=new.plan(item)
    restored=create_upstream_adapter(cfg,solver=solver);restored.load_state_dict(state);d=restored.plan(item)
    np.testing.assert_allclose(c.trajectory.controls,d.trajectory.controls,atol=1e-7,rtol=1e-6)
    with pytest.raises(ValueError):restored.load_state_dict(old.state_dict())


@pytest.mark.parametrize('kind',['hold','expired','zero_move'])
def test_recovery_preserves_real_hold_expiry_and_zero_move(setup,kind):
    cfg,item,solver=setup;adapter=create_upstream_adapter(cfg,solver=solver);stationary(adapter,item)
    intent=replace(item.intent,hold=kind=='hold',goal_xy=item.ego_pose[:2] if kind=='zero_move' else item.intent.goal_xy,
        timestamp=0. if kind=='expired' else item.intent.timestamp,expires_at=.2 if kind=='expired' else item.intent.expires_at)
    output=adapter.plan(replace(item,intent=intent))
    if kind=='expired':assert not output.valid and output.fallback_reason=='expired_intent'
    np.testing.assert_allclose(output.trajectory.controls,0.,atol=1e-7)


def test_real_cached_solver_load_preserves_built_library_and_pair(setup,tmp_path):
    cfg,item,_=setup
    first=create_upstream_adapter(cfg,cache_dir=tmp_path);a=first.plan(item)
    libraries=list(tmp_path.rglob('libacados_ocp_solver*.so'));assert len(libraries)==1
    before=libraries[0].stat().st_mtime_ns
    second=create_upstream_adapter(cfg,cache_dir=tmp_path);b=second.plan(item)
    assert libraries[0].stat().st_mtime_ns==before
    np.testing.assert_allclose(a.trajectory.controls,b.trajectory.controls,atol=1e-7,rtol=1e-6)


def test_real_acados_frame_review_matches_existing_complete_review(setup):
    cfg,item,solver=setup
    reference=create_upstream_adapter(cfg,solver=solver).plan(item)
    fast=create_upstream_adapter(replace(cfg,review_acceleration_version='flat_frame_pose_cache_v3'),solver=solver).plan(item)
    assert (reference.valid,reference.fallback_reason)==(fast.valid,fast.fallback_reason)
    np.testing.assert_allclose(reference.trajectory.states,fast.trajectory.states,atol=1e-7,rtol=1e-6)
    np.testing.assert_allclose(reference.trajectory.controls,fast.trajectory.controls,atol=1e-7,rtol=1e-6)
