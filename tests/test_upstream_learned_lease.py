"""Learned speed promises must cover the trajectory actually being reviewed."""
from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from run_upstream_t11 import make_collector
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_ablation_learning import renew_legal_bounds, LEARNED_BOUND_VERSION
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_speed_lease import braking_clearance_limit

BASE = 'configs/experiment/upstream_scale_s04_braking_lease_cuda_v5.yaml'


def test_learned_promises_fit_full_horizon_and_use_received_states_only():
    torch.set_num_threads(1)
    collector = make_collector(BASE, 'A1', 23, samples=16, num_envs=2)
    env = collector.env
    # Reproduce the cause: a stopping-distance promise allows a reachable ball
    # to consume more than the available gap during the eight-second horizon.
    gap = 4.74
    assert braking_clearance_limit(env.spec, gap, .5) * env.spec.grid.horizon > gap
    expected = []
    for e in range(2):
        for a in range(4):
            pose = env.local_pose(e, a)
            snapshot = env.radio.snapshot(e, a, 0., pose)
            gaps = [np.linalg.norm(pose[:2] - m.pose[:2]) - 2 * env.spec.vehicle.effective_radius
                    for i, m in enumerate(snapshot.messages) if snapshot.state_valid[i]]
            clearance = min(gaps, default=np.inf)
            horizon = env.spec.grid.horizon + env.spec.grid.request_validity + env.cfg.simulated_latency
            expected.append(min(env.spec.vehicle.forward_limit, .8 * max(0., clearance) / horizon,
                                braking_clearance_limit(env.spec, clearance, env.cfg.simulated_latency)))
    renew_legal_bounds(env)
    np.testing.assert_allclose([env.bounds[e][a].speed for e in range(2) for a in range(4)], expected)
    reference = env.bounds[0][0].speed
    # Change uncommunicated peer truth and another environment. The receiver's
    # own state and cached packets remain identical, hence so must its pledge.
    env.core.poses[0, 1:, :2] += 40.
    env.core.poses[1, :, :2] += 20.
    renew_legal_bounds(env)
    assert env.bounds[0][0].speed == reference
    assert collector.metadata['execution_bound_policy'] == LEARNED_BOUND_VERSION


def test_existing_actuator_state_and_older_promises_are_preserved():
    torch.set_num_threads(1)
    env = make_collector(BASE, 'A1', 23, samples=16, num_envs=1).env
    env.core.previous_control[0, 0] = [.7, 0.]
    renew_legal_bounds(env)
    assert env.bounds[0][0].speed >= .7  # Never promise an instant speed drop.
    env.pledge_speed(0, 1, 0., 10.)
    renew_legal_bounds(env)
    assert env.ledger.speed_cap(0, 1, 0.) == 0.  # Newer bounds cannot revoke it.


def test_old_learning_bound_identity_requires_explicit_migration():
    torch.set_num_threads(1)
    collector = make_collector(BASE, 'A1', 23, samples=16, num_envs=1)
    state = deepcopy(collector.state_dict())
    state['metadata']['execution_bound_policy'] = 'braking_clearance_lease_v3'
    with pytest.raises(ValueError, match='restore contract mismatch'):
        collector.load_state_dict(state)
