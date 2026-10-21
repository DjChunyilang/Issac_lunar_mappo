"""New cohort reset/restore and bounded admission; 16 physical steps total."""
from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from run_upstream_t11 import make_collector
from upstream_learning_cohort import COHORT, mixed_initial_conditions, near_poses
from upstream_process_rollout import ProcessUpstreamRollout
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_decision import LearningTimeConfig
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_training import PreparedUpstreamTrainer, TrainingOptions

BASE = 'configs/experiment/upstream_scale_s05_benchmark_cpu_review_v2.yaml'


def test_near_geometry_and_moving_resets_survive_restore():
    torch.set_num_threads(1)
    c = make_collector(BASE, 'A1', 23, samples=8, num_envs=4, initial_cohort=COHORT)
    env = c.env
    xy = near_poses()[..., :2]
    distances = np.linalg.norm(xy[:, :, None]-xy[:, None, :], axis=-1)
    distances[:, np.arange(4), np.arange(4)] = np.inf
    assert np.all(distances.min((1, 2)) >= 2*env.spec.vehicle.effective_radius)
    centered = xy-xy.mean(1, keepdims=True)
    assert np.all((centered**2).sum(-1).mean(-1) <= env.spec.task.dispersion)
    assert np.all(np.linalg.norm(centered, axis=-1)+env.spec.vehicle.effective_radius <= env.spec.task.worksite_radius)
    distances[:, np.arange(4), np.arange(4)] = 0.
    assert np.all(distances.max((1, 2)) <= env.spec.task.dmax)
    poses, controls = mixed_initial_conditions(np.zeros((16, 4, 3)), 23, COHORT)
    assert (controls[..., 0].mean(-1) > 0).sum() == 8
    for owner in range(4):
        p, v = mixed_initial_conditions(np.zeros((4, 4, 3)), 23+4*owner, COHORT)
        np.testing.assert_array_equal(p, poses[owner*4:(owner+1)*4])
        np.testing.assert_array_equal(v, controls[owner*4:(owner+1)*4])
    state = env.state_dict()
    env.core.previous_control[:]=0.; env.core.actual_velocity[:]=0.
    env.load_state_dict(state)
    env.core.previous_control[:]=0.; env.core.actual_velocity[:]=0.
    env.reset(np.array([1, 3]))
    np.testing.assert_allclose(env.core.actual_velocity[[1, 3], :, 0], .1)
    np.testing.assert_allclose(env.core.previous_control[[1, 3], :, 0], .1)
    np.testing.assert_allclose(env.core.actual_velocity[[0, 2]], 0.)
    original = make_collector(BASE, 'A1', 23, samples=8, num_envs=4)
    assert 'initial_control' not in original.env.metadata
    with pytest.raises(ValueError, match='identity mismatch'):
        original.env.load_state_dict(state)


def test_mixed_owner_closed_loop_admission_and_budget():
    torch.set_num_threads(1)
    timing = LearningTimeConfig(5., 1.)
    c = ProcessUpstreamRollout(BASE, 'A1', 23, owners=4, envs_per_owner=4, samples=8,
        episode_seconds=.2, cpu_workers=1, learning_time=timing, mode='signal', initial_cohort=COHORT)
    try:
        t = PreparedUpstreamTrainer(c, TrainingOptions(rollout_macro_steps=1, epochs=1,
            gamma=c.gamma_step, lam=c.lambda_step, gradient_clipping='actor_critic', terminal_actor_credit='effective'))
        batch, _, _ = t.collect()
        assert c.base_steps == 16 and batch['learning_valid'].all()
        assert not batch['action_effective'].any()
        audit = {'schema_version': 't11_learning_repair_validation_v1', 'scope': 'learning_repair_validation',
            'maximum_base_steps': 20000, 'profile_fingerprint': c.env.spec.fingerprint,
            'learning_time': timing.metadata(.2), 'formal_training_ready': False}
        with pytest.raises(ValueError, match='paused'):
            t.update(batch, readiness=audit)
        t._perform_update = lambda batch: {'admitted': True}
        assert t.update(batch, readiness=audit, training_authorized=True) == {'admitted': True}
        invalid = deepcopy(audit); invalid['maximum_base_steps'] = 200000
        with pytest.raises(ValueError, match='paused'):
            t.update(batch, readiness=invalid, training_authorized=True)
        c.base_steps = 20001
        with pytest.raises(ValueError, match='paused'):
            t.update(batch, readiness=audit, training_authorized=True)
    finally:
        c.close()
