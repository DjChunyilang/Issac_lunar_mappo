"""Regression tests for the causal diagnostic's information and target boundaries."""
from types import SimpleNamespace

import numpy as np

from scripts.diagnose_upstream_decision_credit import local_centroid_requests, renew_targets
from scripts.analyze_upstream_decision_credit import frozen_hold_reference
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_planning import UpstreamIntent
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_reward import UpstreamReward


def test_renewal_keeps_fixed_frame_target_and_original_ticket():
    intent = UpstreamIntent(np.array([2., -1.]), False, 0., 1.)
    renewed = renew_targets([[intent]], np.array([4.]), 1.)[0][0]
    np.testing.assert_array_equal(renewed.goal_xy, intent.goal_xy)
    assert (renewed.timestamp, renewed.expires_at) == (4., 5.)
    assert (intent.timestamp, intent.expires_at) == (0., 1.)


def fake_env(messages, valid):
    pose = np.array([2., 3., 1.7])
    snapshot = SimpleNamespace(messages=[SimpleNamespace(pose=np.array(p)) for p in messages], state_valid=valid)
    return SimpleNamespace(e=1, core=SimpleNamespace(times=np.array([2.])),
        local_pose=lambda e, a: pose.copy(),
        radio=SimpleNamespace(snapshot=lambda *args: snapshot),
        spec=SimpleNamespace(grid=SimpleNamespace(request_validity=1.)))


def test_centroid_uses_only_valid_received_states_not_hidden_team_state():
    env = fake_env([[8., 3., 0.], [-100., -100., 0.]], [True, False])
    intent = local_centroid_requests(env)[0][0]
    np.testing.assert_allclose(intent.goal_xy, [5., 3.])
    assert not intent.hold
    env.radio = fake_env([[-100., -100., 0.], [8., 3., 0.]], [False, True]).radio
    np.testing.assert_allclose(local_centroid_requests(env)[0][0].goal_xy, intent.goal_xy)


def test_missing_messages_hold_and_toward_away_share_same_limit():
    env = fake_env([], [])
    intent = local_centroid_requests(env)[0][0]
    assert intent.hold
    np.testing.assert_array_equal(intent.goal_xy, [2., 3.])
    env = fake_env([[100., 3., 0.]], [True])
    np.testing.assert_allclose(local_centroid_requests(env)[0][0].goal_xy, [6., 3.])
    np.testing.assert_allclose(local_centroid_requests(env, away=True)[0][0].goal_xy, [-2., 3.])


def test_hold_reference_matches_actual_reward_and_terminal_potential_cancellation():
    spec = SimpleNamespace(task=SimpleNamespace(dmax=12., dispersion=30., hold_duration=5.),
                           grid=SimpleNamespace(dt=.2), vehicle=SimpleNamespace(effective_radius=3.274))
    tracker = UpstreamReward(spec, 1)
    facts = {key: np.array([value]) for key, value in
             {'dmax': 24., 'dispersion': 60., 'hold_count': 0., 'flatness_cost': 1., 'minimum_distance': 10.}.items()}
    features = np.zeros((1, 4, 5)); features[..., 4] = 1.
    total = 0.
    for step in range(3):
        task = {'learning_valid': np.array([True]), 'ended_this_step': np.array([step == 2]),
                'terminal_reason': np.array(['task_deadline' if step == 2 else 'none'])}
        reward, _ = tracker.advance(facts, facts, task, np.zeros((1, 4, 2)), features,
                                    np.ones((1, 4)), np.zeros((1, 4)))
        total += .99**step*reward[0]
    np.testing.assert_allclose(frozen_hold_reference(24., 60., 3), total, atol=1e-10, rtol=0)
