"""Affected contracts: scale, reward, asynchronous curriculum and exact resume."""
import sys
import math
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from _common import cfg_from_experiment
from audit_upstream_curriculum import geometry_audit, reward_audit
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_curriculum import (
    SpawnCurriculum,
    update_progress,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_decision import (
    remaining_decisions,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_skrl import (
    SkrlUpstreamTrainer,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_training import (
    TrainingOptions,
)
from run_upstream_t11 import make_collector
from upstream_checkpoint import read_checkpoint, save_checkpoint
from upstream_curriculum_experiment import (
    admission, load_curriculum, timing_for, collection_decisions, initialize_fresh_policy,
    evaluation_panels, evaluation_reservation, evaluation_budget_upper_bound, evaluate, convergence_report,
)
from upstream_process_rollout import ProcessUpstreamRollout

BASE = 'configs/experiment/upstream_1km_curriculum_v1.yaml'
COURSE = 'configs/training/upstream_1km_curriculum.yaml'
REVIEWED = 'configs/training/upstream_1km_curriculum_v2.yaml'


@pytest.fixture
def cpu_base(tmp_path):
    p = tmp_path/'cpu.yaml'
    p.write_text(f'extends: {Path(BASE).resolve()}\nupstream_scale:\n  planner_device: cpu\n')
    return str(p)


def test_geometry_reward_and_seeded_curriculum():
    spec = cfg_from_experiment(BASE).spec
    config = load_curriculum(COURSE)
    assert (spec.world_width, spec.world_height) == (1000., 1000.)
    assert (spec.communication.full_radius, spec.communication.sparse_radius) == (300., 600.)
    assert len(geometry_audit(spec, config)) == 7
    assert reward_audit(spec, config)['terminal_and_nonterminal_telescoping_passed']
    c = SpawnCurriculum(spec, config, 23, 4)
    c.sample(range(4)); c.set_level(6)
    saved = c.state_dict()
    expected = c.sample([1, 3])
    c.load_state_dict(saved)
    np.testing.assert_array_equal(expected, c.sample([1, 3]))
    assert c.episode_levels[[0, 2]].tolist() == [0, 0]
    p = {'level': 1, 'stage_start_base_steps': 0, 'outcomes': [], 'history': []}
    old_successes = [{'curriculum_level': 0, 'terminal_reason': 'success'}]*40
    q, promoted = update_progress(config, p, old_successes, 20000)
    assert not promoted and not q['outcomes']
    q, promoted = update_progress(config, p, [{'curriculum_level': 1, 'terminal_reason': 'success'}]*40, 20000)
    assert promoted and q['level'] == 2 and q['history'][0]['base_steps'] == 20000


def test_async_reset_deadlines_clock_and_restore(cpu_base):
    torch.set_num_threads(1)
    config = load_curriculum(COURSE)
    c = make_collector(cpu_base, 'A1', 23, samples=8, num_envs=2,
                       curriculum_config=config, learning_time=timing_for(config))
    env = c.env
    old = env.core.poses.copy()
    env.curriculum.set_level(1)
    env.reset(np.array([1]))
    np.testing.assert_array_equal(old[0], env.core.poses[0])
    assert not np.array_equal(old[1], env.core.poses[1])
    assert env.task.deadlines.tolist() == [30., 60.]
    # Actor and Critic must see the per-episode clock, not the 1200s maximum.
    env.core.times[:] = 15.
    env._publish()
    actor, state = env.observe()
    torch.testing.assert_close(actor[:, 0, -1], torch.tensor([.5, .75]))
    torch.testing.assert_close(state[:, -1], torch.tensor([.5, .75]))
    env.core.times[:] = 0.
    env.reset(np.array([0, 1]))
    saved = env.state_dict()
    env.curriculum.set_level(6)
    env.reset(np.array([0]))
    env.load_state_dict(saved)
    assert env.curriculum.level == 1
    np.testing.assert_array_equal(env.task.deadlines, saved['curriculum']['deadlines'])
    expected = env.curriculum.sample([0])
    env.load_state_dict(saved)
    env.reset(np.array([0]))
    np.testing.assert_array_equal(env.core.poses[[0]], expected)


def test_owner_learning_budget_and_full_checkpoint(cpu_base, tmp_path):
    torch.set_num_threads(1)
    config = load_curriculum(COURSE)
    # 16 + 32 physical steps: early terminal + independently reset new stage.
    for i, stage in enumerate(config['stages']):
        stage['deadline_s'] = .2 if i == 0 else .4
    config['easy_replay_fraction'] = 0.
    timing = timing_for(config)
    c = ProcessUpstreamRollout(cpu_base, 'A1', 23, owners=4, envs_per_owner=4, samples=8,
        episode_seconds=1200., mode='train', cpu_workers=1, learning_time=timing, curriculum_config=config)
    try:
        t = SkrlUpstreamTrainer(c, TrainingOptions(rollout_macro_steps=1, epochs=1, gamma=c.gamma_step,
            lam=c.lambda_step, gradient_clipping='actor_critic', terminal_actor_credit='effective', value_normalization='running'))
        batch, episodes, _ = t.collect()
        assert c.base_steps == 16 and len(episodes) == 16
        assert all(e['curriculum_level'] == 0 for e in episodes)
        assert not batch['bootstrap_allowed'].any()
        audit = admission(c, 2000000)
        with pytest.raises(ValueError, match='paused'):
            t.update(batch, readiness=audit)
        t.update(batch, readiness=audit, training_authorized=True)
        c.curriculum_progress.update(level=1, stage_start_base_steps=16)
        c.set_curriculum_level(1)
        target = tmp_path/'saved.pt'
        save_checkpoint(t, {'test': True}, 1, target, scope='curriculum_1km')
        batch, episodes, _ = t.collect()
        assert c.base_steps == 48 and batch['duration'].eq(2).all()
        assert all(e['curriculum_level'] == 1 for e in episodes)
        assert batch['hidden'].eq(0).all()
        t.load_state_dict(read_checkpoint(target)['trainer'])
        assert c.base_steps == 16 and c.curriculum_progress['level'] == 1
        states = c._request('state', [None]*4)
        assert all(s['curriculum']['level'] == 1 and s['curriculum']['episode_levels'].tolist() == [0]*4 for s in states)
        assert remaining_decisions(2000000, 1999600, 16, 25, 16) == 1
        assert remaining_decisions(2000000, 1999601, 16, 25, 16) == 0
        invalid = deepcopy(audit); invalid['maximum_base_steps'] = 4000000
        with pytest.raises(ValueError, match='paused'):
            t.update(batch, readiness=invalid, training_authorized=True)
    finally:
        c.close()


def test_reviewed_spawn_geometry_credit_and_version_isolation():
    spec = cfg_from_experiment(BASE).spec
    config = load_curriculum(REVIEWED)
    old = load_curriculum(COURSE)
    assert len(geometry_audit(spec, config)) == 7
    assert reward_audit(spec, config)['terminal_and_nonterminal_telescoping_passed']
    c = SpawnCurriculum(spec, config, 23, 1024)
    assert c.level == 2
    c.sample(range(1024))
    proportions = np.bincount(c.episode_levels, minlength=7)/1024
    assert .06 < proportions[0] < .14 and .06 < proportions[1] < .14
    assert .74 < proportions[2] < .86 and proportions[3:].sum() == 0
    saved = c.state_dict()
    expected = c.sample([0, 7])
    c.load_state_dict(saved)
    np.testing.assert_array_equal(expected, c.sample([0, 7]))
    with pytest.raises(ValueError, match='Cross-curriculum'):
        SpawnCurriculum(spec, old, 23, 1024).load_state_dict(saved)
    g, l = timing_for(config).discounts(.2)
    assert math.isclose((g*l)**25, (.997*.99)**5)
    assert 53 < math.log(.5)/math.log(.997*.99) < 54
    assert SpawnCurriculum(spec, old, 23, 4).level == 0


def test_stage_reservation_promotion_and_total_budget():
    config = load_curriculum(REVIEWED)
    p = {'level': 2, 'stage_start_base_steps': 0, 'outcomes': [], 'history': []}
    assert collection_decisions(config, p, 2000000, 0, 16, 25) == 32
    assert collection_decisions(config, p, 2000000, 249600, 16, 25) == 1
    assert collection_decisions(config, p, 2000000, 249601, 16, 25) == 0
    # No automatic advance for failures, even at the stage budget.
    q, promoted = update_progress(config, p,
        [{'curriculum_level': 2, 'terminal_reason': 'task_deadline'}]*40, 250000)
    assert not promoted and collection_decisions(config, q, 2000000, 250000, 16, 25) == 0
    q, promoted = update_progress(config, p,
        [{'curriculum_level': 2, 'terminal_reason': 'success'}]*40, 250000)
    assert promoted and collection_decisions(config, q, 2000000, 250000, 16, 25) == 32
    q['stage_start_base_steps'] = 1990000
    assert collection_decisions(config, q, 2000000, 1999600, 16, 25) == 1
    assert collection_decisions(config, q, 2000000, 1999601, 16, 25) == 0
    # V1 still has no stage cap.
    assert collection_decisions(load_curriculum(COURSE), p, 2000000, 250000, 16, 25) == 16


def test_fixed_evaluation_schedule_fits_budget_and_has_three_full_checkpoints():
    config = load_curriculum(REVIEWED)
    assert evaluation_reservation(evaluation_panels(config, 2, 'initial')) == 4400
    assert evaluation_reservation(evaluation_panels(config, 2, 'scheduled')) == 3200
    assert evaluation_reservation(evaluation_panels(config, 2, 'scheduled', full=True)) == 52400
    assert evaluation_reservation(evaluation_panels(config, 2, 'final', final=True)) == 71600
    assert evaluation_budget_upper_bound(config, 2000000) == 235200
    assert evaluation_budget_upper_bound(config, 2000000) <= config['evaluation_max_base_steps']
    assert len(config['full_evaluation_at_base_steps']) + 1 == 3
    assert evaluation_panels(config, 2, 'scheduled') == evaluation_panels(config, 6, 'scheduled')
    old = load_curriculum(COURSE)
    assert [p[1] for p in evaluation_panels(old, 0, 'initial')] == [0, 1]


@pytest.mark.parametrize('change', [
    {'initial_level': 7}, {'initial_hold_probability': 1.},
    {'final_evaluation_reservation': 1000}, {'trend_evaluation_seconds': 21.},
    {'full_evaluation_at_base_steps': [500001]},
])
def test_reject_invalid_reviewed_configuration(tmp_path, change):
    import yaml
    config = load_curriculum(REVIEWED)
    config.update(change)
    path = tmp_path/'invalid.yaml'
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError):
        load_curriculum(path)


def test_reviewed_initialization_and_owner_start_resume(cpu_base, tmp_path):
    torch.set_num_threads(1)
    config = load_curriculum(REVIEWED)
    config['easy_replay_fraction'] = 0.
    timing = timing_for(config)
    single = make_collector(cpu_base, 'A1', 23, samples=8, num_envs=2,
                            curriculum_config=config, learning_time=timing)
    initialize_fresh_policy(single.policy, config)
    assert torch.allclose(single.policy.hold_logit(torch.zeros(2, 128)).sigmoid(), torch.full((2, 1), .35))
    assert torch.allclose(single.policy.log_std.exp(), torch.full((2,), math.exp(-.5)*.25))
    assert single.env.curriculum.episode_levels.tolist() == [2, 2]
    assert single.env.task.deadlines.tolist() == [300., 300.]
    c = ProcessUpstreamRollout(cpu_base, 'A1', 23, owners=2, envs_per_owner=1, samples=8,
        episode_seconds=1200., mode='train', cpu_workers=1, learning_time=timing, curriculum_config=config)
    try:
        assert c.curriculum_progress['level'] == 2
        t = SkrlUpstreamTrainer(c, TrainingOptions(rollout_macro_steps=32, epochs=5,
            gamma=c.gamma_step, lam=c.lambda_step, gradient_clipping='actor_critic',
            terminal_actor_credit='effective', value_normalization='running'))
        initialize_fresh_policy(c.policy, config)
        target = tmp_path/'initial.pt'
        save_checkpoint(t, {'test': 'reviewed'}, 0, target, scope='curriculum_1km')
        c.curriculum_progress['level'] = 6
        c.set_curriculum_level(6)
        t.load_state_dict(read_checkpoint(target)['trainer'])
        assert c.curriculum_progress['level'] == 2 and c.base_steps == 0
        states = c._request('state', [None]*2)
        assert all(s['curriculum']['level'] == 2 and s['curriculum']['episode_levels'].tolist() == [2]
                   and s['curriculum']['deadlines'].tolist() == [300.] for s in states)
    finally:
        c.close()


def test_trend_evaluation_does_not_require_terminal_or_count_as_convergence(tmp_path, monkeypatch):
    import run_upstream_t11
    import upstream_decision_evaluation
    config = load_curriculum(REVIEWED)
    (tmp_path/'metrics').mkdir()

    def make_fake(*args, **kwargs):
        class Curriculum:
            level = 2
            deadlines = np.full(4, 300.)

            def set_level(self, level):
                self.level = level
                self.deadlines = np.full(4, config['stages'][level]['deadline_s'])

            def sample(self, ids, replay):
                return np.zeros((4, 4, 3))
        return SimpleNamespace(env=SimpleNamespace(curriculum=Curriculum(),
            task=SimpleNamespace(deadlines=np.zeros(4)), reset=lambda *a: None))

    def fake_panel(c, *, deterministic, seconds):
        complete = seconds >= c.env.curriculum.deadlines[0]
        return {'episodes': [{}]*4 if complete else [], 'successes': 4 if complete else 0,
                'trace': [], 'actual_base_steps': round(4*seconds/.2)}

    monkeypatch.setattr(run_upstream_t11, 'make_collector', make_fake)
    monkeypatch.setattr(upstream_decision_evaluation, 'panel', fake_panel)
    collector = SimpleNamespace(curriculum_progress={'level': 2}, base_steps=0,
        policy=torch.nn.Linear(1, 1), learning_time=timing_for(config))
    row = evaluate(tmp_path, 'initial', collector, SimpleNamespace(config=BASE), config)
    assert row['actual_base_steps'] == 4400 and not row['full_episode_evaluation']
    assert row['results']['trend_level_6_mean']['episodes'] == []
    assert row['results']['level_0_mean']['full_episode_evaluation']
    assert not convergence_report(tmp_path, config)['stable_convergence_screen_passed']
    for steps in (500000, 1000000, 1250000):
        collector.base_steps = steps
        evaluate(tmp_path, f'step_{steps}', collector, SimpleNamespace(config=BASE), config, full=True)
    report = convergence_report(tmp_path, config)
    assert report['stable_convergence_screen_passed'] and len(report['final_level_panels']) == 3
