"""Bounded medium-crater cost and isolated reward/checkpoint semantics."""
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from _common import cfg_from_experiment
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_reward import UpstreamReward,UpstreamRewardSettings


def fixture():
    cfg=cfg_from_experiment('configs/experiment/upstream_1km_crater_classes_v1.yaml')
    tracker=UpstreamReward(cfg.spec,1,UpstreamRewardSettings(failure_penalty=100.),obstacle_input=cfg.obstacle_input)
    facts={'dmax':np.array([12.]),'dispersion':np.array([30.]),'minimum_distance':np.array([10.]),
           'flatness_cost':np.ones(1),'hold_count':np.zeros(1)}
    task={'learning_valid':np.ones(1,bool),'ended_this_step':np.zeros(1,bool),'terminal_reason':np.array(['none'])}
    features=np.zeros((1,4,5));features[...,4]=1.
    def advance(medium=-1.,event=None):
        return tracker.advance(facts,facts,task if event is None else event,np.zeros((1,4,2)),features,
            np.ones((1,4)),np.zeros((1,4)),obstacle_distances=np.ones((1,4)),
            medium_crater_distances=np.full((1,4),medium))[1]
    return cfg,tracker,task,advance


def test_medium_cost_is_light_and_capped_without_turning_into_failure():
    cfg,tracker,task,advance=fixture()
    assert advance(1.)['crater_medium'][0]==0.
    first=advance()['crater_medium'][0]
    assert first==pytest.approx(-.05*cfg.spec.grid.dt)
    for _ in range(500):
        terms=advance()
        assert terms['terminal'][0]==0.
    assert tracker.medium_crater_paid[0]==pytest.approx(2.)
    assert tracker.component_returns['crater_medium'][0]==pytest.approx(-2.)
    assert advance()['crater_medium'][0]==0.
    ended={**task,'ended_this_step':np.ones(1,bool),'terminal_reason':np.array(['terrain_failure'])}
    assert advance(event=ended)['terminal'][0]==-100.


def test_medium_cap_restores_resets_and_ignores_invalid_transitions():
    cfg,tracker,task,advance=fixture()
    invalid={**task,'learning_valid':np.zeros(1,bool)}
    advance(event=invalid)
    assert tracker.medium_crater_paid[0]==0.
    for _ in range(20):advance()
    restored=UpstreamReward(cfg.spec,1,tracker.settings,obstacle_input=cfg.obstacle_input)
    restored.load_state_dict(tracker.state_dict())
    np.testing.assert_array_equal(restored.medium_crater_paid,tracker.medium_crater_paid)
    restored.reset(np.array([0]))
    assert restored.medium_crater_paid[0]==0.
    assert restored.component_returns['crater_medium'][0]==0.
    old=cfg_from_experiment('configs/experiment/upstream_1km_obstacle_input_v1.yaml')
    legacy=UpstreamReward(old.spec,1,tracker.settings,obstacle_input=old.obstacle_input)
    with pytest.raises(ValueError,match='identity'):
        restored.load_state_dict(legacy.state_dict())
