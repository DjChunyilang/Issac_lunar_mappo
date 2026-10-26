"""Continuous-cost scaling preserves potential shaping and terminal events."""
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from _common import cfg_from_experiment
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_reward import (
    UpstreamReward,UpstreamRewardSettings,configured_reward_settings,
)


@pytest.fixture(scope='module')
def cfg():
    return cfg_from_experiment('configs/experiment/upstream_1km_crater_classes_v1.yaml')


@pytest.mark.parametrize('gather_scaling',['linear','log1p'])
@pytest.mark.parametrize('reason',['none','success','collision','terrain_failure','task_deadline'])
def test_only_legacy_continuous_costs_scale(cfg,gather_scaling,reason):
    settings=UpstreamRewardSettings(gather_scaling=gather_scaling,hazard_cost=.05,failure_penalty=100.)
    old=UpstreamReward(cfg.spec,1,settings,obstacle_input=cfg.obstacle_input)
    new=UpstreamReward(cfg.spec,1,replace(settings,continuous_cost_scale=.02),obstacle_input=cfg.obstacle_input)
    before={'dmax':np.array([20.]),'dispersion':np.array([100.]),'minimum_distance':np.array([5.]),
            'flatness_cost':np.array([2.]),'hold_count':np.zeros(1)}
    after={**before,'dmax':np.array([18.]),'dispersion':np.array([90.])}
    task={'learning_valid':np.ones(1,bool),'ended_this_step':np.array([reason!='none']),
          'terminal_reason':np.array([reason])}
    f=np.broadcast_to([.1,.1,.1,.03,.6],(1,4,5)).copy()
    arguments=(before,after,task,np.full((1,4,2),.1),f,np.full((1,4),.6),np.full((1,4),.04))
    keyword={'obstacle_distances':np.full((1,4),-.5),'medium_crater_distances':np.full((1,4),-1.)}
    _,reference=old.advance(*arguments,**keyword)
    _,candidate=new.advance(*arguments,**keyword)
    old_potential,new_potential=old.potentials(before),old.potentials(after)
    collision=-settings.collision_cost if reason=='collision' else 0.
    for name,value in reference.items():
        if name in ('terminal','crater_medium'):
            np.testing.assert_array_equal(candidate[name],value)
        else:
            fixed=np.zeros(1)
            if name in old_potential:
                fixed=settings.gamma*np.where(task['ended_this_step'],0.,new_potential[name])-old_potential[name]
            if name=='safety':fixed+=collision
            np.testing.assert_allclose(candidate[name]-fixed,.02*(value-fixed),atol=1e-12,rtol=1e-12)
    assert new.medium_crater_paid[0]==old.medium_crater_paid[0]


def test_scale_configuration_and_checkpoint_identity(cfg):
    base=UpstreamRewardSettings()
    assert 'continuous_cost_scale' not in base.metadata
    assert replace(base,continuous_cost_scale=1.).metadata==base.metadata
    scaled=configured_reward_settings(.999,.99,{'reward_scaling':'log1p','reward_overrides':{'continuous_cost_scale':.02}})
    assert scaled.continuous_cost_scale==.02
    old=UpstreamReward(cfg.spec,1,base,obstacle_input=cfg.obstacle_input)
    new=UpstreamReward(cfg.spec,1,scaled,obstacle_input=cfg.obstacle_input)
    assert 'continuous_cost_scope' not in old.metadata
    assert new.metadata['continuous_cost_scope']=='legacy_nonpotential_excluding_crater_medium_v1'
    with pytest.raises(ValueError,match='identity'):
        new.load_state_dict(old.state_dict())
    restored=UpstreamReward(cfg.spec,1,scaled,obstacle_input=cfg.obstacle_input)
    restored.load_state_dict(new.state_dict())
    assert restored.metadata==new.metadata


@pytest.mark.parametrize('scale',[0.,-1.,float('nan'),True])
def test_invalid_continuous_scale_is_rejected(scale):
    with pytest.raises(ValueError):UpstreamRewardSettings(continuous_cost_scale=scale)
