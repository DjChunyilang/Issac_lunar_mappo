"""Canonical +/- pi jumps are equivalent odometry, actual yaw defects are not."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from _common import cfg_from_experiment
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.contracts import wrap
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_timing import ActualControlHistory


@pytest.mark.parametrize('sign',[-1.,1.])
def test_actual_midpoint_history_crosses_angle_branch_without_invented_motion(sign):
    cfg=cfg_from_experiment('configs/experiment/upstream_scale_s03_environment_v2.yaml')
    heading=sign*(np.pi-.003);observed=wrap(heading+sign*.03*.2)
    history=ActualControlHistory(0,2,cfg.spec.fingerprint,[0.,.2],[[0.,0.,heading],[0.,0.,observed]],
        [[0.,sign*.03]],[[0.,sign*.03]],[0.,0.])
    history.validate(cfg.spec,0,2)
    bad=ActualControlHistory(0,2,cfg.spec.fingerprint,[0.,.2],[[0.,0.,heading],[0.,0.,observed+.01]],
        [[0.,sign*.03]],[[0.,sign*.03]],[0.,0.])
    with pytest.raises(ValueError,match='odometry'):bad.validate(cfg.spec,0,2)
