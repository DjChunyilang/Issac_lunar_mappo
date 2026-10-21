"""Independent 36-case inputs bind dense sensing, actual history and sender data."""
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from _common import cfg_from_experiment
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_cases import calibration_cases,FAMILIES,CALIBRATION_SEEDS
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_planning import UpstreamIntent


def test_calibration_grid_has_exact_independent_seeds_real_frames_and_private_inputs():
    import torch
    torch.set_num_threads(1)
    cfg=cfg_from_experiment('configs/experiment/upstream_scale_s04_calibration_cuda_v4.yaml')
    cases=calibration_cases(cfg)
    assert len(cases)==36 and len({c.fingerprint for c in cases})==36
    assert {c.family for c in cases}==set(FAMILIES) and {c.seed for c in cases}==set(CALIBRATION_SEEDS)
    assert not {23,31,47}&{c.seed for c in cases}
    for c in cases:
        c.input.__post_init__();c.input.actual_history.validate(cfg.spec,0,0)
        snapshot=c.input.terrain.snapshot()
        assert len(snapshot.xy)<=12231 and len(snapshot.xy)>8000
        assert (snapshot.observed_at==0.).all() and c.input.information_time==0.
        assert abs(c.input.now-.6)<1e-10 and c.input.times.shape==(41,)
        assert not hasattr(c.input.terrain,'_runtime') and not hasattr(c.input.terrain,'_core')
        assert len({p.sender_id for p in c.input.neighbors})==3
    unknown=[c for c in cases if c.family=='unknown']
    assert all(not c.input.terrain.lookup([8.,0.])[1][0] for c in unknown)
    # Goal/intent semantics enter the case identity, independently of code SHA.
    c=cases[0];changed=replace(c,input=replace(c.input,intent=UpstreamIntent([1.,0.],False,.6,1.6)))
    assert changed.fingerprint!=c.fingerprint


def test_initial_high_speed_cases_bind_actual_delay_without_instantaneous_clipping():
    import torch
    torch.set_num_threads(1)
    cfg=cfg_from_experiment('configs/experiment/upstream_scale_s04_calibration_cuda_v4.yaml')
    cases=calibration_cases(cfg)
    full=next(c for c in cases if c.family=='flat' and c.seed==109)
    np.testing.assert_allclose(full.input.previous_control,[25/9,0.],atol=1e-10)
    assert full.input.ego_pose[0]>1.6 and len(full.input.actual_history.controls)==3
    assert full.input.intent.speed_cap is None and not full.input.intent.stop_at_goal
