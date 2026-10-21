"""Cost-scale regression and unchanged actuator predictions for S04 v3."""
from dataclasses import replace
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from _common import cfg_from_experiment
from test_upstream_mppi import request,terrain as terrain_fixture
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_planning import UpstreamIntent
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_mppi import UpstreamMPPISettings
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_mppi_precision import UpstreamMPPIKernelV2
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_mppi_calibration import UpstreamMPPIKernelV3,UpstreamMPPIKernelV4,UpstreamMPPISettingsV3
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_factory import create_upstream_adapter


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_progress_cost_overcomes_spurious_one_step_smooth_penalty_with_identical_physics(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.fail('Required CUDA correctness is unavailable')
    torch.set_num_threads(1)
    cfg=replace(cfg_from_experiment('configs/experiment/upstream_scale_s04_cost_v3.yaml'),planner_device=device)
    terrain=terrain_fixture.__wrapped__(cfg)
    item=request(cfg,terrain,intent=UpstreamIntent([.2,0.],False,0.,1.))
    old=UpstreamMPPIKernelV2(cfg,UpstreamMPPISettings(samples=2,device=device))
    new=UpstreamMPPIKernelV3(cfg,UpstreamMPPISettingsV3(samples=2,device=device))
    raw=torch.zeros((1,2,40,2),device=device);raw[0,1,0,0]=.04
    a=old.evaluate([item],raw);b=new.evaluate([item],raw)
    for i in (0,1,2,4,5):assert torch.equal(a[i],b[i]) # Same states, controls, velocities and validity.
    assert a[-1]['goal'][0,0].item()==pytest.approx(.23,abs=1e-12)
    assert a[-1]['smooth'][0,1].item()==pytest.approx(1.,abs=1e-7)
    expected=.04**2/(25/9)**2 # Two changes, each weighted 0.5.
    assert b[-1]['smooth'][0,1].item()==pytest.approx(expected,abs=1e-10)
    assert a[3][0,1]>a[3][0,0] and b[3][0,1]<b[3][0,0]
    for key in ('control','reverse','unknown','terrain','neighbor'):
        torch.testing.assert_close(a[-1][key],b[-1][key],rtol=0,atol=0)


def test_cost_version_rejects_old_hot_state_and_freezes_settings_identity():
    cfg=cfg_from_experiment('configs/experiment/upstream_scale_s04_cost_v3.yaml')
    new=create_upstream_adapter(cfg);old=create_upstream_adapter(replace(cfg,planner_adapter='mppi_upstream_v2'))
    with pytest.raises(ValueError):new.load_state_dict(old.state_dict())
    assert new.state_version==3 and cfg.metadata['mppi_cost_version']=='upstream_mppi_speed_normalized_cost_v2'
    settings=UpstreamMPPISettingsV3(samples=512,goal_scale=2.)
    modified=create_upstream_adapter(cfg,mppi_settings=settings)
    assert modified.batch_kernel.metadata['settings']['goal_scale']==2.
    with pytest.raises(ValueError):new.load_state_dict(modified.state_dict())


def test_near_goal_initialization_and_configured_weights_are_explicit():
    cfg=cfg_from_experiment('configs/experiment/upstream_scale_s04_guide_probe_cuda_v3.yaml')
    adapter=create_upstream_adapter(cfg);terrain=terrain_fixture.__wrapped__(cfg)
    item=request(cfg,terrain,intent=UpstreamIntent([.8,0.],False,0.,1.))
    guide=adapter.batch_kernel.seed_controls([item])
    assert guide[0,0,0].item()==pytest.approx(.1,abs=1e-7)
    assert adapter.batch_kernel.settings.goal_scale==2. and adapter.batch_kernel.settings.temperature==.5
    assert cfg.metadata['mppi_parameters']=={'goal_scale':2.,'temperature':.5}
    # This initialization creates no additional execution speed ceiling.
    assert item.intent.speed_cap is None and cfg.spec.vehicle.forward_limit==25/9


def test_target_normalization_preserves_explicit_yaw_weight_and_hard_validity():
    cfg=cfg_from_experiment('configs/experiment/upstream_scale_s04_target_probe_cuda_v4.yaml')
    terrain=terrain_fixture.__wrapped__(cfg)
    item=request(cfg,terrain,intent=UpstreamIntent([.8,0.],False,0.,1.,terminal_yaw=3.141592653589793))
    settings=UpstreamMPPISettingsV3(samples=2,device='cuda',goal_scale=2.,temperature=.5)
    old=UpstreamMPPIKernelV3(cfg,settings);new=UpstreamMPPIKernelV4(cfg,settings)
    raw=torch.zeros((1,2,40,2),device='cuda')
    a=old.evaluate([item],raw);b=new.evaluate([item],raw)
    for i in (0,1,2,4,5):assert torch.equal(a[i],b[i])
    assert a[-1]['goal'][0,0].item()==pytest.approx(15.36)
    assert b[-1]['goal'][0,0].item()==pytest.approx(192.) # Relative XY cost 184 + unchanged explicit yaw cost 8.
    zero=request(cfg,terrain,intent=UpstreamIntent([0.,0.],False,0.,1.))
    assert torch.isfinite(new.evaluate([zero],raw)[3]).all()
    assert cfg.metadata['planner_state_version']==4
