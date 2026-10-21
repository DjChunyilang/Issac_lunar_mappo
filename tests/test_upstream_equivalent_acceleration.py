"""Equivalent fast paths preserve hazards, swept failures and public state."""
from dataclasses import replace

import numpy as np
import pytest

from test_upstream_sensing import cfg, frame, sparse  # noqa: F401
from test_upstream_visibility_flat_review import field
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_sensing import DenseTerrainMemory
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_flat_review import FlatFramePoseReviewer
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_footprint import FootprintNeighbor
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_execution import rollout_numpy


def test_repeated_direct_frame_preserves_old_hazards_times_and_restore(cfg, frame):
    fast, reference = DenseTerrainMemory(cfg.spec), DenseTerrainMemory(cfg.spec)
    frames = [sparse(frame, height=.4), sparse(frame, time=1., offset=.04),
              sparse(frame, time=2., offset=.04), sparse(frame, time=3., offset=.04),
              sparse(frame, time=4., offset=.06, traversability=.1),
              sparse(frame, time=5., offset=.06, traversability=.1)]
    for incoming in frames:
        reference._last_frame = None  # Execute the conservative merge reference.
        fast.observe(incoming); reference.observe(incoming)
        a, b = fast.state_dict(), reference.state_dict()
        for key in a:
            np.testing.assert_equal(a[key], b[key])
    assert len(fast.view().snapshot().xy) > 1
    old = fast.view()
    fast.observe(replace(frames[-1], timestamp=6.))
    assert fast.view() is not old and fast.view()._tree is old._tree
    assert not fast.view()._xy.flags.writeable
    restored = DenseTerrainMemory(cfg.spec); restored.load_state_dict(fast.state_dict())
    incoming = replace(frames[-1], timestamp=7.)
    restored.observe(incoming); fast.observe(incoming)
    for key in fast.state_dict():
        np.testing.assert_equal(fast.state_dict()[key], restored.state_dict()[key])


@pytest.mark.parametrize('case', ['clear', 'crossing', 'expanding_radius', 'workspace', 'unknown', 'rotation'])
def test_batched_sweep_matches_scalar_including_first_failure(cfg, case):
    spec = cfg.spec; terrain = field(spec)
    fast, scalar = FlatFramePoseReviewer(spec), FlatFramePoseReviewer(spec)
    scalar._flat_trajectory = lambda *args: None
    scalar._clear_sweep = lambda *args: False
    pose = [0., 0., .2] if case != 'unknown' else [14., 0., .2]
    control = [0., .1] if case == 'rotation' else [.2, 0.]
    plan = rollout_numpy(pose, [control] * 5, spec)
    times = plan.times
    positions = np.tile([12., 0.], (len(times), 1))
    radii = np.full(len(times), 2 * spec.vehicle.effective_radius)
    if case == 'crossing':positions[:, 0] = np.linspace(12., -12., len(times))
    if case == 'expanding_radius':radii += np.linspace(0., 9., len(times))
    neighbor = FootprintNeighbor.for_spec(2, times, positions, radii, spec)
    workspace = [[-4., -4.], [3.3, 4.]] if case == 'workspace' else None
    args = dict(neighbors=[neighbor], workspace=workspace)
    assert fast.trajectory(plan, control, terrain, **args) == scalar.trajectory(plan, control, terrain, **args)


@pytest.mark.parametrize('case', ['clear','growth','crossing','boundary','sampling_boundary','unknown','bad_control','bad_state','parking'])
def test_stationary_batch_checks_future_uncertainty_and_original_failures(cfg,case):
    spec=cfg.spec;terrain=field(spec)
    fast,scalar=FlatFramePoseReviewer(spec),FlatFramePoseReviewer(spec)
    scalar._stationary_trajectory=lambda *args:None
    scalar._flat_trajectory = lambda *args: None
    scalar._clear_sweep=lambda *args:False
    pose=[14.,0.,0.] if case=='unknown' else [0.,0.,0.]
    plan=rollout_numpy(pose,np.zeros((40,2)),spec)
    positions=np.tile([12.,0.],(41,1));radii=np.full(41,2*spec.vehicle.effective_radius)
    if case=='growth':radii+=np.linspace(0.,15.,41)
    if case=='crossing':positions[:,0]=np.linspace(12.,-12.,41)
    if case=='sampling_boundary':positions[:,0]=12.-np.arange(41)*.8
    if case=='boundary':positions[:,0]=radii[0]-1e-8
    if case=='bad_control':plan.controls[7]=[.4,0.]
    if case=='bad_state':plan.states[12,0]+=.01
    peer=FootprintNeighbor.for_spec(2,plan.times,positions,radii,spec)
    args=dict(neighbors=[peer],workspace=[[-100.,-100.],[100.,100.]],require_stop=True,parking_end=case=='parking')
    assert fast.trajectory(plan,[0.,0.],terrain,**args)==scalar.trajectory(plan,[0.,0.],terrain,**args)

