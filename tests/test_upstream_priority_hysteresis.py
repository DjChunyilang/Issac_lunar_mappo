"""Parked-slot hysteresis cannot permanently block lower-priority followers."""
from dataclasses import replace

import pytest
import numpy as np

from test_upstream_gathering import cfg, terrain, messages, decide
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_visibility import (
    LeaderSquareVisibilityScript, LeaderSquareVisibilityHysteresisScript,
    clearance_waypoint,
)


def test_historical_priority_wait_and_new_release(cfg, terrain):
    pose = [7.2, -4., 0.]
    old = LeaderSquareVisibilityScript(cfg.spec, 0, 2)
    new = LeaderSquareVisibilityHysteresisScript(cfg.spec, 0, 2)
    peers = {1: [0., 7.2 + .3079473168699153]}
    assert decide(old, cfg, terrain, pose, peer_positions=peers).reason == 'lower_sender_priority'
    released = decide(new, cfg, terrain, pose, peer_positions=peers)
    assert not released.intent.hold
    assert new.metadata != old.metadata
    with pytest.raises(ValueError):
        new.load_state_dict(old.state_dict())


@pytest.mark.parametrize('violation', ['outside_exit', 'moving', 'not_hold', 'stale'])
def test_release_still_requires_legal_stopped_parked_peer(cfg, terrain, violation):
    script = LeaderSquareVisibilityHysteresisScript(cfg.spec, 0, 2)
    pose = [7.2, -4., 0.]
    snapshot = messages(cfg, 2, pose, now=2., peer_positions={1: [0., 7.2 + (.41 if violation == 'outside_exit' else .31)]})
    packets = list(snapshot.messages)
    index = next(i for i, p in enumerate(packets) if p.sender_id == 1)
    if violation == 'moving':packets[index] = replace(packets[index], actual_velocity=[.1, 0.])
    if violation == 'not_hold':packets[index] = replace(packets[index], hold=False)
    if violation == 'stale':packets[index] = replace(packets[index], timestamp=0., received_at=0.)
    snapshot = replace(snapshot, messages=tuple(packets))
    result = script.decide(pose, [0., 0.], [0., 0.], terrain, snapshot, 2., 0)
    assert result.intent.hold and result.reason == 'lower_sender_priority'


@pytest.mark.parametrize('pose,center,goal', [
    ([9.30469792,.82949223],[8.52639398,7.39488906],[15.72639398,7.39488906]),
    ([13.37951363,13.12892948],[17.33678729,7.85124835],[17.33678729,15.05124835]),
])
def test_saved_nmpc_stall_waypoint_increases_clearance(pose,center,goal):
    pose=np.asarray(pose);center=np.asarray(center);radius=6.548968397966155
    waypoint,detour=clearance_waypoint(pose,goal,[(0,center,radius)])
    assert detour and np.linalg.norm(waypoint-pose)<=4.
    path=pose+np.linspace(0.,1.,101)[:,None]*(waypoint-pose)
    assert np.min(np.linalg.norm(path-center,axis=1))>=radius
    assert np.linalg.norm(waypoint-center)>np.linalg.norm(pose-center)+.4
