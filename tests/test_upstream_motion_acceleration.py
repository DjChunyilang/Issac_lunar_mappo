"""Stable native memory updates against the original scalar merge contract."""
from dataclasses import replace

import numpy as np
import pytest

from test_upstream_sensing import cfg, frame
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_native import memory_merge_cpu
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_sensing import DenseTerrainMemory


@pytest.mark.parametrize('seed', range(8))
def test_native_memory_matches_scalar_hazards_duplicates_and_order(cfg, seed):
    rng = np.random.default_rng(seed)
    memory = DenseTerrainMemory(cfg.spec)
    for step in range(5):
        xy = rng.integers(-10, 11, size=(300, 2)) * .1 + rng.uniform(-.04, .04, (300, 2))
        features = rng.uniform(0., 1., (300, 5))
        features[:, 0] *= .3; features[:, 1:3] -= .5; features[:, 1:3] *= .4
        features[:, 3] *= .07
        if step:
            xy[:20] = memory.view()._xy[:20]
        xy[20:25] = xy[25:30]  # Merge duplicates at their first insertion slot.
        xy[30:32] = [[0., -0.], [-0., 0.]]
        xy[32:36] = [[.05, .15], [-.05, -.15], [.25, -.25], [-.25, .25]]
        expected = memory._merge_reference(xy, features, float(step))
        actual = memory_merge_cpu(memory.view(), xy, features, float(step), cfg.spec)
        assert actual is not None
        assert actual[3] == expected[3]
        for a, b in zip(actual[:3], expected[:3], strict=True):
            np.testing.assert_array_equal(a.view(np.uint64), b.view(np.uint64))
        old = memory.view(); old_xy = old._xy.copy()
        memory._view = memory._make_view(*actual[:3])
        np.testing.assert_array_equal(old._xy, old_xy)
        assert not old._xy.flags.writeable


def test_native_memory_capacity_preserves_fresh_and_eviction_order(cfg):
    memory = DenseTerrainMemory(cfg.spec)
    count = cfg.spec.perception.capacity
    xy = np.column_stack((np.arange(count) * .2, np.zeros(count)))
    features = np.zeros((count, 5)); features[:, 4] = 1.
    times = np.arange(count, dtype=float)
    memory._view = memory._make_view(xy, features, times)
    fresh = np.column_stack((np.arange(count - 10, count + 80) * .2, np.zeros(90)))
    new_features = np.zeros((90, 5)); new_features[:, 4] = 1.
    expected = memory._merge_reference(fresh, new_features, float(count))
    actual = memory_merge_cpu(memory.view(), fresh, new_features, float(count), cfg.spec)
    assert actual is not None and len(actual[0]) == count
    for a, b in zip(actual[:3], expected[:3], strict=True):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(actual[0][-90:], fresh)


def test_native_memory_boundary_uses_scalar_authority(cfg, frame):
    memory = DenseTerrainMemory(cfg.spec)
    valid = np.zeros(len(frame.xy), dtype=bool)
    index = np.linalg.norm(frame.xy, axis=1).argmin(); valid[index] = True
    f = np.zeros_like(frame.features)
    f[index] = [0., cfg.spec.terrain.driving_slope, 0., 0., 1.]
    first = replace(frame, features=f, valid=valid, roughness_valid=valid)
    memory.observe(first)
    xy = first.xy + [.01, 0.]
    f[index] = [0., 0., 0., 0., 1.]
    second = replace(first, xy=xy, sensor_pose=first.sensor_pose + [.01, 0., 0.], features=f, timestamp=.2)
    expected = memory._merge_reference(second.xy[valid], second.features[valid], .2)
    assert memory_merge_cpu(memory.view(), second.xy[valid], second.features[valid], .2, cfg.spec) is None
    memory.observe(second)
    for a, b in zip((memory.view()._xy, memory.view()._features, memory.view()._timestamps), expected[:3], strict=True):
        np.testing.assert_array_equal(a, b)
