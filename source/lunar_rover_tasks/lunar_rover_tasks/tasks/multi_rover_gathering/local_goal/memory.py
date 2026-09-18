"""Bounded sensor-only point memory. No terrain backend is available here."""

from collections import OrderedDict

import numpy as np
from scipy.spatial import cKDTree


class LocalTerrainMemory:
    def __init__(self, capacity=4096, resolution=0.1, support_radius=0.155):
        self.capacity = capacity
        self.resolution = resolution
        self.support_radius = support_radius
        self.reset()

    def reset(self):
        self.samples = OrderedDict()
        self.tree = None
        self._xy = np.empty((self.capacity, 2))
        self._features = np.empty((self.capacity, 5))
        self._timestamps = np.empty(self.capacity)

    @property
    def xy(self):
        return self._xy[: len(self.samples)]

    @property
    def features(self):
        return self._features[: len(self.samples)]

    def observe(self, xy, features, timestamp):
        # Only caller-supplied direct observations enter this collection.
        for point, feature in zip(xy, features):
            key = tuple(np.rint(point / self.resolution).astype(int))
            slot = self.samples.pop(key, None)
            if slot is None:
                if len(self.samples) == self.capacity:
                    _, slot = self.samples.popitem(last=False)
                else:
                    slot = len(self.samples)
            self.samples[key] = slot
            self._xy[slot] = point
            self._features[slot] = feature
            self._timestamps[slot] = timestamp
        self.tree = cKDTree(self.xy)

    def lookup(self, xy):
        xy = np.atleast_2d(xy)
        if self.tree is None:
            return np.zeros((len(xy), 5)), np.zeros(len(xy), bool), np.zeros(len(xy), bool)
        distance, index = self.tree.query(xy)
        # Supported estimates and directly observed samples are DISTINCT flags.
        return self.features[index], distance <= self.support_radius, distance < 1e-5

    def safe(self, xy):
        features, known, _ = self.lookup(xy)
        # Road limits, deliberately unrelated to terminal-site flatness limits.
        return known & (features[:, 4] >= 0.25) & (np.linalg.norm(features[:, 1:3], axis=1) <= 0.8)

    def local_ball(self, xy):
        """Conservative observed support disk for a sequential-convex terrain constraint."""
        if self.tree is None:
            return np.asarray(xy), 0.0
        _, index = self.tree.query(xy)
        feature = self.features[index]
        radius = (
            self.support_radius
            if feature[4] >= 0.25 and np.linalg.norm(feature[1:3]) <= 0.8
            else 0.0
        )
        return self.xy[index].copy(), radius

    def free_disks(self, center, count=128):
        """A time-independent union of observed support disks, not a timed corridor.

        Restrict each disk by the nearest impassable observation so that its
        interior cannot be closer to a known bad sample than to its own center.
        The subset is conservative and never introduces unobserved free space.
        """
        disks = np.tile([1000.0, 1000.0, 0.0], (count, 1))
        if self.tree is None:
            return disks
        valid = (self.features[:, 4] >= 0.25) & (
            np.linalg.norm(self.features[:, 1:3], axis=1) <= 0.8
        )
        free = self.xy[valid]
        if not len(free):
            return disks
        indices = np.argsort(np.linalg.norm(free - np.asarray(center), axis=1), kind="stable")[
            :count
        ]
        points = free[indices]
        radii = np.full(len(points), self.support_radius)
        if (~valid).any():
            distance, _ = cKDTree(self.xy[~valid]).query(points)
            radii = np.minimum(radii, 0.5 * distance)
        disks[: len(points), :2] = points
        disks[: len(points), 2] = radii
        return disks
