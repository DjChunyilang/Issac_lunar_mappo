"""Bounded sensor-only point memory. No terrain backend is available here."""

from collections import OrderedDict

import numpy as np
from scipy.spatial import cKDTree


class LocalTerrainMemory:
    def __init__(self, capacity=4096, resolution=0.1, support_radius=0.155,
                 version="quantized_latest_v1"):
        if version not in ("quantized_latest_v1", "direct_batch_v2", "direct_batch_v3"):
            raise ValueError("Unsupported local terrain memory version")
        if capacity <= 0 or not np.isfinite([resolution, support_radius]).all() or min(resolution, support_radius) <= 0:
            raise ValueError("Invalid local terrain memory bounds")
        self.capacity = capacity
        self.resolution = resolution
        self.support_radius = support_radius
        self.version = version
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
        if getattr(self, "version", "quantized_latest_v1") in ("direct_batch_v2", "direct_batch_v3"):
            return self._observe_direct_batch(xy, features, timestamp)
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
        self.tree = cKDTree(self.xy) if len(self.samples) else None

    def _observe_direct_batch(self, xy, features, timestamp):
        """Replace touched indexing cells with ALL their current direct samples.

        A cell is only an indexing/eviction unit, never a measurement location.
        Distinct positions in one cell survive; exact duplicate positions merge
        safety attributes conservatively. No support radius is enlarged.
        """
        xy, features = np.asarray(xy, float), np.asarray(features, float)
        if xy.ndim != 2 or xy.shape[1:] != (2,) or features.shape != (len(xy), 5):
            raise ValueError("Invalid direct terrain observation shape")
        if not np.isfinite(xy).all() or not np.isfinite(features).all() or not np.isfinite(timestamp):
            raise ValueError("Nonfinite direct terrain observation")
        fresh = OrderedDict()
        cells = set()
        for point, feature in zip(xy, features):
            cell = tuple(np.rint(point / self.resolution).astype(int))
            cells.add(cell)
            key = (*cell, float(point[0]), float(point[1]))
            if key in fresh:
                merged = fresh[key][1].copy()
                if np.linalg.norm(feature[1:3]) > np.linalg.norm(merged[1:3]):
                    merged[1:3] = feature[1:3]
                merged[4] = min(merged[4], feature[4])
                merged[0] = feature[0]
                fresh[key] = (point.copy(), merged, float(timestamp))
            else:
                fresh[key] = (point.copy(), feature.copy(), float(timestamp))
        if len(fresh) > self.capacity:
            raise ValueError("Memory capacity cannot retain a complete direct observation batch")
        preserve_hazards = self.version == "direct_batch_v3"
        rows = OrderedDict((key, (self._xy[slot].copy(), self._features[slot].copy(),
                                 float(self._timestamps[slot])))
                           for key, slot in self.samples.items() if key[:2] not in cells or
                           (preserve_hazards and (self._features[slot, 4] < .25 or
                             np.linalg.norm(self._features[slot, 1:3]) > .8)))
        if preserve_hazards:
            for key in fresh.keys() & rows.keys():
                point, feature, observed = fresh[key]
                previous = rows[key][1]
                if np.linalg.norm(previous[1:3]) > np.linalg.norm(feature[1:3]):
                    feature[1:3] = previous[1:3]
                feature[4] = min(feature[4], previous[4])
                fresh[key] = point, feature, observed
            # Reobserved hazards are current direct samples too; refresh their
            # LRU position before capacity eviction so the whole batch survives.
            for key in fresh:
                rows.pop(key, None)
        rows.update(fresh)
        while len(rows) > self.capacity:
            rows.popitem(last=False)
        self.samples = OrderedDict()
        for slot, (key, (point, feature, observed)) in enumerate(rows.items()):
            self.samples[key] = slot
            self._xy[slot], self._features[slot], self._timestamps[slot] = point, feature, observed
        self.tree = cKDTree(self.xy) if len(self.samples) else None

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
