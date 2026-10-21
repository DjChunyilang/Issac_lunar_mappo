"""Exact pose-only cache scoped to one footprint trajectory review.

No time-dependent neighbor, control, dynamics, workspace or acceptance result
is cached. A stationary tail still executes every required swept/time check.
The dictionary is discarded even after an exception; nothing enters snapshots.
"""
from contextvars import ContextVar

import numpy as np

from .upstream_footprint import FootprintReviewer

CACHE_VERSION="pose_scope_cache_v1"


class PoseCachedReviewer(FootprintReviewer):
    acceleration_version=CACHE_VERSION

    def __init__(self,spec,**kwargs):
        super().__init__(spec,**kwargs)
        self._scope=ContextVar(f'pose_review_{id(self)}',default=None)

    def pose(self,pose,terrain,*,parking=False):
        scope=self._scope.get()
        array=np.asarray(pose,dtype=float)
        if scope is None or scope[0] is not terrain or array.shape!=(3,) or not np.isfinite(array).all():
            return super().pose(pose,terrain,parking=parking)
        # Use exact bytes. Near-equal poses and different parking thresholds
        # must receive independent physical checks.
        key=(array.tobytes(),bool(parking))
        if key not in scope[1]:scope[1][key]=super().pose(array,terrain,parking=parking)
        return scope[1][key]

    def trajectory(self,trajectory,previous,terrain,**kwargs):
        token=self._scope.set((terrain,{}))
        try:return super().trajectory(trajectory,previous,terrain,**kwargs)
        finally:self._scope.reset(token)
