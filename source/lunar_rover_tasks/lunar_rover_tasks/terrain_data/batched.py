"""Shared bounded CPU/CUDA scene cache, with explicit missing-data support."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional as F

from .scene_pack import SCHEMA, ScenePack


@dataclass
class RasterQuery:
    features: torch.Tensor
    height_valid: torch.Tensor
    metrics_valid: torch.Tensor
    height_error: torch.Tensor
    error_valid: torch.Tensor


class SceneBank:
    """Scene data are shared across environments and only loaded at reset.

    query never reads files. prefetch must retain the union of active scenes.
    Tensor invalid values are zero-filled ONLY internally; all consumers must
    use returned support masks. Quality is not provided to the actor by default.
    """

    def __init__(self, manifests, *, device="cpu", max_bytes=1536 * 1024**2, storage="tensor"):
        if not manifests:
            raise ValueError("At least one scene required")
        self.scenes = [ScenePack(p) for p in manifests]
        self.device = torch.device(device)
        if storage not in ("tensor", "mmap") or (storage == "mmap" and self.device.type != "cpu"):
            raise ValueError("mmap scene queries require CPU storage")
        self.storage = storage
        self.max_bytes = max_bytes
        self.resident = OrderedDict()
        self.bytes = 0

    def __deepcopy__(self, memo):
        # Immutable scene content and cache are shared; episode references are
        # copied by TerrainRuntime. Never duplicate map tensors in checkpoints.
        return self

    def prefetch(self, active_scene_ids):
        ids = sorted(set(torch.as_tensor(active_scene_ids).detach().cpu().tolist()))
        if any(i < 0 or i >= len(self.scenes) for i in ids):
            raise ValueError("Scene index outside bank")
        if self.storage == "mmap":
            # Immutable npy mappings share the OS page cache across owners.
            # Only queried cells are gathered; never materialize an entire atlas.
            self.resident = OrderedDict((i, self.scenes[i].arrays) for i in ids)
            return
        sizes = {i: np.prod(self.scenes[i].meta["shape"]) * 8 * 4 for i in ids}
        if sum(sizes.values()) > self.max_bytes:
            raise MemoryError(
                "Active scene set exceeds cache budget; reduce resident maps or environment batch"
            )
        for key in list(self.resident):
            if key not in ids:
                self.bytes -= self.resident.pop(key).numel() * 4
        for i in ids:
            if i in self.resident:
                continue
            scene = self.scenes[i]
            a = scene.arrays
            error = a.get("quality_height_error", np.full(scene.meta["shape"], np.nan, np.float32))
            error_good = np.isfinite(error) & (error >= 0)
            stack = np.concatenate(
                (
                    np.nan_to_num(a["features"], nan=0, posinf=0, neginf=0),
                    a["height_valid"][None],
                    a["metrics_valid"][None],
                    np.where(error_good, error, 0)[None],
                    error_good[None],
                ),
                axis=0,
            ).astype(np.float32)
            tensor = torch.from_numpy(stack).unsqueeze(0).to(self.device)
            self.resident[i] = tensor
            self.bytes += tensor.numel() * 4

    def query(self, xy, scene_ids):
        if xy.shape[0] != len(scene_ids) or xy.shape[-1] != 2:
            raise ValueError("Queries require environment-leading xy and scene_ids")
        if self.storage == "mmap":
            return self._query_mmap(xy, scene_ids)
        target_device = xy.device
        points = xy.to(self.device, dtype=torch.float32)
        ids = scene_ids.to(self.device, dtype=torch.long)
        n = points.shape[0]
        points = points.reshape(n, -1, 2)
        result = torch.zeros(n, points.shape[1], 8, device=self.device)
        present = torch.zeros(n, dtype=torch.bool, device=self.device)
        for key, tensor in self.resident.items():
            selected = ids == key
            # Empty batches are cheap; scalar synchronization is one per scene,
            # never one per rover/sample. Full GPU atlas kernels can replace this.
            if not bool(selected.any()):
                continue
            extent = self.scenes[key].meta["extent_xy_m"]
            selected_points = points[selected]
            gx = 2 * selected_points[..., 0] / extent[0]
            gy = -2 * selected_points[..., 1] / extent[1]
            grid = torch.stack((gx, gy), -1)
            finite = torch.isfinite(grid).all(-1)
            h, w = self.scenes[key].meta["shape"]
            inside = (
                finite
                & (gx >= -1 + 1 / w)
                & (gx <= 1 - 1 / w)
                & (gy >= -1 + 1 / h)
                & (gy <= 1 - 1 / h)
            )
            sampled = F.grid_sample(
                tensor,
                torch.nan_to_num(grid).reshape(1, 1, -1, 2),
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
            sampled = sampled[0, :, 0].T.reshape(*grid.shape[:-1], 8)
            sampled[..., 4:6] *= inside[..., None]
            sampled[..., 7] *= inside
            result[selected] = sampled
            present |= selected
        if not bool(present.all()):
            raise RuntimeError("Scene not prefetched; loading is allowed only at reset")
        result = result.reshape(*xy.shape[:-1], 8).to(target_device)
        hv = result[..., 4] >= 1 - 1e-6
        mv = (result[..., 5] >= 1 - 1e-6) & hv
        ev = (result[..., 7] >= 1 - 1e-6) & hv
        features = result[..., :4].clone()
        features[..., 0] = torch.where(hv, features[..., 0], float("nan"))
        features[..., 1:4] = torch.where(mv[..., None], features[..., 1:4], float("nan"))
        error = torch.where(ev, result[..., 6], float("nan"))
        return RasterQuery(features, hv, mv, error, ev)

    def _query_mmap(self, xy, scene_ids):
        points = xy.detach().cpu().float().numpy().reshape(len(scene_ids), -1, 2)
        ids = scene_ids.detach().cpu().numpy()
        result = np.zeros((*points.shape[:-1], 8), np.float32)
        if not set(ids).issubset(self.resident):
            raise RuntimeError("Scene not prefetched; loading is allowed only at reset")
        for key, arrays in self.resident.items():
            selected = ids == key
            if not selected.any():
                continue
            h, w = self.scenes[key].meta["shape"]
            ex, ey = self.scenes[key].meta["extent_xy_m"]
            p = points[selected]
            gx, gy = 2 * p[..., 0] / ex, -2 * p[..., 1] / ey
            inside = (np.isfinite(p).all(-1) & (gx >= -1 + 1/w) & (gx <= 1 - 1/w)
                      & (gy >= -1 + 1/h) & (gy <= 1 - 1/h))
            cx = ((np.nan_to_num(gx, nan=0, posinf=2, neginf=-2) + 1) * w - 1) / 2
            cy = ((np.nan_to_num(gy, nan=0, posinf=2, neginf=-2) + 1) * h - 1) / 2
            x0, y0 = np.floor(cx).astype(np.int64), np.floor(cy).astype(np.int64)
            fx, fy = cx-x0, cy-y0
            sampled = np.zeros((*cx.shape, 8), np.float32)
            for ox, oy in ((0, 0), (1, 0), (0, 1), (1, 1)):
                x, y = x0 + ox, y0 + oy
                valid = (x >= 0) & (x < w) & (y >= 0) & (y < h)
                x, y = np.clip(x, 0, w-1), np.clip(y, 0, h-1)
                weight = ((fx if ox else 1-fx) * (fy if oy else 1-fy) * valid).astype(np.float32)
                values = np.zeros((*cx.shape, 8), np.float32)
                values[..., :4] = np.moveaxis(np.nan_to_num(arrays['features'][:, y, x], nan=0, posinf=0, neginf=0), 0, -1)
                values[..., 4] = arrays['height_valid'][y, x]
                values[..., 5] = arrays['metrics_valid'][y, x]
                if 'quality_height_error' in arrays:
                    error = arrays['quality_height_error'][y, x]
                    good = np.isfinite(error) & (error >= 0)
                    values[..., 6] = np.where(good, error, 0)
                    values[..., 7] = good
                sampled += weight[..., None] * values
            sampled[..., 4:6] *= inside[..., None]
            sampled[..., 7] *= inside
            result[selected] = sampled
        result = torch.from_numpy(result.reshape(*xy.shape[:-1], 8)).to(xy.device)
        hv = result[..., 4] >= 1-1e-6
        mv = (result[..., 5] >= 1-1e-6) & hv
        ev = (result[..., 7] >= 1-1e-6) & hv
        features = result[..., :4].clone()
        features[..., 0] = torch.where(hv, features[..., 0], float('nan'))
        features[..., 1:] = torch.where(mv[..., None], features[..., 1:], float('nan'))
        return RasterQuery(features, hv, mv, torch.where(ev, result[..., 6], float('nan')), ev)

    def contract(self):
        return {"schema": SCHEMA, "scenes": [s.checkpoint_contract() for s in self.scenes]}

    def close(self):
        self.resident.clear()
        self.bytes = 0
