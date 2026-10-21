"""Local spatial-hash queries for new-scale Torch planning pre-screening.

Only owned sensor snapshots and identified predictions are packed. These
center queries do not replace the independent CPU contact/body/swept review.
"""

import math
import time

import numpy as np
import torch

from .upstream_footprint import FootprintTerrain


PAIR_WORK_BYTES = 256


def _device(value):
    device = torch.device(value)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Local pre-screen supports CPU or CUDA devices")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA planner is unavailable")
        device = torch.device("cuda", torch.cuda.current_device() if device.index is None else device.index)
    return device


class LocalTerrainBatch:
    version = "local_hash_prescreen_v1"

    def __init__(self, views, *, device="cpu", chunk=256, memory_budget=64 * 1024**2):
        if not views or any(type(view) is not FootprintTerrain for view in views):
            raise ValueError("Expected nonempty batch of owned local terrain views")
        if any(view.profile_fingerprint != views[0].profile_fingerprint for view in views):
            raise ValueError("Mixed terrain profiles")
        if isinstance(chunk, bool) or not isinstance(chunk, int) or chunk < 1 or isinstance(memory_budget, bool) or not isinstance(memory_budget, int) or memory_budget < 4096:
            raise ValueError("Invalid local query memory budget")
        self.device = _device(device)
        self.batch_size, self.chunk, self.memory_budget = len(views), chunk, memory_budget
        self.support = views[0].support_radius
        # Widen the cell by the complete nearest-tie tolerance. A 3 x 3
        # panel then contains every supported nearest point and its ties,
        # including a dangerous point just outside the support boundary.
        self.cell = self.support + 1e-8 + 1e-10
        self._views = tuple(views)
        self.ambiguous_queries = 0
        self.cpu_tie_query_s = 0.
        self.boundary_queries = 0
        snapshots = [view.readonly_snapshot() for view in views]
        xy = np.concatenate([s.xy for s in snapshots], axis=0)
        features = np.concatenate([s.features for s in snapshots], axis=0)
        senders = np.concatenate([np.full(len(s.xy), b, dtype=np.int64) for b, s in enumerate(snapshots)])
        self.empty = len(xy) == 0
        self.max_pair_entries = 0
        if self.empty:
            return
        cells = np.floor(xy / self.cell)
        if not np.isfinite(cells).all() or np.abs(cells).max() > 2**40:
            raise ValueError("Unsupported local coordinate magnitude")
        cells = cells.astype(np.int64)
        self.origin = cells.min(0)
        span = cells.max(0) - self.origin + 1
        self.nx, self.ny = int(span[0]), int(span[1])
        if self.batch_size * self.nx * self.ny >= 2**62:
            raise ValueError("Local spatial index overflow")
        cells -= self.origin
        keys = (senders * self.nx + cells[:, 0]) * self.ny + cells[:, 1]
        order = np.argsort(keys, kind="stable")
        keys, first, counts = np.unique(keys[order], return_index=True, return_counts=True)
        # A small known occupancy bound avoids a device-to-host scalar read in
        # every query block. Dense buckets still use their actual local bound.
        self._maximum_occupancy = int(counts.max())
        self.keys = torch.as_tensor(keys, device=self.device)
        self.first = torch.as_tensor(first, device=self.device)
        self.counts = torch.as_tensor(counts, device=self.device)
        # Query geometry uses float64 even for float32 rollout, preserving local
        # snapshot coordinates and conservative 1e-8 nearest-distance ties.
        self.xy = torch.as_tensor(xy[order], device=self.device, dtype=torch.float64)
        self.features = torch.as_tensor(features[order], device=self.device, dtype=torch.float64)
        self.origin_t = torch.as_tensor(self.origin, device=self.device)
        self.offsets = torch.cartesian_prod(torch.arange(-1, 2, device=self.device),
                                            torch.arange(-1, 2, device=self.device))

    @classmethod
    def from_rows(cls,views,rows):
        """Compose immutable per-rover device indexes without sorting/uploading unchanged rows."""
        present=[row for row in rows if not row.empty]
        if not present:
            return cls(views,device=rows[0].device,chunk=rows[0].chunk,memory_budget=rows[0].memory_budget)
        result=object.__new__(cls);result.__dict__.update(present[0].__dict__)
        result._views=tuple(views);result.batch_size=len(rows)
        result.origin=np.min([row.origin for row in present],axis=0)
        maximum=np.max([row.origin+np.array([row.nx,row.ny]) for row in present],axis=0)
        result.nx,result.ny=map(int,maximum-result.origin)
        if result.batch_size*result.nx*result.ny>=2**62:raise ValueError('Local spatial index overflow')
        keys=[];first=[];counts=[];xy=[];features=[];offset=0
        for i,row in enumerate(rows):
            if row.empty:continue
            x=row.keys//row.ny+int(row.origin[0]-result.origin[0])
            y=row.keys%row.ny+int(row.origin[1]-result.origin[1])
            keys.append((i*result.nx+x)*result.ny+y)
            first.append(row.first+offset);counts.append(row.counts)
            xy.append(row.xy);features.append(row.features);offset+=len(row.xy)
        result.keys=torch.cat(keys);result.first=torch.cat(first);result.counts=torch.cat(counts)
        result.xy=torch.cat(xy);result.features=torch.cat(features)
        result.origin_t=torch.as_tensor(result.origin,device=result.device)
        result._maximum_occupancy=max(row._maximum_occupancy for row in present)
        result.ambiguous_queries=result.boundary_queries=result.max_pair_entries=0
        result.cpu_tie_query_s=0.;result._rollout_query=None
        return result

    def select(self, indices):
        """GPU-resident row selection: reuse points/features and bucket offsets."""
        result = object.__new__(type(self)); result.__dict__.update(self.__dict__)
        result.batch_size = len(indices); result._views = tuple(self._views[i] for i in indices)
        result.ambiguous_queries = result.boundary_queries = result.max_pair_entries = 0
        result.cpu_tie_query_s = 0.; result._rollout_query = None
        if not self.empty:
            width = self.nx * self.ny
            masks = [(self.keys >= i * width) & (self.keys < (i + 1) * width) for i in indices]
            result.keys = torch.cat([self.keys[m] + (j-i)*width for j,(i,m) in enumerate(zip(indices,masks))])
            result.first = torch.cat([self.first[m] for m in masks])
            result.counts = torch.cat([self.counts[m] for m in masks])
            result.empty = not len(result.keys)
        return result

    def lookup(self, points):
        """Return features and support masks, with zero features outside support.

        Unique supported neighbors use the configured Torch device. Ambiguous
        nearest neighbors use the same owned CPU snapshot query to preserve
        cKDTree source-channel and equal-slope ordering exactly. That transfer
        and query cost is exposed, rather than advertised as an all-GPU path.
        """
        if (not isinstance(points, torch.Tensor) or points.ndim < 2 or points.shape[0] != self.batch_size
                or points.shape[-1] != 2 or points.dtype not in (torch.float32, torch.float64)
                or points.device != self.device or not torch.isfinite(points).all()):
            raise ValueError("Local tensor queries require finite [B,...,2] on configured device")
        cached = getattr(self, '_rollout_query', None)
        if cached is not None:
            states, features, known, ambiguous, state_version, known_version = cached
            if (points.shape == states[..., :2].shape and points.stride() == states[..., :2].stride()
                    and points.data_ptr() == states.data_ptr() and states._version == state_version
                    and known._version == known_version):
                self.ambiguous_queries += int(ambiguous.sum())
                return features, known
        shape = points.shape[:-1]
        flat = points.reshape(-1, 2).double()
        per_env = math.prod(points.shape[1:-1])
        result = torch.zeros((len(flat), 5), device=self.device, dtype=torch.float64)
        known = torch.zeros(len(flat), device=self.device, dtype=torch.bool)
        if not len(flat) or self.empty:
            return result.to(points.dtype).reshape(*shape, 5), known.reshape(shape)
        if self.device.type == "cuda":
            from .upstream_native import lookup_cuda
            result, known, ambiguous_all, boundary_all = lookup_cuda(self, flat, per_env)
        else:
            ambiguous_all = torch.zeros_like(known)
            boundary_all = torch.zeros_like(known)
            # Chunk queries as well as bucket occupants. A small caller-specified
            # budget must remain effective even with a large requested chunk.
            query_chunk = max(1, min(self.chunk, self.memory_budget // (9 * PAIR_WORK_BYTES * 2)))
            for start in range(0, len(flat), query_chunk):
                query = flat[start:start + query_chunk]
                batch = torch.arange(start, start + len(query), device=self.device) // per_env
                scaled = torch.floor(query / self.cell)
                coordinate_ok = torch.isfinite(scaled).all(-1) & (scaled.abs() <= 2**40).all(-1)
                # Never cast unbounded/Inf floats to integers and never calculate a
                # hash from out-of-grid cells: both can wrap into a valid bucket.
                scaled = torch.nan_to_num(scaled, nan=0., posinf=0., neginf=0.).clamp(-2**40, 2**40)
                cells = scaled.long()[:, None] - self.origin_t + self.offsets
                inside = coordinate_ok[:, None] & (cells >= 0).all(-1) & (cells[..., 0] < self.nx) & (cells[..., 1] < self.ny)
                safe_x = cells[..., 0].clamp(0, self.nx - 1)
                safe_y = cells[..., 1].clamp(0, self.ny - 1)
                keys = (batch[:, None] * self.nx + safe_x) * self.ny + safe_y
                slots = torch.searchsorted(self.keys, keys).clamp(max=len(self.keys) - 1)
                matched = inside & (self.keys[slots] == keys)
                counts = torch.where(matched, self.counts[slots], 0)
                maximum = self._maximum_occupancy if self._maximum_occupancy <= 16 else int(counts.max())
                nearest = torch.full((len(query),), torch.inf, device=self.device, dtype=torch.float64)
                nearest_ids = torch.zeros(len(query), device=self.device, dtype=torch.long)
                width = max(1, self.memory_budget // (len(query) * 9 * PAIR_WORK_BYTES * 2))

                def panel(offset):
                    positions = torch.arange(offset, min(offset + width, maximum), device=self.device)
                    index = self.first[slots, None] + positions
                    occupied = matched[..., None] & (positions < counts[..., None])
                    index = index.clamp(max=len(self.xy) - 1)
                    self.max_pair_entries = max(self.max_pair_entries, index.numel())
                    distance = (self.xy[index] - query[:, None, None]).norm(dim=-1)
                    distance = torch.where(occupied, distance, torch.inf).flatten(1)
                    return distance, index.flatten(1), occupied.flatten(1)

                # First find the final nearest distance. Streaming reductions of
                # almost-equal minima can otherwise retain an earlier point more
                # than 1e-8 from the final minimum (non-transitive tie chaining).
                for offset in range(0, maximum, width):
                    distance, ids, _ = panel(offset)
                    best, pick = distance.min(-1)
                    selected = ids.gather(1, pick[:, None]).squeeze(1)
                    nearest_ids = torch.where(best < nearest, selected, nearest_ids)
                    nearest = torch.minimum(nearest, best)
                count_ties = torch.zeros(len(query), device=self.device, dtype=torch.long)
                for offset in range(0, maximum, width):
                    distance, _, occupied = panel(offset)
                    # A tiny widened ambiguity detector delegates boundary cases
                    # to CPU; it never broadens the actual conservative merge.
                    count_ties += (occupied & ((distance - nearest[:, None]).abs() <= 1e-8 + 2e-12)).sum(-1)
                mask = nearest <= self.support + 1e-10
                aggregate = torch.where(mask[:, None], self.features[nearest_ids], 0.)
                boundary = torch.isfinite(nearest) & ((nearest - self.support).abs() <= 2e-10)
                ambiguous = mask & (count_ties > 1)
                result[start:start + len(query)] = aggregate
                known[start:start + len(query)] = mask
                ambiguous_all[start:start + len(query)] = ambiguous
                boundary_all[start:start + len(query)] = boundary
        # Correct the same owned queries once per lookup rather than repeatedly
        # synchronizing the host inside every small CUDA block. The CPU lookup
        # keeps its bounded query/tie allocation and original source ordering.
        correction = ambiguous_all | boundary_all
        if bool(correction.any()):
            started = time.perf_counter()
            ids = torch.nonzero(correction, as_tuple=False).flatten()
            host_query = flat[ids].detach().cpu().numpy()
            host_batch = (ids // per_env).detach().cpu().numpy()
            corrected = np.zeros((len(ids), 5)); supported = np.zeros(len(ids), dtype=bool)
            for identity in np.unique(host_batch):
                same = host_batch == identity
                features, valid = self._views[int(identity)].lookup(host_query[same])
                corrected[same] = np.where(valid[:, None], features, 0.)
                supported[same] = valid
            result[ids] = torch.as_tensor(corrected, device=self.device)
            known[ids] = torch.as_tensor(supported, device=self.device)
            self.cpu_tie_query_s += time.perf_counter() - started
            self.ambiguous_queries += int(ambiguous_all.sum())
            self.boundary_queries += int(boundary_all.sum())
        return result.to(points.dtype).reshape(*shape, 5), known.reshape(shape)


class LocalNeighborBatch:
    """Sender-preserving discrete-node pre-screen; swept checks remain on CPU."""

    def __init__(self, inputs, *, device="cpu", dtype=torch.float64):
        from .upstream_planning import UpstreamPlannerInput
        if not inputs or any(type(item) is not UpstreamPlannerInput for item in inputs):
            raise ValueError("Neighbors require a nonempty batch of typed planner inputs")
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("Neighbor pre-screen requires float32 or float64")
        self.device = _device(device)
        for item in inputs:
            item.__post_init__()
        if any(item.spec.fingerprint != inputs[0].spec.fingerprint for item in inputs):
            raise ValueError("Mixed neighbor geometry profiles")
        self.batch_size = len(inputs)
        self.steps = steps = inputs[0].spec.grid.steps
        count = max(len(item.neighbors) for item in inputs)
        positions = np.zeros((len(inputs), steps + 1, count, 2))
        radii = np.zeros((len(inputs), steps + 1, count))
        senders = np.full((len(inputs), count), -1, dtype=np.int64)
        for b, request in enumerate(inputs):
            for p, neighbor in enumerate(request.neighbors):
                senders[b, p] = neighbor.sender_id
                for t, now in enumerate(request.times):
                    positions[b, t, p], radii[b, t, p] = neighbor.at(now)
        # Preserve prediction precision even when rollout storage is float32:
        # rounding a conservative radius down must not turn an unsafe sample
        # into a pre-screen success.
        self.positions = torch.as_tensor(positions, device=self.device, dtype=torch.float64)
        self.radii = torch.as_tensor(radii, device=self.device, dtype=torch.float64)
        self.senders = torch.as_tensor(senders, device=self.device)
        self.valid = self.senders >= 0

    def select(self, indices):
        result = object.__new__(type(self)); result.__dict__.update(self.__dict__)
        result.batch_size = len(indices)
        for name in ('positions', 'radii', 'senders', 'valid', 'actuator_cap_curve_v2'):
            if hasattr(self, name):setattr(result, name, getattr(self, name)[indices])
        return result

    def risk_and_safe(self, states):
        if (not isinstance(states, torch.Tensor) or states.ndim != 4 or states.shape[0] != self.batch_size
                or states.shape[-2:] != (self.steps + 1, 3) or states.device != self.device
                or states.dtype not in (torch.float32, torch.float64) or not torch.isfinite(states).all()):
            raise ValueError("Neighbor pre-screen states require finite [B,K,41,3] on configured device")
        states = states.double()
        if not self.positions.shape[2]:
            shape = states.shape[:2]
            return states.new_zeros(shape), torch.ones(shape, device=states.device, dtype=torch.bool)
        distance = (states[:, :, :, None, :2] - self.positions[:, None]).norm(dim=-1)
        margin = self.radii[:, None] + .2
        risk = torch.where(self.valid[:, None, None], ((margin - distance) / .2).clamp_min(0).square(), 0.)
        safe = torch.where(self.valid[:, None, None], distance >= self.radii[:, None], True).all((-1, -2))
        return 8 * risk.sum((-1, -2)), safe
