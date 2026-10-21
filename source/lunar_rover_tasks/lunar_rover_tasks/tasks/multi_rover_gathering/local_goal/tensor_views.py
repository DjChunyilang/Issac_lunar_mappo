"""Sensor snapshot packing and bounded-memory Torch queries, without owner references."""

import numpy as np
import torch


class TerrainBatch:
    def __init__(self, views, tensor, chunk=256, *, memory_budget_bytes=None):
        snapshots = [v.snapshot() for v in views]
        count = max(1, max(len(s.xy) for s in snapshots))
        xy = np.zeros((len(views), count, 2))
        features = np.zeros((len(views), count, 5))
        valid = np.zeros((len(views), count), bool)
        times = np.zeros((len(views), count))
        for b, s in enumerate(snapshots):  # packing only
            n = len(s.xy)
            xy[b, :n], features[b, :n], times[b, :n], valid[b, :n] = (
                s.xy,
                s.features,
                s.observed_at,
                True,
            )
        self.xy, self.features, self.observed_at = tensor(xy), tensor(features), tensor(times)
        self.valid = torch.as_tensor(valid, device=self.xy.device)
        self.support = tensor([s.support_radius for s in snapshots])
        self.chunk = chunk
        if memory_budget_bytes is not None:
            if (
                isinstance(memory_budget_bytes, bool)
                or not isinstance(memory_budget_bytes, int)
                or memory_budget_bytes < 1
            ):
                raise ValueError("Invalid terrain query memory budget")
            # Account for coordinate deltas/squares, distance/tie reductions
            # and masks. This bounds temporary pairwise tensors, not all live
            # planner buffers or the CUDA allocator's reserved memory.
            bytes_per_query = len(views) * count * (8 * self.xy.element_size() + 4)
            self.chunk = min(chunk, max(1, memory_budget_bytes // bytes_per_query))
        self.max_query_block = 0

    def lookup(self, points):
        shape = points.shape[:-1]
        flat = points.reshape(points.shape[0], -1, 2)
        chunks, known_chunks, direct_chunks = [], [], []
        slope = self.features[..., 1:3].norm(dim=-1)
        for start in range(0, flat.shape[1], self.chunk):
            query = flat[:, start : start + self.chunk]
            self.max_query_block = max(self.max_query_block, query.shape[1])
            distances = (query[:, :, None] - self.xy[:, None]).square().sum(-1).sqrt()
            distances = distances.masked_fill(~self.valid[:, None], torch.inf)
            nearest, index = distances.min(-1)
            ties = ((distances - nearest[..., None]).abs() <= 1e-7) & self.valid[:, None]
            # Select the greatest slope norm and smallest traversability among ties.
            slope_index = torch.where(ties, slope[:, None], -torch.inf).argmax(-1)
            feature = self.features.gather(1, index[..., None].expand(-1, -1, 5)).clone()
            worst = self.features.gather(1, slope_index[..., None].expand(-1, -1, 5))
            feature[..., 1:3] = worst[..., 1:3]
            feature[..., 4] = torch.where(ties, self.features[:, None, :, 4], torch.inf).amin(-1)
            nonempty = self.valid.any(-1)[:, None]
            feature = torch.where(nonempty[..., None], feature, torch.zeros_like(feature))
            known = nonempty & torch.isfinite(nearest) & (nearest <= self.support[:, None])
            chunks.append(feature)
            known_chunks.append(known)
            direct_chunks.append(known & (nearest < 1e-5))
        if not chunks:
            return (
                points.new_empty((*shape, 5)),
                torch.empty(shape, device=points.device, dtype=torch.bool),
                torch.empty(shape, device=points.device, dtype=torch.bool),
            )
        return (
            torch.cat(chunks, 1).reshape(*shape, 5),
            torch.cat(known_chunks, 1).reshape(shape),
            torch.cat(direct_chunks, 1).reshape(shape),
        )


class NeighborBatch:
    def __init__(self, requests, tensor):
        count = max((len(r.neighbors) for r in requests), default=0)
        positions = np.zeros((len(requests), count, 16, 2))
        radii = np.zeros((len(requests), count, 16))
        valid = np.zeros((len(requests), count), bool)
        senders = np.full((len(requests), count), -1)
        for b, r in enumerate(requests):
            for slot, neighbor in enumerate(r.neighbors):  # message packing only
                positions[b, slot], radii[b, slot] = (
                    neighbor.prediction.positions,
                    neighbor.prediction.radii,
                )
                valid[b, slot], senders[b, slot] = True, neighbor.sender_id
        self.positions, self.radii = tensor(positions), tensor(radii)
        self.valid = torch.as_tensor(valid, device=self.positions.device)
        self.senders = torch.as_tensor(senders, device=self.positions.device)

    def cost(self, states, *, accumulation_dtype=None):
        positions, radii = self.positions, self.radii
        if accumulation_dtype is not None:
            # Convert before subtraction, squaring and reduction. Casting only
            # the final sum cannot recover distance/risk rounding losses.
            states = states.to(accumulation_dtype)
            positions, radii = positions.to(accumulation_dtype), radii.to(accumulation_dtype)
        delta = states[:, :, 1:, None, :2] - positions[:, None, :, 1:].transpose(2, 3)
        # positions transpose above is [B,1,N,P,2].
        distance = delta.norm(dim=-1)
        radius = radii[:, None, :, 1:].transpose(2, 3)
        risk = ((radius + 0.2 - distance) / 0.2).clamp_min(0).square()
        return 8 * torch.where(self.valid[:, None, None], risk, 0).sum((-1, -2))

    def safe(self, states, points):
        alpha = torch.linspace(0, 1, 9, device=states.device, dtype=states.dtype)
        p = self.positions.transpose(1, 2)  # B,T,P,2
        r = self.radii.transpose(1, 2)
        centers = (1 - alpha[None, None, :, None, None]) * p[:, :-1, None] + alpha[
            None, None, :, None, None
        ] * p[:, 1:, None]
        radii = (1 - alpha[None, None, :, None]) * r[:, :-1, None] + alpha[None, None, :, None] * r[
            :, 1:, None
        ]
        distance = (points[..., None, :] - centers[:, None]).norm(dim=-1)
        sweep = (
            torch.diff(states[..., :2], dim=2)[..., None, :] - torch.diff(p, dim=1)[:, None]
        ).norm(dim=-1) / 16
        collision = distance < radii[:, None] + sweep[:, :, :, None] - 1e-4
        return (~(collision & self.valid[:, None, None, None])).all((-1, -2, -3))


def terrain_cost(states, terrain):
    features, known, _ = terrain.lookup(states[:, :, 1:, :2])
    risk = (features[..., 1:3].norm(dim=-1) / 0.8).square() + (
        (1 - features[..., 4]) / 0.75
    ).square()
    return 2 * torch.where(known, risk, 0).sum(-1), 20 * (~known).sum(-1)


def hard_validity(states, controls, previous, caps, terrain, neighbors):
    alpha = torch.linspace(0, 1, 9, device=states.device, dtype=states.dtype)
    points = (1 - alpha[None, None, None, :, None]) * states[:, :, :-1, None, :2] + alpha[
        None, None, None, :, None
    ] * states[:, :, 1:, None, :2]
    features, known, _ = terrain.lookup(points)
    safe = known & (features[..., 4] >= 0.25) & (features[..., 1:3].norm(dim=-1) <= 0.8)
    valid = safe.all((-1, -2)) & neighbors.safe(states, points)
    valid &= (controls[..., 0].abs() <= caps[:, None, None] + 1e-4).all(-1)
    valid &= (controls[..., 1].abs() <= 2.4001).all(-1)
    wheel = controls[..., 0, None] + controls[..., 1, None] * controls.new_tensor([-0.188, 0.188])
    valid &= (wheel.abs() <= 1.7641).all((-1, -2))
    delta = torch.diff(
        torch.cat((previous[:, None, None].expand(-1, controls.shape[1], 1, -1), controls), 2),
        dim=2,
    )
    valid &= (delta.abs() <= controls.new_tensor([0.3001, 0.6001])).all((-1, -2))
    return valid
