"""Batched Torch MPPI. Raw Gaussian importance weights precede actuator mapping.

Only time/iteration loops are used by the rollout kernel. Python row loops are
input packing, independent RNG ownership and final protocol export/validation.
"""

from dataclasses import dataclass, asdict, replace
from copy import deepcopy
import hashlib
import time

import numpy as np
import torch

from .contracts import DT, N
from .baselines import TrajectoryAdapter
from .execution import brake_controls, shifted_sequence


@dataclass(frozen=True)
class MPPIConfig:
    adapter: str = "mppi_base_v1"
    samples: int = 512
    iterations: int = 2
    temperature: float = 1.0
    noise_v: float = 0.35
    noise_w: float = 0.7
    device: str = "cpu"
    dtype: str = "float32"
    query_chunk: int = 256
    terrain_cost: bool = False
    neighbor_cost: bool = False
    unknown_cost: bool = False
    terrain_dynamics: bool = False
    slope_scale: float = 0.75
    deadline_s: float | None = None

    def __post_init__(self):
        if self.adapter not in (
            "mppi_base_v1", "mppi_terrain_neighbor_v1", "mppi_terrain_neighbor_stable_v2"
        ):
            raise ValueError("Unknown MPPI revision")
        for name in ("samples", "iterations", "query_chunk"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"Invalid MPPI {name}")
        for name in ("temperature", "noise_v", "noise_w", "slope_scale"):
            value = getattr(self, name)
            if isinstance(value, bool) or not np.isfinite(value) or value <= 0:
                raise ValueError(f"Invalid MPPI {name}")
        for name in ("terrain_cost", "neighbor_cost", "unknown_cost", "terrain_dynamics"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError("MPPI switches must be bool")
        if self.adapter == "mppi_base_v1" and any(
            (self.terrain_cost, self.neighbor_cost, self.unknown_cost, self.terrain_dynamics)
        ):
            raise ValueError("Base MPPI must use flat dynamics without T07 soft costs")
        if self.dtype not in ("float32", "float64") or torch.device(self.device).type not in (
            "cpu",
            "cuda",
        ):
            raise ValueError("Unsupported MPPI device/dtype")
        if self.deadline_s is not None and (
            isinstance(self.deadline_s, bool)
            or not np.isfinite(self.deadline_s)
            or self.deadline_s <= 0
        ):
            raise ValueError("Invalid MPPI deadline")

    @classmethod
    def from_contract(cls, c, slope_scale=0.75):
        extended = c.planner_adapter in (
            "mppi_terrain_neighbor_v1", "mppi_terrain_neighbor_stable_v2"
        )
        if (
            isinstance(c.planner_seed, bool)
            or not isinstance(c.planner_seed, int)
            or c.planner_seed < 0
        ):
            raise ValueError("Invalid MPPI seed")
        return cls(
            adapter=c.planner_adapter,
            samples=c.mppi_samples,
            iterations=c.mppi_iterations,
            temperature=c.mppi_temperature,
            noise_v=c.mppi_noise_v,
            noise_w=c.mppi_noise_w,
            device=c.planner_device,
            query_chunk=c.mppi_query_chunk,
            terrain_cost=extended and c.mppi_terrain_cost,
            neighbor_cost=extended and c.mppi_neighbor_cost,
            unknown_cost=extended and c.mppi_unknown_cost,
            terrain_dynamics=extended and c.mppi_terrain_dynamics,
            slope_scale=slope_scale,
            deadline_s=c.mppi_deadline_s,
        )

    @property
    def stable_numerics(self):
        return self.adapter == "mppi_terrain_neighbor_stable_v2"

    @property
    def query_memory_budget(self):
        return 256 * 1024**2 if self.stable_numerics else None

    def metadata(self):
        metadata = {
            **asdict(self),
            "time_grid": "15x0.2",
            "cost_version": "mppi_cost_v1",
            "actuator_mapping": "convex_limits_v1",
            "sampling": "raw_gaussian_importance_v1",
            "dynamics_version": "analytic_terrain_midpoint_v1"
            if self.terrain_dynamics
            else "flat_midpoint_v1",
            "terrain_snapshot": "local_terrain_samples_v1",
        }
        if self.stable_numerics:
            metadata.update(
                numerics_version="fp32_fp64_accum_v2",
                accumulation_dtype="float64",
                terrain_query_memory_budget_bytes=self.query_memory_budget,
                terrain_query_budget_version="pairwise_temporaries_v1",
                terrain_query_budget_formula="max(1,min(query_chunk,budget//(B*M*(8*element_size+4))))",
            )
        return metadata


def map_control(raw, previous, cap):
    """Convex mapping to actuator bounds and per-step slew intersection."""
    vmax = torch.as_tensor(cap, device=raw.device, dtype=raw.dtype)
    while vmax.ndim < raw.ndim - 1:
        vmax = vmax.unsqueeze(-1)
    v = torch.maximum(torch.minimum(raw[..., 0], vmax), -vmax)
    desired = torch.stack((v, raw[..., 1].clamp(-2.4, 2.4)), dim=-1)
    wheel = torch.maximum(
        abs(desired[..., 0] - 0.188 * desired[..., 1]),
        abs(desired[..., 0] + 0.188 * desired[..., 1]),
    )
    desired = desired * (1.764 / wheel.clamp_min(1e-12)).clamp_max(1).unsqueeze(-1)
    delta = desired - previous
    slew = raw.new_tensor([0.3, 0.6])
    alpha = (slew / delta.abs().clamp_min(1e-12)).amin(dim=-1).clamp_max(1)
    return previous + alpha.unsqueeze(-1) * delta


def step_dynamics(pose, control, features=None, slope_scale=0.75):
    scale = torch.ones_like(control[..., 0])
    if features is not None:
        slope = (
            features[..., 1] * pose[..., 2].cos() + features[..., 2] * pose[..., 2].sin()
        ).abs()
        scale = (features[..., 4] * torch.exp(-slope * slope_scale)).clamp(0.25, 1)
    velocity = control * scale.unsqueeze(-1)
    midpoint = pose[..., 2] + 0.5 * DT * velocity[..., 1]
    next_pose = pose + DT * torch.stack(
        (velocity[..., 0] * midpoint.cos(), velocity[..., 0] * midpoint.sin(), velocity[..., 1]),
        dim=-1,
    )
    return next_pose, velocity


def rollout_batch(pose, raw, previous, cap, *, terrain=None, slope_scale=0.75):
    state = pose[:, None, :].expand(-1, raw.shape[1], -1)
    last = previous[:, None, :].expand(-1, raw.shape[1], -1)
    states, controls, velocities = [state], [], []
    for t in range(N):
        last = map_control(raw[:, :, t], last, cap)
        features = None if terrain is None else terrain.lookup(state[..., :2])[0]
        state, velocity = step_dynamics(state, last, features, slope_scale)
        states.append(state)
        controls.append(last)
        velocities.append(velocity)
    return torch.stack(states, 2), torch.stack(controls, 2), torch.stack(velocities, 2)


def base_cost(states, controls, previous, goals, yaws, yaw_active):
    distance2 = (states[:, :, 1:, :2] - goals[:, None, None, :]).square().sum(-1)
    goal = 2 * distance2.sum(-1) + 12 * distance2[:, :, -1]
    diff = torch.diff(
        torch.cat((previous[:, None, None, :].expand(-1, controls.shape[1], 1, -1), controls), 2),
        dim=2,
    )
    control = (controls.square() * controls.new_tensor([0.05, 0.03])).sum((-1, -2))
    smooth = (diff.square() * controls.new_tensor([0.5, 0.2])).sum((-1, -2))
    reverse = 0.2 * controls[..., 0].clamp_max(0).square().sum(-1)
    angle = states[:, :, -1, 2] - yaws[:, None]
    goal = goal + yaw_active[:, None] * (angle.sin().square() + (1 - angle.cos()).square())
    return {"goal": goal, "control": control, "smooth": smooth, "reverse": reverse}


def importance_update(mean, noise, cost, valid, temperature, sigma, *, accumulation_dtype=None):
    """Numerically stable raw-Gaussian update, including the all-invalid case."""
    output_dtype = mean.dtype
    if accumulation_dtype is not None:
        mean, noise, cost, sigma = (
            value.to(accumulation_dtype) for value in (mean, noise, cost, sigma)
        )
    score = cost + temperature * (mean[:, None] * noise / sigma.square()).sum((-1, -2))
    usable = valid & torch.isfinite(score) & torch.isfinite(noise).all(dim=(-1, -2))
    safe_score = torch.where(usable, score, torch.inf)
    minimum = safe_score.amin(-1, keepdim=True)
    any_valid = usable.any(-1, keepdim=True)
    minimum = torch.where(any_valid, minimum, torch.zeros_like(minimum))
    logits = -(safe_score - minimum) / temperature
    weights = torch.where(usable, torch.exp(logits), torch.zeros_like(logits))
    weights = weights / weights.sum(-1, keepdim=True).clamp_min(torch.finfo(weights.dtype).tiny)
    clean_noise = torch.where(usable[..., None, None], noise, torch.zeros_like(noise))
    updated = mean + (weights[..., None, None] * clean_noise).sum(1)
    if accumulation_dtype is not None:
        updated = updated.to(output_dtype)
    return updated, weights, score, usable


def feasible_samples_cpu(request, states, controls):
    """Vectorized legacy hard checks for base CPU fixtures; final output rechecked."""
    cap = 1.15 if request.intent.speed_cap is None else request.intent.speed_cap
    valid = np.isfinite(states).all((1, 2)) & np.isfinite(controls).all((1, 2))
    valid &= (abs(controls[..., 0]) <= cap + 1e-4).all(1) & (abs(controls[..., 1]) <= 2.4001).all(1)
    wheel = controls[..., 0, None] + controls[..., 1, None] * np.array([-0.188, 0.188])
    valid &= (abs(wheel) <= 1.7641).all((1, 2))
    delta = np.diff(
        np.concatenate(
            (np.broadcast_to(request.previous_control, (len(controls), 1, 2)), controls), 1
        ),
        axis=1,
    )
    valid &= (abs(delta) <= np.array([0.3001, 0.6001])).all((1, 2))
    clean = np.where(np.isfinite(states), states, 0)
    alphas = np.linspace(0, 1, 9)
    points = (1 - alphas[None, None, :, None]) * clean[:, :-1, None, :2] + alphas[
        None, None, :, None
    ] * clean[:, 1:, None, :2]
    valid &= request.terrain.safe(points.reshape(-1, 2)).reshape(len(states), N, 9).all((1, 2))
    for peer in request.neighbors:  # sensor payload packing boundary, not rollout computation
        p = peer.prediction
        centers = (1 - alphas[None, :, None]) * p.positions[:-1, None] + alphas[
            None, :, None
        ] * p.positions[1:, None]
        radii = (1 - alphas[None, :]) * p.radii[:-1, None] + alphas[None, :] * p.radii[1:, None]
        sweep = (
            np.linalg.norm(np.diff(clean[..., :2], axis=1) - np.diff(p.positions, axis=0), axis=-1)
            / 16
        )
        valid &= (
            np.linalg.norm(points - centers[None], axis=-1) >= radii[None] + sweep[..., None] - 1e-4
        ).all((1, 2))
    return valid


class MPPIKernel:
    """Shared compute only; persistent warm/RNG state belongs to each handle."""

    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA planner device is unavailable")
        self.dtype = getattr(torch, config.dtype)
        self.batch_calls = 0

    def tensor(self, value):
        return torch.as_tensor(value, device=self.device, dtype=self.dtype)

    def seed_controls(self, requests):
        pose = self.tensor(np.stack([r.ego_pose for r in requests]))
        goals = self.tensor(np.stack([r.intent.goal_xy for r in requests]))
        previous = self.tensor(np.stack([r.previous_control for r in requests]))
        caps = self.tensor(
            [1.15 if r.intent.speed_cap is None else r.intent.speed_cap for r in requests]
        )
        yaws = self.tensor(
            [
                r.ego_pose[2] if r.intent.terminal_yaw is None else r.intent.terminal_yaw
                for r in requests
            ]
        )
        state, last, controls = pose, previous, []
        for _ in range(N):
            delta = goals - state[:, :2]
            distance = delta.norm(dim=-1)
            angle = torch.atan2(delta[:, 1], delta[:, 0]) - state[:, 2]
            angle = torch.atan2(angle.sin(), angle.cos())
            direction = torch.where(angle.abs() <= np.pi / 2, 1.0, -1.0)
            reverse_angle = angle + np.pi
            angle = torch.where(
                direction < 0, torch.atan2(reverse_angle.sin(), reverse_angle.cos()), angle
            )
            turn = torch.atan2((yaws - state[:, 2]).sin(), (yaws - state[:, 2]).cos())
            angular = torch.where(distance < 0.1, turn, angle) * 2
            linear = direction * torch.minimum(caps, distance * 2) * angle.cos().clamp_min(0)
            desired = torch.stack((linear, angular.clamp(-2.4, 2.4)), dim=-1)
            last = map_control(desired, last, caps)
            controls.append(last)
            state, _ = step_dynamics(state, last)
        return torch.stack(controls, 1)

    def evaluate(self, requests, raw):
        from .tensor_views import TerrainBatch, NeighborBatch, terrain_cost, hard_validity

        begin = time.monotonic()
        pose = self.tensor(np.stack([r.ego_pose for r in requests]))
        previous = self.tensor(np.stack([r.previous_control for r in requests]))
        goals = self.tensor(np.stack([r.intent.goal_xy for r in requests]))
        cap = self.tensor(
            [1.15 if r.intent.speed_cap is None else r.intent.speed_cap for r in requests]
        )
        yaw = self.tensor(
            [
                r.ego_pose[2] if r.intent.terminal_yaw is None else r.intent.terminal_yaw
                for r in requests
            ]
        )
        active = self.tensor([r.intent.terminal_yaw is not None for r in requests])
        extended = self.config.adapter != "mppi_base_v1"
        terrain = TerrainBatch(
            [r.terrain for r in requests], self.tensor, self.config.query_chunk,
            memory_budget_bytes=self.config.query_memory_budget,
        )
        neighbors = NeighborBatch(requests, self.tensor)
        packed = time.monotonic()
        states, controls, velocities = rollout_batch(
            pose,
            raw,
            previous,
            cap,
            terrain=terrain if self.config.terrain_dynamics else None,
            slope_scale=self.config.slope_scale,
        )
        predicted = time.monotonic()
        parts = base_cost(states, controls, previous, goals, yaw, active)
        if extended:
            risk, unknown = terrain_cost(states, terrain)
            parts.update(
                terrain=risk if self.config.terrain_cost else torch.zeros_like(risk),
                unknown=unknown if self.config.unknown_cost else torch.zeros_like(risk),
                neighbor=neighbors.cost(
                    states, accumulation_dtype=torch.float64 if self.config.stable_numerics else None
                )
                if self.config.neighbor_cost
                else torch.zeros_like(risk),
            )
        cost = sum(parts.values())
        finite = (
            torch.isfinite(states).all((-1, -2))
            & torch.isfinite(controls).all((-1, -2))
            & torch.isfinite(cost)
        )
        scored = time.monotonic()
        valid = finite & hard_validity(states, controls, previous, cap, terrain, neighbors)
        self.last_timings = {
            "packing_s": packed - begin,
            "prediction_s": predicted - packed,
            "cost_s": scored - predicted,
            "sample_review_s": time.monotonic() - scored,
        }
        return states, controls, velocities, cost, valid, parts

    def score_selected(self, requests, trajectories):
        from .tensor_views import TerrainBatch, NeighborBatch, terrain_cost

        states = self.tensor(np.stack([t.states for t in trajectories]))[:, None]
        controls = self.tensor(np.stack([t.controls for t in trajectories]))[:, None]
        previous = self.tensor(np.stack([r.previous_control for r in requests]))
        goals = self.tensor(np.stack([r.intent.goal_xy for r in requests]))
        yaw = self.tensor(
            [
                r.ego_pose[2] if r.intent.terminal_yaw is None else r.intent.terminal_yaw
                for r in requests
            ]
        )
        active = self.tensor([r.intent.terminal_yaw is not None for r in requests])
        parts = base_cost(states, controls, previous, goals, yaw, active)
        if self.config.adapter != "mppi_base_v1":
            terrain = TerrainBatch(
                [r.terrain for r in requests], self.tensor, self.config.query_chunk,
                memory_budget_bytes=self.config.query_memory_budget,
            )
            neighbors = NeighborBatch(requests, self.tensor)
            risk, unknown = terrain_cost(states, terrain)
            parts.update(
                terrain=risk if self.config.terrain_cost else torch.zeros_like(risk),
                unknown=unknown if self.config.unknown_cost else torch.zeros_like(risk),
                neighbor=neighbors.cost(
                    states, accumulation_dtype=torch.float64 if self.config.stable_numerics else None
                )
                if self.config.neighbor_cost
                else torch.zeros_like(risk),
            )
        return parts

    def plan_batch(self, planners, inputs, *, noise=None, clock=time.monotonic):
        start = clock()
        if len(planners) != len(inputs) or len({id(p) for p in planners}) != len(planners):
            raise ValueError("One independent MPPI handle required per input")
        if len({(r.env_id, r.agent_id) for r in inputs}) != len(inputs):
            raise ValueError("Duplicate MPPI batch identity")
        for r in inputs:
            r.validate()
        inputs = [replace(r, terrain=r.terrain.conservative()) for r in inputs]
        for p, r in zip(planners, inputs):
            if p.batch_kernel.config != self.config:
                raise ValueError("Incompatible MPPI batch configuration")
            if p.last_now is not None and r.now < p.last_now:
                raise ValueError("Planning time moved backwards")
        for p, r in zip(planners, inputs):
            p.bind(r)
        self.batch_calls += 1
        outputs = [None] * len(inputs)
        active = []
        for i, (p, r) in enumerate(zip(planners, inputs)):
            if r.now > r.intent.expires_at or r.intent.hold:
                candidate = p.predict(r, brake_controls(r.previous_control))
                outputs[i] = p.finish(
                    r,
                    [candidate],
                    reason="expired_intent" if r.now > r.intent.expires_at else "",
                    force_brake=r.now > r.intent.expires_at,
                )
                p.warm = None
                p.last_now = r.now
                p.last_samples = None
                p.last_diagnostics = {"samples": 0, "parking_branch": True}
            else:
                active.append(i)
        if not active:
            return outputs
        req = [inputs[i] for i in active]
        handles = [planners[i] for i in active]
        mean = self.seed_controls(req)
        for i, (p, r) in enumerate(zip(handles, req)):
            if p.warm is not None and p.last_now is not None:
                steps = int(round((r.now - p.last_now) / DT))
                if steps < 0:
                    raise ValueError("Planning time moved backwards")
                if steps < N:
                    mean[i] = self.tensor(shifted_sequence(p.warm, steps))
        sigma = self.tensor([self.config.noise_v, self.config.noise_w])
        if noise is not None:
            expected = (self.config.iterations, len(active), self.config.samples, N, 2)
            if tuple(noise.shape) != expected:
                raise ValueError("Injected noise shape mismatch")
            noise = self.tensor(noise)
        timed_out = False
        timings = {}
        for iteration in range(self.config.iterations):
            eps = (
                torch.stack([p.sample_noise(self.config.samples) for p in handles])
                if noise is None
                else noise[iteration]
            )
            raw = mean[:, None] + eps
            states, controls, velocities, cost, valid, parts = self.evaluate(req, raw)
            for key, value in self.last_timings.items():
                timings[key] = timings.get(key, 0) + value
            update_start = time.monotonic()
            updated, weights, score, usable = importance_update(
                mean,
                eps,
                cost,
                torch.isfinite(cost)
                & torch.isfinite(states).all((-1, -2))
                & torch.isfinite(controls).all((-1, -2)),
                self.config.temperature,
                sigma,
                accumulation_dtype=torch.float64 if self.config.stable_numerics else None,
            )
            mean = updated
            timings["update_s"] = timings.get("update_s", 0) + time.monotonic() - update_start
            if self.config.deadline_s is not None and clock() - start > self.config.deadline_s:
                timed_out = True
                break
        final_states, final_controls, final_velocities, _, _, _ = self.evaluate(req, mean[:, None])
        for key, value in self.last_timings.items():
            timings[key] = timings.get(key, 0) + value
        for row, (p, r) in enumerate(zip(handles, req)):

            def trajectory(s, c, v):
                # Export in host precision from the same commands, preserving exact initial pose.
                return p.predict(r, c.detach().cpu().double().numpy())

            fresh = trajectory(
                final_states[row, 0], final_controls[row, 0], final_velocities[row, 0]
            )
            candidates = [fresh]
            feasible = torch.nonzero(valid[row], as_tuple=False).flatten()
            if len(feasible):
                best = int(feasible[cost[row, feasible].argmin()])
                candidates.append(
                    trajectory(states[row, best], controls[row, best], velocities[row, best])
                )
            reason = (
                "planner_timeout"
                if timed_out
                else "all_samples_invalid"
                if not bool(usable[row].any())
                else "no_feasible_samples"
                if not bool(valid[row].any())
                else ""
            )
            if timed_out or not bool(usable[row].any()):
                candidates = []
            review_start = time.monotonic()
            result = p.finish(r, candidates, reason=reason)
            final_review = time.monotonic() - review_start
            p.warm = mean[row].detach().cpu().double().numpy() if result.valid else None
            p.last_now = r.now
            p.last_diagnostics = {
                "samples": self.config.samples,
                "iterations": iteration + 1,
                "valid_fraction": float(valid[row].float().mean()),
                "numeric_valid_fraction": float(torch.isfinite(cost[row]).float().mean()),
                "effective_samples": float(
                    torch.where(
                        weights[row].sum() > 0, 1 / weights[row].square().sum().clamp_min(1e-12), 0
                    )
                ),
                "components": {},
                "raw_noise_sha256": hashlib.sha256(
                    eps[row].detach().cpu().numpy().tobytes()
                ).hexdigest(),
                "batch_size": len(active),
                "timed_out": timed_out,
                "timings": {
                    **timings,
                    "final_review_s": final_review,
                    "end_to_end_s": clock() - start,
                },
                "fallback_source": result.feedback.status,
                "candidate_rejection": result.feedback.candidate_rejection,
                "importance_correction_mean": float((score[row] - cost[row])[usable[row]].mean())
                if usable[row].any()
                else None,
            }
            p.last_samples = {
                "raw": raw[row].detach().cpu(),
                "cost": cost[row].detach().cpu(),
                "score": score[row].detach().cpu(),
                "weights": weights[row].detach().cpu(),
                "noise": eps[row].detach().cpu(),
                "valid": valid[row].detach().cpu(),
                "states": states[row].detach().cpu(),
                "controls": controls[row].detach().cpu(),
                "velocities": velocities[row].detach().cpu(),
                **{f"component_{name}": value[row].detach().cpu() for name, value in parts.items()},
            }
            outputs[active[row]] = result
        # Score the actual host-validated selected states, without another
        # actuator mapping or a per-instance Torch rollout.
        score_begin = time.monotonic()
        selected_parts = self.score_selected(req, [outputs[i].trajectory for i in active])
        selected_cost = sum(selected_parts.values())
        selected_seconds = time.monotonic() - score_begin
        elapsed = clock() - start
        for row, p in enumerate(handles):  # protocol/diagnostic export only
            result = outputs[active[row]]
            if result.valid:
                outputs[active[row]] = type(result)(
                    result.trajectory,
                    result.feedback,
                    result.valid,
                    result.fallback_reason,
                    float(selected_cost[row, 0]),
                )
            p.last_diagnostics["components"] = {
                name: float(value[row, 0]) if torch.isfinite(value[row, 0]) else None
                for name, value in selected_parts.items()
            }
            p.last_diagnostics["timings"].update(
                selected_cost_s=selected_seconds, end_to_end_s=elapsed
            )
            result.feedback.solve_seconds = elapsed
        return outputs


class MPPIAdapter(TrajectoryAdapter):
    def __init__(self, kernel, seed=23):
        super().__init__(kernel.config.slope_scale, flat=not kernel.config.terrain_dynamics)
        self.batch_kernel = kernel
        self.adapter_id = self.revision = kernel.config.adapter
        self.seed = seed
        self.episode = 0
        self.warm = self.last_now = None
        self.rng = torch.Generator(device=kernel.device)
        self.last_samples = None
        self.intent = None

    def configuration(self):
        return {
            **super().configuration(),
            "mppi": self.batch_kernel.config.metadata(),
            "seed": self.seed,
        }

    def bind(self, request):
        first = self.identity is None
        super().bind(request)
        self.intent = deepcopy(request.intent)
        if first:
            self._seed_rng()

    def _seed_rng(self):
        sender_key = None if self.identity is None else tuple(int(value) for value in self.identity)
        identity = (self.seed, sender_key, self.episode)
        seed = int.from_bytes(hashlib.sha256(repr(identity).encode()).digest()[:8], "little") % (
            2**63 - 1
        )
        self.rng.manual_seed(seed)

    def reset(self):
        super().reset()
        self.warm = self.last_now = self.last_samples = None
        self.intent = None
        self.episode += 1
        self._seed_rng()

    def sample_noise(self, count):
        c = self.batch_kernel.config
        sigma = self.batch_kernel.tensor([c.noise_v, c.noise_w])
        return (
            torch.randn(
                (count, N, 2),
                generator=self.rng,
                device=self.batch_kernel.device,
                dtype=self.batch_kernel.dtype,
            )
            * sigma
        )

    def state_dict(self):
        return {
            **super().state_dict(),
            "state_version": 2 if self.batch_kernel.config.stable_numerics else 1,
            "warm": deepcopy(self.warm),
            "last_now": self.last_now,
            "rng_state": self.rng.get_state().cpu().clone(),
            "rng_device": self.batch_kernel.device.type,
            "episode": self.episode,
            "intent": deepcopy(self.intent),
        }

    def validate_state_dict(self, state):
        super().validate_state_dict(state)
        from .protocol import LocalIntent

        if "intent" not in state or (
            state["intent"] is not None and not isinstance(state["intent"], LocalIntent)
        ):
            raise ValueError("Missing or invalid saved intent")
        if state["intent"] is not None:
            state["intent"].validate(state.get("last_now") or state["intent"].timestamp)
        if state.get("warm") is not None and state["intent"] is None:
            raise ValueError("Warm state requires saved intent")
        if state.get("rng_device") != self.batch_kernel.device.type:
            raise ValueError("Cross-device planner RNG restoration is not supported")
        if (
            not isinstance(state.get("episode"), int)
            or isinstance(state["episode"], bool)
            or state["episode"] < 0
        ):
            raise ValueError("Invalid episode RNG state")
        if state.get("warm") is not None:
            from .protocol import finite_array

            finite_array(state["warm"], (N, 2), "warm")
            if state.get("last_now") is None or not np.isfinite(state["last_now"]):
                raise ValueError("Missing warm planning time")
        if state.get("last_now") is not None and (
            not np.isfinite(state["last_now"]) or state["last_now"] < 0
        ):
            raise ValueError("Invalid planning time")
        try:
            test_rng = torch.Generator(device=self.batch_kernel.device)
            test_rng.set_state(state["rng_state"])
        except (KeyError, RuntimeError, TypeError) as error:
            raise ValueError("Invalid planner RNG state") from error

    def load_state_dict(self, state):
        self.validate_state_dict(state)
        super().load_state_dict(state)
        self.warm, self.last_now, self.episode = (
            deepcopy(state["warm"]),
            state["last_now"],
            state["episode"],
        )
        self.rng.set_state(state["rng_state"])
        self.intent = deepcopy(state["intent"])

    def plan(self, request):
        return self.batch_kernel.plan_batch([self], [request])[0]
