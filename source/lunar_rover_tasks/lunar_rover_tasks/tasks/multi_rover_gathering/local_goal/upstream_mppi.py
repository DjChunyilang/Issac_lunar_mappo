"""Independent 40-step, asymmetric-vehicle MPPI with local-only tensor inputs."""

from copy import deepcopy
from dataclasses import asdict, dataclass, replace
import hashlib
import math
import time

import numpy as np
import torch

from .mppi import importance_update
from .upstream_execution import control_error, limit_control_torch, midpoint_torch, terrain_gain_torch
from .upstream_planning import Candidate, PlanningAdapter
from .upstream_tensor_views import LocalNeighborBatch, LocalTerrainBatch


@dataclass(frozen=True)
class UpstreamMPPISettings:
    samples: int = 512
    iterations: int = 2
    device: str = "cpu"
    dtype: str = "float32"
    noise_v: float = .25
    noise_w: float = .10
    temperature: float = 1.
    query_chunk: int = 256
    query_memory_bytes: int = 64 * 1024**2
    rollout_chunk: int = 4096
    candidate_limit: int = 16
    deadline_s: float | None = None

    def __post_init__(self):
        for name in ("samples", "iterations", "query_chunk", "query_memory_bytes", "candidate_limit", "rollout_chunk"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError("Invalid MPPI integer setting: " + name)
        if self.candidate_limit > 16 or self.query_memory_bytes < 4096:
            raise ValueError("Unsupported MPPI review/query budget")
        if self.device not in ("cpu", "cuda") or self.dtype not in ("float32", "float64"):
            raise ValueError("Unsupported MPPI device/dtype")
        for value in (self.noise_v, self.noise_w, self.temperature):
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError("Invalid MPPI sampling setting")
        if self.deadline_s is not None and (not math.isfinite(self.deadline_s) or self.deadline_s <= 0):
            raise ValueError("Invalid MPPI deadline")


class UpstreamMPPIKernel:
    """Shared tensor compute; only time, optimizer and memory-block loops."""
    version = "upstream_mppi_kernel_v1"
    cost_version = "upstream_mppi_normalized_cost_v1"

    def __init__(self, cfg, settings=None):
        self.cfg, self.spec = cfg, cfg.spec
        self.settings = settings or UpstreamMPPISettings(
            samples=cfg.mppi_samples, iterations=cfg.mppi_iterations, device=cfg.planner_device)
        self.device = torch.device(self.settings.device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA MPPI is unavailable; no CPU fallback")
        self.dtype = getattr(torch, self.settings.dtype)
        self.batch_calls = 0
        self.last_evaluation_times = {}

    def synchronize(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    @property
    def metadata(self):
        return {"kernel": self.version, "cost": self.cost_version,
                "settings": asdict(self.settings), "profile_fingerprint": self.spec.fingerprint,
                "time_grid": asdict(self.spec.grid), "accumulation_dtype": "float64",
                "zero_roundoff": 1e-7 if self.dtype == torch.float32 else 1e-10,
                "terrain_query": LocalTerrainBatch.version, "sampling": "raw_gaussian_importance_v1"}

    def tensor(self, value):
        return torch.as_tensor(value, dtype=self.dtype, device=self.device)

    def pack(self, requests):
        views = [r.terrain for r in requests]
        if self.cfg.obstacle_input is not None:
            from .upstream_grid import GridTerrainBatch
            terrain=GridTerrainBatch(views,self.device,getattr(self,'_grid_tile_cache',None),max_owners=self.cfg.simulation.num_envs*4)
            self._grid_tile_cache=terrain.cache
            self.last_grid_upload_bytes=terrain.upload_bytes
            return terrain,LocalNeighborBatch(requests,device=self.device)
        cached = getattr(self, '_packed_terrain', None)
        if (cached is not None and len(cached._views) == len(views)
                and all(a._xy is b._xy and a._features is b._features for a,b in zip(cached._views, views))):
            terrain = cached
            terrain._views = tuple(views)
            terrain.ambiguous_queries = terrain.boundary_queries = terrain.max_pair_entries = 0
            terrain.cpu_tie_query_s = 0.
            terrain._rollout_query = None
        else:
            cache=getattr(self,'_terrain_rows',{});rows=[]
            for request,view in zip(requests,views):
                key=(request.env_id,request.agent_id);row=cache.get(key)
                if row is None or row._views[0]._xy is not view._xy or row._views[0]._features is not view._features:
                    row=LocalTerrainBatch([view],device=self.device,chunk=self.settings.query_chunk,
                                          memory_budget=self.settings.query_memory_bytes)
                    cache[key]=row
                rows.append(row)
            self._terrain_rows=cache
            terrain=LocalTerrainBatch.from_rows(views,rows)
            self._packed_terrain = terrain
        neighbors = LocalNeighborBatch(requests, device=self.device)
        return terrain, neighbors

    def seed_controls(self, requests):
        pose = self.tensor(np.stack([r.ego_pose for r in requests]))
        goals = self.tensor(np.stack([r.intent.goal_xy for r in requests]))
        delta = goals - pose[:, :2]
        angle = torch.atan2(delta[:, 1], delta[:, 0]) - pose[:, 2]
        angle = torch.atan2(angle.sin(), angle.cos())
        direction = torch.where(angle.abs() <= math.pi / 2, 1., -1.)
        backward = angle + math.pi
        turn = torch.where(direction < 0, torch.atan2(backward.sin(), backward.cos()), angle)
        caps = self.tensor([self.spec.vehicle.forward_limit if r.intent.speed_cap is None else r.intent.speed_cap for r in requests])
        caps = torch.where(direction < 0, caps.clamp_max(self.spec.vehicle.reverse_limit), caps)
        speed = direction * caps * turn.cos().clamp_min(0)
        speed = torch.where(delta.norm(dim=-1) < 1e-8, 0., speed)
        command = torch.stack((speed, turn.clamp(-self.spec.vehicle.yaw_limit, self.spec.vehicle.yaw_limit)), -1)
        return command[:, None].expand(-1, self.spec.grid.steps, -1).clone()

    def rollout(self, requests, raw, terrain):
        if raw.ndim != 4 or raw.shape[0] != len(requests) or raw.shape[-2:] != (40, 2):
            raise ValueError("MPPI controls require [B,K,40,2]")
        pose = self.tensor(np.stack([r.ego_pose for r in requests]))
        previous = self.tensor(np.stack([r.previous_control for r in requests]))
        state = pose[:, None].expand(-1, raw.shape[1], -1)
        last = previous[:, None].expand(-1, raw.shape[1], -1)
        caps = self.tensor([self.spec.vehicle.forward_limit if r.intent.speed_cap is None else r.intent.speed_cap for r in requests])[:, None]
        reverse = caps.clamp_max(self.spec.vehicle.reverse_limit)
        states, controls, velocities, known_nodes = [state], [], [], []
        numeric = torch.isfinite(raw).all((-1, -2))
        clean = torch.where(torch.isfinite(raw), raw, 0.)
        v = self.spec.vehicle
        stopping = torch.zeros(raw.shape[:2], device=self.device, dtype=torch.bool)
        for t in range(40):
            desired = clean[:, :, t].clone()
            desired[..., 0] = torch.maximum(-reverse, torch.minimum(caps, desired[..., 0]))
            # Conservative joint-stop bound. Once braking begins within this
            # sampled horizon, it cannot return to MOVE later in the sequence.
            step_roundoff = 1e-5 if self.dtype == torch.float32 else 1e-10
            stop_bound = (torch.ceil(last[..., 0].abs() / (v.linear_deceleration * self.spec.grid.dt) - step_roundoff)
                          + torch.ceil(last[..., 1].abs() / (v.angular_deceleration * self.spec.grid.dt) - step_roundoff) + 1)
            stopping |= stop_bound >= 40 - t
            desired = torch.where(stopping[..., None], 0., desired)
            flat_desired, flat_last = desired.reshape(-1, 2), last.reshape(-1, 2)
            # Polygon candidate tensors dominate memory. Blocking this flattened
            # axis retains every environment and every sample at the declared K.
            chunks = [limit_control_torch(flat_desired[start:start + self.settings.rollout_chunk],
                                         flat_last[start:start + self.settings.rollout_chunk], self.spec)
                      for start in range(0, len(flat_last), self.settings.rollout_chunk)]
            last = torch.cat(chunks).reshape_as(last)
            # Polygon intersections at zero can leave machine roundoff. A
            # nonzero sign would incorrectly delay a later direction switch.
            last = torch.where(last.abs() <= (1e-7 if self.dtype == torch.float32 else 1e-10), 0., last)
            features, known = terrain.lookup(state[..., :2])
            gain = terrain_gain_torch(state, features, self.spec)
            state, velocity = midpoint_torch(state, last, self.spec, gain=gain)
            states.append(state); controls.append(last); velocities.append(velocity); known_nodes.append(known)
        features, known = terrain.lookup(state[..., :2])
        known_nodes.append(known)
        return (torch.stack(states, 2), torch.stack(controls, 2), torch.stack(velocities, 2),
                numeric, torch.stack(known_nodes, 2))

    def _cost_constants(self, requests):
        """Keep only the current request group; never retain old terrain history."""
        previous=getattr(self,'_cost_requests',())
        if previous and len(previous)==len(requests) and all(a is b for a,b in zip(previous,requests)):
            return self._cost_values
        def exact(value):
            return torch.as_tensor(value, device=self.device, dtype=torch.float64)
        values=dict(poses=exact(np.stack([r.ego_pose for r in requests])),
            start=exact(np.stack([r.ego_pose[:2] for r in requests])),
            goals=exact(np.stack([r.intent.goal_xy for r in requests])),
            previous=exact(np.stack([r.previous_control for r in requests])),
            stop=torch.as_tensor([r.intent.stop_at_goal for r in requests],device=self.device),
            yaw=exact([r.ego_pose[2] if r.intent.terminal_yaw is None else r.intent.terminal_yaw for r in requests]),
            yaw_active=torch.as_tensor([r.intent.terminal_yaw is not None for r in requests],device=self.device),
            workspace=exact([r.workspace if r.workspace is not None else (-1e12,-1e12,1e12,1e12) for r in requests]),
            limits=exact([self.spec.vehicle.forward_limit,self.spec.vehicle.yaw_limit]),
            slew=exact([self.spec.vehicle.linear_acceleration,self.spec.vehicle.angular_acceleration])*self.spec.grid.dt,
            control_weight=exact([.05,.03]),smooth_weight=exact([.5,.2]))
        self._cost_requests=tuple(requests);self._cost_values=values
        return values

    def cost(self, requests, states, controls, terrain, neighbors):
        s, u = states.double(), controls.double()
        constants=self._cost_constants(requests)
        start,goals=constants['start'],constants['goals']
        vector = goals - start
        length = vector.norm(dim=-1)
        direction = vector / length.clamp_min(1e-12)[:, None]
        offset = s[:, :, 1:, :2] - start[:, None, None]
        progress = (offset * direction[:, None, None]).sum(-1)
        lateral = offset - progress[..., None] * direction[:, None, None]
        rolling_error = (length[:, None, None] - progress).clamp_min(0).square() + lateral.square().sum(-1)
        stop = constants['stop'] | (length < 1e-8)
        euclidean = (s[:, :, 1:, :2] - goals[:, None, None]).square().sum(-1)
        error = torch.where(stop[:, None, None], euclidean, rolling_error) / self.spec.rho_max**2
        goal = 2 * error.sum(-1) + 12 * error[..., -1]
        yaw,yaw_active=constants['yaw'],constants['yaw_active']
        goal += yaw_active[:, None] * 2 * (1 - (s[:, :, -1, 2] - yaw[:, None]).cos())
        previous = constants['previous']
        change = torch.diff(torch.cat((previous[:, None, None].expand(-1, u.shape[1], 1, -1), u), 2), dim=2)
        limits,slew=constants['limits'],constants['slew']
        parts = {"goal": goal,
                 "control": ((u / limits).square() * constants['control_weight']).sum((-1, -2)),
                 "smooth": ((change / slew).square() * constants['smooth_weight']).sum((-1, -2)),
                 "reverse": .2 * (u[..., 0].clamp_max(0) / self.spec.vehicle.reverse_limit).square().sum(-1)}
        if self.cfg.obstacle_input is not None:
            features,known,ground=terrain.lookup_ground(s[...,:2])
        else:
            features,known=terrain.lookup(s[...,:2])
        f = features.double()
        if self.cfg.obstacle_input is not None:
            risk=ground.double()
        else:
            risk=(f[...,1:3].norm(dim=-1)/self.spec.terrain.driving_slope).square()+((1-f[...,4])/.75).square()
        if self.cfg.obstacle_input is None and not self.cfg.predictive_footprint_checks:
            # Reuse the center query; no contact grids or swept-pose queries.
            rough = (f[..., 3] / self.spec.terrain.roughness_rms).clamp(0, 10)
            step = torch.diff(f[..., 0], dim=-1, prepend=f[..., :1, 0]).abs()
            step = (step / self.spec.terrain.step_height).clamp_max(10)
            risk = risk + rough.square() + step.square()
        parts["terrain"] = 2 * torch.where(known, risk, 0.).sum(-1)
        parts["unknown"] = 20 * (~known).double().sum(-1)
        parts["neighbor"], neighbor_safe = neighbors.risk_and_safe(s)
        terrain_safe = known & (f[..., 4] >= self.spec.terrain.minimum_traversability) & (f[..., 1:3].norm(dim=-1) <= self.spec.terrain.driving_slope + 1e-8)
        safe = terrain_safe.all(-1) & neighbor_safe & (u[..., -1, :].abs().amax(-1) <= 1e-7)
        workspace = constants['workspace']
        radius = self.spec.vehicle.effective_radius
        safe &= ((s[..., :2] - radius >= workspace[:, None, None, :2])
                 & (s[..., :2] + radius <= workspace[:, None, None, 2:])).all((-1, -2))
        return parts, safe

    def evaluate(self, requests, raw, packed=None):
        terrain, neighbors = self.pack(requests) if packed is None else packed
        use_events=self.device.type=='cuda' and self.settings.deadline_s is None
        events=[torch.cuda.Event(enable_timing=True) for _ in range(3)] if use_events else None
        if events:events[0].record()
        else:self.synchronize()
        started = time.perf_counter()
        states, controls, velocities, numeric, known = self.rollout(requests, raw, terrain)
        if events:events[1].record()
        else:self.synchronize()
        predicted = time.perf_counter()
        parts, prescreen = self.cost(requests, states, controls, terrain, neighbors)
        cost = sum(parts.values())
        numeric &= torch.isfinite(cost) & torch.isfinite(states).all((-1, -2))
        if events:events[2].record()
        else:self.synchronize()
        self.last_evaluation_events=events
        self.last_evaluation_times = {"prediction_s": 0. if events else predicted - started,
                                      "cost_s": 0. if events else time.perf_counter() - predicted}
        return states, controls, velocities, cost, numeric, prescreen & numeric, parts

    def plan_batch(self, planners, inputs, *, noise=None, clock=time.monotonic, record_samples=True):
        # Selection uses review order, never the final physical cost. Keep the
        # resulting drafts private until all actual NumPy trajectories have
        # been scored together and the original timeout rule has run.
        deferred=self.cfg.obstacle_input is not None
        if deferred:
            for planner in planners:planner._defer_output_scoring=True
        try:
            return self._plan_batch(planners,inputs,noise=noise,clock=clock,record_samples=record_samples)
        finally:
            if deferred:
                for planner in planners:planner.__dict__.pop('_defer_output_scoring',None)
            self.__dict__.pop('_cost_requests',None)
            self.__dict__.pop('_cost_values',None)

    def _score_output_batch(self,planners,requests,outputs,packed):
        if self.cfg.obstacle_input is None:return
        indices=[i for i,output in enumerate(outputs) if output.valid]
        if not indices:return
        items=[requests[i] for i in indices]
        trajectories=[outputs[i].trajectory for i in indices]
        states=torch.as_tensor(np.stack([t.states for t in trajectories]),device=self.device,dtype=torch.float64)[:,None]
        controls=torch.as_tensor(np.stack([t.controls for t in trajectories]),device=self.device,dtype=torch.float64)[:,None]
        terrain,neighbors=(value.select(indices) for value in packed)
        parts,_=self.cost(items,states,controls,terrain,neighbors)
        names=list(parts)
        # A single D2H transfer replaces one Python scalar sync per component.
        values=torch.stack([parts[name][:,0] for name in names],-1).cpu().tolist()
        costs=[sum(row) for row in values]
        if not all(map(math.isfinite,costs)):
            raise ValueError('Final trajectory cost must be finite or None')
        for index,row,cost in zip(indices,values,costs):
            selected=dict(zip(names,row))
            outputs[index]=replace(outputs[index],total_cost=cost)
            outputs[index].diagnostics['selected_cost_parts']=dict(selected)

    def _commit_output_batch(self,planners,requests,outputs):
        if self.cfg.obstacle_input is not None:
            for planner,item,output in zip(planners,requests,outputs):
                planner._commit_output(item,output)
                planner.warm=(output.trajectory.controls.copy() if output.valid
                              and not output.diagnostics.get('parking_branch',False) else None)
                if output.valid:planner.selected_cost_parts=dict(output.diagnostics['selected_cost_parts'])
                planner.last_diagnostics=dict(output.diagnostics)

    def _plan_batch(self, planners, inputs, *, noise=None, clock=time.monotonic, record_samples=True):
        if len(planners) != len(inputs) or len({id(p) for p in planners}) != len(planners):
            raise ValueError("One independent MPPI handle per batch input")
        if not inputs:
            return []
        if len({(r.env_id, r.agent_id) for r in inputs}) != len(inputs):
            raise ValueError("Duplicate MPPI input identity")
        for planner, request in zip(planners, inputs):
            if planner.batch_kernel.metadata != self.metadata:
                raise ValueError("Mixed MPPI batch configuration")
            planner.validate_input(request)
        expected = (self.settings.iterations, len(inputs), self.settings.samples, 40, 2)
        if noise is not None and (not hasattr(noise, "shape") or tuple(noise.shape) != expected):
            raise ValueError("Injected MPPI noise shape mismatch")
        # Validate tensor conversion before binding any handle or consuming RNG.
        injected = None if noise is None else self.tensor(noise)
        started_batch = clock()
        # One owned all-vehicle index serves HOLD scoring, active proposals,
        # and final execution-trajectory scoring without host repacking.
        all_pack_start=time.perf_counter()
        all_packed = self.pack(inputs)
        all_pack_seconds=time.perf_counter()-all_pack_start
        for i, (planner, request) in enumerate(zip(planners, inputs)):
            planner._score_context = request
            planner._score_packed = tuple(value.select([i]) for value in all_packed)
        outputs = [None] * len(inputs)
        active = []
        for i, (planner, request) in enumerate(zip(planners, inputs)):
            outputs[i] = planner.prepare(request)
            if outputs[i] is None:
                active.append(i)
            else:
                if self.cfg.obstacle_input is None:planner.warm = None
                planner.last_samples = None
                details = {**outputs[i].diagnostics, "samples": 0, "parking_branch": True,
                    "selected_cost_parts": dict(planner.selected_cost_parts) if outputs[i].valid else {}}
                outputs[i].diagnostics.update(details)
                if self.cfg.obstacle_input is None:planner.last_diagnostics=details
        if not active:
            self._score_output_batch(planners,inputs,outputs,all_packed)
            self._commit_output_batch(planners,inputs,outputs)
            elapsed = clock() - started_batch
            for output in outputs:
                output.diagnostics["end_to_end_s"] = elapsed
            for planner,output in zip(planners,outputs):planner.last_diagnostics=dict(output.diagnostics)
            return outputs
        self.batch_calls += 1
        requests, handles = [inputs[i] for i in active], [planners[i] for i in active]
        start = started_batch
        pack_start = time.perf_counter()
        packed = tuple(value.select(active) for value in all_packed)
        if self.settings.deadline_s is not None:self.synchronize()
        evaluation_events=[];update_events=[]
        stages = {"pack_s": all_pack_seconds + time.perf_counter() - pack_start, "prediction_s": 0., "cost_s": 0., "update_s": 0.}
        mean = self.seed_controls(requests)
        for i, (planner, request) in enumerate(zip(handles, requests)):
            if planner.warm is not None and planner._last_time is not None:
                shift = round((request.now - planner._last_time) / self.spec.grid.dt)
                if 0 <= shift < 40:
                    mean[i] = self.tensor(np.vstack((planner.warm[shift:], np.zeros((shift, 2)))))
        sigma = self.tensor([self.settings.noise_v, self.settings.noise_w])
        complete = 0; last = None; timed_out = False
        for iteration in range(self.settings.iterations):
            if self.settings.deadline_s is not None and clock() - start > self.settings.deadline_s:
                timed_out = True
                break
            eps = (injected[iteration, active] if injected is not None else torch.stack([
                torch.randn((self.settings.samples, 40, 2), generator=p.generator,
                            device=self.device, dtype=self.dtype) for p in handles]) * sigma)
            proposal_mean = mean
            raw_controls = mean[:, None] + eps
            evaluated = self.evaluate(requests, raw_controls, packed)
            if self.last_evaluation_events:evaluation_events.append(self.last_evaluation_events)
            for key, value in self.last_evaluation_times.items():stages[key] += value
            states, controls, velocities, cost, numeric, pre, parts = evaluated
            update_start = time.perf_counter()
            pair=[torch.cuda.Event(enable_timing=True) for _ in range(2)] if self.device.type=='cuda' and self.settings.deadline_s is None else None
            if pair:pair[0].record()
            mean, weights, score, usable = importance_update(mean, eps, cost, numeric,
                self.settings.temperature, sigma, accumulation_dtype=torch.float64)
            if pair:
                pair[1].record();update_events.append(pair)
            else:
                self.synchronize()
                stages["update_s"] += time.perf_counter() - update_start
            last = (evaluated, weights, score, usable, eps, raw_controls, proposal_mean)
            complete += 1
            if self.settings.deadline_s is not None and clock() - start > self.settings.deadline_s:
                timed_out = True
                break
        updated = self.evaluate(requests, mean[:, None], packed) if complete else None
        if updated is not None:
            if self.last_evaluation_events:evaluation_events.append(self.last_evaluation_events)
            for key, value in self.last_evaluation_times.items():stages[key] += value
        if self.settings.deadline_s is not None and clock() - start > self.settings.deadline_s:
            timed_out = True
        # Compute reductions and stable ranking on the original device, then
        # transfer each dtype as one contiguous slab for all vehicles. Reuse
        # those arrays for candidates and the complete diagnostic snapshot.
        if complete:
            evaluated,weights,score,usable,eps,raw_controls,proposal_mean=last
            states,controls,velocities,cost,numeric,pre,parts=evaluated
            rank=torch.argsort(torch.where(pre,cost,torch.inf),dim=-1,stable=True)
            selected=rank[:,:self.settings.candidate_limit]
            selected_controls=torch.gather(controls,1,selected[:,:,None,None].expand(-1,-1,40,2))
            payload=dict(selected_indices=selected,selected_valid=torch.gather(pre,1,selected),
                selected_controls=selected_controls,usable_any=usable.any(-1),
                correction=score-cost,updated_controls=updated[1],updated_numeric=updated[4],
                **({'updated_prescreen':updated[5]} if not self.cfg.predictive_footprint_checks else {}),
                numeric_fraction=numeric.double().mean(-1),pre_fraction=pre.double().mean(-1),
                pre_count=pre.sum(-1),effective=1/weights.square().sum(-1))
            if record_samples:
                payload.update(states=states,controls=controls,cost=cost,numeric=numeric,pre=pre,
                    weights=weights,eps=eps,raw_controls=raw_controls,proposal_mean=proposal_mean,
                    **{'part_'+k:v for k,v in parts.items()})
            transfer_bytes=sum(v.numel()*v.element_size() for v in payload.values())
            host={}
            for dtype in {v.dtype for v in payload.values()}:
                group=[(k,v) for k,v in payload.items() if v.dtype==dtype]
                slab=torch.cat([v.detach().reshape(-1) for _,v in group]).cpu()
                offset=0
                for key,value in group:
                    host[key]=slab[offset:offset+value.numel()].view(value.shape);offset+=value.numel()
            if record_samples:
                states,controls,cost,numeric,pre,weights,eps,raw_controls,proposal_mean=[host[k] for k in
                    ('states','controls','cost','numeric','pre','weights','eps','raw_controls','proposal_mean')]
                parts={k:host['part_'+k] for k in parts}
        for events in evaluation_events:
            events[-1].synchronize()
            stages['prediction_s']+=events[0].elapsed_time(events[1])/1000
            stages['cost_s']+=events[1].elapsed_time(events[2])/1000
        for events in update_events:
            events[-1].synchronize();stages['update_s']+=events[0].elapsed_time(events[1])/1000
        prior_trajectories = [deepcopy(p._previous_trajectory) for p in handles]
        review_start = time.perf_counter()
        for b, (planner, request) in enumerate(zip(handles, requests)):
            candidates = []
            details = {"samples": self.settings.samples, "iterations_completed": complete,
                       "timed_out": timed_out, "batch_size": len(active), "kernel": self.version,
                       "query_max_pair_entries": packed[0].max_pair_entries,
                       "terrain_ambiguous_queries": packed[0].ambiguous_queries,
                       "terrain_boundary_queries": packed[0].boundary_queries,
                       "terrain_cpu_tie_query_s": packed[0].cpu_tie_query_s,
                       "numerical_valid_fraction": 0., "prescreen_fraction": 0.,
                       "prescreen_count": 0, "sample_review_limit": self.settings.candidate_limit,
                       "full_review_passed": False, "sampling": "raw_gaussian_importance_v1"}
            if self.cfg.obstacle_input is not None:
                details['grid_upload_bytes']=packed[0].upload_bytes
            if complete:
                details.update(numerical_valid_fraction=float(host['numeric_fraction'][b]),
                               prescreen_fraction=float(host['pre_fraction'][b]),
                               prescreen_count=int(host['pre_count'][b]),
                               effective_samples=float(host['effective'][b]) if bool(host['usable_any'][b]) else 0.,
                               full_samples_recorded=record_samples,host_transfer_bytes=transfer_bytes,
                               importance_correction=[value if math.isfinite(value) else None
                                   for value in host['correction'][b].tolist()])
                if (bool(host['usable_any'][b]) and bool(host['updated_numeric'][b, 0])
                        and (self.cfg.predictive_footprint_checks or bool(host['updated_prescreen'][b,0]))):
                    candidates.append(Candidate(host['updated_controls'][b,0].numpy(), "updated_mean"))
                for slot,index in enumerate(host['selected_indices'][b].tolist()):
                    if bool(host['selected_valid'][b,slot]):
                        candidates.append(Candidate(host['selected_controls'][b,slot].numpy(), "sample",
                                                    diagnostics={"sample_index": index}))
                planner.last_samples = {"recorded_iteration": complete - 1, "total_iterations": self.settings.iterations,
                                        "proposal_mean": proposal_mean[b].detach().cpu().clone(),
                                        "raw_controls": raw_controls[b].detach().cpu().clone(),
                                        "controls": controls[b].detach().cpu().clone(),
                                        "states": states[b].detach().cpu().clone(),
                                        "cost": cost[b].detach().cpu().clone(), "numerical_valid": numeric[b].cpu().clone(),
                                        "prescreen": pre[b].cpu().clone(), "raw_noise": eps[b].detach().cpu().clone(),
                                        "weights": weights[b].detach().cpu().clone(),
                                        "physical_cost_parts": {name: value[b].detach().cpu().clone() for name, value in parts.items()}} if record_samples else None
                if not bool(host['usable_any'][b]):
                    details["failure_reason"] = "all_samples_numerically_invalid"
                    candidates = []
            else:
                planner.last_samples = None
                details["failure_reason"] = "planner_timeout"
            if timed_out:
                candidates = []
                details["failure_reason"] = "planner_timeout"
            output = planner.finish(request, candidates, diagnostics=details)
            if self.cfg.obstacle_input is None:planner.warm = output.trajectory.controls.copy() if output.valid else None
            details = {**output.diagnostics, "full_review_passed": output.valid and self.cfg.predictive_footprint_checks,
                                         "selected_cost_parts": dict(planner.selected_cost_parts) if output.valid else {}}
            output.diagnostics.update(details)
            if self.cfg.obstacle_input is None:planner.last_diagnostics=details
            outputs[active[b]] = output
        self._score_output_batch(planners,inputs,outputs,all_packed)
        # The receiver cannot use any output before all full reviews finish.
        elapsed = clock() - started_batch
        if self.settings.deadline_s is not None and elapsed > self.settings.deadline_s and not timed_out:
            timed_out = True
            for b, (planner, request) in enumerate(zip(handles, requests)):
                planner._previous_trajectory = prior_trajectories[b]
                details = dict(outputs[active[b]].diagnostics)
                details.update(timed_out=True, failure_reason="planner_timeout", timeout_phase="full_review",
                               full_review_passed=False, selected_cost_parts={})
                outputs[active[b]] = planner._brake(request, "planner_timeout", diagnostics=details)
                planner.warm = None
            elapsed = clock() - started_batch
        for i, (planner, output) in enumerate(zip(planners, outputs)):
            d = output.diagnostics
            reviewed = d.get("sampled_candidate_reviews", 0)
            unreviewed = max(0, d.get("prescreen_count", 0) - reviewed)
            d.update(end_to_end_s=elapsed, unreviewed_sample_count=unreviewed,
                     sample_review_budget_exhausted=(reviewed >= self.settings.candidate_limit and unreviewed > 0
                         and d.get("selected_source") not in ("sample", "updated_mean")))
            output.feedback.solve_seconds = elapsed
            planner.last_diagnostics = dict(d)
        self._commit_output_batch(planners,inputs,outputs)
        stages["full_review_and_scoring_s"] = time.perf_counter() - review_start
        for planner, output in zip(planners, outputs):
            output.diagnostics["batch_stage_times"] = dict(stages)
            planner.last_diagnostics = dict(output.diagnostics)
        return outputs


class UpstreamMPPIAdapter(PlanningAdapter):
    adapter_id = "mppi_upstream_v1"

    def __init__(self, cfg, kernel=None):
        self.batch_kernel = kernel or UpstreamMPPIKernel(cfg)
        self.generator = torch.Generator(device=self.batch_kernel.device)
        self.warm = None
        self.last_samples = None
        self.last_diagnostics = {}
        self._rng_bound = False
        self.selected_cost_parts = {}
        super().__init__(cfg)

    def prepare(self, item):
        result = super().prepare(item)
        if not self._rng_bound:
            key = f"{self.cfg.seed}:{item.env_id}:{item.agent_id}".encode()
            self.generator.manual_seed(int.from_bytes(hashlib.sha256(key).digest()[:8], "little") % (2**63 - 1))
            self._rng_bound = True
        return result

    def plan(self, item):
        return self.batch_kernel.plan_batch([self], [item])[0]

    def candidate_controls(self, item):
        raise RuntimeError("MPPI candidates are produced by shared plan_batch")

    def score_trajectory(self, item, trajectory):
        if getattr(self, '_score_context', None) is item:
            terrain, neighbors = self._score_packed
        else:
            terrain, neighbors = self.batch_kernel.pack([item])
        states = torch.as_tensor(trajectory.states, dtype=torch.float64, device=self.batch_kernel.device)[None, None]
        controls = torch.as_tensor(trajectory.controls, dtype=torch.float64, device=self.batch_kernel.device)[None, None]
        parts, _ = self.batch_kernel.cost([item], states, controls, terrain, neighbors)
        self.selected_cost_parts = {name: float(value[0, 0]) for name, value in parts.items()}
        return sum(self.selected_cost_parts.values())

    def _adapter_state(self):
        return {"kernel": self.batch_kernel.metadata, "warm": deepcopy(self.warm),
                "generator": self.generator.get_state().cpu().clone(), "rng_bound": self._rng_bound}

    def _validate_adapter_state(self, state):
        if not isinstance(state, dict) or state.get("kernel") != self.batch_kernel.metadata:
            raise ValueError("MPPI device/version/settings mismatch")
        if type(state.get("rng_bound")) is not bool:
            raise ValueError("Invalid MPPI RNG binding")
        warm = state.get("warm")
        if warm is not None and (np.shape(warm) != (40, 2) or not np.isfinite(warm).all()):
            raise ValueError("Invalid MPPI warm sequence")
        if warm is not None:
            if not state["rng_bound"] or np.max(abs(warm[-1])) > 1e-8:
                raise ValueError("Invalid MPPI warm stop/binding")
            for step, control in enumerate(warm):
                if control_error(control, warm[step - 1] if step else control, self.spec):
                    raise ValueError("Illegal MPPI warm control")
        probe = torch.Generator(device=self.batch_kernel.device)
        try:
            probe.set_state(state["generator"])
        except (KeyError, TypeError, RuntimeError) as error:
            raise ValueError("Invalid MPPI generator state") from error

    def _load_adapter_state(self, state):
        self.generator.set_state(state["generator"])
        self._rng_bound = state["rng_bound"]
        self.warm = deepcopy(state["warm"])
        self.last_samples = None
        self.last_diagnostics = {}
        self.selected_cost_parts = {}

    def load_state_dict(self, state):
        if not isinstance(state, dict) or not isinstance(state.get("adapter_state"), dict):
            raise ValueError("Invalid MPPI state")
        extra = state["adapter_state"]
        if extra.get("rng_bound") != (state.get("identity") is not None):
            raise ValueError("MPPI RNG and vehicle identity disagree")
        warm = extra.get("warm")
        previous = state.get("previous_trajectory")
        if warm is not None and (previous is None or not np.array_equal(warm, previous.controls)):
            raise ValueError("MPPI warm state and selected trajectory disagree")
        super().load_state_dict(state)

    def _reset_adapter(self):
        self.warm = None
        self.last_samples = None
        self.last_diagnostics = {}
        self._rng_bound = False
        self.selected_cost_parts = {}
