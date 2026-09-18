"""Sensor/transport boundary and macro transitions around the unchanged proxy core."""

import copy
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass

import numpy as np
import torch

from .. import terrain_features as terrain
from ..gathering_env import MultiRoverGatheringCore
from ..metrics import compute_team_metrics
from ..reward import (
    compute_centroid_flatness_cost,
    compute_centroid_flatness_reward,
    compute_reward,
)
from ..simple_controller import ControlCommand
from ..termination import compute_done, compute_success_gates
from .contracts import DT, LocalGoalRequest, N, NeighborPrediction, PlannerFeedback, rotation, wrap
from .memory import LocalTerrainMemory
from .planner import NMPCPlanner, build_solver, track_trajectory


@dataclass
class MacroTransition:
    reward: torch.Tensor
    duration: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    terminal_actor: torch.Tensor
    terminal_critic: torch.Tensor
    actor: torch.Tensor
    critic: torch.Tensor
    episodes: list
    records: list


class LocalGoalEnvironment:
    def __init__(self, cfg, cache_dir, *, solver_factory=build_solver):
        if cfg.simulation.device != "cpu" or cfg.task.n_agents != 4:
            raise ValueError("First-round route supports four rovers on CPU")
        if not cfg.safety.collision_termination_enabled:
            raise ValueError("Collision termination must be explicitly enabled")
        if (
            cfg.reward_weights.oracle
            or cfg.reward_weights.motion
            or cfg.reward_weights.consistency
            or cfg.reward_weights.active_dstc
        ):
            raise ValueError("New route forbids oracle/primitive/DSTC reward weights")
        self.core = MultiRoverGatheringCore(cfg)
        self.e, self.a = self.core.num_envs, 4
        self.pool = ThreadPoolExecutor(max_workers=4)
        self.planners = []
        for i in range(self.e * self.a):
            solver = solver_factory(cache_dir, cfg.terrain.slope_speed_scale, build=(i == 0))
            self.planners.append(NMPCPlanner(solver, cfg.terrain.slope_speed_scale))
        self.memories = [LocalTerrainMemory() for _ in self.planners]
        self.origins = np.zeros((self.e, self.a, 3))
        self.height_origins = np.zeros((self.e, self.a))
        self.goals = [None for _ in self.planners]
        self.plans = [None for _ in self.planners]
        self.feedback = [PlannerFeedback() for _ in self.planners]
        self.commands = np.zeros((self.e, self.a, 2))
        self.received = {}
        self.distance = np.zeros(self.e)
        self.metrics = []
        self.reset(np.arange(self.e), reset_core=False)

    def close(self):
        self.pool.shutdown(wait=True)

    def poses(self):
        p = np.concatenate(
            [self.core.positions[..., :2].numpy(), self.core.yaws[..., None].numpy()], -1
        )
        for e in range(self.e):
            for a in range(self.a):
                origin = self.origins[e, a]
                p[e, a, :2] = rotation(-origin[2]) @ (p[e, a, :2] - origin[:2])
                p[e, a, 2] = wrap(p[e, a, 2] - origin[2])
        return p

    def reset(self, ids, *, reset_core=True):
        ids = np.asarray(ids, dtype=int)
        if reset_core:
            self.core.reset(torch.as_tensor(ids))
        for e in ids:
            self.origins[e, :, :2] = self.core.positions[e, :, :2].numpy()
            self.origins[e, :, 2] = self.core.yaws[e].numpy()
            self.height_origins[e] = self.core.positions[e, :, 2].numpy()
            self.commands[e] = 0
            self.distance[e] = 0
            for a in range(self.a):
                i = e * self.a + a
                self.planners[i].reset()
                self.memories[i].reset()
                self.plans[i] = None
                self.feedback[i] = PlannerFeedback()
                self.goals[i] = LocalGoalRequest(np.zeros(3), 0.0, 0.0, 1.0)
            self.received = {k: v for k, v in self.received.items() if k[0] != e}
        self.observe()

    def _sensors(self, base, poses):
        # Consume ONLY the existing 112 two-channel samples. Estimate slopes
        # from their heights; never expose the terrain runtime to the planner.
        specs = [
            (terrain.MULTISCALE_TERRAIN_FINE_X, terrain.MULTISCALE_TERRAIN_FINE_Y),
            (terrain.MULTISCALE_TERRAIN_MEDIUM_X, terrain.MULTISCALE_TERRAIN_MEDIUM_Y),
            (terrain.MULTISCALE_TERRAIN_COARSE_X, terrain.MULTISCALE_TERRAIN_COARSE_Y),
        ]
        samples = base[..., 66:290].numpy()
        for e in range(self.e):
            for a in range(self.a):
                offset = 0
                points, features = [], []
                for xs, ys in specs:
                    count = len(xs) * len(ys) * 2
                    grid = samples[e, a, offset : offset + count].reshape(len(xs), len(ys), 2)
                    offset += count
                    xy = np.stack(np.meshgrid(xs, ys, indexing="ij"), -1).reshape(-1, 2)
                    slope = np.stack(np.gradient(grid[..., 0], xs, ys, edge_order=1), -1).reshape(
                        -1, 2
                    )
                    matrix = rotation(poses[e, a, 2])
                    points.append(xy @ matrix.T + poses[e, a, :2])
                    feature = np.zeros((len(xy), 5))
                    feature[:, 0] = (
                        grid[..., 0].reshape(-1)
                        + float(self.core.positions[e, a, 2])
                        - self.height_origins[e, a]
                    )
                    feature[:, 1:3] = slope @ matrix.T
                    feature[:, 4] = 1 - grid[..., 1].reshape(-1)
                    features.append(feature)
                self.memories[e * self.a + a].observe(
                    np.concatenate(points),
                    np.concatenate(features),
                    float(self.core.step_count[e]) * DT,
                )

    def _transport(self, poses):
        cache = self.core.communication_cache
        for e in range(self.e):
            now = float(self.core.step_count[e]) * DT
            for receiver in range(self.a):
                pose = poses[e, receiver]
                own_velocity = (
                    rotation(-self.origins[e, receiver, 2])
                    @ self.core.velocities_xy[e, receiver].numpy()
                )
                for sender in range(self.a):
                    if sender == receiver or not cache.valid[e, receiver, sender]:
                        continue
                    key = e, receiver, sender
                    full = bool(cache.full[e, receiver, sender])
                    if key in self.received and self.received[key]["timestamp"] == now:
                        continue
                    if float(cache.age[e, receiver, sender]) > 1e-6:
                        if key in self.received and not full:
                            self.received[key]["plan"] = None
                            self.received[key]["goal"] = None
                        continue
                    feature = cache.features[e, receiver, sender].numpy()
                    position = pose[:2] + rotation(pose[2]) @ feature[:2]
                    velocity = (
                        own_velocity + rotation(pose[2]) @ feature[2:4] if full else np.zeros(2)
                    )
                    message = {
                        "position": position,
                        "velocity": velocity,
                        "timestamp": now,
                        "expires_at": now + 0.6,
                        "version": 1,
                        "full": full,
                        "plan": None,
                        "goal": None,
                    }
                    # Simulator transport transforms ONLY the sender's published payload.
                    # Receiver planner never accesses live peer states or these transforms.
                    if full:
                        origin_s, origin_r = self.origins[e, sender], self.origins[e, receiver]
                        transform = lambda xy, origin_s=origin_s, origin_r=origin_r: (
                            (xy @ rotation(origin_s[2]).T + origin_s[:2] - origin_r[:2])
                            @ rotation(origin_r[2])
                        )
                        goal = self.goals[e * self.a + sender]
                        message["goal"] = transform(goal.target[:2])
                        message["stop_intent"] = bool(
                            goal.speed_cap < 0.05
                            and now <= goal.expires_at
                            and np.linalg.norm(velocity) < 0.05
                        )
                        plan = self.plans[e * self.a + sender]
                        if plan is not None and now <= plan.times[-1]:
                            message["plan"] = (plan.times.copy(), transform(plan.states[:, :2]))
                            message["expires_at"] = min(now + 0.6, plan.times[-1])
                    self.received[key] = message

    def neighbors(self, e, a):
        now = float(self.core.step_count[e]) * DT
        times = np.arange(N + 1) * DT
        result = []
        for sender in range(self.a):
            message = self.received.get((e, a, sender))
            if message is None:
                continue
            age = max(0.0, now - message["timestamp"])
            fresh = now <= message["expires_at"]
            if fresh and message["plan"] is not None:
                ts, xy = message["plan"]
                points = np.stack([np.interp(now + times, ts, xy[:, j]) for j in range(2)], -1)
                # A published stationary tail is a terminal hold, not an
                # unspecified max-speed continuation. Only trust fresh FULL
                # payloads, and keep age/time tracking uncertainty in either case.
                stopped_tail = bool(message["full"] and np.linalg.norm(xy[-1] - xy[-2]) < 1e-5)
                radius = (
                    0.42
                    + 0.04
                    + 0.08 * age
                    + 0.08 * times
                    + (0.0 if stopped_tail else 1.15) * np.maximum(0, now + times - ts[-1])
                )
            else:
                points = (
                    message["position"] + (age + times[:, None]) * message["velocity"]
                    if fresh
                    else np.tile(message["position"], (N + 1, 1))
                )
                trusted_stop = fresh and message["full"] and message.get("stop_intent", False)
                growth = 0.08 if trusted_stop else 0.15 if fresh and message["full"] else 1.15
                radius = 0.42 + 0.04 + growth * (age + times)
            result.append(
                NeighborPrediction(points, radius, message["timestamp"], message["expires_at"])
            )
        return result

    def observe(self):
        base, state = self.core.get_observations()
        poses = self.poses()
        self._sensors(base, poses)
        self._transport(poses)
        extra = np.zeros((self.e, self.a, 115), dtype=np.float32)
        offsets = np.stack(
            np.meshgrid(np.linspace(-1.6, 1.6, 5), np.linspace(-1.6, 1.6, 5), indexing="ij"), -1
        ).reshape(-1, 2)
        for e in range(self.e):
            for a in range(self.a):
                i = e * self.a + a
                pose = poses[e, a]
                goal, fb = self.goals[i], self.feedback[i]
                delta = rotation(-pose[2]) @ (goal.target[:2] - pose[:2])
                velocity = (
                    rotation(-float(self.core.yaws[e, a])) @ self.core.velocities_xy[e, a].numpy()
                )
                features, known, _ = self.memories[i].lookup(
                    offsets @ rotation(pose[2]).T + pose[:2]
                )
                extra[e, a, :10] = [
                    *delta,
                    np.sin(goal.target[2] - pose[2]),
                    np.cos(goal.target[2] - pose[2]),
                    goal.speed_cap,
                    fb.blocked_seconds / 10,
                    {
                        "reset": 0,
                        "feasible": 1,
                        "feasible_iteration_limit": 2,
                        "feasible_prefix": 5,
                        "previous": 3,
                        "brake": 4,
                    }[fb.status]
                    / 5,
                    velocity[0],
                    float(self.core.angular_velocities[e, a]),
                    known.mean(),
                ]
                extra[e, a, 10:85] = np.stack(
                    [
                        np.where(known, features[:, 0], 0),
                        np.where(known, 1 - features[:, 4], 0),
                        known,
                    ],
                    -1,
                ).reshape(-1)
                for slot, sender in enumerate(j for j in range(4) if j != a):
                    msg = self.received.get((e, a, sender))
                    now = float(self.core.step_count[e]) * DT
                    if msg is None:
                        continue
                    age = max(0.0, now - msg["timestamp"])
                    values = np.zeros(10)
                    values[:4] = [
                        age / 10,
                        float(now <= msg["expires_at"]),
                        msg["version"],
                        max(0, msg["expires_at"] - now),
                    ]
                    if now <= msg["expires_at"] and msg["goal"] is not None:
                        values[4:6] = rotation(-pose[2]) @ (msg["goal"] - pose[:2])
                    if now <= msg["expires_at"] and msg["plan"] is not None:
                        ts, xy = msg["plan"]
                        for k, t in enumerate((0.4, 1.0)):
                            point = np.array([np.interp(now + t, ts, xy[:, j]) for j in range(2)])
                            values[6 + k * 2 : 8 + k * 2] = rotation(-pose[2]) @ (point - pose[:2])
                    extra[e, a, 85 + slot * 10 : 95 + slot * 10] = values
        extra = torch.from_numpy(extra)
        return torch.cat([base, extra], -1), torch.cat([state, extra[:, :, :10].flatten(1)], -1)

    def _physics(self, commands, active):
        c = self.core
        old = c.positions.clone()
        previous = compute_team_metrics(c.positions, c.velocities_xy)
        previous_flat = c.evaluate_current_gather_point_flatness(previous)
        cost_old = compute_centroid_flatness_cost(
            previous_flat.height_range, previous_flat.max_slope, c.cfg
        )
        control = torch.as_tensor(commands, dtype=torch.float32)
        control[~active] = 0
        c._integrate(ControlCommand(control[..., 0], control[..., 1]))
        c.step_count += active.long()
        c.global_step_count += 1
        c._advance_communication()
        metrics = compute_team_metrics(c.positions, c.velocities_xy)
        flat = c.evaluate_current_gather_point_flatness(metrics)
        cost = compute_centroid_flatness_cost(flat.height_range, flat.max_slope, c.cfg)
        flat_reward, _, _ = compute_centroid_flatness_reward(
            cost_old, cost, previous.dmax, metrics.dmax, c.cfg
        )
        done, hold = compute_done(
            c.positions,
            c.velocities_xy,
            metrics,
            c.success_hold_count,
            c.step_count,
            c.max_episode_steps,
            c.cfg.success_thresholds,
            c.cfg.safety,
            flatness_ok=flat.is_flat,
        )
        c.success_hold_count = torch.where(active, hold, c.success_hold_count)
        physical = torch.stack(
            [
                torch.linalg.norm(c.positions[..., :2] - old[..., :2], dim=-1),
                c.angular_velocities.abs() * DT,
            ],
            -1,
        )
        terms, _ = compute_reward(
            c.positions,
            c.oracle_point,
            previous,
            metrics,
            c.prev_mean_oracle_distance,
            physical,
            c.previous_physical_action,
            done,
            c.success_hold_count,
            c.last_terrain_features,
            c.cfg,
            terrain_speed_scale=c.last_terrain_speed_scale,
            height_delta=c.last_height_delta,
            centroid_flatness_reward=flat_reward,
        )
        c.previous_physical_action = physical
        c.metrics = metrics
        gates = compute_success_gates(
            metrics, c.velocities_xy, c.cfg.success_thresholds, flatness_ok=flat.is_flat
        )
        self.distance += physical[..., 0].mean(-1).numpy() * active.numpy()
        return terms.total, done, gates

    def state_dict(self):
        return {
            "core": copy.deepcopy(self.core.__dict__),
            "local": {
                name: copy.deepcopy(getattr(self, name))
                for name in (
                    "origins",
                    "height_origins",
                    "goals",
                    "plans",
                    "feedback",
                    "commands",
                    "received",
                    "distance",
                    "memories",
                )
            },
            "warm": [(p.previous, p.blocked) for p in self.planners],
        }

    def load_state_dict(self, state):
        self.core.__dict__.update(state["core"])
        for name, value in state["local"].items():
            setattr(self, name, value)
        for planner, (previous, blocked) in zip(self.planners, state["warm"]):
            planner.reset()
            planner.previous, planner.blocked = previous, blocked

    def step(self, actions, *, gamma=0.99, max_low_steps=5, deadline=float("inf"), enabled=None):
        poses = self.poses()
        actions = np.asarray(actions)
        for e in range(self.e):
            for a in range(self.a):
                i = e * self.a + a
                self.goals[i] = LocalGoalRequest.from_action(
                    actions[e, a], poses[e, a], float(self.core.step_count[e]) * DT
                )
                goal = self.goals[i]
                origin = self.origins[e, a]
                self.core.committed_plan_world_subgoal[e, a, :2] = torch.as_tensor(
                    origin[:2] + rotation(origin[2]) @ goal.target[:2]
                )
                self.core.committed_plan_local_xy[e, a] = torch.as_tensor(actions[e, a, :2] * 1.6)
                self.core.committed_reference_speed[e, a] = goal.speed_cap
                self.core.committed_planned_yaw_delta[e, a] = float(actions[e, a, 2] * np.pi)
        active = torch.ones(self.e, dtype=torch.bool) if enabled is None else enabled.clone()
        reward = torch.zeros(self.e)
        duration = torch.zeros(self.e, dtype=torch.long)
        terminated, truncated = torch.zeros_like(active), torch.zeros_like(active)
        terminal_actor, terminal_critic = self.observe()
        episodes, records = [], []
        for _ in range(max_low_steps):
            if duration.max() > 0 and time.monotonic() >= deadline:
                break
            poses = self.poses()
            futures = []
            for e in range(self.e):
                if not active[e]:
                    continue
                for a in range(self.a):
                    i = e * self.a + a
                    futures.append(
                        (
                            e,
                            a,
                            self.pool.submit(
                                self.planners[i].plan,
                                poses[e, a],
                                self.goals[i],
                                self.memories[i],
                                self.neighbors(e, a),
                                self.commands[e, a].copy(),
                                float(self.core.step_count[e]) * DT,
                            ),
                        )
                    )
            commands = np.zeros_like(self.commands)
            for e, a, future in futures:
                i = e * self.a + a
                plan, feedback = future.result()
                self.plans[i], self.feedback[i] = plan, feedback
                commands[e, a] = track_trajectory(poses[e, a], plan)
                self.metrics.append(feedback)
            self.commands = commands.copy()
            r, done, gates = self._physics(commands, active)
            reward += gamma**duration * r * active
            duration += active.long()
            obs, state = self.observe()
            terminal_actor[active] = obs[active]
            terminal_critic[active] = state[active]
            ended = active & done.done
            for e in torch.where(ended)[0].tolist():
                episodes.append(
                    {
                        "env": e,
                        "success": bool(done.success[e]),
                        "collision": bool(done.collision[e]),
                        "timeout": bool(done.truncated[e]),
                        "out_of_bounds": bool(done.out_of_bounds[e]),
                        "dmax": float(self.core.metrics.dmax[e]),
                        "dispersion": float(self.core.metrics.dispersion[e]),
                        "motion_progress_m": float(self.distance[e]),
                        "hold": int(self.core.success_hold_count[e]),
                        **{key: bool(getattr(gates, key)[e]) for key in gates.__dataclass_fields__},
                    }
                )
            records.append(
                {
                    "positions": self.core.positions.tolist(),
                    "request": actions.tolist(),
                    "commands": commands.tolist(),
                    "actual_velocity": self.core.velocities_xy.tolist(),
                    "actual_yaw_rate": self.core.angular_velocities.tolist(),
                    "active": active.tolist(),
                    "plans": [
                        {
                            "states": p.states.tolist(),
                            "times": p.times.tolist(),
                            "velocities": p.velocities.tolist(),
                            "controls": p.controls.tolist(),
                            "version": p.version,
                        }
                        if p
                        else None
                        for p in self.plans
                    ],
                    "planner_feedback": [asdict(f) for f in self.feedback],
                }
            )
            terminated |= ended & done.terminated
            truncated |= ended & done.truncated
            active &= ~ended
            if not active.any():
                break
        ids = torch.where(terminated | truncated)[0].numpy()
        if len(ids):
            self.reset(ids)
        obs, state = self.observe()
        return MacroTransition(
            reward,
            duration,
            terminated,
            truncated,
            terminal_actor,
            terminal_critic,
            obs,
            state,
            episodes,
            records,
        )
