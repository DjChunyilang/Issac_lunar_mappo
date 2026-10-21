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
    apply_finite_task_reward,
)
from ..simple_controller import ControlCommand
from ..termination import TerminalReason, compute_done, compute_success_gates
from .contracts import DT, LocalGoalRequest, N, NeighborPrediction, PlannerFeedback, rotation, wrap
from .contracts import contract_metadata
from .memory import LocalTerrainMemory
from .actions import ACTION_VERSION, PolicyAction
from .registry import create_adapters
from .planner import NMPCPlanner, build_solver, track_trajectory
from .protocol import (
    LocalIntent,
    PlannerInput,
    TerrainView,
    IdentifiedNeighbor,
    plan_batch,
    execution_passthrough,
)


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
    terminal_reason: torch.Tensor
    bootstrap_allowed: torch.Tensor
    learning_valid: torch.Tensor


class LocalGoalEnvironment:
    def __init__(
        self,
        cfg,
        cache_dir,
        *,
        solver_factory=build_solver,
        policy_device="cpu",
        solver_device="cpu",
    ):
        if getattr(cfg, "runtime_family", None) == "upstream_scale_v1":
            raise ValueError("Upstream scale requires its versioned execution environment; legacy core refused")
        if cfg.simulation.device != "cpu" or solver_device != "cpu" or cfg.task.n_agents != 4:
            raise ValueError(
                "NMPC host core requires four rovers and CPU solver; policy/raster devices are separate"
            )
        if not cfg.safety.collision_termination_enabled:
            raise ValueError("Collision termination must be explicitly enabled")
        if (
            cfg.reward_weights.oracle
            or cfg.reward_weights.motion
            or cfg.reward_weights.consistency
            or cfg.reward_weights.active_dstc
        ):
            raise ValueError("New route forbids oracle/primitive/DSTC reward weights")
        self.contracts = contract_metadata(cfg)
        self.protocol_enabled = "planner_protocol" in self.contracts
        self.hybrid = self.contracts["action_version"] == ACTION_VERSION
        self.finite_task = self.contracts["task_contract_version"] == "finite_task_v2"
        if abs(cfg.simulation.planning_dt - DT) > 1e-9:
            raise ValueError("Local-goal execution requires the fixed 0.2 s control period")
        if self.finite_task and (
            cfg.success_thresholds.angular_speed <= 0 or cfg.success_thresholds.hold_steps <= 0
        ):
            raise ValueError("Finite task requires positive angular-speed and dwell thresholds")
        self.reward_gamma = 0.99
        self.policy_device = torch.device(policy_device)
        if self.policy_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA policy device is unavailable")
        self.terrain_channels = 3 if cfg.terrain.type == "lunar_scene_pack" else 2
        self.core = MultiRoverGatheringCore(cfg)
        self.e, self.a = self.core.num_envs, 4
        self.pool = ThreadPoolExecutor(max_workers=4)
        if self.protocol_enabled:
            self.planners = create_adapters(
                self.e * self.a,
                cache_dir,
                cfg.terrain.slope_speed_scale,
                cfg.local_goal_contract,
                solver_factory=solver_factory,
            )
        else:
            self.planners = []
            for i in range(self.e * self.a):
                solver = solver_factory(cache_dir, cfg.terrain.slope_speed_scale, build=(i == 0))
                self.planners.append(NMPCPlanner(solver, cfg.terrain.slope_speed_scale))
        memory_kwargs = {"version": getattr(cfg.local_goal_contract, "terrain_memory_version", "quantized_latest_v1")}
        if self.terrain_channels == 3:
            if (
                min(
                    cfg.terrain.local_memory_resolution_m, cfg.terrain.local_memory_support_radius_m
                )
                <= 0
            ):
                raise ValueError("Real maps require explicit sensor-memory scales and task basis")
            memory_kwargs.update({
                "resolution": cfg.terrain.local_memory_resolution_m,
                "support_radius": cfg.terrain.local_memory_support_radius_m,
            })
        if self.contracts.get("planner_adapter") == "primitive47_adapter_v1":
            raise ValueError("Native primitives require explicit replay payload; use planner diagnostics")
        self.memories = [LocalTerrainMemory(**memory_kwargs) for _ in self.planners]
        self.origins = np.zeros((self.e, self.a, 3))
        self.height_origins = np.zeros((self.e, self.a))
        self.goals = [None for _ in self.planners]
        self.intents = [None for _ in self.planners]
        self.plans = [None for _ in self.planners]
        self.projectors = []
        self.execution_plans = [None for _ in self.planners]
        if "safety" in self.contracts:
            from .safety import SafetyProjector
            self.projectors = [SafetyProjector(cfg.terrain.slope_speed_scale,
                deadline_s=cfg.local_goal_contract.safety_deadline_s) for _ in self.planners]
        self.feedback = [PlannerFeedback() for _ in self.planners]
        self.commands = np.zeros((self.e, self.a, 2))
        self.received = {}
        self.distance = np.zeros(self.e)
        self.metrics = []
        self.episode_return = np.zeros(self.e)
        self.episode_discounted_return = np.zeros(self.e)
        self.episode_steps = np.zeros(self.e, dtype=np.int64)
        self.episode_terms = {}
        self.last_reward_terms = {}
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

    def script_observations(self):
        """Public local script input; never exposes the core or peer truth."""
        from .gathering_script import ScriptObservation, ScriptMessage
        poses = self.poses()
        observations = []
        for e in range(self.e):
            now = float(self.core.step_count[e])*DT
            for a in range(self.a):
                messages = tuple(ScriptMessage(sender, m["position"], m["goal"],
                    m.get("yaw"), m["full"], m["timestamp"], m["expires_at"],
                    m.get("stop_intent", False))
                    for (env, receiver, sender), m in sorted(self.received.items())
                    if env == e and receiver == a)
                yaw = float(self.core.yaws[e, a])
                velocity = np.array([self.core.velocities_xy[e, a].numpy() @
                    np.array([np.cos(yaw),np.sin(yaw)]),float(self.core.angular_velocities[e,a])])
                observations.append(ScriptObservation(e,a,poses[e,a],velocity,now,
                    TerrainView(self.memories[e*self.a+a]).conservative(),messages,
                    tuple(self.neighbors(e,a,identified=True))))
        return tuple(observations)

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
            self.episode_return[e] = 0
            self.episode_discounted_return[e] = 0
            self.episode_steps[e] = 0
            for values in self.episode_terms.values():
                values[e] = 0
            for a in range(self.a):
                i = e * self.a + a
                self.planners[i].reset()
                if self.projectors:
                    self.projectors[i].reset()
                self.execution_plans[i] = None
                self.memories[i].reset()
                self.plans[i] = None
                self.feedback[i] = PlannerFeedback()
                self.goals[i] = (
                    None if self.hybrid else LocalGoalRequest(np.zeros(3), 0.0, 0.0, 1.0)
                )
                self.intents[i] = None if self.hybrid else LocalIntent.from_request(self.goals[i])
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
        samples = base[..., 66 : 66 + 112 * self.terrain_channels].numpy()
        for e in range(self.e):
            for a in range(self.a):
                offset = 0
                points, features = [], []
                for xs, ys in specs:
                    count = len(xs) * len(ys) * self.terrain_channels
                    grid = samples[e, a, offset : offset + count].reshape(
                        len(xs), len(ys), self.terrain_channels
                    )
                    offset += count
                    xy = np.stack(np.meshgrid(xs, ys, indexing="ij"), -1).reshape(-1, 2)
                    slope = np.stack(np.gradient(grid[..., 0], xs, ys, edge_order=1), -1).reshape(
                        -1, 2
                    )
                    matrix = rotation(poses[e, a, 2])
                    valid = np.ones(grid.shape[:2], dtype=bool)
                    if self.terrain_channels == 3:
                        from scipy.ndimage import minimum_filter

                        # Gradients require neighboring height samples too.
                        valid = minimum_filter(grid[..., 2], size=3, mode="nearest") > 0.5
                    keep = valid.reshape(-1)
                    points.append((xy @ matrix.T + poses[e, a, :2])[keep])
                    feature = np.zeros((len(xy), 5))
                    feature[:, 0] = (
                        grid[..., 0].reshape(-1)
                        + float(self.core.positions[e, a, 2])
                        - self.height_origins[e, a]
                    )
                    feature[:, 1:3] = slope @ matrix.T
                    feature[:, 4] = 1 - grid[..., 1].reshape(-1)
                    features.append(feature[keep])
                self.memories[e * self.a + a].observe(
                    np.concatenate(points),
                    np.concatenate(features),
                    float(self.core.step_count[e]) * DT,
                )

    def _transport(self, poses):
        cache = self.core.communication_cache
        published_stop = {}
        if getattr(self, "hybrid", False):
            # Sender-side publication; the receiver only consumes this permitted FULL payload.
            for e in range(self.e):
                now = float(self.core.step_count[e]) * DT
                for a in range(self.a):
                    intent = self.intents[e * self.a + a]
                    published_stop[e, a] = bool(
                        intent is not None
                        and intent.hold
                        and now <= intent.expires_at
                        and torch.linalg.vector_norm(self.core.velocities_xy[e, a]) <= 0.05
                        and self.core.angular_velocities[e, a].abs() <= 0.1
                    )
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
                        if self.finite_task:
                            self.received.pop((e, receiver, sender), None)
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
                    if self.projectors or "script_version" in self.contracts:
                        message["yaw"] = wrap(pose[2] + np.arctan2(feature[5], feature[4]))
                    # Simulator transport transforms ONLY the sender's published payload.
                    # Receiver planner never accesses live peer states or these transforms.
                    if full:
                        origin_s, origin_r = self.origins[e, sender], self.origins[e, receiver]
                        transform = lambda xy, origin_s=origin_s, origin_r=origin_r: (
                            (xy @ rotation(origin_s[2]).T + origin_s[:2] - origin_r[:2])
                            @ rotation(origin_r[2])
                        )
                        if getattr(self, "hybrid", False):
                            intent = self.intents[e * self.a + sender]
                            message["goal"] = None if intent is None else transform(intent.goal_xy)
                            message["stop_intent"] = published_stop[e, sender]
                        else:
                            goal = self.goals[e * self.a + sender]
                            message["goal"] = transform(goal.target[:2])
                            message["stop_intent"] = bool(
                                goal.speed_cap < 0.05
                                and now <= goal.expires_at
                                and np.linalg.norm(velocity) < 0.05
                            )
                        plan = (self.execution_plans if self.projectors else self.plans)[
                            e * self.a + sender]
                        if plan is not None and now <= plan.times[-1]:
                            message["plan"] = (plan.times.copy(), transform(plan.states[:, :2]))
                            message["expires_at"] = min(now + 0.6, plan.times[-1])
                    self.received[key] = message

    def neighbors(self, e, a, *, identified=False):
        now = float(self.core.step_count[e]) * DT
        times = np.arange(N + 1) * DT
        result = []
        for sender in range(self.a):
            if self.finite_task and not self.core.communication_cache.valid[e, a, sender]:
                continue
            message = self.received.get((e, a, sender))
            if message is None:
                continue
            if self.finite_task:
                if message["version"] != 1:
                    continue
                if not self.core.communication_cache.full[e, a, sender]:
                    message = {
                        **message,
                        "full": False,
                        "plan": None,
                        "velocity": np.zeros(2),
                        "stop_intent": False,
                    }
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
            prediction = NeighborPrediction(
                points, radius, message["timestamp"], message["expires_at"]
            )
            result.append(IdentifiedNeighbor(sender, prediction) if identified else prediction)
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
                if getattr(self, "hybrid", False):
                    intent = self.intents[i]
                    delta = (
                        np.zeros(2)
                        if intent is None
                        else rotation(-pose[2]) @ (intent.goal_xy - pose[:2])
                    )
                else:
                    delta = rotation(-pose[2]) @ (goal.target[:2] - pose[:2])
                velocity = (
                    rotation(-float(self.core.yaws[e, a])) @ self.core.velocities_xy[e, a].numpy()
                )
                features, known, _ = self.memories[i].lookup(
                    offsets @ rotation(pose[2]).T + pose[:2]
                )
                if getattr(self, "hybrid", False):
                    now = float(self.core.step_count[e]) * DT
                    intent_valid = intent is not None and now <= intent.expires_at
                    remaining = (
                        0.0
                        if intent is None
                        else np.clip(
                            (intent.expires_at - now) / (intent.expires_at - intent.timestamp), 0, 1
                        )
                    )
                    extra[e, a, :10] = [
                        *delta,
                        float(intent.hold) if intent is not None else 0.0,
                        float(intent_valid),
                        remaining,
                        fb.blocked_seconds / 10,
                        {
                            "reset": 0,
                            "feasible": 1,
                            "feasible_iteration_limit": 2,
                            "previous": 3,
                            "brake": 4,
                            "feasible_prefix": 5,
                        }[fb.status]
                        / 5,
                        velocity[0],
                        float(self.core.angular_velocities[e, a]),
                        known.mean(),
                    ]
                else:
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
                senders = (
                    self.core.last_communication_snapshot.sender_indices[e, a].tolist()
                    if self.finite_task
                    else [j for j in range(4) if j != a]
                )
                for slot, sender in enumerate(senders):
                    if sender < 0:
                        continue
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
                    payload_allowed = now <= msg["expires_at"]
                    if self.finite_task:
                        values[1] = float(payload_allowed and msg["version"] == 1)
                        payload_allowed &= (
                            msg.get("full", False)
                            and msg["version"] == 1
                            and bool(
                                self.core.last_communication_snapshot.full_messages[e, a, slot]
                            )
                        )
                    if payload_allowed and msg["goal"] is not None:
                        values[4:6] = rotation(-pose[2]) @ (msg["goal"] - pose[:2])
                    if payload_allowed and msg["plan"] is not None:
                        ts, xy = msg["plan"]
                        for k, t in enumerate((0.4, 1.0)):
                            point = np.array([np.interp(now + t, ts, xy[:, j]) for j in range(2)])
                            values[6 + k * 2 : 8 + k * 2] = rotation(-pose[2]) @ (point - pose[:2])
                    extra[e, a, 85 + slot * 10 : 95 + slot * 10] = values
        extra = torch.from_numpy(extra)
        actor = torch.cat([base, extra], -1)
        critic = torch.cat([state, extra[:, :, :10].flatten(1)], -1)
        if self.finite_task:
            remaining = (1.0 - self.core.step_count.float() / self.core.max_episode_steps).clamp(
                0.0, 1.0
            )[:, None]
            actor = torch.cat([actor, remaining[:, None].expand(-1, self.a, -1)], -1)
            critic = torch.cat([critic, remaining], -1)
        return actor.to(self.policy_device), critic.to(self.policy_device)

    def _physics(self, commands, active):
        c = self.core
        old = c.positions.clone()
        previous_hold = c.success_hold_count.clone()
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
            terrain_unknown=c.last_terrain_unknown.any(-1),
            task_contract_version=self.contracts["task_contract_version"],
            angular_velocities=c.angular_velocities if self.finite_task else None,
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
        if self.finite_task:
            terms = apply_finite_task_reward(
                terms,
                previous,
                metrics,
                cost_old,
                cost,
                previous_hold,
                c.success_hold_count,
                done,
                c.cfg,
                gamma=self.reward_gamma,
            )
        self.last_reward_terms = {
            key: getattr(terms, key).detach().numpy().copy()
            * (1.0 if key in ("total", "success_hold") else getattr(c.cfg.reward_weights, key))
            for key in terms.__dataclass_fields__
        }
        c.previous_physical_action = physical
        c.metrics = metrics
        gates = compute_success_gates(
            metrics,
            c.velocities_xy,
            c.cfg.success_thresholds,
            flatness_ok=flat.is_flat,
            angular_velocities=c.angular_velocities if self.finite_task else None,
        )
        self.distance += physical[..., 0].mean(-1).numpy() * active.numpy()
        return terms.total, done, gates

    def terrain_contract(self):
        bank = self.core.terrain_runtime.scene_bank
        return {
            "observation": self.core.cfg.observation.schema_version,
            "terrain": bank.contract() if bank is not None else "legacy_terrain_v1",
            **({"local_memory_version": self.core.cfg.local_goal_contract.terrain_memory_version}
               if getattr(self.core.cfg.local_goal_contract, "terrain_memory_version", "quantized_latest_v1") != "quantized_latest_v1" else {}),
        }

    def state_dict(self):
        core = copy.deepcopy(self.core.__dict__)
        core["terrain_runtime"].scene_bank = None
        return {
            "task_contract": self.contracts,
            "terrain_contract": self.terrain_contract(),
            **({"safety_states": [p.state_dict() for p in self.projectors],
                 "execution_plans": copy.deepcopy(self.execution_plans)}
               if self.projectors else {}),
            "core": core,
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
                    "episode_return",
                    "episode_discounted_return",
                    "episode_steps",
                    "episode_terms",
                    "memories",
                )
            },
            **(
                {
                    "planner_states": [p.state_dict() for p in self.planners],
                    "intents": copy.deepcopy(self.intents),
                }
                if self.protocol_enabled
                else {"warm": [(p.previous, p.blocked) for p in self.planners]}
            ),
        }

    def load_state_dict(self, state):
        saved_contract = state.get("task_contract")
        if saved_contract != self.contracts and (self.finite_task or saved_contract is not None):
            raise ValueError("Checkpoint task/observation/action/planner contract mismatch")
        if getattr(self.core.cfg.local_goal_contract, "terrain_memory_version", "quantized_latest_v1") != "quantized_latest_v1":
            memories = state.get("local", {}).get("memories", [])
            if len(memories) != len(self.memories) or any(
                type(saved) is not LocalTerrainMemory or
                getattr(saved, "version", "quantized_latest_v1") != current.version or
                (saved.capacity, saved.resolution, saved.support_radius) !=
                (current.capacity, current.resolution, current.support_radius)
                for saved, current in zip(memories, self.memories)):
                raise ValueError("Checkpoint local-memory version/bounds mismatch")
        if self.projectors:
            if len(state.get("safety_states", [])) != len(self.projectors) or len(
                state.get("execution_plans", [])) != len(self.projectors):
                raise ValueError("Missing safety states/execution predictions")
            # Validate on detached handles before mutating any live state.
            from .safety import SafetyProjector
            for saved in state["safety_states"]:
                SafetyProjector(self.core.cfg.terrain.slope_speed_scale,
                    deadline_s=self.core.cfg.local_goal_contract.safety_deadline_s).load_state_dict(saved)
        bank = self.core.terrain_runtime.scene_bank
        if (bank is not None or "terrain_contract" in state) and state.get(
            "terrain_contract"
        ) != self.terrain_contract():
            raise ValueError("Checkpoint terrain/observation contract mismatch")
        if self.protocol_enabled:
            states = state.get("planner_states", [])
            if len(states) != len(self.planners) or len(state.get("intents", [])) != len(
                self.planners
            ):
                raise ValueError("Missing or mismatched planner states/intents")
            previous_states = [p.state_dict() for p in self.planners]
            try:
                for planner, saved in zip(self.planners, states):
                    planner.load_state_dict(saved)
            except (ValueError, TypeError, KeyError):
                for planner, previous in zip(self.planners, previous_states):
                    planner.load_state_dict(previous)
                raise
        self.core.__dict__.update(copy.deepcopy(state["core"]))
        self.core.terrain_runtime.scene_bank = bank
        if bank is not None:
            bank.prefetch(self.core.terrain_runtime.scene_indices)
        for name, value in state["local"].items():
            setattr(self, name, copy.deepcopy(value))
        if self.protocol_enabled:
            self.intents = copy.deepcopy(state["intents"])
            if self.projectors:
                for projector, saved in zip(self.projectors, state["safety_states"]):
                    projector.load_state_dict(saved)
                self.execution_plans = copy.deepcopy(state["execution_plans"])
        else:
            for planner, (previous, blocked) in zip(self.planners, state["warm"]):
                planner.reset()
                planner.previous, planner.blocked = previous, blocked

    def step(
        self,
        actions,
        *,
        gamma=0.99,
        max_low_steps=5,
        deadline=float("inf"),
        enabled=None,
        hold_mask=None,
        diagnostic_callback=None,
    ):
        # hold_mask is a diagnostic request, never an Actor post-sampling override in training.
        if max_low_steps < 1:
            raise ValueError("At least one low-level step is required")
        if getattr(self, "hybrid", False):
            if hold_mask is not None:
                raise ValueError("Hybrid policy HOLD cannot be overridden by hold_mask")
            if not isinstance(actions, PolicyAction):
                raise ValueError("polar_hold_v2 requires structured PolicyAction")
            action = actions.validate(
                (self.e, self.a), self.contracts["rho_max"], self.contracts["beta_max"]
            )
            poses = self.poses()
            intents = action.decode(
                poses,
                self.core.step_count.numpy() * DT,
                self.contracts["rho_max"],
                self.contracts["beta_max"],
            )
            # Decode the entire batch before committing any state.
            self.intents = intents
            actions, holds = action.physical, action.hold
            for e in range(self.e):
                for a in range(self.a):
                    intent, pose, origin = intents[e * self.a + a], poses[e, a], self.origins[e, a]
                    self.core.committed_plan_world_subgoal[e, a, :2] = torch.as_tensor(
                        origin[:2] + rotation(origin[2]) @ intent.goal_xy
                    )
                    self.core.committed_plan_local_xy[e, a] = torch.as_tensor(
                        rotation(-pose[2]) @ (intent.goal_xy - pose[:2])
                    )
                    self.core.committed_reference_speed[e, a] = 0
                    self.core.committed_planned_yaw_delta[e, a] = 0
        else:
            holds = (
                np.zeros((self.e, self.a), dtype=bool)
                if hold_mask is None
                else np.asarray(hold_mask)
            )
            if holds.shape != (self.e, self.a) or holds.dtype != np.bool_:
                raise ValueError("hold_mask must be a boolean [env, agent] diagnostic mask")
            poses = self.poses()
            actions = (
                actions.detach().cpu().numpy() if torch.is_tensor(actions) else np.asarray(actions)
            )
            for e in range(self.e):
                for a in range(self.a):
                    i = e * self.a + a
                    self.goals[i] = LocalGoalRequest.from_action(
                        actions[e, a], poses[e, a], float(self.core.step_count[e]) * DT
                    )
                    if holds[e, a]:
                        self.goals[i] = LocalGoalRequest.for_hold(
                            poses[e, a], float(self.core.step_count[e]) * DT
                        )
                    goal = self.goals[i]
                    if getattr(self, "protocol_enabled", False):
                        self.intents[i] = LocalIntent.from_request(goal, hold=bool(holds[e, a]))
                    origin = self.origins[e, a]
                    self.core.committed_plan_world_subgoal[e, a, :2] = torch.as_tensor(
                        origin[:2] + rotation(origin[2]) @ goal.target[:2]
                    )
                    self.core.committed_plan_local_xy[e, a] = torch.as_tensor(
                        np.zeros(2) if holds[e, a] else actions[e, a, :2] * 1.6
                    )
                    self.core.committed_reference_speed[e, a] = goal.speed_cap
                    self.core.committed_planned_yaw_delta[e, a] = float(
                        0 if holds[e, a] else actions[e, a, 2] * np.pi
                    )
        self.reward_gamma = gamma
        active = (
            torch.ones(self.e, dtype=torch.bool)
            if enabled is None
            else enabled.detach().cpu().clone()
        )
        reward = torch.zeros(self.e)
        duration = torch.zeros(self.e, dtype=torch.long)
        terminated, truncated = torch.zeros_like(active), torch.zeros_like(active)
        terminal_reason = torch.zeros(self.e, dtype=torch.long)
        learning_valid = active.clone()
        terminal_actor, terminal_critic = self.observe()
        episodes, records = [], []
        for _ in range(max_low_steps):
            if duration.max() > 0 and time.monotonic() >= deadline:
                break
            poses = self.poses()
            protocol_records = []
            if getattr(self, "protocol_enabled", False):
                indices, inputs = [], []
                for e in range(self.e):
                    if not active[e]:
                        continue
                    for a in range(self.a):
                        i = e * self.a + a
                        now = float(self.core.step_count[e]) * DT
                        yaw = float(self.core.yaws[e, a])
                        vxy = self.core.velocities_xy[e, a].numpy()
                        velocity = np.array(
                            [
                                vxy @ np.array([np.cos(yaw), np.sin(yaw)]),
                                float(self.core.angular_velocities[e, a]),
                            ]
                        )
                        indices.append((e, a, i))
                        inputs.append(
                            PlannerInput(
                                e,
                                a,
                                poses[e, a],
                                velocity,
                                self.intents[i],
                                TerrainView(self.memories[i]),
                                tuple(self.neighbors(e, a, identified=True)),
                                self.commands[e, a],
                                now,
                                now + np.arange(N + 1) * DT,
                            )
                        )
                outputs = plan_batch(
                    [self.planners[i] for _, _, i in indices], inputs, pool=self.pool
                )
                commands = np.zeros_like(self.commands)
                for (e, a, i), request, output in zip(indices, inputs, outputs):
                    self.plans[i], self.feedback[i] = output.trajectory, output.feedback
                    decision = execution_passthrough(poses[e, a], output)
                    if self.projectors:
                        from .safety import SafetyInput, decision_record
                        # Public task boundary transformed by this rover's own odometry origin.
                        A = np.vstack([np.eye(2), -np.eye(2)])
                        origin = self.origins[e, a]
                        context = SafetyInput(request, A @ rotation(origin[2]),
                            np.full(4, self.core.cfg.safety.world_xy_limit)-A@origin[:2])
                        decision = self.projectors[i].project(context, decision.nominal_control)
                        self.execution_plans[i] = self.projectors[i].execution_prediction(
                            context, decision.applied_control)
                    commands[e, a] = decision.applied_control
                    self.metrics.append(output.feedback)
                    protocol_records.append(
                        {
                            "env_id": e,
                            "agent_id": a,
                            "valid": output.valid,
                            "fallback_reason": output.fallback_reason,
                            "total_cost": output.total_cost,
                            "nominal_control": decision.nominal_control.tolist(),
                            "applied_control": decision.applied_control.tolist(),
                            "safety_status": decision.safety_status,
                            **({"diagnostics": copy.deepcopy(getattr(self.planners[i], "last_diagnostics", {}))}
                               if diagnostic_callback is not None else {}),
                            **({"safety": decision_record(decision),
                                "execution_prediction_published": self.execution_plans[i] is not None}
                               if self.projectors else {}),
                        }
                    )
            else:
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
            active_np = active.numpy()
            r_np = r.detach().numpy()
            self.episode_return += np.where(active_np, r_np, 0)
            self.episode_discounted_return += np.where(
                active_np,
                gamma**self.episode_steps * r_np,
                0,
            )
            self.episode_steps += active_np
            for key, values in self.last_reward_terms.items():
                if key != "total":
                    self.episode_terms.setdefault(key, np.zeros(self.e))
                    self.episode_terms[key] += np.where(active_np, values, 0)
            if done.learning_valid is not None:
                learning_valid &= ~active | done.learning_valid
            duration += active.long()
            obs, state = self.observe()
            terminal_actor[active.to(self.policy_device)] = obs[active.to(self.policy_device)]
            terminal_critic[active.to(self.policy_device)] = state[active.to(self.policy_device)]
            ended = active & done.done
            for e in torch.where(ended)[0].tolist():
                episodes.append(
                    {
                        "env": e,
                        "success": bool(done.success[e]),
                        "collision": bool(done.collision[e]),
                        "timeout": bool(
                            done.timeout[e]
                            if self.finite_task
                            else (done.truncated[e] and not done.invalid_terrain[e])
                        ),
                        "terminal_reason": TerminalReason(
                            int(done.terminal_reason[e])
                        ).name.lower(),
                        "raw_return": float(self.episode_return[e]),
                        "discounted_return": float(self.episode_discounted_return[e]),
                        "duration_low_steps": int(self.episode_steps[e]),
                        "reward_terms": {
                            key: float(value[e]) for key, value in self.episode_terms.items()
                        },
                        "invalid_terrain": bool(done.invalid_terrain[e]),
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
                    "request": (
                        {
                            "physical": actions.tolist(),
                            "hold": holds.tolist(),
                            "version": ACTION_VERSION,
                        }
                        if getattr(self, "hybrid", False)
                        else actions.tolist()
                    ),
                    "hold_request": holds.tolist(),
                    "actual_stopped": (
                        (
                            torch.linalg.norm(self.core.velocities_xy, dim=-1)
                            <= self.core.cfg.success_thresholds.speed
                        )
                        & (
                            self.core.angular_velocities.abs()
                            <= self.core.cfg.success_thresholds.angular_speed
                        )
                    ).tolist(),
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
                    **(
                        {
                            "planner_protocol": self.contracts["planner_protocol"],
                            "planner_execution": protocol_records,
                            **({"execution_predictions": [
                                {"states": p.states.tolist(), "controls": p.controls.tolist(),
                                 "velocities": p.velocities.tolist(), "times": p.times.tolist(),
                                 "version": p.version} if p is not None else None
                                for p in self.execution_plans
                            ]} if self.projectors else {}),
                        }
                        if getattr(self, "protocol_enabled", False)
                        else {}
                    ),
                }
            )
            if diagnostic_callback is not None:
                diagnostic_callback(records[-1])
            if done.terminal_reason is not None:
                terminal_reason = torch.where(ended, done.terminal_reason, terminal_reason)
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
            reward.to(self.policy_device),
            duration.to(self.policy_device),
            terminated.to(self.policy_device),
            truncated.to(self.policy_device),
            terminal_actor,
            terminal_critic,
            obs,
            state,
            episodes,
            records,
            terminal_reason.to(self.policy_device),
            ((~terminated & learning_valid) if self.finite_task else ~terminated).to(
                self.policy_device
            ),
            (learning_valid & (duration > 0)).to(self.policy_device),
        )
