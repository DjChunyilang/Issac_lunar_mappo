#!/usr/bin/env python3
"""Unified zero-training planner replay and local closed-loop diagnostics."""

import argparse
from dataclasses import replace
import json
import time
from pathlib import Path
import resource
from dataclasses import asdict
import hashlib
import platform
import yaml

import numpy as np
import torch

from _common import cfg_from_experiment, load_yaml
from train_local_goal_nmpc import implementation_hash
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.contracts import (
    DT,
    N,
    contract_metadata,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.memory import LocalTerrainMemory
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.protocol import (
    LocalIntent,
    PlannerInput,
    TerrainView,
    IdentifiedNeighbor,
    plan_batch,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.contracts import NeighborPrediction
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.registry import (
    create_adapters,
    ADAPTERS,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.baselines import (
    PrimitiveReplayInput,
    primitive_fingerprint,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.execution import dynamics
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.contracts import rotation
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.environment import (
    LocalGoalEnvironment,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.actions import PolicyAction


def flat_view():
    memory = LocalTerrainMemory(capacity=4096)
    xy = np.stack(np.meshgrid(np.arange(-2, 2.01, 0.1), np.arange(-2, 2.01, 0.1)), axis=-1).reshape(
        -1, 2
    )
    features = np.zeros((len(xy), 5))
    features[:, 4] = 1
    memory.observe(xy, features, 0)
    return TerrainView(memory)


def fixture(goal=(0.8, 0), *, agent=0, env=0, view=None):
    return PlannerInput(
        env,
        agent,
        np.zeros(3),
        np.zeros(2),
        LocalIntent(np.array(goal), False, 0, 1),
        flat_view() if view is None else view,
        (),
        np.zeros(2),
        0,
        np.arange(N + 1) * DT,
    )


def scenario_fixture(goal, scene):
    item = fixture(goal)
    if scene in ("risk", "blocked", "unknown"):
        snapshot = item.terrain.snapshot()
        memory = LocalTerrainMemory()
        xy, features = snapshot.xy.copy(), snapshot.features.copy()
        if scene == "risk":
            features[xy[:, 1] < -0.02, 4] = 0.35
        if scene == "blocked":
            features[(xy[:, 0] > 0.4) & (xy[:, 0] < 0.7), 4] = 0.1
        if scene == "unknown":
            keep = xy[:, 0] <= 0.4
            xy, features = xy[keep], features[keep]
        memory.observe(xy, features, 0)
        item = replace(item, terrain=TerrainView(memory))
    if scene in ("cross", "head_on", "stopped", "expired"):
        t = np.arange(16) * DT
        positions = (
            np.stack((np.full(16, 0.45), 0.7 - 0.5 * t), -1)
            if scene == "cross"
            else np.stack((1 - 0.5 * t, np.zeros(16)), -1)
        )
        if scene in ("stopped", "expired"):
            positions = np.tile([0.6, 0.25], (16, 1))
        prediction = NeighborPrediction(
            positions,
            np.full(16, 0.46),
            -1 if scene == "expired" else 0,
            -0.1 if scene == "expired" else 1,
        )
        item = replace(item, neighbors=(IdentifiedNeighbor(2, prediction),))
    return item


def output_row(output):
    return {
        "valid": output.valid,
        "status": output.feedback.status,
        "reason": output.fallback_reason,
        "total_cost": output.total_cost,
        "first_control": output.trajectory.controls[0].tolist(),
        "states": output.trajectory.states.tolist(),
        "controls": output.trajectory.controls.tolist(),
        "velocities": output.trajectory.velocities.tolist(),
        "times": output.trajectory.times.tolist(),
        "prefix_steps": output.feedback.prefix_steps,
        "seconds": output.feedback.solve_seconds,
        "safety_status": "projection_disabled",
    }


def sample_artifact(planner, directory, index):
    samples = getattr(planner, "last_samples", None)
    if samples is None or directory is None:
        return None
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"samples_{index:04d}.npz"
    if path.exists():
        raise FileExistsError("Refusing to overwrite samples")
    np.savez_compressed(path, **{k: v.numpy() for k, v in samples.items()})
    return str(path)


def run(
    raw_path,
    adapter,
    *,
    samples=None,
    seed=23,
    steps=0,
    primitive=None,
    cache_dir=None,
    goal=(0.8, 0),
    artifact_dir=None,
    scene="flat",
    requests=None,
):
    torch.set_num_threads(1)
    cfg = cfg_from_experiment(raw_path)
    c = cfg.local_goal_contract
    c.planner_adapter, c.planner_revision = adapter, ADAPTERS[adapter][0]
    c.planner_seed = seed
    if samples is not None:
        c.mppi_samples = samples
    payload = None if requests is None else json.loads(Path(requests).read_text())
    if payload is not None:
        if not isinstance(payload, list) or not payload or steps:
            raise ValueError("Replay requires a nonempty ordered request list and steps=0")
        identities = list(
            dict.fromkeys((p.get("env_id", 0), p.get("agent_id", 0)) for p in payload)
        )
    else:
        identities = [(0, 0)]
    planners = create_adapters(len(identities), cache_dir, cfg.terrain.slope_speed_scale, c)
    planner = planners[0]
    item = scenario_fixture(goal, scene)
    if adapter.startswith("mppi_"):
        item = replace(item, terrain=item.terrain.conservative())
    rows = []
    count = range(47) if primitive == "all" else [int(primitive or 0)]
    if payload is not None:
        for index, p in enumerate(payload):
            pose = np.asarray(p.get("pose", [0, 0, 0]), dtype=float)
            now = p.get("now", 0)
            intent = (
                LocalIntent.for_hold(pose, now)
                if p.get("hold", False)
                else LocalIntent(
                    np.asarray(p["goal"]),
                    False,
                    p.get("timestamp", now),
                    p.get("expires_at", now + 1),
                    p.get("terminal_yaw"),
                    p.get("speed_cap"),
                )
            )
            if p.get("hold", False):
                intent = replace(intent, expires_at=p.get("expires_at", now + 1))
            identity = (p.get("env_id", 0), p.get("agent_id", 0))
            planner = planners[identities.index(identity)]
            replay = replace(
                item,
                env_id=identity[0],
                agent_id=identity[1],
                ego_pose=pose,
                ego_velocity=np.asarray(p.get("velocity", [0, 0])),
                previous_control=np.asarray(p.get("previous_control", [0, 0])),
                intent=intent,
                now=now,
                times=now + np.arange(16) * DT,
            )
            request = (
                PrimitiveReplayInput(replay, p["primitive_id"])
                if adapter == "primitive47_adapter_v1"
                else replay
            )
            output = planner.plan(request)
            rows.append(
                {
                    "request_index": index,
                    "env_id": identity[0],
                    "agent_id": identity[1],
                    **output_row(output),
                    "diagnostics": getattr(planner, "last_diagnostics", {}),
                    "sample_artifact": sample_artifact(planner, artifact_dir, index),
                }
            )
    elif adapter == "primitive47_adapter_v1":
        for index in count:
            planner.reset()
            out = planner.plan(PrimitiveReplayInput(item, index))
            rows.append({"primitive_id": index, **output_row(out)})
    else:
        rows.append(output_row(plan_batch(planners, [item])[0]))
        rows[-1]["diagnostics"] = getattr(planner, "last_diagnostics", {})
        rows[-1]["sample_artifact"] = sample_artifact(planner, artifact_dir, 0)
    loop = []
    if steps:
        if adapter == "primitive47_adapter_v1":
            raise ValueError("Primitive closed-loop requires explicit native command schedule")
        planner.reset()
        item = scenario_fixture(goal, scene)
        if adapter.startswith("mppi_"):
            item = replace(item, terrain=item.terrain.conservative())
        for k in range(steps):
            # Renew the same fixed odometry goal; do not move it with ego.
            if k % 5 == 0:
                item = replace(
                    item, intent=replace(item.intent, timestamp=item.now, expires_at=item.now + 1)
                )
            out = plan_batch(planners, [item])[0]
            u = out.trajectory.controls[0]
            feature = (
                item.terrain.lookup(item.ego_pose[:2])[0][0]
                if getattr(planner, "flat", False) is False
                else np.array([0, 0, 0, 0, 1])
            )
            next_pose, velocity = dynamics(item.ego_pose, u, feature, cfg.terrain.slope_speed_scale)
            loop.append({"now": item.now, **output_row(out), "actual_pose": next_pose.tolist()})
            loop[-1]["diagnostics"] = getattr(planner, "last_diagnostics", {})
            loop[-1]["sample_artifact"] = sample_artifact(planner, artifact_dir, k + 1)
            item = replace(
                item,
                ego_pose=next_pose,
                ego_velocity=velocity,
                previous_control=u,
                now=(k + 1) * DT,
                times=((k + 1) + np.arange(N + 1)) * DT,
            )
            if np.linalg.norm(next_pose[:2] - item.intent.goal_xy) <= 0.2:
                break
    return {
        "config_source": str(raw_path),
        "contracts": contract_metadata(cfg),
        "seed": seed,
        "implementation_sha256": implementation_hash(),
        "training_interactions": 0,
        "optimizer_updates": 0,
        "observation_source": "given_legal_local_observations",
        "scene": scene,
        "request_replay": requests,
        "map_identity": "local_fixture_grid_v1",
        "rows": rows,
        "local_loop": loop,
        "final_error": None
        if not loop
        else float(np.linalg.norm(item.ego_pose[:2] - item.intent.goal_xy)),
        "primitive_library_sha256": primitive_fingerprint(),
        "cuda_validation": "deferred",
        "max_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    }


def sensor_run(
    raw_path,
    adapter,
    *,
    samples=128,
    seed=23,
    steps=100,
    goal=(0.8, 0),
    artifact_dir=None,
    cache_dir=None,
    scene="flat",
):
    """Fixed odometry goals through the real sensor, communication and physics chain."""
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    cfg = cfg_from_experiment(raw_path)
    cfg.simulation.num_envs = 2
    if scene == "flat":
        # Configure a flat backend, then use its existing sensor pipeline.
        # No planner memory is replaced or filled from a test grid.
        cfg.terrain.type = "flat_proxy"
        cfg.terrain.amplitude = 0
        cfg.terrain.crater_count = 0
        cfg.terrain.topology_profile = "open"
    elif scene != "inherited":
        raise ValueError("Sensor diagnostics support flat or inherited terrain")
    c = cfg.local_goal_contract
    c.planner_adapter, c.planner_revision = adapter, ADAPTERS[adapter][0]
    c.mppi_samples, c.planner_seed = samples, seed
    env = LocalGoalEnvironment(cfg, cache_dir)
    rows = []
    collision = False
    limit_violation = False
    fixed = env.poses()[..., :2] + np.asarray(goal)
    reached = np.zeros((2, 4), bool)
    try:
        for k in range(min(100, steps)):
            pose = env.poses()
            physical = np.zeros((2, 4, 2))
            for e in range(2):
                for a in range(4):
                    delta = rotation(-pose[e, a, 2]) @ (fixed[e, a] - pose[e, a, :2])
                    physical[e, a] = [
                        min(np.linalg.norm(delta), 1.6),
                        np.arctan2(delta[1], delta[0]),
                    ]
            physical[reached] = 0  # canonical continuous placeholders for HOLD
            # Renew the same fixed goal. A rover that reached its goal requests HOLD.
            previous = env.commands.copy()
            transition = env.step(
                PolicyAction(torch.tensor(physical, dtype=torch.float32), torch.tensor(reached)),
                max_low_steps=1,
            )
            collision |= any(e["collision"] for e in transition.episodes)
            commands = np.asarray(transition.records[0]["commands"])
            wheel = commands[..., 0, None] + commands[..., 1, None] * np.array([-0.188, 0.188])
            limit_violation |= bool(
                (abs(commands[..., 0]) > 1.1501).any()
                or (abs(commands[..., 1]) > 2.4001).any()
                or (abs(wheel) > 1.7641).any()
                or (abs(commands - previous) > np.array([0.3001, 0.6001])).any()
            )
            if transition.episodes:
                rows.append(
                    {"step": k, "episodes": transition.episodes, "ended_before_local_goal": True}
                )
                break
            reached |= np.linalg.norm(env.poses()[..., :2] - fixed, axis=-1) <= 0.2
            sample_paths = []
            for i, p in enumerate(env.planners):
                sample_paths.append(sample_artifact(p, artifact_dir, k * 8 + i))
            rows.append(
                {
                    "step": k,
                    "records": transition.records,
                    "diagnostics": [getattr(p, "last_diagnostics", {}) for p in env.planners],
                    "sample_artifacts": sample_paths,
                    "errors": np.linalg.norm(env.poses()[..., :2] - fixed, axis=-1).tolist(),
                }
            )
            if reached.all() and abs(env.commands).max() < 1e-6:
                break
        error = np.linalg.norm(env.poses()[..., :2] - fixed, axis=-1)
        return {
            "contracts": contract_metadata(cfg),
            "seed": seed,
            "training_interactions": 0,
            "optimizer_updates": 0,
            "implementation_sha256": implementation_hash(),
            "observation_source": "existing_sensor_communication_chain",
            "map_identity": {"terrain_config": asdict(cfg.terrain), "seed": seed},
            "scene": scene,
            "batch_size": 8,
            "samples": samples,
            "fixed_goals": fixed.tolist(),
            "rows": rows,
            "final_errors": error.tolist(),
            "all_reached": bool(reached.all()),
            "collision": collision,
            "limit_violation": limit_violation,
            "local_pass": bool((error <= 0.2).all() and not collision and not limit_violation),
            "sensor_memory_counts": [len(m.samples) for m in env.memories],
            "cuda_validation": "deferred",
            "max_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        }
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment/t05_planner_baselines.yaml")
    parser.add_argument("--planner", choices=ADAPTERS, default="nmpc_polar_v1")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--primitive", default="all")
    parser.add_argument("--sensor-env", action="store_true")
    parser.add_argument("--goal", type=float, nargs=2, default=(0.8, 0))
    parser.add_argument(
        "--scene",
        choices=(
            "flat",
            "risk",
            "blocked",
            "unknown",
            "cross",
            "head_on",
            "stopped",
            "missing",
            "expired",
            "inherited",
        ),
        default="flat",
    )
    parser.add_argument("--requests", type=Path)
    args = parser.parse_args()
    if (
        not 0 <= args.steps <= 100
        or (args.samples is not None and args.samples < 1)
        or args.seed < 0
    ):
        raise ValueError("Expected steps in [0,100], positive samples and nonnegative seed")
    if args.sensor_env and args.requests is not None:
        raise ValueError("Sensor integration and fixture request replay are separate diagnostics")
    if not args.sensor_env and args.scene == "inherited":
        raise ValueError("Inherited terrain requires the sensor environment")
    if args.output.exists():
        raise FileExistsError("Refusing to overwrite diagnostics")
    start = time.monotonic()
    source_start = implementation_hash()
    options = dict(
        samples=args.samples,
        seed=args.seed,
        steps=args.steps,
        goal=args.goal,
        artifact_dir=args.output.parent / (args.output.stem + "_samples"),
        cache_dir=args.output.parent.parent / "solver_cache",
    )
    if args.sensor_env:
        if args.planner == "primitive47_adapter_v1":
            raise ValueError("Native primitives use explicit replay")
        options["samples"] = 128 if args.samples is None else args.samples
        result = sensor_run(args.config, args.planner, scene=args.scene, **options)
    else:
        result = run(
            args.config,
            args.planner,
            primitive=args.primitive,
            scene=args.scene,
            requests=None if args.requests is None else str(args.requests),
            **options,
        )
    result["seconds"] = time.monotonic() - start
    result["resolved_config"] = load_yaml(args.config)
    result["resolved_config"]["local_goal"].update(
        planner_adapter=args.planner,
        planner_revision=ADAPTERS[args.planner][0],
        planner_seed=args.seed,
    )
    if args.samples is not None:
        result["resolved_config"]["local_goal"]["mppi_samples"] = args.samples
    if args.sensor_env:
        result["resolved_config"]["experiment"]["num_envs"] = 2
        result["resolved_config"]["terrain"] = result["map_identity"]["terrain_config"]
        result["resolved_config"]["local_goal"]["mppi_samples"] = options["samples"]
    result["implementation_start_sha256"] = source_start
    result["source_stable_during_run"] = source_start == implementation_hash()
    result["diagnostic_script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result["config_sha256"] = hashlib.sha256(
        json.dumps(result["resolved_config"], sort_keys=True).encode()
    ).hexdigest()
    result["host"] = {
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False))
    run_dir = args.output.parent.parent
    manifest = run_dir / "run_manifest.json"
    if not manifest.exists():
        config_dir = run_dir / "config"
        config_dir.mkdir(parents=True, exist_ok=True)
        (config_dir / "experiment.yaml").write_text(
            yaml.safe_dump(result["resolved_config"], sort_keys=False)
        )
        manifest.write_text(
            json.dumps(
                {
                    "entry": "diagnose_planners",
                    "seed": args.seed,
                    "contracts": result["contracts"],
                    "evidence": str(args.output),
                    "source_sha256": source_start,
                    "config_sha256": result["config_sha256"],
                    "training_interactions": 0,
                    "optimizer_updates": 0,
                    "cuda_validation": "deferred",
                },
                indent=2,
            )
        )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "rows": len(result["rows"]),
                "final_error": result.get("final_error"),
                "local_pass": result.get("local_pass"),
            }
        )
    )


if __name__ == "__main__":
    main()
