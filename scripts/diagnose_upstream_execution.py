#!/usr/bin/env python3
"""S01 no-learning execution witnesses; does not claim S03/S05 acceptance."""

import argparse
from dataclasses import asdict
import math
from pathlib import Path
import time

import numpy as np
import torch

from _common import cfg_from_experiment
from planner_evidence import EvidenceRun, json_write, source_snapshot
from upstream_evidence import require_contract_tests
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.protocol import TerrainSnapshot
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_execution import control_error
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_footprint import FootprintTerrain, admit_control
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_proxy import UpstreamActuatorProxy

TEST_FILES = ("tests/test_upstream_spec.py", "tests/test_upstream_execution.py",
              "tests/test_config_wiring.py", "tests/test_cuda_evidence_gates.py")
REVIEW_V2_TEST_FILES = (*TEST_FILES, "tests/test_upstream_review_v2.py")


def serial(value):
    if isinstance(value, np.ndarray):return value.tolist()
    if isinstance(value, np.generic):return value.item()
    if isinstance(value, dict):return {key: serial(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):return [serial(item) for item in value]
    return value


def synthetic_local_frame(pose, spec, now, *, query_version="scalar_ties_v1"):
    """Explicitly provided flat observations, not the S03 sensor chain."""
    p = spec.perception
    x = np.linspace(-p.reverse, p.forward, p.frame_shape[0])
    y = np.linspace(-p.lateral, p.lateral, p.frame_shape[1])
    xx, yy = np.meshgrid(x, y, indexing="ij")
    local = np.c_[xx.ravel(), yy.ravel()]
    c, s = np.cos(pose[2]), np.sin(pose[2])
    xy = local @ np.array([[c, s], [-s, c]]) + pose[:2]
    features = np.zeros((len(xy), 5)); features[:, 4] = 1.
    return FootprintTerrain(TerrainSnapshot(xy, features, np.full(len(xy), now), p.support_radius), spec, query_version=query_version)


def diagnose(cfg, run):
    if cfg.clock_mode != "offline_simulated_latency" or cfg.planner_device != "cpu":
        raise ValueError("S01 actuator probe requires explicit CPU/offline diagnostics; realtime and planner CUDA remain pending")
    spec = cfg.spec
    initial = np.zeros((cfg.simulation.num_envs, 4, 3))
    initial[:, :, 1] = np.arange(4) * 10.
    core = UpstreamActuatorProxy(cfg, initial)
    records = []; cases = []
    def advance(target):
        record = core.step(target)
        for command, previous in zip(record["applied_control"].reshape(-1, 2), record["previous_control"].reshape(-1, 2)):
            if control_error(command, previous, spec):raise ValueError("Actual applied actuator violation")
        if record["geometry_overlap"].any() or record["clearance_violation"].any():
            raise ValueError("Actual footprint envelope overlap in diagnostic")
        records.append(serial(record))
        return record
    for speed in (spec.initial_test_speed, 1.5, 2., spec.vehicle.forward_limit):
        target = np.zeros_like(core.previous_control); target[..., 0] = speed
        for _ in range(math.ceil(spec.vehicle.forward_limit / (spec.vehicle.linear_acceleration * spec.grid.dt)) + 2):
            if np.allclose(core.actual_velocity[..., 0], speed, atol=1e-10, rtol=0):break
            advance(target)
        if not np.allclose(core.actual_velocity[..., 0], speed, atol=1e-10, rtol=0):
            raise ValueError("Declared speed never actually reached")
        pose = core.poses[0, 0].copy(); now = float(core.times[0])
        terrain = synthetic_local_frame(pose, spec, now, query_version=cfg.terrain_query_version)
        started = time.perf_counter()
        decision = admit_control([speed, 0.], core.previous_control[0, 0], pose, terrain, spec,
                                 information_time=now, latency=cfg.simulated_latency, review_version=cfg.execution_review_version)
        elapsed = time.perf_counter() - started
        if not decision.verified:raise ValueError(f"Stop corridor rejected speed {speed}: {decision.reason}")
        envelope = decision.envelope
        # Explicit actual held-control steps account for the injected delay once.
        for _ in range(envelope.delay_steps):advance(core.previous_control.copy())
        if not np.allclose(core.poses[0, 0], envelope.trajectory.states[envelope.delay_steps], atol=1e-8, rtol=1e-8):
            raise ValueError("Injected delay disagrees with information-time reserve")
        activation_record = advance(target)
        if not math.isclose(float(activation_record["activation_time"][0]), envelope.activation_time, abs_tol=1e-8):
            raise ValueError("Delay counted twice or wrong activation timestamp")
        json_write(run.path / "samples" / f"stop_reserve_{len(cases):02d}.json", serial(asdict(envelope)))
        cases.append({"requested_speed": speed, "actual_reached_speed": float(core.actual_velocity[0, 0, 0]),
                      "information_time": now, "actual_activation_time": envelope.activation_time,
                      "stop_review_wall_time_s": elapsed, "sensor_samples": spec.perception.sample_count,
                      "reserve_steps": len(envelope.trajectory.controls), "fits_horizon": envelope.fits_horizon,
                      "conservative_center_distance_m": envelope.conservative_center_distance,
                      "input_kind": "provided_synthetic_local_observation", "full_stop_review": True})
    hot = core.state_dict(); run.snapshot("full_speed_hot_state.pkl", hot)
    restored = UpstreamActuatorProxy(cfg, initial); restored.load_state_dict(hot)
    before_stop = core.poses.copy()
    for _ in range(100):
        if not core.previous_control.any():break
        advance(np.zeros_like(core.previous_control)); restored.step(np.zeros_like(restored.previous_control))
    if core.previous_control.any() or core.actual_velocity.any():raise ValueError("HOLD did not actually stop")
    for name in ("poses", "previous_control", "actual_velocity", "times"):
        if not np.array_equal(getattr(core, name), getattr(restored, name)):
            raise ValueError("Hot-state restore changed actual brake execution")
    stop_displacement = np.linalg.norm(core.poses - before_stop, axis=-1)
    untouched = core.state_dict()
    core.reset([0], initial[:1])
    if cfg.simulation.num_envs > 1 and not np.array_equal(core.poses[1:], untouched["poses"][1:]):
        raise ValueError("Asynchronous reset affected another environment")
    json_write(run.path / "metrics/execution_records.json", records)
    return {"stage": "S01_shared_execution_foundation", "actual_speed_cases": cases,
            "actual_stop_displacement_m": stop_displacement.tolist(), "actual_stopped": True,
            "batch_size": cfg.simulation.num_envs * 4, "hot_restore_exact": True,
            "async_reset_isolated": True, "steps_recorded": len(records),
            "zero_actuator_violations": True, "zero_geometry_overlap": True,
            "information_boundary": "provided_local_observations; full S03 chain not integrated",
            "safety_projection": "S02 integration pending", "dynamic_limiter": "stop_corridor_v1_foundation",
            "formal_training": "paused", "physx": "not_validated"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment/upstream_scale_s01_review_v2.yaml")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--junit", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    cfg = cfg_from_experiment(args.config)
    run = EvidenceRun(args.run_dir, args.config, resume=args.resume,
                      settings={"stage": "S01_shared_execution_foundation", "torch_threads": 1})
    strict = cfg.execution_review_version == "footprint_review_v2"
    junit = require_contract_tests(args.junit, run, REVIEW_V2_TEST_FILES if strict else TEST_FILES)
    run.resolved(cfg)
    summary = diagnose(cfg, run)
    rejection = None
    if strict:
        from profile_upstream_footprint import reproductions
        rejection = reproductions(cfg.spec)
        if any(value["reproduced"] for value in rejection.values()):
            raise ValueError("Required review counterexample still accepted")
    if source_snapshot() != run.identity["source_snapshot"]:
        raise ValueError("Source changed during execution diagnostic")
    json_write(run.path / "metrics/summary.json", summary)
    json_write(run.path / "metrics/strict_acceptance.json", {
        "s00_contract_foundation_pass": True, "s01_shared_execution_foundation_pass": True,
        "s01_full_environment_integration_pass": False, "s02_planners_pass": False,
        "s03_perception_communication_task_pass": False, "s05_gate_b_pass": False,
        "contracts": cfg.metadata, "contract_tests": junit, "training_interactions": 0,
        "optimizer_updates": 0, "formal_training": "paused"})
    if strict:
        json_write(run.path / "metrics/review_v2_rejections.json", rejection)
        json_write(run.path / "metrics/review_v2_acceptance.json", {
            "geometry_rejection_pass": True, "terrain_dynamics_rejection_pass": True,
            "query_version": cfg.terrain_query_version, "review_version": cfg.execution_review_version,
            "realtime_pass": False, "full_environment_integration_pass": False,
            "contract_tests": junit, "formal_training": "paused"})
    print({"actual_speed_cases": [c["actual_reached_speed"] for c in summary["actual_speed_cases"]],
           "batch_size": summary["batch_size"], "actual_stopped": True, "tests": junit["tests"],
           "s01_full_integration": "pending"})


if __name__ == "__main__":
    main()
