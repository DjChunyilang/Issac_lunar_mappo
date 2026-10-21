#!/usr/bin/env python3
"""Hardware, local-input and synchronized MPPI acceptance; never trains."""
import argparse
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import resource
import time
from unittest.mock import patch
import xml.etree.ElementTree as ET

import numpy as np
import torch

from _common import cfg_from_experiment
from planner_evidence import EvidenceRun, json_write, source_snapshot
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal import mppi, tensor_views
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.benchmark import FAMILIES, make_case, to_input
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.execution import validate_trajectory
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.memory import LocalTerrainMemory
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.protocol import TerrainView


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def noise_for(config, batch, seed):
    return (np.random.default_rng(seed).normal(size=(config.iterations, batch, config.samples, 15, 2))
            * [config.noise_v, config.noise_w]).astype(np.float32)


def synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def measured_call(kernel, requests, noise):
    handles = [mppi.MPPIAdapter(kernel, seed=23) for _ in requests]
    synchronize(kernel.device)
    start = time.perf_counter()
    outputs = kernel.plan_batch(handles, requests, noise=noise)
    synchronize(kernel.device)
    elapsed = time.perf_counter() - start
    valid = []
    for item, output in zip(requests, outputs):
        trajectory = output.trajectory
        finite = all(np.isfinite(value).all() for value in
                     (trajectory.states, trajectory.controls, trajectory.velocities, trajectory.times))
        violations = validate_trajectory(
            trajectory, item.terrain, [p.prediction for p in item.neighbors],
            1.15 if item.intent.speed_cap is None else item.intent.speed_cap, item.previous_control,
        ) if output.valid else []
        valid.append(finite and not violations)
    return elapsed, outputs, handles, all(valid)


def panel(config, requests, noise, repeats):
    started = time.perf_counter()
    kernel = mppi.MPPIKernel(config)
    construction_s = time.perf_counter() - started
    warmup_s, _, _, warm_ok = measured_call(kernel, requests, noise)
    if kernel.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(kernel.device)
    durations, passed = [], warm_ok
    for _ in range(repeats):
        seconds, outputs, handles, correct = measured_call(kernel, requests, noise)
        durations.append(seconds)
        passed &= correct
    row = {
        "planner_metadata": config.metadata(), "batch_size": len(requests), "samples": config.samples,
        "construction_s": construction_s, "warmup_s": warmup_s, "seconds": durations,
        "p50_s": float(np.median(durations)), "throughput_instances_s": len(requests) / float(np.median(durations)),
        "p50_p95_p99_s": np.percentile(durations, [50, 95, 99]).tolist(),
        "percentile_scope": "Three or more calls per panel; descriptive only, not a latency-tail certificate",
        "rss_peak_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20 if kernel.device.type == "cuda" else None,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20 if kernel.device.type == "cuda" else None,
        "valid": [o.valid for o in outputs], "feedback": [o.feedback.status for o in outputs],
        "fallback_reason": [o.fallback_reason for o in outputs],
        "selected_controls": [o.trajectory.controls.tolist() for o in outputs],
        "selected_states": [o.trajectory.states.tolist() for o in outputs],
        "passed": bool(passed), "noise_sha256": hashlib.sha256(noise.tobytes()).hexdigest(),
        "timing_scope": "Warmed kernels, cold handles; synchronized full call including transfers and final CPU validation",
        "input_identities": [[r.env_id, r.agent_id] for r in requests],
        "neighbor_sender_ids": [[p.sender_id for p in r.neighbors] for r in requests],
        "input_layout": "B independent environments, one identical ego per environment; actual four-agent integration is separate",
    }
    return row, handles


@contextmanager
def stage_barriers(kernel):
    """A separate intrusive profile; barriers are not used for main latency panels."""
    totals = defaultdict(float)
    stack = []
    targets = [(mppi, "rollout_batch", "prediction_s"), (mppi, "base_cost", "base_cost_s"),
               (mppi, "importance_update", "update_s"), (tensor_views, "terrain_cost", "terrain_cost_s"),
               (tensor_views, "hard_validity", "sample_review_s"),
               (tensor_views.NeighborBatch, "cost", "neighbor_cost_s"),
               (mppi.MPPIAdapter, "finish", "final_host_review_s"),
               (mppi.MPPIKernel, "tensor", "tensor_conversion_and_transfer_s")]

    def wrapped(original, name):
        def call(*args, **kwargs):
            synchronize(kernel.device)
            start = time.perf_counter()
            result = original(*args, **kwargs)
            synchronize(kernel.device)
            totals[name] += time.perf_counter() - start
            return result
        return call

    try:
        for owner, attribute, name in targets:
            context = patch.object(owner, attribute, wrapped(getattr(owner, attribute), name))
            context.__enter__()
            stack.append(context)
        yield totals
    finally:
        for context in reversed(stack):
            context.__exit__(None, None, None)


def memory_boundary(config):
    kernel = mppi.MPPIKernel(config)
    axis = np.arange(64) * .1 - 3.15
    xy = np.stack(np.meshgrid(axis, axis), -1).reshape(-1, 2)
    features = np.zeros((4096, 5)); features[:, 4] = 1
    memory = LocalTerrainMemory(4096, .05, .155)
    memory.observe(xy, features, 0)
    view = TerrainView(memory).conservative()
    if len(view.snapshot().xy) != 4096:
        raise AssertionError("Memory capacity fixture must retain all 4096 local samples")
    batch = tensor_views.TerrainBatch([view] * 64, kernel.tensor, config.query_chunk,
                                     memory_budget_bytes=config.metadata()["terrain_query_memory_budget_bytes"])
    points = np.zeros((64, 128, 2)); points[:, -1] = [10, 10]
    torch.cuda.reset_peak_memory_stats()
    synchronize(kernel.device); begin = time.perf_counter()
    values, known, direct = batch.lookup(kernel.tensor(points))
    synchronize(kernel.device)
    correct = bool(torch.isfinite(values).all() and known[:, :-1].all() and not known[:, -1].any())
    return {"batch_size": 64, "memory_samples": 4096, "query_points_per_instance": 128,
            "configured_query_chunk": config.query_chunk, "effective_query_chunk": batch.chunk,
            "max_query_block": batch.max_query_block, "passed": correct,
            "seconds": time.perf_counter() - begin, "known_count": int(known.sum()),
            "direct_count": int(direct.sum()), "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
            "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
            "scope": "Maximum local-memory capacity query, not a full B64/K2048 planning latency"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment/t09_planner_benchmark_cuda_v2.yaml")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--cuda-junit", required=True, type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Actual CUDA required; CPU fallback forbidden")
    if args.repeats < 3:
        raise ValueError("Acceptance requires at least three timed calls per panel")
    torch.set_num_threads(1)
    cfg = cfg_from_experiment(args.config)
    config = mppi.MPPIConfig.from_contract(cfg.local_goal_contract, cfg.terrain.slope_speed_scale)
    if config.adapter != "mppi_terrain_neighbor_stable_v2" or config.device != "cuda" or not cfg.local_goal_contract.training_paused:
        raise ValueError("Stable independent CUDA diagnostic configuration required")
    tests = list(ET.parse(args.cuda_junit).iter("testcase"))
    test_sources = {"tests/test_mppi_cuda.py": sha("tests/test_mppi_cuda.py")}
    provenance = json.loads(args.cuda_junit.with_suffix(".source.json").read_text())
    if provenance != {"source_snapshot": source_snapshot(), "test_sources": test_sources}:
        raise ValueError("CUDA JUnit source or test provenance mismatch")
    no_skip = bool(tests) and all(t.find("skipped") is None for t in tests)
    numeric_pass = bool(tests) and all(t.find("failure") is None and t.find("error") is None for t in tests)
    if not no_skip or not numeric_pass:
        raise ValueError("Required CUDA contracts did not all pass")
    run = EvidenceRun(args.run_dir, args.config, settings={"suite": "mppi_cuda_acceptance_v1", "repeats": args.repeats,
                       "samples": [512, 1024, 2048], "batches": [1, 4, 16, 64], "memory_capacity": 4096})
    run.resolved(cfg)
    json_write(run.path / "config/hardware.json", {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
               "runtime": torch.version.cuda, "memory_mib": torch.cuda.get_device_properties(0).total_memory / 2**20,
               "capability": list(torch.cuda.get_device_capability()), "torch_threads": torch.get_num_threads()})
    panels, pairs, complex_rows = [], [], []
    for family in ("flat", "slope", "cross"):
        case = make_case(family, 0, 23)
        json_write(run.path / f"config/{family}.json", case)
        for samples in (512, 1024, 2048):
            current = replace(config, samples=samples)
            for batch_size in (1, 4, 16, 64):
                requests = [to_input(case, identity=(i, 0)) for i in range(batch_size)]
                for request in requests:
                    request.validate()
                noise = noise_for(current, batch_size, 23)
                gpu, handles = panel(current, requests, noise, args.repeats)
                tag = f"{family}_b{batch_size}_k{samples}"
                json_write(run.path / f"metrics/panel_{tag}.json", gpu)
                if handles[0].last_samples is not None:
                    np.savez_compressed(run.path / f"samples/{tag}_first_instance.npz",
                                        **{k: v.numpy() for k, v in handles[0].last_samples.items()})
                panels.append({"family": family, "samples": samples, "batch_size": batch_size, "passed": gpu["passed"]})
                if batch_size == 1:
                    cpu, _ = panel(replace(current, device="cpu"), requests, noise, args.repeats)
                    json_write(run.path / f"metrics/cpu_{tag}.json", cpu)
                    errors = {name: float(np.max(np.abs(np.array(cpu[name]) - np.array(gpu[name]))))
                              for name in ("selected_controls", "selected_states")}
                    matched = all(np.allclose(cpu[name], gpu[name], atol=1e-5, rtol=1e-5)
                                  for name in ("selected_controls", "selected_states"))
                    pairs.append({"tag": tag, "passed": matched and cpu["valid"] == gpu["valid"] and cpu["feedback"] == gpu["feedback"],
                                  "errors": errors, "speedup": cpu["p50_s"] / gpu["p50_s"]})
                print(json.dumps({"phase": "performance", "tag": tag, "passed": gpu["passed"],
                                  "p50_s": gpu["p50_s"], "allocated_mib": gpu["peak_allocated_mib"]}), flush=True)
    for family in FAMILIES:
        for seed in (23, 31, 47):
            case = make_case(family, 1, seed)
            current = replace(config, samples=128)
            requests = [to_input(case)]
            noise = noise_for(current, 1, seed)
            outputs = {}
            for device in ("cpu", "cuda"):
                _, value, _, correct = measured_call(mppi.MPPIKernel(replace(current, device=device)), requests, noise)
                outputs[device] = (value[0], correct)
            cpu, gpu = outputs["cpu"][0], outputs["cuda"][0]
            passed = all(v[1] for v in outputs.values()) and (cpu.valid, cpu.feedback.status) == (gpu.valid, gpu.feedback.status)
            passed &= all(np.allclose(getattr(cpu.trajectory, name), getattr(gpu.trajectory, name), atol=1e-5, rtol=1e-5)
                          for name in ("states", "controls", "velocities"))
            complex_rows.append({"family": family, "seed": seed, "samples": 128, "passed": bool(passed),
                                 "gpu_valid": gpu.valid, "feedback": gpu.feedback.status,
                                 "max_control_error": float(np.max(abs(cpu.trajectory.controls - gpu.trajectory.controls)))})
    json_write(run.path / "metrics/complex_inputs.json", {"rows": complex_rows, "pairs": pairs})
    capacity = memory_boundary(config)
    json_write(run.path / "metrics/memory_capacity.json", capacity)
    profiler = mppi.MPPIKernel(config)
    with stage_barriers(profiler) as stages:
        elapsed, _, _, correct = measured_call(profiler, [to_input(make_case("cross", 0, 23))], noise_for(config, 1, 23))
    json_write(run.path / "metrics/synchronized_stages.json", {"seconds": elapsed, "stages": dict(stages), "passed": correct,
               "scope": "Separate intrusive diagnostic call with stage barriers; not ordinary latency or GPU-only kernel time"})
    required = [{"samples": k, "batch_size": b, "passed": all(row["passed"] for row in panels
                 if row["samples"] == k and row["batch_size"] == b)} for k in (512, 1024, 2048) for b in (1, 4, 16, 64)]
    checks = {"numeric_test_pass": numeric_pass, "no_skip": no_skip, "matrix_complete": len(panels) == 36 and all(p["passed"] for p in required),
              "complex_inputs": len(complex_rows) == 27 and all(r["passed"] for r in complex_rows) and all(p["passed"] for p in pairs),
              "calibration": any("float64" in t.attrib["name"] for t in tests),
              "snapshot": any("snapshot" in t.attrib["name"] or "restore" in t.attrib["name"] for t in tests),
              "device": capacity["passed"] and correct, "precision": config.dtype == "float32" and config.metadata()["accumulation_dtype"] == "float64"}
    artifacts = [{"path": str(path.resolve()), "sha256": sha(path)} for path in sorted((run.path / "metrics").glob("*.json"))]
    result = {"schema_version": "mppi_cuda_acceptance_v1", "passed": all(checks.values()), "source_snapshot": source_snapshot(),
              "test_sources": test_sources, "planner_metadata": config.metadata(), "checks": checks, "panels": required,
              "complex_families": list(FAMILIES), "contract_junit": {"path": str(args.cuda_junit.resolve()), "sha256": sha(args.cuda_junit)},
              "manifest": {"path": str((run.path / 'run_manifest.json').resolve()), "sha256": sha(run.path / "run_manifest.json")},
              "artifacts": artifacts, "performance_panels": len(panels), "complex_pairs": len(complex_rows),
              "sampling_coverage": "Final iteration of first instance per performance panel; HOLD has no samples",
              "formal_training": "paused", "optimizer_updates": 0, "full_t09_t10": "Separate staged acceptance"}
    json_write(run.path / "metrics/strict_acceptance.json", result)
    print(json.dumps({"passed": result["passed"], "checks": checks}), flush=True)
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
