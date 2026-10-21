#!/usr/bin/env python3
"""Synchronized, no-learning MPPI device/chunk measurements on frozen local inputs."""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from _common import cfg_from_experiment
from planner_evidence import EvidenceRun, json_write
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.benchmark import make_case, to_input
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.mppi import MPPIAdapter, MPPIConfig, MPPIKernel


def measure(config, requests, noise, repeats):
    start = time.perf_counter()
    kernel = MPPIKernel(config)
    handles = [MPPIAdapter(kernel, seed=23) for _ in requests]
    construction_s = time.perf_counter() - start

    def call():
        for handle in handles:
            handle.reset()
        if kernel.device.type == "cuda":
            torch.cuda.synchronize(kernel.device)
        begin = time.perf_counter()
        outputs = kernel.plan_batch(handles, requests, noise=noise)
        if kernel.device.type == "cuda":
            torch.cuda.synchronize(kernel.device)
        return time.perf_counter() - begin, outputs

    warmup_s, _ = call()
    if kernel.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(kernel.device)
    durations = []
    for _ in range(repeats):
        elapsed, outputs = call()
        durations.append(elapsed)
    row = {
        "config": asdict(config), "batch": len(requests), "repeats": repeats,
        "construction_s": construction_s, "warmup_s": warmup_s,
        "seconds": durations, "p50_s": float(np.median(durations)),
        "throughput_instances_s": len(requests) / float(np.median(durations)),
        "valid": [output.valid for output in outputs],
        "status": [output.feedback.status for output in outputs],
        "selected_controls": [output.trajectory.controls.tolist() for output in outputs],
        "selected_states": [output.trajectory.states.tolist() for output in outputs],
        "selected_cost": [output.total_cost for output in outputs],
        "peak_allocated_mib": (torch.cuda.max_memory_allocated(kernel.device) / 2**20
                               if kernel.device.type == "cuda" else None),
        "peak_reserved_mib": (torch.cuda.max_memory_reserved(kernel.device) / 2**20
                              if kernel.device.type == "cuda" else None),
        "timing_scope": "Cold planner state; warmed kernels; CPU packaging, transfers, planning and final host validation included",
        "phase_timing_scope": "Existing CUDA per-phase monotonic timings are dispatch-only; not used for performance conclusions",
    }
    samples = {name: value.numpy() for name, value in handles[0].last_samples.items()}
    return row, samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment/t07_mppi_terrain_neighbor_cuda.yaml")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--samples", nargs="+", type=int, default=[512, 1024, 2048])
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 4, 16, 64])
    parser.add_argument("--cpu-batches", nargs="+", type=int, default=[1])
    parser.add_argument("--chunks", nargs="+", type=int, default=[256, 1024, 2048])
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device required; CPU fallback is not allowed")
    if args.repeats < 1 or any(n < 1 for n in args.samples + args.batches + args.cpu_batches + args.chunks):
        raise ValueError("Positive measurement sizes required")
    torch.set_num_threads(1)
    cfg = cfg_from_experiment(args.config)
    if cfg.local_goal_contract.planner_device != "cuda" or not cfg.local_goal_contract.training_paused:
        raise ValueError("Independent CUDA diagnostic configuration required")
    settings = {"samples": args.samples, "batches": args.batches, "cpu_batches": args.cpu_batches,
                "chunks": args.chunks, "repeats": args.repeats, "noise_seed": 23,
                "scope": "Flat frozen legal local input, not a full task or terrain/neighbor performance suite"}
    run = EvidenceRun(args.run_dir, args.config, settings=settings)
    case = make_case("flat", 0, 23)
    json_write(run.path / "config/case.json", case)
    json_write(run.path / "metrics/device.json", {
        "torch": torch.__version__, "runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0), "capability": torch.cuda.get_device_capability(0),
        "memory_mib": torch.cuda.get_device_properties(0).total_memory / 2**20,
        "torch_threads": torch.get_num_threads(), "timing": "cuda.synchronize before/after every measured call",
    })
    base = MPPIConfig.from_contract(cfg.local_goal_contract, cfg.terrain.slope_speed_scale)
    rows = []
    for batch in sorted(set(args.batches + args.cpu_batches)):
        requests = [to_input(case, identity=(i // 4, i % 4)) for i in range(batch)]
        for samples in args.samples:
            noise = (np.random.default_rng(23).normal(size=(base.iterations, batch, samples, 15, 2))
                     * [base.noise_v, base.noise_w]).astype(np.float32)
            noise_hash = hashlib.sha256(noise.tobytes()).hexdigest()
            devices = (["cpu"] if batch in args.cpu_batches else []) + (["cuda"] if batch in args.batches else [])
            for device in devices:
                chunks = args.chunks if device == "cuda" and batch == 1 and samples == args.samples[0] else [256 if device == "cpu" else base.query_chunk]
                for chunk in chunks:
                    tag = f"{device}_b{batch}_k{samples}_q{chunk}"
                    row, raw = measure(replace(base, device=device, samples=samples, query_chunk=chunk),
                                       requests, noise, args.repeats)
                    row.update(tag=tag, noise_sha256=noise_hash)
                    json_write(run.path / f"metrics/{tag}.json", row)
                    np.savez_compressed(run.path / f"samples/{tag}_first_instance.npz", **raw)
                    rows.append(row)
                    print(json.dumps({"tag": tag, "p50_s": row["p50_s"],
                                      "peak_mib": row["peak_allocated_mib"]}), flush=True)
    pairs = []
    for gpu in [row for row in rows if row["config"]["device"] == "cuda"]:
        cpu = next((row for row in rows if row["config"]["device"] == "cpu"
                    and row["batch"] == gpu["batch"] and row["config"]["samples"] == gpu["config"]["samples"]), None)
        if cpu is not None:
            c, g = np.asarray(cpu["selected_controls"]), np.asarray(gpu["selected_controls"])
            pairs.append({"cpu": cpu["tag"], "gpu": gpu["tag"],
                          "same_noise": cpu["noise_sha256"] == gpu["noise_sha256"],
                          "controls_match": bool(np.allclose(c, g, atol=1e-5, rtol=1e-5)),
                          "max_control_error": float(np.max(np.abs(c-g))),
                          "valid_status_match": cpu["valid"] == gpu["valid"] and cpu["status"] == gpu["status"],
                          "speedup": cpu["p50_s"] / gpu["p50_s"]})
    json_write(run.path / "metrics/device_benchmark.json", {
        "completed_measurements": len(rows), "pairs": pairs,
        "all_outputs_valid": all(all(row["valid"]) for row in rows),
        "sampling_artifacts": "Last optimization iteration of first instance per panel; not all instances",
        "cuda_availability_and_execution": True, "full_cuda_numeric_gate": "Requires separate contract tests",
        "formal_training": "paused", "optimizer_updates": 0, "training_interactions": 0,
    })


if __name__ == "__main__":
    main()
