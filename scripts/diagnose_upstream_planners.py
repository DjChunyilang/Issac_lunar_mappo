#!/usr/bin/env python3
"""Append-only S02 seven-setting smoke evidence, explicitly short of full S02.

All scenes use provided synthetic local observations. This entry does not call
S03 sensors, task termination, safety projection or a learner. Prepare a run,
execute the complete frozen contract suite into it, then resume the diagnostic.
"""

import argparse
from copy import deepcopy
from dataclasses import asdict, replace
import json
import hashlib
import importlib.util
import os
import math
import resource
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import torch

from _common import ROOT, cfg_from_experiment, load_yaml
from diagnose_upstream_execution import synthetic_local_frame
from planner_evidence import EvidenceRun, file_sha256, json_write, source_snapshot
from upstream_evidence import require_contract_tests
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_baselines import (
    UpstreamPrimitiveInput, primitive_fingerprint, primitive_library,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_execution import (
    control_error, rollout_numpy, stop_envelope,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_factory import create_upstream_adapter
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_footprint import FootprintReviewer
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_planning import UpstreamIntent, UpstreamPlannerInput
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_proxy import UpstreamActuatorProxy
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_spec import canonical

DEFAULT_CONFIG = "configs/experiment/upstream_scale_s02.yaml"
STAGE = "S02_seven_setting_smoke_partial"
TEST_FILES = (
    "tests/test_upstream_spec.py", "tests/test_upstream_execution.py", "tests/test_upstream_review_v2.py",
    "tests/test_upstream_planning.py", "tests/test_upstream_nmpc.py", "tests/test_upstream_mppi.py",
    "tests/test_upstream_tensor_views.py",
    "tests/test_upstream_baselines.py", "tests/test_upstream_factory.py",
    "tests/test_upstream_planner_diagnostics.py", "tests/test_config_wiring.py",
)
SETTINGS = (("line", "line_upstream_v1", None), ("quintic", "quintic_upstream_v1", None),
            ("primitive47", "primitive47_upstream_v1", None), ("nmpc", "nmpc_upstream_v1", None),
            ("mppi512", "mppi_upstream_v1", 512), ("mppi1024", "mppi_upstream_v1", 1024),
            ("mppi2048", "mppi_upstream_v1", 2048))


def serial(value):
    if isinstance(value, torch.Tensor):return serial(value.detach().cpu().numpy())
    if isinstance(value, np.ndarray):return serial(value.tolist())
    if isinstance(value, np.generic):return serial(value.item())
    if isinstance(value, dict):return {str(key): serial(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):return [serial(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):return {"nonfinite": repr(value)}
    return value


def contract_test_sources():
    return {name: file_sha256(ROOT / name) for name in TEST_FILES}


def require_unchanged(run, tests):
    if (source_snapshot() != run.identity["source_snapshot"] or contract_test_sources() != tests
            or json.loads(Path(run.config).read_text()) != run.identity["resolved_config"]):
        raise ValueError("Producer or contract test source changed; preserve run and start a new one")


def requested_panels(device):
    panels = list(SETTINGS)
    if device == "both":
        panels += [(name + "_cuda", adapter, samples) for name, adapter, samples in SETTINGS if adapter == "mppi_upstream_v1"]
    return panels


def run_settings(device, restore_panel):
    return {"stage": STAGE, "planner_device": device, "torch_threads": 1,
            "settings": [name for name, _, _ in requested_panels(device)], "primitive_ids": list(range(47)),
            "execution_devices": ["cpu", "cuda"] if device == "both" else [device],
            "restore_panel": bool(restore_panel), "input_kind": "provided_synthetic_local_observation",
            "training_interactions": 0, "optimizer_updates": 0}


def create_run(path, config=DEFAULT_CONFIG, *, device="cpu", resume=False, restore_panel=False):
    if device not in {"cpu", "cuda", "both"}:raise ValueError("Explicit CPU, CUDA or both devices required")
    data = deepcopy(load_yaml(config))
    if data.get("upstream_scale", {}).get("planner_execution_version") != "upstream_planning_v1":
        raise ValueError("S02 diagnostic requires explicit upstream_planning_v1")
    data["upstream_scale"]["planner_device"] = "cpu" if device == "both" else device
    data["upstream_scale"]["planner_adapter"] = "mppi_upstream_v1"
    path = Path(path)
    config_path = path / "config/requested_experiment.json"
    if (path / "run_manifest.json").exists() and not resume:
        raise FileExistsError("Existing diagnostic run requires --resume; artifacts are never overwritten")
    if resume and not (path / "run_manifest.json").exists():
        raise ValueError("Cannot resume a missing diagnostic run")
    if config_path.exists():
        if json.loads(config_path.read_text()) != data:raise ValueError("Diagnostic device/configuration mismatch")
    else:
        json_write(config_path, data)
    run = EvidenceRun(path, config_path, resume=resume, settings=run_settings(device, restore_panel))
    cfg = cfg_from_experiment(config_path)
    if cfg.clock_mode != "offline_simulated_latency" or not cfg.training_paused:
        raise ValueError("S02 smoke requires paused, explicit offline simulation")
    run.resolved(cfg)
    return cfg, run


def archive_inputs(run, cfg):
    # The worktree is intentionally uncommitted. Keep the producing bytes in
    # addition to hashes so later edits cannot erase diagnostic provenance.
    for relative, expected in run.identity["source_snapshot"].items():
        content = (ROOT / relative).read_bytes()
        if hashlib.sha256(content).hexdigest() != expected:
            raise ValueError("Producer changed before source archive was frozen")
        target = run.path / "config/source_snapshot" / relative
        if target.exists():
            if file_sha256(target) != expected:raise ValueError("Archived source checksum mismatch")
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as stream:stream.write(content)
    profile = run.path / "config/design_profile.json"
    design = json.loads(cfg.spec.profile_json)
    if profile.exists():
        if json.loads(profile.read_text()) != design:raise ValueError("Archived profile mismatch")
    else:json_write(profile, design)
    path = run.path / "config/runtime_dependencies.json"
    if not path.exists():
        libraries = {}
        source = os.environ.get("ACADOS_SOURCE_DIR")
        if source:
            for name in ("libacados.so", "libhpipm.so", "libblasfeo.so"):
                library = Path(source) / "lib" / name
                libraries[name] = {"path": str(library.resolve()), "sha256": file_sha256(library)} if library.is_file() else None
        module = importlib.util.find_spec("acados_template")
        json_write(path, {"packages": run.identity["dependencies"], "torch_cuda": torch.version.cuda,
                         "acados_template_module": None if module is None else module.origin,
                         "acados_source_dir": source, "acados_libraries": libraries})


def freeze_tests(run, *, create=False):
    tests = contract_test_sources()
    path = run.path / "metrics/contracts.source.json"
    value = {"source_snapshot": run.identity["source_snapshot"], "test_sources": tests}
    if path.exists():
        if json.loads(path.read_text()) != value:raise ValueError("Contract provenance already belongs to different sources")
    elif create:
        if (run.path / "metrics/contracts.xml").exists():
            raise ValueError("Cannot attach provenance after contract execution; prepare a new run first")
        json_write(path, value)
    else:
        raise ValueError("Missing pre-execution contract provenance; use --prepare before running tests")
    return tests


def panel_config(cfg, adapter, samples, *, device=None):
    data = json.loads(cfg.configuration_json)
    data["upstream_scale"]["planner_adapter"] = adapter
    device = (cfg.planner_device if device is None else device) if adapter == "mppi_upstream_v1" else "cpu"
    data["upstream_scale"]["planner_device"] = device
    count = cfg.mppi_samples if samples is None else samples
    data["upstream_scale"]["mppi_samples"] = count
    return replace(cfg, planner_adapter=adapter, planner_device=device,
                   mppi_samples=count, configuration_json=canonical(data))


def initial_input(cfg):
    information_time = 0.
    activation_time = math.ceil(cfg.simulated_latency / cfg.spec.grid.dt - 1e-12) * cfg.spec.grid.dt
    terrain = synthetic_local_frame(np.zeros(3), cfg.spec, information_time, query_version=cfg.terrain_query_version)
    return UpstreamPlannerInput(0, 0, np.zeros(3), np.zeros(2),
        UpstreamIntent([4., 0.], False, information_time, information_time + cfg.spec.grid.request_validity),
        terrain, (), np.zeros(2), activation_time, cfg.spec.grid.times(activation_time),
        (-cfg.spec.world_width/2, -cfg.spec.world_height/2, cfg.spec.world_width/2, cfg.spec.world_height/2),
        cfg.spec, information_pose=np.zeros(3), information_time=information_time)


def payload(context, primitive_id):
    return context if primitive_id is None else UpstreamPrimitiveInput.for_context(context, primitive_id)


def synchronize(cfg):
    if cfg.planner_device == "cuda":torch.cuda.synchronize()


def execute_first_control(cfg, context, output):
    """Actual actuator-proxy transition, including exactly one injected delay."""
    initial = np.zeros((cfg.simulation.num_envs, 4, 3))
    initial[:, :, 1] = np.arange(4) * 10.
    proxy = UpstreamActuatorProxy(cfg, initial)
    desired = np.zeros((cfg.simulation.num_envs, 4, 2))
    held = []
    for _ in range(round(context.now / cfg.spec.grid.dt)):
        held.append(proxy.step(desired))
    desired[0, 0] = output.trajectory.controls[0]
    actual = proxy.step(desired)
    np.testing.assert_allclose(actual["activation_time"][0], context.now, atol=1e-10, rtol=0)
    np.testing.assert_allclose(actual["applied_control"][0, 0], output.trajectory.controls[0], atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(actual["after"][0, 0], output.trajectory.states[1], atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(actual["actual_velocity"][0, 0], output.trajectory.velocities[0], atol=1e-8, rtol=1e-8)
    if control_error(actual["applied_control"][0, 0], context.previous_control, cfg.spec):
        raise ValueError("Actual proxy first control violates actuator limits")
    if actual["geometry_overlap"].any() or actual["clearance_violation"].any():
        raise ValueError("Actual proxy first step violated fleet envelope clearance")
    return {"held_delay_steps": len(held), "held_delay_records": held, "actual_step": actual,
            "actual_control_equals_planned_first": True, "proxy_task_loop": "not_integrated"}


def reserve_evidence(cfg, context, output):
    reserve = stop_envelope(context.information_pose, context.previous_control, output.trajectory.controls[0],
                            cfg.spec, information_time=context.information_time, latency=cfg.simulated_latency)
    report = FootprintReviewer(cfg.spec).trajectory(reserve.trajectory, context.previous_control, context.terrain,
        neighbors=context.neighbors, workspace=np.asarray(context.workspace).reshape(2, 2),
        require_stop=True, conservative_reserve=True)
    if output.valid and (not reserve.fits_horizon or not report.passed):
        raise ValueError("Planner valid output failed independent information-time reserve review")
    return {"envelope": asdict(reserve), "independent_review": asdict(report)}


def restore_case(cfg, context, primitive_id, planner, first_output, cache_dir):
    # Use a fresh, explicitly provided information snapshot and hold the actual
    # first control during its injected delay before making the next request.
    saved = planner.state_dict()
    first_control = first_output.trajectory.controls[0].copy()
    information_pose = first_output.trajectory.states[1].copy()
    info_time = context.now + cfg.spec.grid.dt
    steps = math.ceil(cfg.simulated_latency / cfg.spec.grid.dt - 1e-12)
    held = rollout_numpy(information_pose, np.tile(first_control, (steps, 1)), cfg.spec, info_time)
    now = info_time + steps * cfg.spec.grid.dt
    next_context = replace(context, ego_pose=held.states[-1], ego_velocity=first_control,
        previous_control=first_control, information_pose=information_pose, information_time=info_time,
        now=now, times=cfg.spec.grid.times(now),
        terrain=synthetic_local_frame(information_pose, cfg.spec, info_time, query_version=cfg.terrain_query_version),
        intent=UpstreamIntent(context.intent.goal_xy, False, info_time, info_time + cfg.spec.grid.request_validity))
    restored = create_upstream_adapter(cfg, cache_dir=cache_dir)
    restored.load_state_dict(saved)
    expected = planner.plan(payload(next_context, primitive_id))
    actual = restored.plan(payload(next_context, primitive_id))
    errors = {name: float(np.max(abs(getattr(actual.trajectory, name) - getattr(expected.trajectory, name))))
              for name in ("states", "controls", "velocities", "times")}
    for name in errors:
        np.testing.assert_allclose(getattr(actual.trajectory, name), getattr(expected.trajectory, name), atol=1e-7, rtol=1e-6)
    if (actual.valid, actual.feedback.status, actual.fallback_reason) != (expected.valid, expected.feedback.status, expected.fallback_reason):
        raise ValueError("Restored planner feedback disagrees with uninterrupted execution")
    return {"passed": True, "maximum_absolute_errors": errors, "atol": 1e-7, "rtol": 1e-6,
            "same_device": cfg.planner_device, "expected": asdict(expected), "restored": asdict(actual)}


def diagnose(cfg, run, *, restore_panel=False):
    records = []; kernels = {}
    panels = requested_panels(run.identity["settings"]["planner_device"])
    library = primitive_library(cfg.spec)
    json_write(run.path / "config/primitive47_library.json", library)
    for name, adapter_id, samples in panels:
        current = panel_config(cfg, adapter_id, samples, device="cuda" if name.endswith("_cuda") else None)
        json_write(run.path / f"config/panel_{name}.json", json.loads(current.configuration_json))
        json_write(run.path / f"config/contracts_{name}.json", current.metadata)
        context = initial_input(current)
        json_write(run.path / f"samples/{name}_input.json", {
            "input_kind": "provided_synthetic_local_observation", "map_identity": "synthetic_flat_s02_v1",
            "seed": current.seed, "information_time": context.information_time, "activation_time": context.now,
            "ego_pose": serial(context.ego_pose), "intent": serial(asdict(context.intent)),
            "terrain_snapshot": serial(asdict(context.terrain.snapshot())), "workspace": list(context.workspace),
            "neighbors": [], "sensor_chain": "not_executed"})
        cache = run.path / "state" / "acados_cache" / name
        started = time.perf_counter()
        try:
            planner = create_upstream_adapter(current, kernel_cache=kernels, cache_dir=cache)
            synchronize(current)
            construction_seconds = time.perf_counter() - started
        except Exception as error:
            record = {"setting": name, "invocation_completed": False, "error_type": type(error).__name__,
                      "error": str(error), "construction_seconds": time.perf_counter() - started}
            records.append(record)
            json_write(run.path / f"metrics/{name}_construction_failure.json", record)
            (run.path / f"logs/{name}_construction_failure.txt").write_text(traceback.format_exc())
            continue
        ids = range(47) if adapter_id == "primitive47_upstream_v1" else (None,)
        for native_id in ids:
            case = name if native_id is None else f"primitive47_{native_id:02d}"
            record = {"setting": name, "case": case, "primitive_id": native_id,
                      "adapter_id": adapter_id, "planner_device": current.planner_device,
                      "samples": samples, "iterations": current.mppi_iterations if samples else None,
                      "construction_seconds": construction_seconds, "seed": current.seed}
            try:
                planner.reset()
                synchronize(current); started = time.perf_counter()
                output = planner.plan(payload(context, native_id))
                synchronize(current)
                record["planning_and_review_seconds"] = time.perf_counter() - started
                record["process_peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                record["cuda_peak_allocated_bytes"] = (torch.cuda.max_memory_allocated()
                    if current.planner_device == "cuda" else None)
                started = time.perf_counter()
                reserve = reserve_evidence(current, context, output)
                actual = execute_first_control(current, context, output)
                record.update(invocation_completed=True, valid_execution=bool(output.valid),
                    independent_reserve_passed=bool(reserve["independent_review"]["passed"]),
                    fallback_reason=output.fallback_reason, feedback_status=output.feedback.status,
                    additional_evidence_seconds=time.perf_counter() - started,
                    predicted_terminal_goal_error=float(np.linalg.norm(output.trajectory.states[-1, :2] - context.intent.goal_xy)),
                    actual_first_step_displacement=float(np.linalg.norm(output.trajectory.states[1, :2] - context.ego_pose[:2])))
                json_write(run.path / f"samples/{case}_execution.json", serial({"output": asdict(output),
                    "information_time_reserve": reserve, "actual_proxy_execution": actual,
                    "safety_projection": "not_integrated", "goal_completion": "not_evaluated"}))
                run.snapshot(f"{case}_planner.pkl", planner.state_dict())
                raw_samples = getattr(planner, "last_samples", None)
                if raw_samples is not None:run.snapshot(f"{case}_raw_samples.pkl", raw_samples)
                record["raw_samples_recorded"] = raw_samples is not None
                if restore_panel and (native_id is None or native_id == 0):
                    restored = restore_case(current, context, native_id, planner, output, cache)
                    json_write(run.path / f"samples/{case}_restore.json", serial(restored))
                    record["restore_passed"] = restored["passed"]
            except Exception as error:
                record.update(invocation_completed=False, error_type=type(error).__name__, error=str(error))
                (run.path / f"logs/{case}_failure.txt").write_text(traceback.format_exc())
            json_write(run.path / f"metrics/{case}.json", serial(record))
            records.append(record)
    expected_cases = sum(47 if adapter == "primitive47_upstream_v1" else 1 for _, adapter, _ in panels)
    return {"stage": STAGE, "records": records, "settings_required": [name for name, _, _ in panels],
            "expected_cases": expected_cases, "recorded_cases": len(records),
            "smoke_pass": len(records) == expected_cases and all(
                r.get("invocation_completed") and r.get("valid_execution") and r.get("independent_reserve_passed") for r in records),
            "primitive_library_fingerprint": primitive_fingerprint(cfg.spec),
            "input_kind": "provided_synthetic_local_observation", "sensor_chain": "not_executed",
            "cpu_cuda_pairing_at_run_samples": "not_executed", "batch_load_matrix": "not_executed",
            "restore_panel_requested": restore_panel, "legacy_23_real_acados": "requires_separate_actual_regression",
            "training_interactions": 0, "optimizer_updates": 0, "formal_training": "paused"}


def strict_record(summary, junit):
    return {"stage": STAGE, "smoke_pass": summary["smoke_pass"], "s02_full_acceptance_pass": False,
            "s03_perception_communication_task_pass": False, "s05_gate_b_pass": False,
            "t11_training_acceptance_pass": False, "legacy_23_real_acados_pass": False,
            "cpu_cuda_full_matrix_pass": False, "realtime_pass": False, "physx_pass": False,
            "contract_tests": junit, "training_interactions": 0, "optimizer_updates": 0, "formal_training": "paused"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda", "both"), default="cpu")
    parser.add_argument("--prepare", action="store_true", help="Create immutable run and frozen contract provenance only")
    parser.add_argument("--resume", action="store_true", help="Continue the prepared run without overwriting artifacts")
    parser.add_argument("--junit", type=Path)
    parser.add_argument("--restore-panel", action="store_true")
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    cfg, run = create_run(args.run_dir, args.config, device=args.device, resume=args.resume, restore_panel=args.restore_panel)
    if args.prepare:archive_inputs(run, cfg)
    tests = freeze_tests(run, create=args.prepare)
    if args.prepare:
        print("Prepared contract command:")
        print(" ".join([sys.executable, "-m", "pytest", "-q", *TEST_FILES,
                        "--junitxml=" + str(run.path / "metrics/contracts.xml")]))
        return 0
    if (run.path / "metrics/summary.json").exists() or (run.path / "metrics/strict_acceptance.json").exists():
        raise FileExistsError("Diagnostic already produced a terminal record; use a new run")
    try:
        require_unchanged(run, tests)
        junit = require_contract_tests(args.junit or run.path / "metrics/contracts.xml", run, TEST_FILES)
        summary = diagnose(cfg, run, restore_panel=args.restore_panel)
        require_unchanged(run, tests)
        json_write(run.path / "metrics/summary.json", serial(summary))
        json_write(run.path / "metrics/strict_acceptance.json", strict_record(summary, junit))
    except Exception as error:
        json_write(run.path / "metrics/entry_failure.json", {"stage": STAGE,
            "error_type": type(error).__name__, "error": str(error), "s02_full_acceptance_pass": False,
            "training_interactions": 0, "optimizer_updates": 0})
        (run.path / "logs/entry_failure.txt").write_text(traceback.format_exc())
        raise
    print({"stage": STAGE, "smoke_pass": summary["smoke_pass"], "cases": summary["recorded_cases"],
           "s02_full_acceptance_pass": False, "formal_training": "paused"})
    return 0 if summary["smoke_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
