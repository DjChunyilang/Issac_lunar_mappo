#!/usr/bin/env python3
"""Stage-specific upstream scale evidence. No training or legacy execution."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from _common import cfg_from_experiment
from planner_evidence import EvidenceRun, json_write, source_snapshot
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_spec import UpstreamRuntimeConfig
from upstream_evidence import require_contract_tests


def audit_s00(cfg):
    if not isinstance(cfg, UpstreamRuntimeConfig):
        raise ValueError("S00 requires the independent upstream runtime configuration")
    s = cfg.spec
    audit = s.geometry_audit()
    assert cfg.training_paused and s.vehicle.forward_limit == 10 / 3.6
    assert s.communication.absolute_velocity_bound == s.vehicle.absolute_speed_bound
    assert s.grid.steps == 40 and s.grid.horizon == 8
    assert s.perception.sample_count <= s.perception.capacity
    assert cfg.metadata["planner_protocol"] == "planner_protocol_v2"
    return {
        "stage": "S00_contract_foundation",
        "configuration_parsed": True,
        "immutable_profile": True,
        "geometry": audit,
        "normalization": {"forward_speed": s.vehicle.forward_limit,
            "reverse_speed": s.vehicle.reverse_limit, "yaw_rate": s.vehicle.yaw_limit,
            "local_goal_distance": s.rho_max, "team_dmax": s.task.dmax,
            "team_dispersion": s.task.dispersion,
            "linear_acceleration_step": s.vehicle.linear_acceleration * s.grid.dt,
            "angular_acceleration_step": s.vehicle.angular_acceleration * s.grid.dt},
        "contracts": cfg.metadata,
        "execution_environment": "not_connected_at_S00",
        "dynamic_speed_limit": "not_verified_at_S00",
        "real_acados": "required_at_S02_not_run_at_S00",
        "cuda": "new_scale_validation_pending_S02",
        "formal_training": "paused", "optimizer_updates": 0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment/upstream_scale_s00.yaml")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--junit", type=Path, required=True)
    args = parser.parse_args()
    cfg = cfg_from_experiment(args.config)
    if not isinstance(cfg, UpstreamRuntimeConfig):
        raise ValueError("Wrong runtime family")
    run = EvidenceRun(args.run_dir, args.config, resume=args.resume,
        settings={"stage": "S00_contract_foundation", "profile_sha256": cfg.spec.profile_sha256})
    before = run.identity["source_snapshot"]
    junit = ET.parse(args.junit)
    cases = list(junit.iter("testcase"))
    errors = [case.attrib.get("name") for case in cases if
              any(case.find(tag) is not None for tag in ("failure", "error", "skipped"))]
    upstream = [case for case in cases if case.attrib.get("classname", "").endswith("test_upstream_spec")]
    if errors or len(upstream) < 20:
        raise ValueError(f"Required S00 tests incomplete or failing: {errors}")
    require_contract_tests(args.junit, run, ("tests/test_upstream_spec.py", "tests/test_config_wiring.py",
                                           "tests/test_cuda_evidence_gates.py"))
    provenance = run.path / "metrics/contracts.source.json"
    if not provenance.exists() or json.loads(provenance.read_text())["source_snapshot"] != before:
        raise ValueError("JUnit requires matching pre-test production source provenance")
    tests = json.loads(provenance.read_text())["test_sources"]
    if any(hashlib.sha256(Path(name).read_bytes()).hexdigest() != checksum for name, checksum in tests.items()):
        raise ValueError("S00 test source changed after execution")
    value = audit_s00(cfg)
    json_write(run.path / "config/resolved_upstream.json", asdict(cfg))
    json_write(run.path / "metrics/s00_audit.json", value)
    if before != source_snapshot():
        raise ValueError("Production code changed during S00 acceptance")
    json_write(run.path / "metrics/strict_acceptance.json", {
        "s00_contract_foundation_pass": True, "tests_passed": len(cases),
        "upstream_tests_passed": len(upstream), "failures": 0, "skipped": 0,
        "source_matches": True, "junit": str(args.junit),
        "junit_sha256": hashlib.sha256(args.junit.read_bytes()).hexdigest(),
        "profile_sha256": cfg.spec.profile_sha256, "runtime_family": cfg.runtime_family,
        "s01_execution_pass": False, "s02_planners_pass": False, "s05_gate_b_pass": False,
        "formal_training": "paused", "physx": "not_validated"})
    print(json.dumps({"s00_contract_foundation_pass": True, "tests_passed": len(cases),
                      "upstream_tests": len(upstream), "full_speed_execution": "not_tested"}))


if __name__ == "__main__":
    main()
