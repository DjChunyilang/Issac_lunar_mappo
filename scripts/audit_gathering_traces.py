#!/usr/bin/env python3
"""Read-only audit of completed no-learning control and sampling evidence."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import pickle
import re
import zipfile

import numpy as np

from planner_evidence import json_write


def audit(root, *, require_full=False, cuda=False):
    latest = {}
    paths = [p for p in (root / "metrics").glob("trace_*.json")
             if re.fullmatch(r"trace_.+_\d{4}_a\d+", p.stem)]
    for path in paths:
        prefix, macro, attempt = path.stem.rsplit("_", 2)
        key = (prefix[6:], int(macro))
        if key not in latest or int(attempt[1:]) > latest[key][0]:
            latest[key] = (int(attempt[1:]), path)
    errors, seen, previous = [], set(), {}
    calls = eligible = captured = required = 0
    step_counts = Counter()
    for (tag, macro), (_, path) in sorted(latest.items()):
        trace = json.loads(path.read_text())
        if len(trace["records"]) != len(trace["sample_artifacts"]):
            errors.append(f"Sampling trace length differs: {path.name}")
            continue
        for offset, record in enumerate(trace["records"]):
            assets = trace["sample_artifacts"][offset]
            ids = {int(Path(asset).stem.rsplit("_", 3)[-2]) for asset in assets}
            for asset in assets:
                if not (root / asset).exists():
                    errors.append(f"Missing sample: {asset}")
            for row in record["planner_execution"]:
                env, agent = row["env_id"], row["agent_id"]
                identity = env * 4 + agent
                calls += 1
                step_counts[(tag, env)] += 1
                if not np.allclose(row["applied_control"], record["commands"][env][agent], atol=1e-7, rtol=0):
                    errors.append(f"Applied mismatch: {path.name} {identity}")
                prediction = record.get("execution_predictions", [None] * 8)[identity]
                if prediction is not None and not np.allclose(prediction["controls"][0], row["applied_control"], atol=1e-7, rtol=0):
                    errors.append(f"Broadcast prediction mismatch: {path.name} {identity}")
                diagnostic = row.get("diagnostics", {})
                available = diagnostic.get("samples", 0) > 0
                if available:
                    eligible += 1
                    if diagnostic["samples"] != 512 or diagnostic["iterations"] != 2:
                        errors.append(f"K/iteration mismatch: {path.name} {identity}")
                reason = row.get("fallback_reason") or row.get("safety", {}).get("reason")
                failure = reason if reason and reason not in ("hold_request", "expired_request") and not str(reason).startswith("hold") else None
                key = (tag, identity)
                new_failure = failure is not None and failure != previous.get(key)
                previous[key] = failure
                needed = available and (key not in seen or (macro * 5 + offset) % 25 == 0 or new_failure)
                if available:
                    seen.add(key)
                if needed:
                    required += 1
                    if identity not in ids:
                        errors.append(f"Required sample missing: {path.name} step{offset} id{identity}")
                if available and identity in ids:
                    captured += 1
    archives = list((root / "samples").glob("*.npz"))
    for path in archives:
        with zipfile.ZipFile(path) as archive:
            with archive.open("raw.npy") as stream:
                version = np.lib.format.read_magic(stream)
                shape, _, _ = (np.lib.format.read_array_header_1_0(stream) if version == (1, 0)
                               else np.lib.format.read_array_header_2_0(stream))
                if shape != (512, 15, 2):
                    errors.append(f"Raw K shape mismatch: {path.name} {shape}")
    episodes = [json.loads(p.read_text()) for p in (root / "metrics").glob("episode_*.json")]
    expected_calls = sum(row["episode"]["duration_low_steps"] * 4 for row in episodes)
    if require_full and len(episodes) != 144:
        errors.append(f"Incomplete episode matrix: {len(episodes)} != 144")
    if len(episodes) == 144 and calls != expected_calls:
        errors.append(f"Call coverage mismatch: {calls} != {expected_calls}")
    for row in episodes:
        if row["training_interactions"] != 0 or row["optimizer_updates"] != 0:
            errors.append(f"Unexpected learning: {row['scenario']['case_id']}")
        if cuda and row["variant"] == "mppi512":
            config = row["contracts"]["mppi"]
            if config["device"] != "cuda" or config["adapter"] != "mppi_terrain_neighbor_stable_v2":
                errors.append(f"Incorrect GPU contract: {row['scenario']['case_id']}")
    states = list((root / "state").glob("*.pkl"))
    committed = 0
    for path in states:
        metadata = path.with_suffix(".pkl.json")
        if not metadata.exists():
            continue  # An interrupted publication is preserved, not committed evidence.
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != json.loads(metadata.read_text())["sha256"]:
            errors.append(f"State checksum mismatch: {path.name}")
            continue
        committed += 1
        if cuda and "mppi512" in path.name:
            # Only trusted, locally generated diagnostic states are accepted.
            state = pickle.loads(data)
            for planner in state["environment"]["planner_states"]:
                if planner["rng_device"] != "cuda" or planner["state_version"] != 2:
                    errors.append(f"Incorrect GPU state version: {path.name}")
    return {"completed_episodes": len(episodes), "trace_files_total": len(paths), "committed_macro_traces": len(latest),
            "planner_execution_calls": calls, "expected_calls_finished_episodes": expected_calls,
            "eligible_mppi_sampling_calls": eligible, "retained_sampling_calls": captured,
            "required_event_sampling_calls": required, "sampling_call_coverage": captured / max(1, eligible),
            "raw_archives_including_preserved_attempts": len(archives), "committed_state_snapshots": committed,
            "sampling_scope": "First available call, every 25 low steps, and new failure; HOLD has no samples",
            "errors": errors, "passed": not errors, "cuda_required": cuda, "full_matrix_required": require_full}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--require-full", action="store_true")
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()
    result = audit(args.run_dir, require_full=args.require_full, cuda=args.cuda)
    index = len(list((args.run_dir / "metrics").glob("trace_audit_*.json")))
    json_write(args.run_dir / f"metrics/trace_audit_{index:04d}.json", result)
    print(json.dumps(result), flush=True)
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
