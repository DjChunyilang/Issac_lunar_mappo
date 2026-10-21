#!/usr/bin/env python3
"""Complete selected failure-repair episodes; never issues a T09/T10 gate."""
import argparse
import json
from pathlib import Path

import torch

from evaluate_gathering_no_learning import scene_cases, execute_pair, map_sources
from planner_evidence import EvidenceRun, json_write


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--family", choices=["npb", "slope", "crossing", "flat_compact", "flat_irregular", "flat_reverse"], required=True)
    parser.add_argument("--variant", choices=["nmpc", "mppi512"], required=True)
    parser.add_argument("--safety", choices=["on", "off"], required=True)
    parser.add_argument("--duration", type=int, choices=[96, 180], required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-macros", type=int)
    args = parser.parse_args()
    torch.set_num_threads(1)
    settings = {"purpose": "targeted_failure_repair_diagnostic_v1", "formal_t10_acceptance": False,
                "map_sources": map_sources(), "samples": 512}
    run = EvidenceRun(args.run_dir, args.config, resume=args.resume, settings=settings)
    cases = [case for case in scene_cases() if case["family"] == args.family]
    if not (run.path/"config/scenes.json").exists():
        json_write(run.path/"config/scenes.json", scene_cases())
    execute_pair(run, cases[:2], args.variant, args.safety == "on", args.duration,
                 max_macros=args.max_macros)
    execute_pair(run, [cases[2], cases[2]], args.variant, args.safety == "on", args.duration,
                 max_macros=args.max_macros)
    records = [json.loads(path.read_text()) for path in (run.path/"metrics").glob("episode_*.json")]
    result = {"completed_episodes": len(records), "successes": sum(row["episode"]["success"] for row in records),
              "collisions": sum(row["episode"]["collision"] for row in records),
              "execution_limit_violations": sum(row["execution_limit_violations"] for row in records),
              "targeted_diagnostic_only": True, "formal_t10_acceptance": False,
              "training_interactions": 0, "optimizer_updates": 0}
    index = len(list((run.path/"metrics").glob("repair_summary_*.json")))
    json_write(run.path/f"metrics/repair_summary_{index:04d}.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
