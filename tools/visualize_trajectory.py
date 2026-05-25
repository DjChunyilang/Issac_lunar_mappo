#!/usr/bin/env python
"""Dump a sample trajectory as JSON for plotting."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source" / "lunar_rover_tasks"))

from lunar_rover_tasks.tasks.multi_rover_gathering.gathering_env import MultiRoverGatheringCore
from lunar_rover_tasks.tasks.multi_rover_gathering.gathering_env_cfg import make_debug_cfg


def main() -> None:
    env = MultiRoverGatheringCore(make_debug_cfg(num_envs=1, device="cpu"))
    out = env.step(env.random_actions())
    traj = out.info["trajectory"].points[0].detach().cpu().tolist()
    print(json.dumps({"trajectory_xyz": traj}, indent=2))


if __name__ == "__main__":
    main()

