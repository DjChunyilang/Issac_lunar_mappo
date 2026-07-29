from __future__ import annotations

import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from _common import cfg_from_experiment, load_yaml  # noqa: E402


BASELINE = (
    ROOT
    / "configs"
    / "experiment"
    / "exp073_structured_bicycle_quintic_map25_robust_flat_slots_radius42.yaml"
)
CONFIG = (
    ROOT
    / "configs"
    / "experiment"
    / "exp075_structured_bicycle_quintic_map25_robust_flat_slots_fast_curriculum.yaml"
)


def _without_exp075_curriculum_delta(raw: dict) -> dict:
    normalized = copy.deepcopy(raw)
    normalized["experiment"]["name"] = "<experiment>"
    normalized["algorithm"]["training_semantics"] = "<semantics>"
    initial_state = normalized["initial_state"]
    initial_state["curriculum_warmup_timesteps"] = 4096
    initial_state["curriculum_ramp_timesteps"] = 8192
    return normalized


def test_exp075_only_accelerates_initial_state_curriculum() -> None:
    baseline = load_yaml(BASELINE)
    exp075 = load_yaml(CONFIG)

    assert _without_exp075_curriculum_delta(exp075) == _without_exp075_curriculum_delta(
        baseline
    )

    cfg = cfg_from_experiment(CONFIG)
    assert cfg.initial_state.curriculum_warmup_timesteps == 256
    assert cfg.initial_state.curriculum_ramp_timesteps == 512
    # The safety/filter curriculum remains unchanged; this isolates exposure
    # to the full initial-state distribution.
    assert cfg.planner.subgoal_filter.warmup_timesteps == 4096
    assert cfg.planner.subgoal_filter.ramp_timesteps == 4096
