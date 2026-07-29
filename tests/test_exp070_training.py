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
    / "exp069_structured_bicycle_quintic_map25_oracle_slots_hard_safety.yaml"
)
CONFIG = (
    ROOT
    / "configs"
    / "experiment"
    / "exp070_structured_bicycle_quintic_map25_oracle_slots_reward.yaml"
)


def _without_exp070_reward_target_delta(raw: dict) -> dict:
    normalized = copy.deepcopy(raw)
    normalized["experiment"]["name"] = "<experiment>"
    normalized["algorithm"]["training_semantics"] = "<semantics>"
    normalized["task"].pop("execution_slot_reward_target", None)
    return normalized


def test_exp070_isolates_slot_aligned_oracle_progress_reward() -> None:
    baseline = load_yaml(BASELINE)
    exp070 = load_yaml(CONFIG)

    assert _without_exp070_reward_target_delta(exp070) == _without_exp070_reward_target_delta(
        baseline
    )

    cfg = cfg_from_experiment(CONFIG)
    assert cfg.observation.schema_version == "ego_v6_gather_slot_goal"
    assert cfg.task.execution_slot_reward_target is True
