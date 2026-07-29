from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from _common import cfg_from_experiment, load_yaml  # noqa: E402


CONFIG = (
    ROOT
    / "configs"
    / "experiment"
    / "exp068_structured_bicycle_quintic_map25_site_slot_goal_bc.yaml"
)


def test_exp068_uses_separate_center_and_slot_execution_features() -> None:
    raw = load_yaml(CONFIG)
    cfg = cfg_from_experiment(CONFIG)

    assert cfg.task.explicit_goal_in_execution is True
    assert cfg.observation.schema_version == "ego_v7_gather_site_and_slot_goal"
    assert cfg.actor_obs_dim == 92
    assert cfg.critic_state_dim == 54
    assert raw["algorithm"]["actor_architecture"] == "branched_v4"
    assert raw["algorithm"]["teacher_mode"] == "oracle_slots"
    assert cfg.gather_point.execution_slot_radius == pytest.approx(0.45)
