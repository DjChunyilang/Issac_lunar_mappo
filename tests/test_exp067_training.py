from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from _common import cfg_from_experiment, load_yaml  # noqa: E402


BASELINE = (
    ROOT
    / "configs"
    / "experiment"
    / "exp066_structured_bicycle_quintic_map25_oracle_ring_bc.yaml"
)
CONFIG = (
    ROOT
    / "configs"
    / "experiment"
    / "exp067_structured_bicycle_quintic_map25_oracle_slots_bc.yaml"
)


def _without_exp067_delta(raw: dict) -> dict:
    normalized = copy.deepcopy(raw)
    normalized["experiment"]["name"] = "<experiment>"
    normalized["observation"]["schema_version"] = "<schema>"
    normalized["gather_point"].pop("execution_slot_radius", None)
    normalized["algorithm"]["training_semantics"] = "<semantics>"
    for key in (
        "bc_updates",
        "bc_batch_size",
        "bc_learning_rate",
        "teacher_mode",
        "teacher_stop_radius",
        "teacher_slow_distance",
        "teacher_max_rho",
        "teacher_terrain_scale",
        "bc_yaw_noise_degrees",
        "bc_min_nearest_distance",
    ):
        normalized["algorithm"].pop(key, None)
    return normalized


def test_exp067_isolates_symmetric_oracle_slots_bc_warm_start() -> None:
    baseline = load_yaml(BASELINE)
    exp067 = load_yaml(CONFIG)

    assert _without_exp067_delta(exp067) == _without_exp067_delta(baseline)

    cfg = cfg_from_experiment(CONFIG)
    assert cfg.task.explicit_goal_in_execution is True
    assert cfg.observation.schema_version == "ego_v6_gather_slot_goal"
    assert cfg.actor_obs_dim == 89
    assert cfg.critic_state_dim == 54
    assert cfg.gather_point.execution_slot_radius == pytest.approx(0.45)
    assert exp067["algorithm"]["teacher_mode"] == "oracle_slots"
    assert exp067["algorithm"]["bc_updates"] == 128
    assert exp067["algorithm"]["teacher_max_rho"] == pytest.approx(1.6)
