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
    / "exp065_structured_bicycle_quintic_map25_oracle_goal_broadcast.yaml"
)
CONFIG = (
    ROOT
    / "configs"
    / "experiment"
    / "exp066_structured_bicycle_quintic_map25_oracle_ring_bc.yaml"
)


def _without_exp066_delta(raw: dict) -> dict:
    normalized = copy.deepcopy(raw)
    normalized["experiment"]["name"] = "<experiment>"
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


def test_exp066_isolates_oracle_ring_bc_warm_start() -> None:
    baseline = load_yaml(BASELINE)
    exp066 = load_yaml(CONFIG)

    assert _without_exp066_delta(exp066) == _without_exp066_delta(baseline)

    cfg = cfg_from_experiment(CONFIG)
    assert cfg.task.explicit_goal_in_execution is True
    assert cfg.actor_obs_dim == 89
    assert cfg.critic_state_dim == 54
    assert exp066["algorithm"]["teacher_mode"] == "oracle_ring"
    assert exp066["algorithm"]["bc_updates"] == 32
    assert exp066["algorithm"]["teacher_stop_radius"] == pytest.approx(0.45)
    assert exp066["success_thresholds"]["min_pairwise_distance"] == pytest.approx(0.42)
