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
    / "exp064_structured_bicycle_quintic_map25_centroid_flatness_reward.yaml"
)
CONFIG = (
    ROOT
    / "configs"
    / "experiment"
    / "exp065_structured_bicycle_quintic_map25_oracle_goal_broadcast.yaml"
)


def _without_exp065_delta(raw: dict) -> dict:
    normalized = copy.deepcopy(raw)
    normalized["experiment"]["name"] = "<experiment>"
    normalized["algorithm"]["training_semantics"] = "<semantics>"
    normalized["algorithm"]["actor_architecture"] = "<actor_architecture>"
    normalized["task"].pop("explicit_goal_in_execution", None)
    normalized["observation"]["schema_version"] = "<observation_schema>"
    return normalized


def test_exp065_isolates_the_terrain_aware_execution_goal() -> None:
    baseline = load_yaml(BASELINE)
    exp065 = load_yaml(CONFIG)

    assert _without_exp065_delta(exp065) == _without_exp065_delta(baseline)

    cfg = cfg_from_experiment(CONFIG)
    assert cfg.task.explicit_goal_in_execution is True
    assert cfg.observation.schema_version == "ego_v5_gather_site_goal"
    assert cfg.actor_obs_dim == 89
    assert cfg.critic_state_dim == 54
    assert cfg.gather_point.search_method == "terrain_aware_multiresolution"
    assert cfg.gather_point.require_flat_for_success
