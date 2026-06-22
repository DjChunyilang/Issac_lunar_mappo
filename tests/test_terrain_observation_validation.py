from __future__ import annotations

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_terrain_observation_validation import build_acceptance


def test_terrain_observation_probe_acceptance_requires_all_training_signals() -> None:
    passing = build_acceptance(
        {
            "nan_flag": False,
            "policy_parameter_delta_l2": 0.2,
            "terrain_input_weight_delta_l2": 0.05,
            "post_training_action_std": 0.1,
        },
        terrain_observation_max_abs=0.3,
    )
    missing_terrain_update = build_acceptance(
        {
            "nan_flag": False,
            "policy_parameter_delta_l2": 0.2,
            "terrain_input_weight_delta_l2": 0.0,
            "post_training_action_std": 0.1,
        },
        terrain_observation_max_abs=0.3,
    )

    assert passing["passed"]
    assert not missing_terrain_update["passed"]
    assert not missing_terrain_update["checks"]["terrain_input_weights_updated"]
