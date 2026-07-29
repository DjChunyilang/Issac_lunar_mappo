from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from _common import cfg_from_experiment, load_yaml  # noqa: E402


def test_exp088_and_exp089_inherit_flatness_gated_warmstart_and_isolate_control_delta() -> None:
    exp087 = load_yaml(
        ROOT / "configs/experiment/exp087_structured_bicycle_quintic_map25_flatness_gated_warmstart.yaml"
    )
    exp088 = load_yaml(
        ROOT / "configs/experiment/exp088_structured_bicycle_quintic_map25_flatness_center_stronger.yaml"
    )
    exp089 = load_yaml(
        ROOT / "configs/experiment/exp089_structured_bicycle_quintic_map25_flatness_center_early.yaml"
    )
    exp090 = load_yaml(
        ROOT / "configs/experiment/exp090_structured_bicycle_quintic_map25_flatness_center_early_radius37.yaml"
    )
    exp091 = load_yaml(
        ROOT / "configs/experiment/exp091_structured_bicycle_quintic_map25_flatness_center_early_radius33.yaml"
    )
    exp092 = load_yaml(
        ROOT / "configs/experiment/exp092_structured_bicycle_quintic_map25_flatness_center_early_radius35.yaml"
    )
    exp093 = load_yaml(
        ROOT / "configs/experiment/exp093_structured_bicycle_quintic_map25_flatness_center_early_radius35_time80.yaml"
    )
    exp094 = load_yaml(
        ROOT / "configs/experiment/exp094_structured_bicycle_quintic_map25_flatness_center_early_radius35_time96.yaml"
    )
    exp095 = load_yaml(
        ROOT / "configs/experiment/exp095_structured_bicycle_quintic_map25_flatness_center_early_radius35_time112.yaml"
    )
    exp096 = load_yaml(
        ROOT / "configs/experiment/exp096_structured_bicycle_quintic_map25_flatness_center_early_radius35_time128.yaml"
    )
    exp097 = load_yaml(
        ROOT
        / "configs/experiment/exp097_structured_bicycle_quintic_map25_flatness_center_early_radius35_time128_ppo_probe.yaml"
    )

    assert exp088["low_level_control"] == {
        **exp087["low_level_control"],
        "formation_center_correction_max_offset": 0.50,
        "formation_center_correction_gain": 0.75,
    }
    assert exp089["low_level_control"] == {
        **exp088["low_level_control"],
        "formation_center_activation_dmax_multiplier": 1.50,
        "formation_center_activation_dispersion_multiplier": 1.50,
    }
    assert exp089["algorithm"]["init_checkpoint"] == exp087["algorithm"]["init_checkpoint"]
    assert exp090["gather_point"] == {
        **exp089["gather_point"],
        "execution_slot_radius": 0.37,
    }
    assert exp090["algorithm"]["init_checkpoint"] == exp087["algorithm"]["init_checkpoint"]
    assert exp091["gather_point"] == {
        **exp090["gather_point"],
        "execution_slot_radius": 0.33,
    }
    assert exp092["gather_point"] == {
        **exp091["gather_point"],
        "execution_slot_radius": 0.35,
    }
    assert exp093["simulation"] == {
        **exp092["simulation"],
        "episode_length_s": 80.0,
    }
    assert exp093["experiment"]["eval_steps"] == 400
    assert exp093["evaluation"]["proxy_eval"]["steps"] == 400
    assert exp094["simulation"] == {
        **exp093["simulation"],
        "episode_length_s": 96.0,
    }
    assert exp094["experiment"]["eval_steps"] == 480
    assert exp095["simulation"] == {
        **exp094["simulation"],
        "episode_length_s": 112.0,
    }
    assert exp095["experiment"]["eval_steps"] == 560
    assert exp095["evaluation"]["proxy_eval"]["steps"] == 560
    assert exp096["simulation"] == {
        **exp095["simulation"],
        "episode_length_s": 128.0,
    }
    assert exp096["experiment"]["eval_steps"] == 640
    assert exp096["evaluation"]["proxy_eval"]["steps"] == 640
    assert exp097["simulation"] == exp096["simulation"]
    assert exp097["experiment"]["checkpoint_interval"] == 256
    assert exp097["algorithm"]["learning_rate"] == exp096["algorithm"]["learning_rate"]
    assert cfg_from_experiment(
        ROOT / "configs/experiment/exp089_structured_bicycle_quintic_map25_flatness_center_early.yaml"
    ).low_level_control.formation_center_correction_gain == pytest.approx(0.75)
