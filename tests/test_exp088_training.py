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
    exp098 = load_yaml(
        ROOT
        / "configs/experiment/exp098_structured_bicycle_quintic_map25_time96_strict_slot_capture.yaml"
    )
    exp099 = load_yaml(
        ROOT
        / "configs/experiment/exp099_structured_bicycle_quintic_map25_time96_full_center_correction.yaml"
    )
    exp100 = load_yaml(
        ROOT
        / "configs/experiment/exp100_structured_bicycle_quintic_map25_time96_local_flatness_center.yaml"
    )
    exp101 = load_yaml(
        ROOT
        / "configs/experiment/exp101_structured_bicycle_quintic_map25_time96_local_flatness_ppo.yaml"
    )
    exp102 = load_yaml(
        ROOT
        / "configs/experiment/exp102_structured_bicycle_quintic_map25_time96_wide_local_flatness_center.yaml"
    )
    exp103 = load_yaml(
        ROOT
        / "configs/experiment/exp103_structured_bicycle_quintic_map25_time96_flat_geometry_capture.yaml"
    )
    exp104 = load_yaml(
        ROOT
        / "configs/experiment/exp104_structured_bicycle_quintic_map25_time96_conditional_terminal_branches.yaml"
    )
    exp105 = load_yaml(
        ROOT
        / "configs/experiment/exp105_structured_bicycle_quintic_map25_time96_dynamic_flat_geometry_capture.yaml"
    )
    exp106 = load_yaml(
        ROOT
        / "configs/experiment/exp106_structured_bicycle_quintic_map25_time96_robust_flat_oracle05.yaml"
    )
    exp107 = load_yaml(
        ROOT
        / "configs/experiment/exp107_structured_bicycle_quintic_map25_time96_robust_flat_oracle075.yaml"
    )
    exp108 = load_yaml(
        ROOT
        / "configs/experiment/exp108_structured_bicycle_quintic_map25_time96_slots_radius33.yaml"
    )
    exp109 = load_yaml(
        ROOT
        / "configs/experiment/exp109_structured_bicycle_quintic_map25_time96_slots_radius34.yaml"
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
    assert exp098["simulation"] == exp094["simulation"]
    assert exp098["low_level_control"] == {
        **exp094["low_level_control"],
        "terminal_slot_capture_enabled": True,
        "terminal_slot_capture_dmax_multiplier": 1.00,
        "terminal_slot_capture_dispersion_multiplier": 1.00,
        "terminal_slot_capture_blend": 0.25,
    }
    assert exp099["simulation"] == exp094["simulation"]
    assert exp099["low_level_control"] == {
        **exp094["low_level_control"],
        "formation_center_correction_max_offset": 0.75,
        "formation_center_correction_gain": 1.00,
    }
    assert exp100["simulation"] == exp094["simulation"]
    assert exp100["low_level_control"] == {
        **exp094["low_level_control"],
        "formation_center_local_flatness_search_enabled": True,
        "formation_center_local_flatness_search_radius": 0.25,
        "formation_center_local_flatness_search_samples": 8,
        "formation_center_correction_max_offset": 0.35,
        "formation_center_correction_gain": 1.00,
    }
    assert exp101["simulation"] == exp100["simulation"]
    assert exp101["experiment"]["checkpoint_interval"] == 512
    assert exp101["algorithm"] == {
        **exp100["algorithm"],
        "learning_rate": 0.00001,
        "training_semantics": "exp101_structured_bicycle_quintic_map25_time96_local_flatness_ppo",
    }
    assert exp102["low_level_control"] == {
        **exp099["low_level_control"],
        "formation_center_local_flatness_search_enabled": True,
        "formation_center_local_flatness_search_radius": 0.50,
        "formation_center_local_flatness_search_samples": 16,
    }
    assert exp103["low_level_control"] == {
        **exp099["low_level_control"],
        "flat_geometry_capture_enabled": True,
        "flat_geometry_capture_dmax_multiplier": 1.75,
        "flat_geometry_capture_dispersion_multiplier": 1.75,
        "flat_geometry_capture_blend": 0.15,
    }
    assert exp104["low_level_control"] == {
        **exp102["low_level_control"],
        "flat_geometry_capture_enabled": True,
        "flat_geometry_capture_dmax_multiplier": 1.75,
        "flat_geometry_capture_dispersion_multiplier": 1.75,
        "flat_geometry_capture_blend": 0.15,
    }
    assert exp105["low_level_control"] == {
        **exp103["low_level_control"],
        "flat_geometry_capture_dynamic_assignment": True,
    }
    assert exp106["gather_point"] == {
        **exp099["gather_point"],
        "robustness_radius": 0.05,
        "robustness_samples": 8,
    }
    assert exp107["gather_point"] == {
        **exp099["gather_point"],
        "robustness_radius": 0.075,
        "robustness_samples": 8,
    }
    assert exp108["gather_point"] == {
        **exp099["gather_point"],
        "execution_slot_radius": 0.33,
    }
    assert exp109["gather_point"] == {
        **exp099["gather_point"],
        "execution_slot_radius": 0.34,
    }
    assert cfg_from_experiment(
        ROOT / "configs/experiment/exp089_structured_bicycle_quintic_map25_flatness_center_early.yaml"
    ).low_level_control.formation_center_correction_gain == pytest.approx(0.75)
    assert cfg_from_experiment(
        ROOT / "configs/experiment/exp105_structured_bicycle_quintic_map25_time96_dynamic_flat_geometry_capture.yaml"
    ).low_level_control.flat_geometry_capture_dynamic_assignment
