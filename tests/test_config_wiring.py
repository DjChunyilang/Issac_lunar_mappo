from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from _common import cfg_from_experiment, load_yaml  # noqa: E402


def test_yaml_extends_deep_merges_nested_experiment_sections(tmp_path: Path) -> None:
    base_path = tmp_path / "base.yaml"
    child_path = tmp_path / "child.yaml"
    base_path.write_text(
        yaml.safe_dump(
            {
                "experiment": {"name": "base", "seed": 23},
                "low_level_control": {
                    "formation_center_correction_gain": 0.55,
                    "formation_center_correction_max_offset": 0.35,
                },
            }
        ),
        encoding="utf-8",
    )
    child_path.write_text(
        yaml.safe_dump(
            {
                "extends": "base.yaml",
                "experiment": {"name": "child"},
                "low_level_control": {"formation_center_correction_gain": 0.75},
            }
        ),
        encoding="utf-8",
    )

    raw = load_yaml(child_path)
    cfg = cfg_from_experiment(child_path)

    assert raw["experiment"] == {"name": "child", "seed": 23}
    assert raw["low_level_control"] == {
        "formation_center_correction_gain": 0.75,
        "formation_center_correction_max_offset": 0.35,
    }
    assert cfg.low_level_control.formation_center_correction_gain == pytest.approx(0.75)
    assert cfg.low_level_control.formation_center_correction_max_offset == pytest.approx(0.35)


def test_yaml_extends_rejects_cycles(tmp_path: Path) -> None:
    first = tmp_path / "first.yaml"
    second = tmp_path / "second.yaml"
    first.write_text("extends: second.yaml\n", encoding="utf-8")
    second.write_text("extends: first.yaml\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Cyclic YAML extends chain"):
        load_yaml(first)


def test_oracle_ablation_config_is_wired() -> None:
    cfg = cfg_from_experiment(ROOT / "configs/experiment/exp_002_oracle_ablation.yaml")
    assert cfg.reward_weights.oracle == 0.0


def test_legacy_oracle_weight_key_fails_fast(tmp_path: Path) -> None:
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "experiment": {"name": "legacy_oracle_key"},
                "reward": {"oracle_weight": 0.0},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"reward\.weights\.oracle"):
        cfg_from_experiment(config_path)


def test_split_reward_config_is_not_treated_as_experiment() -> None:
    with pytest.raises(ValueError, match="does not merge"):
        cfg_from_experiment(ROOT / "configs/reward/reward_ablation_no_oracle.yaml")


def test_legacy_obstacle_collision_coefficient_fails_fast(tmp_path: Path) -> None:
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "experiment": {"name": "legacy_obstacle_collision"},
                "reward": {"coefficients": {"obstacle_collision": 8.0}},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"reward\.coefficients\.obstacle_collision"):
        cfg_from_experiment(config_path)


def test_obstacle_collision_key_is_removed_from_source_and_configs() -> None:
    key = "obstacle_" + "collision"
    checked_roots = [ROOT / "source", ROOT / "scripts", ROOT / "configs"]
    text_suffixes = {".py", ".yaml", ".yml", ".toml", ".md", ".txt"}
    offenders = [
        path.relative_to(ROOT)
        for root in checked_roots
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix in text_suffixes
        and key in path.read_text(encoding="utf-8")
    ]
    assert offenders == []


def test_observation_communication_radius_is_wired(tmp_path: Path) -> None:
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "experiment": {"name": "communication_radius"},
                "observation": {"communication_radius": 2.5},
            }
        ),
        encoding="utf-8",
    )

    cfg = cfg_from_experiment(config_path)
    assert cfg.observation.communication_radius == 2.5


def test_observation_schema_version_is_wired_to_terminal_gate_dims(tmp_path: Path) -> None:
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "experiment": {"name": "terminal_gate_observation"},
                "observation": {"schema_version": "ego_v4_terminal_gate"},
            }
        ),
        encoding="utf-8",
    )

    cfg = cfg_from_experiment(config_path)
    assert cfg.observation.schema_version == "ego_v4_terminal_gate"
    assert cfg.actor_obs_dim == 91
    assert cfg.critic_state_dim == 55


def test_gather_site_goal_schema_requires_an_explicit_execution_contract(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "experiment": {"name": "gather_site_goal"},
                "task": {"explicit_goal_in_execution": True},
                "observation": {"schema_version": "ego_v5_gather_site_goal"},
            }
        ),
        encoding="utf-8",
    )

    cfg = cfg_from_experiment(config_path)
    assert cfg.task.explicit_goal_in_execution is True
    assert cfg.observation.schema_version == "ego_v5_gather_site_goal"
    assert cfg.actor_obs_dim == 89
    assert cfg.critic_state_dim == 54

    config_path.write_text(
        yaml.safe_dump(
            {
                "experiment": {"name": "bad_gather_site_goal"},
                "observation": {"schema_version": "ego_v5_gather_site_goal"},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="explicit_goal_in_execution"):
        cfg_from_experiment(config_path)


def test_gather_slot_goal_schema_uses_the_same_explicit_execution_contract(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "experiment": {"name": "gather_slot_goal"},
                "task": {"explicit_goal_in_execution": True},
                "observation": {"schema_version": "ego_v6_gather_slot_goal"},
                "gather_point": {"execution_slot_radius": 0.35},
            }
        ),
        encoding="utf-8",
    )

    cfg = cfg_from_experiment(config_path)
    assert cfg.observation.schema_version == "ego_v6_gather_slot_goal"
    assert cfg.actor_obs_dim == 89
    assert cfg.gather_point.execution_slot_radius == pytest.approx(0.35)


def test_positive_search_robustness_requires_enough_envelope_samples(tmp_path: Path) -> None:
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "experiment": {"name": "robust_search"},
                "gather_point": {
                    "robustness_radius": 0.12,
                    "robustness_samples": 8,
                },
            }
        ),
        encoding="utf-8",
    )
    assert cfg_from_experiment(config_path).gather_point.robustness_radius == pytest.approx(
        0.12
    )

    config_path.write_text(
        yaml.safe_dump(
            {
                "experiment": {"name": "bad_robust_search"},
                "gather_point": {
                    "robustness_radius": 0.12,
                    "robustness_samples": 3,
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="robustness_samples"):
        cfg_from_experiment(config_path)


def test_formation_center_correction_validation(tmp_path: Path) -> None:
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "experiment": {"name": "formation_center_correction"},
                "low_level_control": {
                    "formation_center_correction_enabled": True,
                    "formation_center_activation_dmax_multiplier": 1.75,
                    "formation_center_activation_dispersion_multiplier": 1.5,
                    "formation_center_correction_max_offset": 0.35,
                    "formation_center_correction_gain": 0.55,
                    "formation_center_local_flatness_search_enabled": True,
                    "formation_center_local_flatness_search_radius": 0.25,
                    "formation_center_local_flatness_search_samples": 8,
                },
            }
        ),
        encoding="utf-8",
    )
    cfg = cfg_from_experiment(config_path)
    assert cfg.low_level_control.formation_center_correction_enabled
    assert cfg.low_level_control.formation_center_correction_gain == pytest.approx(0.55)
    assert cfg.low_level_control.formation_center_local_flatness_search_enabled

    config_path.write_text(
        yaml.safe_dump(
            {
                "experiment": {"name": "bad_formation_center_correction"},
                "low_level_control": {
                    "formation_center_correction_gain": 1.1,
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="formation_center_correction_gain"):
        cfg_from_experiment(config_path)

    config_path.write_text(
        yaml.safe_dump(
            {
                "experiment": {"name": "bad_local_flatness_search"},
                "low_level_control": {"formation_center_local_flatness_search_samples": 3},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="formation_center_local_flatness_search_samples"):
        cfg_from_experiment(config_path)


def test_terminal_slot_capture_validation(tmp_path: Path) -> None:
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "experiment": {"name": "terminal_slot_capture"},
                "low_level_control": {
                    "terminal_slot_capture_enabled": True,
                    "terminal_slot_capture_dmax_multiplier": 1.75,
                    "terminal_slot_capture_dispersion_multiplier": 1.5,
                    "terminal_slot_capture_blend": 0.65,
                },
            }
        ),
        encoding="utf-8",
    )
    cfg = cfg_from_experiment(config_path)
    assert cfg.low_level_control.terminal_slot_capture_enabled

    config_path.write_text(
        yaml.safe_dump(
            {
                "experiment": {"name": "bad_terminal_slot_capture"},
                "low_level_control": {"terminal_slot_capture_blend": -0.1},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="terminal_slot_capture_blend"):
        cfg_from_experiment(config_path)


def test_flat_geometry_capture_validation(tmp_path: Path) -> None:
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "experiment": {"name": "flat_geometry_capture"},
                "low_level_control": {
                    "flat_geometry_capture_enabled": True,
                    "flat_geometry_capture_dmax_multiplier": 1.75,
                    "flat_geometry_capture_dispersion_multiplier": 1.5,
                    "flat_geometry_capture_blend": 0.15,
                },
            }
        ),
        encoding="utf-8",
    )
    assert cfg_from_experiment(config_path).low_level_control.flat_geometry_capture_enabled

    config_path.write_text(
        yaml.safe_dump(
            {
                "experiment": {"name": "bad_flat_geometry_capture"},
                "low_level_control": {"flat_geometry_capture_blend": 1.1},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="flat_geometry_capture_blend"):
        cfg_from_experiment(config_path)


def test_execution_slot_reward_target_requires_a_slot_observation_schema(tmp_path: Path) -> None:
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "experiment": {"name": "slot_reward_target"},
                "task": {
                    "explicit_goal_in_execution": True,
                    "execution_slot_reward_target": True,
                },
                "observation": {"schema_version": "ego_v6_gather_slot_goal"},
            }
        ),
        encoding="utf-8",
    )

    cfg = cfg_from_experiment(config_path)
    assert cfg.task.execution_slot_reward_target is True

    config_path.write_text(
        yaml.safe_dump(
            {
                "experiment": {"name": "bad_slot_reward_target"},
                "task": {"execution_slot_reward_target": True},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="execution_slot_reward_target"):
        cfg_from_experiment(config_path)


def test_site_and_slot_goal_schema_requires_the_explicit_execution_contract(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "experiment": {"name": "site_and_slot_goal"},
                "task": {"explicit_goal_in_execution": True},
                "observation": {"schema_version": "ego_v7_gather_site_and_slot_goal"},
            }
        ),
        encoding="utf-8",
    )

    cfg = cfg_from_experiment(config_path)
    assert cfg.observation.schema_version == "ego_v7_gather_site_and_slot_goal"
    assert cfg.actor_obs_dim == 92
    assert cfg.critic_state_dim == 54


def test_unknown_observation_schema_version_fails_fast(tmp_path: Path) -> None:
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "experiment": {"name": "bad_observation_schema"},
                "observation": {"schema_version": "ego_v999"},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"observation\.schema_version"):
        cfg_from_experiment(config_path)


def test_state_terminal_min_pairwise_is_wired_without_actor_schema_change(tmp_path: Path) -> None:
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "experiment": {"name": "critic_terminal_min_pairwise"},
                "state": {"include_terminal_min_pairwise": True},
            }
        ),
        encoding="utf-8",
    )

    cfg = cfg_from_experiment(config_path)
    assert cfg.observation.schema_version == "ego_v3_local_terrain_grid"
    assert cfg.state.include_terminal_min_pairwise is True
    assert cfg.actor_obs_dim == 86
    assert cfg.critic_state_dim == 55


def test_state_dimension_keys_are_not_opened(tmp_path: Path) -> None:
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "experiment": {"name": "bad_state_dim"},
                "state": {"team_state_dim": 9},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"state\.team_state_dim"):
        cfg_from_experiment(config_path)


def test_observation_dimension_keys_are_not_opened(tmp_path: Path) -> None:
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "experiment": {"name": "bad_observation_dim"},
                "observation": {"max_neighbors": 8},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"observation\.max_neighbors"):
        cfg_from_experiment(config_path)


def test_initial_state_config_is_wired_and_validated(tmp_path: Path) -> None:
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "experiment": {"name": "initial_state"},
                "initial_state": {
                    "spawn_radius_min": 4.5,
                    "spawn_radius_max": 6.5,
                    "center_xy_range": 3.0,
                    "jitter_std": 0.45,
                },
            }
        ),
        encoding="utf-8",
    )

    cfg = cfg_from_experiment(config_path)

    assert cfg.initial_state.spawn_radius_min == pytest.approx(4.5)
    assert cfg.initial_state.spawn_radius_max == pytest.approx(6.5)
    assert cfg.initial_state.center_xy_range == pytest.approx(3.0)
    assert cfg.initial_state.jitter_std == pytest.approx(0.45)

    bad_path = tmp_path / "bad_initial_state.yaml"
    bad_path.write_text(
        yaml.safe_dump(
            {
                "experiment": {"name": "bad_initial_state"},
                "initial_state": {
                    "spawn_radius_min": 7.0,
                    "spawn_radius_max": 6.5,
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"spawn_radius_max"):
        cfg_from_experiment(bad_path)


def test_gather_point_search_and_flatness_config_is_wired(tmp_path: Path) -> None:
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "experiment": {"name": "terrain_aware_gather_point"},
                "gather_point": {
                    "search_method": "terrain_aware_multiresolution",
                    "coarse_grid_size": 7,
                    "refinement_grid_size": 3,
                    "refinement_levels": 1,
                    "search_margin": 2.25,
                    "global_fallback_enabled": False,
                    "global_grid_size": 17,
                    "global_beam_width": 12,
                    "global_refinement_levels": 1,
                    "global_max_envs_per_batch": 4,
                    "flatness_radius": 0.8,
                    "flatness_rings": 2,
                    "flatness_samples_per_ring": 8,
                    "max_height_range": 0.12,
                    "max_slope": 0.18,
                    "mean_distance_weight": 1.5,
                    "max_distance_weight": 0.4,
                    "path_risk_weight": 0.9,
                    "path_height_change_weight": 0.3,
                    "flatness_weight": 0.6,
                    "path_samples": 7,
                    "infeasible_penalty": 2500.0,
                    "max_envs_per_batch": 64,
                    "require_flat_for_success": False,
                },
            }
        ),
        encoding="utf-8",
    )

    cfg = cfg_from_experiment(config_path)

    assert cfg.gather_point.search_method == "terrain_aware_multiresolution"
    assert cfg.gather_point.coarse_grid_size == 7
    assert cfg.gather_point.refinement_grid_size == 3
    assert cfg.gather_point.refinement_levels == 1
    assert cfg.gather_point.search_margin == pytest.approx(2.25)
    assert cfg.gather_point.global_fallback_enabled is False
    assert cfg.gather_point.global_grid_size == 17
    assert cfg.gather_point.global_beam_width == 12
    assert cfg.gather_point.global_refinement_levels == 1
    assert cfg.gather_point.global_max_envs_per_batch == 4
    assert cfg.gather_point.flatness_radius == pytest.approx(0.8)
    assert cfg.gather_point.flatness_rings == 2
    assert cfg.gather_point.flatness_samples_per_ring == 8
    assert cfg.gather_point.max_height_range == pytest.approx(0.12)
    assert cfg.gather_point.max_slope == pytest.approx(0.18)
    assert cfg.gather_point.mean_distance_weight == pytest.approx(1.5)
    assert cfg.gather_point.max_distance_weight == pytest.approx(0.4)
    assert cfg.gather_point.path_risk_weight == pytest.approx(0.9)
    assert cfg.gather_point.path_height_change_weight == pytest.approx(0.3)
    assert cfg.gather_point.flatness_weight == pytest.approx(0.6)
    assert cfg.gather_point.path_samples == 7
    assert cfg.gather_point.infeasible_penalty == pytest.approx(2500.0)
    assert cfg.gather_point.max_envs_per_batch == 64
    assert cfg.gather_point.require_flat_for_success is False


@pytest.mark.parametrize(
    ("gather_point", "error"),
    [
        ({"search_method": "centroid_proxy"}, r"gather_point\.search_method"),
        ({"coarse_grid_size": 4}, r"coarse_grid_size"),
        ({"refinement_grid_size": 2}, r"refinement_grid_size"),
        ({"global_grid_size": 4}, r"global_grid_size"),
        ({"global_beam_width": 0}, r"global_beam_width"),
        ({"global_refinement_levels": -1}, r"global_refinement_levels"),
        ({"global_max_envs_per_batch": 0}, r"global_max_envs_per_batch"),
        ({"flatness_radius": 0.0}, r"flatness_radius"),
        ({"flatness_rings": 0}, r"flatness_rings"),
        ({"flatness_samples_per_ring": 3}, r"flatness_samples_per_ring"),
        ({"path_samples": 0}, r"path_samples"),
        ({"infeasible_penalty": 0.0}, r"infeasible_penalty"),
        ({"max_envs_per_batch": 0}, r"max_envs_per_batch"),
        ({"path_risk_weight": -0.1}, r"objective weights"),
    ],
)
def test_invalid_gather_point_config_fails_fast(
    tmp_path: Path,
    gather_point: dict,
    error: str,
) -> None:
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "experiment": {"name": "bad_gather_point"},
                "gather_point": gather_point,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=error):
        cfg_from_experiment(config_path)


def test_unknown_gather_point_key_fails_fast(tmp_path: Path) -> None:
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "experiment": {"name": "unknown_gather_point_key"},
                "gather_point": {"geometric_midpoint_proxy": True},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"gather_point\.geometric_midpoint_proxy"):
        cfg_from_experiment(config_path)


@pytest.mark.parametrize("config_path", sorted((ROOT / "configs/experiment").glob("*.yaml")))
def test_all_experiment_configs_parse(config_path: Path) -> None:
    cfg = cfg_from_experiment(config_path)
    assert cfg.task.n_agents == 4
    assert cfg.observation.communication_radius >= 0.0
