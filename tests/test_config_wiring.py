from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from _common import cfg_from_experiment  # noqa: E402


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


@pytest.mark.parametrize("config_path", sorted((ROOT / "configs/experiment").glob("*.yaml")))
def test_all_experiment_configs_parse(config_path: Path) -> None:
    cfg = cfg_from_experiment(config_path)
    assert cfg.task.n_agents == 4
    assert cfg.observation.communication_radius >= 0.0
