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
