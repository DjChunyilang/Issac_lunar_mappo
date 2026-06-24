from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from _common import cfg_from_experiment, load_yaml  # noqa: E402
from lunar_rover_tasks.tasks.multi_rover_gathering.gathering_env import (  # noqa: E402
    MultiRoverGatheringCore,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.gathering_env_cfg import (  # noqa: E402
    MultiRoverGatheringEnvCfg,
)


CONFIG = ROOT / "configs/experiment/exp019_randomized_terrain_safe_path_risk.yaml"


def test_exp019_config_enforces_safe_success_and_path_risk_contract() -> None:
    raw = load_yaml(CONFIG)
    cfg = cfg_from_experiment(CONFIG)

    assert raw["experiment"]["name"] == "exp019_randomized_terrain_safe_path_risk"
    assert raw["algorithm"]["mode"] == "pure_rl"
    assert raw["algorithm"]["update_mode"] == "shared_joint"
    assert raw["algorithm"]["bc_updates"] == 0
    assert cfg.safety.collision_distance == pytest.approx(0.28)
    assert cfg.success_thresholds.min_pairwise_distance == pytest.approx(0.42)
    assert cfg.safety.collision_distance < cfg.success_thresholds.min_pairwise_distance
    assert cfg.success_thresholds.min_pairwise_distance < cfg.success_thresholds.dmax
    assert cfg.safety.near_distance == pytest.approx(0.85)
    assert cfg.reward_coefficients.near_distance == pytest.approx(6.0)
    assert cfg.reward_coefficients.inter_agent_collision == pytest.approx(80.0)
    assert cfg.reward_coefficients.failure_penalty == pytest.approx(45.0)
    assert cfg.reward_weights.terrain == pytest.approx(0.30)
    assert cfg.reward_coefficients.path_terrain_mean_cost == pytest.approx(0.60)
    assert cfg.reward_coefficients.path_terrain_max_cost == pytest.approx(0.40)
    assert cfg.reward_coefficients.path_height_change_cost == pytest.approx(0.20)


def test_min_pairwise_distance_defaults_to_zero_for_old_configs() -> None:
    cfg = MultiRoverGatheringEnvCfg()

    assert cfg.success_thresholds.min_pairwise_distance == pytest.approx(0.0)


def test_invalid_min_pairwise_config_is_rejected(tmp_path: Path) -> None:
    raw = load_yaml(CONFIG)
    raw["success_thresholds"]["min_pairwise_distance"] = raw["safety"]["collision_distance"]
    path = tmp_path / "bad_exp019.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="min_pairwise_distance"):
        cfg_from_experiment(path)


def test_exp019_step_reports_path_risk_and_safe_gate_metrics() -> None:
    cfg = cfg_from_experiment(CONFIG)
    cfg.simulation.device = "cpu"
    cfg.simulation.num_envs = 4
    env = MultiRoverGatheringCore(cfg)

    action = torch.zeros(cfg.simulation.num_envs, cfg.task.n_agents, 2)
    out = env.step(action)
    path = out.info["path_terrain"]
    gates = out.info["success_gates"]

    assert set(path) == {"risk_mean", "risk_max", "height_change_mean"}
    assert path["risk_mean"].shape == (cfg.simulation.num_envs, cfg.task.n_agents)
    assert path["risk_max"].shape == (cfg.simulation.num_envs, cfg.task.n_agents)
    assert path["height_change_mean"].shape == (cfg.simulation.num_envs, cfg.task.n_agents)
    assert torch.isfinite(path["risk_mean"]).all()
    assert torch.isfinite(path["risk_max"]).all()
    assert torch.isfinite(path["height_change_mean"]).all()
    assert gates.min_pairwise_ok.shape == (cfg.simulation.num_envs,)
    assert torch.isfinite(out.rewards).all()
