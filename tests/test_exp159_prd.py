from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from _common import cfg_from_experiment, load_yaml  # noqa: E402
from audit_exp158_dae import capture_core_state, restore_core_state  # noqa: E402
from lunar_rover_tasks.tasks.multi_rover_gathering.analytical_prd import (  # noqa: E402
    _other_only_safety_baselines,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.gathering_env import (  # noqa: E402
    MultiRoverGatheringCore,
)
from prd_credit import compute_analytical_prd_advantages  # noqa: E402


def test_exp159_configs_preserve_reward_and_lock_single_step_prd() -> None:
    h1_gae = load_yaml(ROOT / "configs/experiment/exp159_h1_gae.yaml")
    h1_prd = load_yaml(ROOT / "configs/experiment/exp159_h1_prd.yaml")
    strict_gae = load_yaml(ROOT / "configs/experiment/exp159_strict_gae.yaml")
    strict_prd = load_yaml(ROOT / "configs/experiment/exp159_strict_prd.yaml")
    assert h1_gae["observation"]["schema_version"] == "ego_v11_multiscale_site_belief"
    assert strict_gae["observation"]["schema_version"] == "ego_v10_multiscale_diff_intent"
    assert h1_gae["reward"]["weights"]["oracle"] == pytest.approx(0.5)
    assert strict_gae["reward"]["weights"]["oracle"] == pytest.approx(0.0)
    for config in (h1_gae, h1_prd, strict_gae, strict_prd):
        assert config["task"]["analytical_prd_enabled"] is True
        assert config["planner"]["action_dim"] == 47
        assert config["planner"]["subgoal_filter"]["enabled"] is False
        assert config["algorithm"]["actor_credit_assignment"] == "none"
        assert config["algorithm"]["actor_credit_scale"] == 0.0
        assert config["algorithm"]["collision_constraint_enabled"] is False
        assert config["algorithm"]["bc_updates"] == 0
        assert config["algorithm"]["init_checkpoint"] is None
    assert h1_prd["algorithm"]["advantage_estimator"] == "analytical_prd_loo"
    assert strict_prd["algorithm"]["advantage_estimator"] == "analytical_prd_loo"
    assert h1_prd["algorithm"]["prd"] == {
        "baseline_scale": 1.0,
        "temporal_trace": False,
        "preserve_team_reward": True,
    }


def test_prd_diagnostics_do_not_change_team_reward_or_execution() -> None:
    base = cfg_from_experiment(ROOT / "configs/experiment/exp159_h1_gae.yaml")
    base.simulation.device = "cpu"
    base.simulation.num_envs = 4
    base.seed = 159
    enabled = cfg_from_experiment(ROOT / "configs/experiment/exp159_h1_gae.yaml")
    enabled.simulation.device = "cpu"
    enabled.simulation.num_envs = 4
    enabled.seed = 159
    enabled.task.analytical_prd_enabled = True
    base.task.analytical_prd_enabled = False
    base_env = MultiRoverGatheringCore(base)
    prd_env = MultiRoverGatheringCore(enabled)
    generator = torch.Generator().manual_seed(159)
    for _ in range(8):
        actions = torch.randint(0, 47, (4, 4), generator=generator)
        base_output = base_env.step(actions)
        prd_output = prd_env.step(actions)
        assert torch.equal(base_output.rewards, prd_output.rewards)
        assert torch.equal(base_output.actor_obs, prd_output.actor_obs)
        assert torch.equal(
            base_output.info["wheel_commands"]["left_radps"],
            prd_output.info["wheel_commands"]["left_radps"],
        )
        info = prd_output.info["analytical_prd"]
        assert float(info["team_reward_preservation_error"].amax()) == 0.0
        assert float(info["source_reconstruction_error"].amax()) <= 1.0e-6


def test_loo_baseline_is_invariant_to_own_47_actions() -> None:
    cfg = cfg_from_experiment(ROOT / "configs/experiment/exp159_h1_prd.yaml")
    cfg.simulation.device = "cpu"
    cfg.simulation.num_envs = 2
    cfg.seed = 159
    env = MultiRoverGatheringCore(cfg)
    snapshot = capture_core_state(env)
    base_actions = torch.tensor([[1, 10, 20, 30], [5, 15, 25, 35]])
    for agent in range(4):
        values = []
        for action in range(47):
            restore_core_state(env, snapshot)
            candidate = base_actions.clone()
            candidate[:, agent] = action
            output = env.step(candidate)
            values.append(output.info["analytical_prd"]["loo_baseline"][:, agent])
        stacked = torch.stack(values)
        assert float((stacked - stacked[:1]).abs().amax()) <= 1.0e-6


def test_single_collision_pair_is_removed_only_for_nonparticipants() -> None:
    cfg = cfg_from_experiment(ROOT / "configs/experiment/exp159_h1_prd.yaml")
    positions = torch.tensor(
        [[[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [3.0, 0.0, 0.0], [6.0, 0.0, 0.0]]]
    )
    _, collision, failure, participants = _other_only_safety_baselines(positions, cfg)
    assert torch.equal(participants, torch.tensor([[True, True, False, False]]))
    assert collision[0, 0] == 0.0
    assert collision[0, 1] == 0.0
    assert collision[0, 2] == pytest.approx(-100.0)
    assert collision[0, 3] == pytest.approx(-100.0)
    assert failure[0, 0] == 0.0
    assert failure[0, 1] == 0.0
    assert failure[0, 2] == pytest.approx(-55.0)
    assert failure[0, 3] == pytest.approx(-55.0)


def test_no_collision_or_oob_has_no_collision_failure_baseline() -> None:
    cfg = cfg_from_experiment(ROOT / "configs/experiment/exp159_h1_prd.yaml")
    positions = torch.tensor(
        [[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0], [6.0, 0.0, 0.0]]]
    )
    _, collision, failure, participants = _other_only_safety_baselines(positions, cfg)
    assert not participants.any()
    assert torch.equal(collision, torch.zeros_like(collision))
    assert torch.equal(failure, torch.zeros_like(failure))


def test_prd_advantage_is_single_step_subtraction_then_joint_normalization() -> None:
    team = torch.tensor([[[1.0]], [[2.0]], [[3.0]]])
    baseline = torch.tensor(
        [
            [[-1.0, -2.0, -3.0, -4.0]],
            [[0.1, 0.2, 0.3, 0.4]],
            [[5.0, 6.0, 7.0, 8.0]],
        ]
    )
    actual = compute_analytical_prd_advantages(
        team_raw_advantages=team,
        loo_baseline=baseline,
    )
    raw = team.unsqueeze(0) - baseline.permute(2, 0, 1).unsqueeze(-1)
    expected = (raw - raw.mean()) / (raw.std() + 1.0e-8)
    assert torch.equal(actual, expected)
