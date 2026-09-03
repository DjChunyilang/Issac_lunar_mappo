from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from skrl.multi_agents.torch.mappo.mappo import compute_gae


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from _common import load_yaml  # noqa: E402
from dae_credit import (  # noqa: E402
    CounterfactualRewardModel,
    compute_dae_advantages,
    compute_raw_gae,
    dae_beta_schedule,
    factual_reward_model_loss,
)
from audit_exp158_dae import capture_core_state, restore_core_state  # noqa: E402
from compare_exp158_pair import stratified_paired_bootstrap  # noqa: E402
from lunar_rover_tasks.tasks.multi_rover_gathering.gathering_env import (  # noqa: E402
    MultiRoverGatheringCore,
)
from _common import cfg_from_experiment  # noqa: E402
from train_skrl_mappo import skrl_mappo_checkpoint_payload  # noqa: E402


def test_beta_schedule_has_fixed_warmup_ramp_and_plateau() -> None:
    assert dae_beta_schedule(1) == 0.0
    assert dae_beta_schedule(128) == 0.0
    assert dae_beta_schedule(129) == pytest.approx(0.3 / 128.0)
    assert dae_beta_schedule(192) == pytest.approx(0.15)
    assert dae_beta_schedule(256) == pytest.approx(0.3)
    assert dae_beta_schedule(2400) == pytest.approx(0.3)


def test_raw_gae_and_beta_zero_match_existing_shared_team_path() -> None:
    generator = torch.Generator().manual_seed(158)
    rewards = torch.randn((7, 5, 1), generator=generator)
    values = torch.randn((7, 5, 1), generator=generator)
    last_values = torch.randn((5, 1), generator=generator)
    terminated = torch.zeros((7, 5, 1), dtype=torch.bool)
    truncated = torch.zeros_like(terminated)
    terminated[4, 2] = True
    truncated[5, 3] = True
    expected_returns, expected_advantages = compute_gae(
        rewards=rewards,
        terminated=terminated,
        truncated=truncated,
        values=values,
        last_values=last_values,
        discount_factor=0.99,
        lambda_coefficient=0.95,
        time_limit_bootstrap=False,
    )
    returns, raw = compute_raw_gae(
        rewards=rewards,
        terminated=terminated,
        truncated=truncated,
        values=values,
        last_values=last_values,
        discount_factor=0.99,
        lambda_coefficient=0.95,
        time_limit_bootstrap=False,
    )
    actual = compute_dae_advantages(
        team_raw_advantages=raw,
        expected_counterfactual_rewards=torch.randn(
            (7, 5, 4, 1), generator=generator
        ),
        terminated=terminated,
        truncated=truncated,
        beta=0.0,
        discount_factor=0.99,
        lambda_coefficient=0.95,
    )
    assert torch.equal(returns, expected_returns)
    for agent in range(4):
        assert torch.equal(actual[agent], expected_advantages)

    logits = torch.randn((4 * 7 * 5, 47), generator=generator, requires_grad=True)
    taken = torch.randint(0, 47, (4 * 7 * 5,), generator=generator)
    log_prob = torch.log_softmax(logits, dim=-1).gather(1, taken[:, None])
    standard_loss = -(log_prob * expected_advantages.reshape(-1, 1).repeat(4, 1)).mean()
    standard_gradient = torch.autograd.grad(standard_loss, logits, retain_graph=True)[0]
    dae_loss = -(log_prob * actual.reshape(-1, 1)).mean()
    dae_gradient = torch.autograd.grad(dae_loss, logits)[0]
    assert torch.equal(standard_gradient, dae_gradient)


def test_dae_recursion_matches_explicit_counterfactual_sum_and_stops_at_timeout() -> None:
    team = torch.tensor([[[1.0]], [[2.0]], [[3.0]], [[4.0]]])
    expected = torch.tensor(
        [
            [[[0.2]], [[0.4]]],
            [[[0.3]], [[0.5]]],
            [[[0.7]], [[0.9]]],
            [[[1.1]], [[1.3]]],
        ]
    ).permute(0, 2, 1, 3)
    # expected is [T=4, E=1, A=2, 1]
    terminated = torch.zeros((4, 1, 1), dtype=torch.bool)
    truncated = torch.zeros_like(terminated)
    truncated[1] = True
    beta = 0.3
    gamma = 0.99
    lam = 0.95
    actual = compute_dae_advantages(
        team_raw_advantages=team,
        expected_counterfactual_rewards=expected,
        terminated=terminated,
        truncated=truncated,
        beta=beta,
        discount_factor=gamma,
        lambda_coefficient=lam,
    )
    traces = torch.zeros((2, 4, 1, 1))
    for agent in range(2):
        for start in range(4):
            coefficient = 1.0
            for step in range(start, 4):
                traces[agent, start, 0, 0] += coefficient * expected[
                    step, 0, agent, 0
                ]
                if terminated[step, 0, 0] or truncated[step, 0, 0]:
                    break
                coefficient *= gamma * lam * beta
    raw = team.unsqueeze(0) - beta * traces
    explicit = (raw - raw.mean()) / (raw.std() + 1.0e-8)
    assert torch.allclose(actual, explicit, atol=1.0e-7, rtol=0.0)


def test_counterfactual_model_masks_query_and_vectorizes_all_actions() -> None:
    torch.manual_seed(17)
    model = CounterfactualRewardModel()
    states = torch.randn((2, 950))
    actions = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    queries = torch.tensor([1, 3])
    candidates = torch.tensor([9, 10])
    baseline = model(states, actions, queries, candidates)
    changed_query = actions.clone()
    changed_query[0, 1] = 20
    changed_query[1, 3] = 21
    assert torch.allclose(
        baseline,
        model(states, changed_query, queries, candidates),
        atol=0.0,
        rtol=0.0,
    )
    changed_other = actions.clone()
    changed_other[:, 0] = torch.tensor([20, 21])
    assert not torch.allclose(
        baseline,
        model(states, changed_other, queries, candidates),
    )

    all_predictions = model.predict_all_actions(states, actions, chunk_size=31)
    assert all_predictions.shape == (2, 4, 47)
    for sample in range(2):
        for agent in range(4):
            loop = []
            for candidate in range(47):
                loop.append(
                    model(
                        states[sample : sample + 1],
                        actions[sample : sample + 1],
                        torch.tensor([agent]),
                        torch.tensor([candidate]),
                    )[0, 0]
                )
            assert torch.allclose(
                all_predictions[sample, agent],
                torch.stack(loop),
                atol=1.0e-6,
                rtol=0.0,
            )


def test_reward_model_factual_loss_does_not_touch_unrelated_actor() -> None:
    torch.manual_seed(23)
    model = CounterfactualRewardModel()
    actor = torch.nn.Linear(11, 47)
    actor_before = [parameter.detach().clone() for parameter in actor.parameters()]
    states = torch.randn((8, 950))
    actions = torch.randint(0, 47, (8, 4))
    rewards = torch.randn((8, 1))
    model.normalizer.update(rewards)
    optimizer = torch.optim.Adam(model.parameters(), lr=3.0e-4)
    loss = factual_reward_model_loss(model, states, actions, rewards)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    assert all(parameter.grad is None for parameter in actor.parameters())
    for before, after in zip(actor_before, actor.parameters(), strict=True):
        assert torch.equal(before, after)


def test_exp158_configs_lock_interface_and_only_dae_arm_enables_model() -> None:
    h1_gae = load_yaml(ROOT / "configs/experiment/exp158_h1_gae.yaml")
    h1_dae = load_yaml(ROOT / "configs/experiment/exp158_h1_dae.yaml")
    strict_gae = load_yaml(ROOT / "configs/experiment/exp158_strict_gae.yaml")
    strict_dae = load_yaml(ROOT / "configs/experiment/exp158_strict_dae.yaml")
    assert h1_gae["observation"]["schema_version"] == "ego_v11_multiscale_site_belief"
    assert strict_gae["observation"]["schema_version"] == "ego_v10_multiscale_diff_intent"
    assert h1_gae["reward"]["weights"]["oracle"] == pytest.approx(0.5)
    assert strict_gae["reward"]["weights"]["oracle"] == pytest.approx(0.0)
    for config in (h1_gae, h1_dae, strict_gae, strict_dae):
        assert config["planner"]["action_dim"] == 47
        assert config["state"]["include_multiscale_agent_terrain"] is True
        assert config["algorithm"]["actor_architecture"] == "multiscale_n1_cnn"
        assert config["algorithm"]["bc_updates"] == 0
        assert config["algorithm"]["init_checkpoint"] is None
        assert config["bounded_curriculum"]["transition_mode"] == "fixed_schedule"
        assert [stage["policy_iterations"] for stage in config["bounded_curriculum"]["stages"]] == [800, 800, 800]
    assert h1_gae["algorithm"]["advantage_estimator"] == "gae"
    assert strict_gae["algorithm"]["advantage_estimator"] == "gae"
    assert h1_dae["algorithm"]["advantage_estimator"] == "dae"
    assert strict_dae["algorithm"]["advantage_estimator"] == "dae"
    assert h1_dae["algorithm"]["dae"]["beta_target"] == pytest.approx(0.3)


def test_frozen_core_snapshot_replays_one_step_exactly() -> None:
    cfg = cfg_from_experiment(ROOT / "configs/experiment/exp158_h1_gae.yaml")
    cfg.simulation.device = "cpu"
    cfg.simulation.num_envs = 2
    cfg.seed = 158
    core = MultiRoverGatheringCore(cfg)
    snapshot = capture_core_state(core)
    actions = torch.tensor([[1, 43, 40, 45], [2, 44, 41, 46]])
    first = core.step(actions)
    first_observation = first.actor_obs.clone()
    first_reward = first.rewards.clone()
    first_left = first.info["wheel_commands"]["left_radps"].clone()
    first_right = first.info["wheel_commands"]["right_radps"].clone()
    restore_core_state(core, snapshot)
    second = core.step(actions)
    assert torch.equal(first_reward, second.rewards)
    assert torch.equal(first_observation, second.actor_obs)
    assert torch.equal(first_left, second.info["wheel_commands"]["left_radps"])
    assert torch.equal(first_right, second.info["wheel_commands"]["right_radps"])


def test_stratified_paired_bootstrap_preserves_cell_pairing() -> None:
    def report(dae: bool) -> dict:
        cells = []
        for cell in ("near_open", "far_bottleneck"):
            episodes = []
            for index in range(20):
                episodes.append(
                    {
                        "success": dae and index < 10,
                        "collision": False,
                        "timeout": not dae,
                        "dmax_ratio": 0.2 if dae else 0.4,
                    }
                )
            cells.append({"cell": cell, "metrics": {"episode_metrics": episodes}})
        return {"cells": cells}

    effect = stratified_paired_bootstrap(
        report(False),
        report(True),
        metric="success",
        transform=lambda gae, dae: float(dae.mean() - gae.mean()),
        samples=1000,
        seed=158,
    )
    assert effect["point"] == pytest.approx(0.5)
    assert effect["lower_95"] > 0.0


def test_dae_checkpoint_keeps_training_model_outside_deployed_policy_state() -> None:
    policy = torch.nn.Linear(5, 47)
    value = torch.nn.Linear(7, 1)
    possible_agents = [f"rover_{index}" for index in range(4)]
    models = {
        agent: {"policy": policy, "value": value} for agent in possible_agents
    }
    reward_model = CounterfactualRewardModel()
    reward_optimizer = torch.optim.Adam(reward_model.parameters(), lr=3.0e-4)
    raw = load_yaml(ROOT / "configs/experiment/exp158_h1_dae.yaml")
    payload = skrl_mappo_checkpoint_payload(
        models,
        possible_agents,
        raw_cfg=raw,
        shared_actor=True,
        centralized_critic=True,
        shared_value=True,
        timesteps=256,
        observation_schema_version="ego_v11_multiscale_site_belief",
        actor_obs_dim=407,
        critic_state_dim=950,
        dae_reward_model=reward_model,
        dae_reward_model_optimizer=reward_optimizer,
        dae_update_count=4,
        dae_beta=0.3,
    )
    assert payload["dae_training"]["deployable"] is False
    assert payload["dae_training"]["update_count"] == 4
    assert payload["metadata"]["advantage_estimator"] == "dae"
    assert set(payload["rover_0"]) == {"policy", "value"}
    assert all("reward" not in key for key in payload["rover_0"]["policy"])
