from __future__ import annotations

import math

import gymnasium as gym
import pytest
import torch

from _common import cfg_from_experiment, load_yaml
from _skrl_metadata import CheckpointCompatibilityError, validate_checkpoint_compatibility
from exp156_statistics import (
    clopper_pearson_lower,
    clopper_pearson_upper,
    strict_cell_acceptance,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.action_interpreter import (
    DIFFERENTIAL_PRIMITIVE_ACTION_COUNT,
    PRIMITIVE_HOLD,
    PRIMITIVE_REVERSE,
    PRIMITIVE_SPIN,
    PRIMITIVE_YIELD,
    decode_action,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.communication import (
    TieredCommunicationCache,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.gathering_env import (
    MultiRoverGatheringCore,
    MultiRoverGatheringSKRLEnv,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.simple_controller import (
    ControlCommand,
)
from lunar_rover_tasks.utils.math_utils import rotate_2d, wrap_to_pi
from train_skrl_mappo import SKRLCategoricalPolicy, build_skrl_mappo_models


CONFIG = "configs/experiment/exp156_differential_multiscale_ablation.yaml"


def _cfg(num_envs: int = 2):
    cfg = cfg_from_experiment(CONFIG)
    cfg.simulation.device = "cpu"
    cfg.simulation.num_envs = num_envs
    return cfg


def test_exp156_public_contract() -> None:
    cfg = _cfg()
    raw = load_yaml(CONFIG)
    assert cfg.actor_obs_dim == 295
    assert cfg.critic_state_dim == 950
    assert cfg.planner.action_dim == DIFFERENTIAL_PRIMITIVE_ACTION_COUNT == 47
    assert cfg.low_level_control.kinematic_model == "differential_drive"
    assert cfg.state.include_multiscale_agent_terrain
    assert cfg.reward_weights.oracle == 0.0
    assert raw["algorithm"]["entropy_schedule_timesteps"] == 153_600
    assert raw["algorithm"]["bc_updates"] == 0
    assert raw["algorithm"]["init_checkpoint"] is None


def test_all_47_actions_decode_with_expected_escape_semantics() -> None:
    cfg = _cfg(1)
    positions = torch.zeros(1, 47, 3)
    yaws = torch.zeros(1, 47)
    actions = torch.arange(47).unsqueeze(0)
    decoded = decode_action(actions, positions, yaws, cfg.planner)
    assert decoded.local_subgoal_xy.shape == (1, 47, 2)
    assert decoded.primitive_type[0, 0] == PRIMITIVE_HOLD
    assert torch.equal(decoded.local_subgoal_xy[0, 0], torch.zeros(2))
    assert torch.all(decoded.primitive_type[0, 40:43] == PRIMITIVE_REVERSE)
    assert torch.allclose(
        decoded.reference_speed[0, 40:43],
        torch.full((3,), -0.45),
    )
    assert torch.all(decoded.motion_direction[0, 40:43] == -1.0)
    assert torch.all(decoded.primitive_type[0, 43:45] == PRIMITIVE_SPIN)
    assert decoded.planned_yaw_delta[0, 43] == pytest.approx(math.pi / 4.0)
    assert decoded.planned_yaw_delta[0, 44] == pytest.approx(-math.pi / 4.0)
    assert torch.all(decoded.primitive_type[0, 45:47] == PRIMITIVE_YIELD)


def test_hold_spin_reverse_and_yield_execute_differentially() -> None:
    core = MultiRoverGatheringCore(_cfg())
    before = core.positions.clone()
    hold = core.step(torch.zeros(2, 4, dtype=torch.long))
    assert torch.equal(core.positions, before)
    assert torch.count_nonzero(hold.info["wheel_commands"]["left_radps"]) == 0
    assert torch.count_nonzero(hold.info["wheel_commands"]["right_radps"]) == 0

    spin = core.step(torch.full((2, 4), 43, dtype=torch.long))
    assert torch.allclose(spin.info["control"].linear, torch.zeros(2, 4))
    assert torch.all(
        spin.info["wheel_commands"]["left_radps"]
        * spin.info["wheel_commands"]["right_radps"]
        < 0.0
    )

    reverse = core.step(torch.full((2, 4), 41, dtype=torch.long))
    assert torch.all(reverse.info["control"].linear < 0.0)
    yielding = core.step(torch.full((2, 4), 45, dtype=torch.long))
    assert torch.all(yielding.info["control"].linear > 0.0)
    assert torch.all(yielding.info["control"].angular != 0.0)


def test_wheel_clip_precedes_effective_differential_motion() -> None:
    core = MultiRoverGatheringCore(_cfg(1))
    command = ControlCommand(
        linear=torch.full((1, 4), 100.0),
        angular=torch.full((1, 4), 100.0),
    )
    core._integrate(command)
    assert core.last_left_wheel_speed.abs().amax() <= 18.0
    assert core.last_right_wheel_speed.abs().amax() <= 18.0
    expected_v = 0.5 * 0.098 * (
        core.last_left_wheel_speed + core.last_right_wheel_speed
    )
    expected_w = 0.098 * (
        core.last_right_wheel_speed - core.last_left_wheel_speed
    ) / 0.376
    assert torch.allclose(
        torch.linalg.norm(core.velocities_xy, dim=-1),
        expected_v.abs() * core.last_terrain_speed_scale,
    )
    assert torch.allclose(
        core.angular_velocities,
        expected_w * core.last_terrain_speed_scale,
    )


def test_v10_communication_intent_and_12m_clearing() -> None:
    cache = TieredCommunicationCache(
        num_envs=1,
        n_agents=2,
        max_neighbors=1,
        device="cpu",
        include_plan_intent=True,
        include_plan_yaw=True,
    )
    positions = torch.tensor([[[0.0, 0.0, 0.0], [11.9, 0.0, 0.0]]])
    velocities = torch.ones(1, 2, 2)
    yaws = torch.zeros(1, 2)
    terrain = torch.ones(1, 2, 5)
    plan = positions.clone()
    plan[..., 0] += 0.8
    speed = torch.tensor([[0.45, 1.15]])
    yaw_delta = torch.tensor([[0.0, math.pi / 4.0]])
    token = torch.tensor([[-0.5, 0.75]])
    cache.reset(
        torch.tensor([0]),
        positions,
        velocities,
        yaws,
        terrain,
        committed_world_subgoal=plan,
        committed_reference_speed=speed,
        coordination_token=token,
        committed_planned_yaw_delta=yaw_delta,
    )
    full = cache.snapshot().features.reshape(1, 2, 1, 17)[0, 0, 0]
    assert full[14] == pytest.approx(1.15)
    assert full[15] == pytest.approx(math.pi / 4.0)
    assert full[16] == pytest.approx(0.75)

    positions[0, 1, 0] = 12.1
    cache.advance(
        dt=0.2,
        positions=positions,
        velocities_xy=velocities,
        yaws=yaws,
        terrain_summary=terrain,
        committed_world_subgoal=plan,
        committed_reference_speed=speed,
        coordination_token=token,
        committed_planned_yaw_delta=yaw_delta,
    )
    sparse = cache.snapshot().features.reshape(1, 2, 1, 17)[0, 0, 0]
    assert torch.count_nonzero(sparse[2:4]) == 0
    assert torch.count_nonzero(sparse[6:11]) == 0
    assert torch.count_nonzero(sparse[12:17]) == 0


@pytest.mark.parametrize(
    "architecture",
    ["multiscale_n0_mlp", "multiscale_n1_cnn", "multiscale_n2_path_conditioned"],
)
def test_three_actor_candidates_share_295_47_950_contract(architecture: str) -> None:
    env = MultiRoverGatheringSKRLEnv(_cfg())
    assert isinstance(env.action_spaces["rover_0"], gym.spaces.Discrete)
    models = build_skrl_mappo_models(
        env,
        actor_architecture=architecture,
        critic_architecture="structured_multiscale_v3",
    )
    policy = models["rover_0"]["policy"]
    critic = models["rover_0"]["value"]
    assert isinstance(policy, SKRLCategoricalPolicy)
    assert sum(parameter.numel() for parameter in policy.parameters()) <= 120_000
    actor_obs, critic_state = env.core.get_observations()
    logits, _ = policy.compute({"observations": actor_obs.reshape(-1, 295)})
    values, _ = critic.compute({"states": critic_state}, role="value")
    assert logits.shape == (8, 47)
    assert values.shape == (2, 1)
    assert torch.isfinite(logits).all() and torch.isfinite(values).all()


def test_n2_materializes_grid_sample_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    env = MultiRoverGatheringSKRLEnv(_cfg())
    models = build_skrl_mappo_models(
        env,
        actor_architecture="multiscale_n2_path_conditioned",
        critic_architecture="structured_multiscale_v3",
    )
    policy = models["rover_0"]["policy"]
    actor_obs, _ = env.core.get_observations()
    original_grid_sample = torch.nn.functional.grid_sample
    calls = []

    def checked_grid_sample(input_tensor, grid, *args, **kwargs):
        calls.append((input_tensor.is_contiguous(), grid.is_contiguous()))
        return original_grid_sample(input_tensor, grid, *args, **kwargs)

    monkeypatch.setattr(torch.nn.functional, "grid_sample", checked_grid_sample)
    logits, _ = policy.compute({"observations": actor_obs.reshape(-1, 295)})
    assert calls == [(True, True), (True, True)]
    assert torch.isfinite(logits).all()


def test_old_291_by_40_checkpoint_is_rejected() -> None:
    cfg = _cfg()
    checkpoint = {
        "metadata": {
            "observation_schema_version": "ego_v9_multiscale_intent",
            "actor_obs_dim": 291,
            "critic_state_dim": 54,
            "action_type": "spatiotemporal_primitives",
            "action_dim": 40,
            "action_distribution": "categorical",
        }
    }
    with pytest.raises(CheckpointCompatibilityError):
        validate_checkpoint_compatibility(checkpoint, cfg)


def test_oracle_and_unsent_global_diagnostics_cannot_change_execution() -> None:
    first = MultiRoverGatheringCore(_cfg())
    second = MultiRoverGatheringCore(_cfg())
    second.oracle_point.add_(torch.tensor([7.0, -5.0, 0.0]))
    second.gather_slot_points.uniform_(-20.0, 20.0)
    second.execution_slot_points.uniform_(-20.0, 20.0)
    second.oracle_search_objective.add_(1000.0)
    action = torch.tensor([[40, 43, 45, 1], [42, 44, 46, 2]])
    first_output = first.step(action)
    second_output = second.step(action)
    assert torch.equal(
        first_output.info["trajectory"].packed,
        second_output.info["trajectory"].packed,
    )
    assert torch.equal(first_output.info["control"].packed, second_output.info["control"].packed)
    assert torch.equal(
        first_output.info["wheel_commands"]["left_radps"],
        second_output.info["wheel_commands"]["left_radps"],
    )
    assert torch.equal(
        first_output.info["wheel_commands"]["right_radps"],
        second_output.info["wheel_commands"]["right_radps"],
    )


def test_actor_observation_and_logits_are_se2_invariant() -> None:
    cfg = _cfg(1)
    cfg.terrain.type = "flat_proxy"
    cfg.terrain.dynamics_enabled = False
    cfg.terrain.amplitude = 0.0
    cfg.terrain.crater_count = 0
    cfg.terrain.topology_profile = "open"
    first = MultiRoverGatheringCore(cfg)
    second = MultiRoverGatheringCore(cfg)
    first.positions[..., :2] = torch.tensor(
        [[[-1.0, -0.5], [1.2, -0.4], [-0.8, 1.0], [1.1, 0.9]]]
    )
    first.positions[..., 2] = 0.0
    first.yaws[:] = torch.tensor([[0.2, -1.0, 2.2, -2.4]])
    first.velocities_xy[:] = torch.tensor(
        [[[0.2, -0.1], [-0.3, 0.4], [0.1, 0.2], [-0.2, -0.1]]]
    )
    first.angular_velocities[:] = torch.tensor([[0.1, -0.2, 0.3, -0.4]])
    first.committed_plan_local_xy[:] = torch.tensor(
        [[[0.4, 0.2], [-0.4, 0.0], [0.0, 0.0], [0.8, -0.8]]]
    )
    first.committed_plan_world_subgoal[..., :2] = (
        first.positions[..., :2]
        + rotate_2d(first.committed_plan_local_xy, first.yaws)
    )
    first.committed_reference_speed[:] = torch.tensor([[0.45, -0.45, 0.0, 0.45]])
    first.committed_planned_yaw_delta[:] = torch.tensor(
        [[0.0, 0.0, math.pi / 4.0, 0.0]]
    )
    first.coordination_token[:] = torch.tensor([[-0.8, -0.2, 0.3, 0.9]])

    theta = 1.1
    translation = torch.tensor([3.5, -2.7])
    second.positions.copy_(first.positions)
    second.positions[..., :2] = rotate_2d(first.positions[..., :2], torch.tensor(theta)) + translation
    second.yaws[:] = wrap_to_pi(first.yaws + theta)
    second.velocities_xy[:] = rotate_2d(first.velocities_xy, torch.tensor(theta))
    second.angular_velocities.copy_(first.angular_velocities)
    second.committed_plan_local_xy.copy_(first.committed_plan_local_xy)
    second.committed_plan_world_subgoal.copy_(first.committed_plan_world_subgoal)
    second.committed_plan_world_subgoal[..., :2] = (
        rotate_2d(first.committed_plan_world_subgoal[..., :2], torch.tensor(theta))
        + translation
    )
    second.committed_reference_speed.copy_(first.committed_reference_speed)
    second.committed_planned_yaw_delta.copy_(first.committed_planned_yaw_delta)
    second.coordination_token.copy_(first.coordination_token)
    ids = torch.tensor([0])
    first._reset_communication(ids)
    second._reset_communication(ids)
    first_obs, _ = first.get_observations()
    second_obs, _ = second.get_observations()
    assert torch.allclose(first_obs, second_obs, atol=2.0e-5, rtol=1.0e-5)

    env = MultiRoverGatheringSKRLEnv(cfg)
    models = build_skrl_mappo_models(
        env,
        actor_architecture="multiscale_n0_mlp",
        critic_architecture="structured_multiscale_v3",
    )
    policy = models["rover_0"]["policy"]
    first_logits, _ = policy.compute({"observations": first_obs.reshape(-1, 295)})
    second_logits, _ = policy.compute({"observations": second_obs.reshape(-1, 295)})
    assert torch.allclose(first_logits, second_logits, atol=2.0e-5, rtol=1.0e-5)


def test_declared_192_episode_exact_confidence_boundaries() -> None:
    assert clopper_pearson_upper(0, 192) <= 0.02
    assert clopper_pearson_lower(180, 192) >= 0.90
    assert clopper_pearson_upper(12, 192) < 0.10
    assert clopper_pearson_upper(13, 192) >= 0.10
    passed = strict_cell_acceptance(
        success_count=180,
        collision_count=0,
        timeout_count=12,
        dmax_ratios=[0.18] * 192,
        bootstrap_samples=1000,
    )
    assert passed["passed"]
