from __future__ import annotations

import math

import gymnasium as gym
import pytest
import torch

from _common import cfg_from_experiment, load_yaml
from _skrl_metadata import CheckpointCompatibilityError, validate_checkpoint_compatibility
from lunar_rover_tasks.tasks.multi_rover_gathering.action_interpreter import (
    SPATIOTEMPORAL_ACTION_COUNT,
    decode_action,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.communication import (
    TieredCommunicationCache,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.gathering_env import (
    MultiRoverGatheringCore,
    MultiRoverGatheringSKRLEnv,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.gathering_env_cfg import make_debug_cfg
from lunar_rover_tasks.tasks.multi_rover_gathering.terrain_features import (
    MULTISCALE_TERRAIN_DIM,
    build_multiscale_local_terrain_grids,
    build_multiscale_local_terrain_observation,
    make_terrain_runtime,
    query_terrain_features,
    randomize_terrain_runtime,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.mapf_diagnostics import (
    analyze_traversability_topology,
)
from train_skrl_mappo import (
    SKRLCategoricalPolicy,
    build_skrl_mappo_models,
    proxy_acceptance,
    strict_thresholds_from_config,
    _claim_curriculum_evaluation,
    _curriculum_gate_checks,
)


CONFIG = "configs/experiment/exp155_multiscale_network_ablation.yaml"


def _cfg(num_envs: int = 2):
    cfg = make_debug_cfg(num_envs=num_envs, device="cpu")
    cfg.observation.schema_version = "ego_v9_multiscale_intent"
    cfg.observation.communication_radius = 12.0
    cfg.planner.action_type = "spatiotemporal_primitives"
    cfg.planner.action_dim = SPATIOTEMPORAL_ACTION_COUNT
    cfg.planner.rho_max = 1.6
    cfg.planner.beta_max = math.pi / 3.0
    cfg.planner.subgoal_filter.enabled = False
    cfg.low_level_control.safety_projection_enabled = False
    cfg.trajectory_generator.geometry_method = "quintic"
    cfg.trajectory_generator.time_parameterization = "arc_length_reference_speed"
    cfg.low_level_control.tracking_point_mode = "planning_time"
    return cfg


def test_exp155_config_fixes_new_interface_and_pure_rl() -> None:
    cfg = cfg_from_experiment(CONFIG)
    raw = load_yaml(CONFIG)
    assert cfg.actor_obs_dim == 291
    assert cfg.planner.action_type == "spatiotemporal_primitives"
    assert cfg.planner.action_dim == 40
    assert raw["algorithm"]["bc_updates"] == 0
    assert raw["algorithm"]["init_checkpoint"] is None
    assert raw["experiment"]["num_envs"] == 256
    assert raw["experiment"]["rollout_steps"] == 64


def test_multiscale_grid_shapes_and_boundary_samples_match() -> None:
    cfg = _cfg(1)
    positions = torch.zeros(1, 4, 3)
    yaws = torch.zeros(1, 4)
    fine, medium, coarse = build_multiscale_local_terrain_grids(
        positions, yaws, cfg.terrain
    )
    flat = build_multiscale_local_terrain_observation(positions, yaws, cfg.terrain)
    assert fine.shape == (1, 4, 7, 9, 2)
    assert medium.shape == (1, 4, 3, 7, 2)
    assert coarse.shape == (1, 4, 4, 7, 2)
    assert flat.shape == (1, 4, MULTISCALE_TERRAIN_DIM)
    # x=0.8 and x=1.6 are deliberately duplicated across adjacent scales.
    assert torch.equal(
        fine[..., -1, (0, 2, 4, 6, 8), :],
        medium[..., 0, 1:6, :],
    )
    assert torch.equal(
        medium[..., -1, (1, 3, 5), :],
        coarse[..., 0, (2, 3, 4), :],
    )


def test_v9_observation_and_discrete_spaces() -> None:
    cfg = _cfg()
    env = MultiRoverGatheringCore(cfg)
    observation, state = env.get_observations()
    assert observation.shape == (2, 4, 291)
    assert state.shape == (2, 54)
    skrl_env = MultiRoverGatheringSKRLEnv(cfg)
    assert isinstance(skrl_env.action_spaces["rover_0"], gym.spaces.Discrete)
    assert skrl_env.action_spaces["rover_0"].n == 40


def test_spatiotemporal_actions_decode_hold_endpoint_and_speed() -> None:
    cfg = _cfg(1)
    positions = torch.zeros(1, 4, 3)
    yaws = torch.zeros(1, 4)
    decoded = decode_action(torch.tensor([[0, 1, 2, 3]]), positions, yaws, cfg.planner)
    assert torch.equal(decoded.local_subgoal_xy[0, 0], torch.zeros(2))
    assert torch.allclose(decoded.local_subgoal_xy[0, 1:], torch.tensor([[0.4, -0.4]] * 3))
    assert torch.allclose(decoded.reference_speed[0], torch.tensor([0.0, 0.45, 0.80, 1.15]))


def test_intent_message_is_full_inside_and_cleared_outside() -> None:
    cache = TieredCommunicationCache(
        num_envs=1,
        n_agents=2,
        max_neighbors=1,
        device="cpu",
        include_plan_intent=True,
    )
    positions = torch.tensor([[[0.0, 0.0, 0.0], [11.9, 0.0, 0.0]]])
    velocities = torch.zeros(1, 2, 2)
    yaws = torch.zeros(1, 2)
    terrain = torch.ones(1, 2, 5)
    plan = positions.clone()
    plan[0, 1, 0] += 1.0
    speed = torch.tensor([[0.45, 1.15]])
    token = torch.tensor([[-0.5, 0.75]])
    cache.reset(torch.tensor([0]), positions, velocities, yaws, terrain, plan, speed, token)
    message = cache.snapshot().features.reshape(1, 2, 1, 16)[0, 0, 0]
    assert torch.any(message[12:14] != 0.0)
    assert message[14] == pytest.approx(1.15)
    assert message[15] == pytest.approx(0.75)

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
    )
    sparse = cache.snapshot().features.reshape(1, 2, 1, 16)[0, 0, 0]
    assert torch.count_nonzero(sparse[2:4]) == 0
    assert torch.count_nonzero(sparse[6:11]) == 0
    assert torch.count_nonzero(sparse[12:16]) == 0


@pytest.mark.parametrize(
    "architecture",
    ["multiscale_n0_mlp", "multiscale_n1_cnn", "multiscale_n2_path_conditioned"],
)
def test_candidate_networks_are_categorical_finite_and_bounded(architecture: str) -> None:
    env = MultiRoverGatheringSKRLEnv(_cfg())
    models = build_skrl_mappo_models(
        env,
        actor_architecture=architecture,
        critic_architecture="structured_v1",
    )
    policy = models["rover_0"]["policy"]
    assert isinstance(policy, SKRLCategoricalPolicy)
    assert sum(parameter.numel() for parameter in policy.parameters()) <= 120_000
    observation, _ = env.reset()
    action, outputs = policy.act(
        {"observations": observation["rover_0"]}, role="policy"
    )
    assert action.shape == (2, 1)
    assert outputs["net_output"].shape == (2, 40)
    assert torch.isfinite(outputs["net_output"]).all()


def test_old_gaussian_checkpoint_is_rejected_by_v9_contract() -> None:
    cfg = _cfg()
    old = {
        "metadata": {
            "observation_schema_version": "ego_v8_decentralized_tiered",
            "actor_obs_dim": 101,
            "critic_state_dim": 54,
        }
    }
    with pytest.raises(CheckpointCompatibilityError):
        validate_checkpoint_compatibility(old, cfg)


def test_timeout_gate_is_strictly_below_point_one() -> None:
    thresholds = strict_thresholds_from_config(load_yaml(CONFIG))
    metrics = {
        "dmax_reduction_ratio": 0.20,
        "success_rate": 0.90,
        "collision_rate": 0.02,
        "timeout_rate": 0.099,
    }
    assert proxy_acceptance(metrics, thresholds)["passed"]
    metrics["timeout_rate"] = 0.10
    assert not proxy_acceptance(metrics, thresholds)["passed"]


def test_bounded_curriculum_has_fixed_three_stage_budget() -> None:
    raw = load_yaml("configs/experiment/exp155_formal_bounded.yaml")
    bounded = raw["bounded_curriculum"]
    assert bounded["enabled"]
    assert [stage["maximum_policy_iterations"] for stage in bounded["stages"]] == [
        800,
        800,
        800,
    ]
    assert bounded["maximum_extension_iterations"] == 600
    assert bounded["stages"][-1]["gate"]["timeout_operator"] == "lt"
    assert bounded["stages"][-1]["gate"]["timeout_rate"] == pytest.approx(0.10)


def test_full_rl_ablation_uses_equal_non_gated_stage_budgets() -> None:
    raw = load_yaml("configs/experiment/exp155_full_rl_ablation.yaml")
    bounded = raw["bounded_curriculum"]
    assert bounded["enabled"]
    assert bounded["transition_mode"] == "fixed_schedule"
    assert [stage["policy_iterations"] for stage in bounded["stages"]] == [
        800,
        800,
        800,
    ]
    assert bounded["maximum_extension_iterations"] == 0
    assert sum(stage["policy_iterations"] for stage in bounded["stages"]) == 2400


def test_full_rl_ablation_preserves_quintic_terrain_risk_reward() -> None:
    raw = load_yaml("configs/experiment/exp155_full_rl_ablation.yaml")
    coefficients = raw["reward"]["coefficients"]
    assert coefficients["path_terrain_mean_cost"] == pytest.approx(0.26)
    assert coefficients["path_terrain_max_cost"] == pytest.approx(0.16)


def test_curriculum_evaluation_is_claimed_once_per_rollout_boundary() -> None:
    state: dict = {}
    assert _claim_curriculum_evaluation(
        state, stage_index=0, stage_iterations=100, interval=100
    )
    assert not _claim_curriculum_evaluation(
        state, stage_index=0, stage_iterations=100, interval=100
    )
    assert not _claim_curriculum_evaluation(
        state, stage_index=0, stage_iterations=101, interval=100
    )
    assert _claim_curriculum_evaluation(
        state, stage_index=0, stage_iterations=200, interval=100
    )
    assert _claim_curriculum_evaluation(
        state, stage_index=1, stage_iterations=100, interval=100
    )


def test_curriculum_timeout_gate_rejects_exact_point_one() -> None:
    metrics = {
        "eval_success_rate": 0.90,
        "eval_collision_rate": 0.02,
        "eval_timeout_rate": 0.10,
        "eval_dmax_reduction_ratio": 0.20,
    }
    gate = {
        "success_rate": 0.85,
        "collision_rate": 0.04,
        "timeout_rate": 0.10,
        "timeout_operator": "lt",
    }
    assert not _curriculum_gate_checks(metrics, gate)["timeout_rate"]


def test_stage_c_runtime_assigns_equal_mixed_and_bottleneck_buckets() -> None:
    cfg = cfg_from_experiment("configs/experiment/exp155_formal_bounded.yaml")
    cfg.terrain.topology_profile = "runtime_bucketed"
    cfg.terrain.topology_curriculum_stage = "mixed_bottleneck"
    runtime = make_terrain_runtime(8, device="cpu")
    randomize_terrain_runtime(
        runtime,
        torch.arange(8),
        cfg.terrain,
        generator=torch.Generator().manual_seed(23),
    )
    assert torch.equal(runtime.topology_bucket, torch.tensor([1, 2, 1, 2, 1, 2, 1, 2]))


@pytest.mark.parametrize(
    ("expected", "profile", "count", "seed", "depth"),
    [
        ("Open", "open", 0, 11, 0.12),
        ("Mixed", "mixed", 30, 11, 0.12),
        ("Bottleneck", "bottleneck", 100, 4, 0.15),
    ],
)
def test_formal_topology_presets_match_their_mapf_labels(
    expected: str,
    profile: str,
    count: int,
    seed: int,
    depth: float,
) -> None:
    cfg = cfg_from_experiment(CONFIG)
    cfg.terrain.topology_profile = profile
    cfg.terrain.crater_count = count
    cfg.terrain.crater_seed = seed
    cfg.terrain.crater_depth_to_diameter = depth
    cfg.terrain.bottleneck_wall_half_width = 0.50
    cfg.terrain.bottleneck_gap_half_width = 0.50
    axis = torch.linspace(-12.5, 12.5, 51)
    grid_y, grid_x = torch.meshgrid(axis, axis, indexing="ij")
    features = query_terrain_features(torch.stack((grid_x, grid_y), dim=-1), cfg.terrain)
    analysis = analyze_traversability_topology(features[..., 4] >= 0.5, max_sources=256)
    assert analysis.topology_class == expected
