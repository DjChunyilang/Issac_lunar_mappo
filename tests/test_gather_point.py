from __future__ import annotations

from dataclasses import replace

import torch

from lunar_rover_tasks.tasks.multi_rover_gathering.gathering_env import (
    MultiRoverGatheringCore,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.gathering_env_cfg import (
    GatherPointCfg,
    TerrainCfg,
    make_debug_cfg,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.metrics import compute_team_metrics
from lunar_rover_tasks.tasks.multi_rover_gathering.oracle import (
    compute_geometric_median,
    compute_mean_oracle_distance,
    evaluate_gather_point_candidates,
    search_optimal_gather_point,
)
from lunar_rover_tasks.tasks.multi_rover_gathering.terrain_features import (
    evaluate_gather_point_flatness,
    make_terrain_runtime,
    query_height,
    query_terrain_features,
    search_local_flatness_center,
)


def _small_gather_cfg(**overrides) -> GatherPointCfg:
    values = {
        "coarse_grid_size": 3,
        "refinement_grid_size": 3,
        "refinement_levels": 0,
        "search_margin": 0.5,
        "flatness_radius": 0.75,
        "flatness_rings": 1,
        "flatness_samples_per_ring": 4,
        "max_height_range": 0.18,
        "max_slope": 0.25,
        "path_samples": 2,
        "max_envs_per_batch": 32,
    }
    values.update(overrides)
    return GatherPointCfg(**values)


def _crater_terrain() -> TerrainCfg:
    return TerrainCfg(
        type="lunar_crater_proxy",
        dynamics_enabled=False,
        amplitude=0.0,
        crater_count=1,
        crater_min_radius=1.0,
        crater_max_radius=1.0,
        crater_depth_to_diameter=0.12,
        crater_rim_height_to_diameter=0.025,
    )


def _coarse_search_candidates(
    positions: torch.Tensor,
    gather_cfg: GatherPointCfg,
    *,
    world_xy_limit: float,
) -> torch.Tensor:
    usable_limit = world_xy_limit - gather_cfg.flatness_radius
    lower = (positions[..., :2].amin(dim=1) - gather_cfg.search_margin).clamp(
        min=-usable_limit,
        max=usable_limit,
    )
    upper = (positions[..., :2].amax(dim=1) + gather_cfg.search_margin).clamp(
        min=-usable_limit,
        max=usable_limit,
    )
    fractions = torch.linspace(
        0.0,
        1.0,
        gather_cfg.coarse_grid_size,
        dtype=positions.dtype,
        device=positions.device,
    )
    x = lower[:, 0, None] + (upper[:, 0] - lower[:, 0])[:, None] * fractions
    y = lower[:, 1, None] + (upper[:, 1] - lower[:, 1])[:, None] * fractions
    grid_x = x[:, :, None].expand(-1, -1, gather_cfg.coarse_grid_size)
    grid_y = y[:, None, :].expand(-1, gather_cfg.coarse_grid_size, -1)
    grid = torch.stack((grid_x, grid_y), dim=-1).flatten(start_dim=1, end_dim=2)
    geometric_median = compute_geometric_median(positions[..., :2])
    centroid = positions[..., :2].mean(dim=1)
    seeds = torch.stack((geometric_median, centroid), dim=1).clamp(
        min=-usable_limit,
        max=usable_limit,
    )
    return torch.cat((grid, seeds), dim=1)


def test_flatness_checks_the_full_patch_not_only_the_center_slope() -> None:
    terrain = _crater_terrain()
    crater_center = torch.tensor([[0.0, 0.0]])

    center_features = query_terrain_features(crater_center, terrain)
    flatness = evaluate_gather_point_flatness(
        crater_center,
        terrain,
        radius=0.75,
        rings=3,
        samples_per_ring=12,
        max_height_range=0.18,
        max_slope=0.25,
    )

    assert torch.linalg.norm(center_features[..., 1:3], dim=-1).item() < 1.0e-6
    assert flatness.height_range.item() > 0.18
    assert flatness.max_slope.item() > 0.25
    assert flatness.is_flat.tolist() == [False]


def test_flatness_is_batched_and_flat_proxy_is_always_accepted() -> None:
    points = torch.tensor(
        [
            [[0.0, 0.0], [1.0, 2.0], [-3.0, 1.0]],
            [[4.0, -2.0], [0.5, 0.25], [-1.0, -1.0]],
        ]
    )

    flatness = evaluate_gather_point_flatness(
        points,
        TerrainCfg(),
        radius=0.75,
        rings=2,
        samples_per_ring=8,
        max_height_range=0.18,
        max_slope=0.25,
    )

    assert flatness.height_range.shape == (2, 3)
    assert flatness.max_slope.shape == (2, 3)
    assert flatness.mean_slope.shape == (2, 3)
    assert flatness.is_flat.shape == (2, 3)
    assert torch.allclose(flatness.height_range, torch.zeros_like(flatness.height_range))
    assert torch.allclose(flatness.max_slope, torch.zeros_like(flatness.max_slope))
    assert flatness.is_flat.all()


def test_local_flatness_search_keeps_an_already_flat_centroid() -> None:
    centers = torch.tensor([[0.25, -0.15], [-0.5, 0.4]])
    result = search_local_flatness_center(
        centers,
        TerrainCfg(),
        search_radius=0.25,
        samples=8,
        flatness_radius=0.75,
        flatness_rings=3,
        flatness_samples_per_ring=12,
        max_height_range=0.18,
        max_slope=0.25,
    )

    assert result.found_flat.tolist() == [True, True]
    assert torch.allclose(result.target_xy, centers)


def test_search_robustness_envelope_rejects_a_center_only_flat_candidate() -> None:
    positions = torch.tensor(
        [[[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, -0.5, 0.0], [0.0, 0.5, 0.0]]]
    )
    candidates = torch.tensor([[[0.0, 0.0]]])
    center_only = _small_gather_cfg(
        flatness_radius=0.10,
        flatness_rings=1,
        flatness_samples_per_ring=8,
        robustness_radius=0.0,
    )
    robust = replace(center_only, robustness_radius=0.20, robustness_samples=8)

    center_evaluation = evaluate_gather_point_candidates(
        positions,
        candidates,
        _crater_terrain(),
        center_only,
    )
    robust_evaluation = evaluate_gather_point_candidates(
        positions,
        candidates,
        _crater_terrain(),
        robust,
    )

    assert center_evaluation.feasible.tolist() == [[True]]
    assert robust_evaluation.feasible.tolist() == [[False]]
    assert robust_evaluation.flatness.max_slope.item() > robust.max_slope


def test_execution_slots_are_symmetric_and_partial_reset_is_isolated() -> None:
    cfg = make_debug_cfg(num_envs=2, device="cpu")
    cfg.task.explicit_goal_in_execution = True
    cfg.observation.schema_version = "ego_v6_gather_slot_goal"
    cfg.gather_point.execution_slot_radius = 0.35
    env = MultiRoverGatheringCore(cfg)

    assert torch.allclose(
        env.gather_slot_points.mean(dim=1),
        env.oracle_point,
        atol=1.0e-6,
    )
    pairwise = torch.cdist(env.gather_slot_points[..., :2], env.gather_slot_points[..., :2])
    eye = torch.eye(cfg.task.n_agents, dtype=torch.bool).unsqueeze(0)
    nearest = pairwise.masked_fill(eye, float("inf")).amin(dim=-1)
    expected_adjacent = 2.0 * cfg.gather_point.execution_slot_radius * (2.0**0.5) / 2.0
    assert torch.allclose(nearest, torch.full_like(nearest, expected_adjacent), atol=1.0e-5)

    untouched_slots = env.gather_slot_points[1].clone()
    env.positions[0, :, :2] += torch.tensor([0.6, -0.3])
    env.refresh_oracle_point(torch.tensor([0]))

    assert torch.allclose(env.gather_slot_points[1], untouched_slots)
    assert torch.allclose(
        env.gather_slot_points[0].mean(dim=0),
        env.oracle_point[0],
        atol=1.0e-6,
    )


def test_slot_reward_target_tracks_the_assigned_formation_not_the_shared_site() -> None:
    cfg = make_debug_cfg(num_envs=1, device="cpu")
    cfg.task.explicit_goal_in_execution = True
    cfg.task.execution_slot_reward_target = True
    cfg.observation.schema_version = "ego_v6_gather_slot_goal"
    cfg.gather_point.execution_slot_radius = 0.45
    env = MultiRoverGatheringCore(cfg)

    slot_distance = compute_mean_oracle_distance(env.positions, env.gather_slot_points)
    shared_distance = compute_mean_oracle_distance(env.positions, env.oracle_point)

    assert torch.allclose(env.prev_mean_oracle_distance, slot_distance)
    assert torch.allclose(env._oracle_reward_target(), env.gather_slot_points)
    assert not torch.allclose(slot_distance, shared_distance)


def test_multiresolution_search_returns_the_global_best_coarse_candidate() -> None:
    positions = torch.tensor(
        [
            [[-2.0, -1.0, 0.0], [-1.0, 2.0, 0.0], [1.0, 0.5, 0.0], [3.0, -0.2, 0.0]],
            [[2.0, -4.0, 0.0], [3.0, -1.0, 0.0], [5.0, -2.5, 0.0], [7.0, -3.2, 0.0]],
        ]
    )
    terrain = TerrainCfg()
    gather_cfg = _small_gather_cfg(refinement_levels=0)
    world_xy_limit = 12.0
    candidates = _coarse_search_candidates(
        positions,
        gather_cfg,
        world_xy_limit=world_xy_limit,
    )
    evaluation = evaluate_gather_point_candidates(
        positions,
        candidates,
        terrain,
        gather_cfg,
    )
    expected_index = evaluation.objective.argmin(dim=-1)
    expected_xy = torch.gather(
        candidates,
        dim=1,
        index=expected_index[:, None, None].expand(-1, 1, 2),
    ).squeeze(1)
    expected_objective = torch.gather(
        evaluation.objective,
        dim=1,
        index=expected_index[:, None],
    ).squeeze(1)

    result = search_optimal_gather_point(
        positions,
        terrain,
        gather_cfg,
        world_xy_limit=world_xy_limit,
    )

    assert evaluation.feasible.all()
    assert torch.allclose(result.point[..., :2], expected_xy, atol=1.0e-6)
    assert torch.allclose(result.objective, expected_objective, atol=1.0e-6)
    assert result.feasible.all()
    assert torch.allclose(result.point[..., 2], torch.zeros(2))


def test_terrain_aware_search_rejects_rough_geometric_median() -> None:
    positions = torch.tensor(
        [[[-1.5, -1.5, 0.0], [-1.5, 1.5, 0.0], [1.5, -1.5, 0.0], [1.5, 1.5, 0.0]]]
    )
    terrain = _crater_terrain()
    gather_cfg = _small_gather_cfg(
        coarse_grid_size=5,
        refinement_grid_size=5,
        refinement_levels=1,
        search_margin=1.5,
        flatness_rings=3,
        flatness_samples_per_ring=12,
        path_samples=3,
    )
    geometric_median = compute_geometric_median(positions[..., :2])
    median_evaluation = evaluate_gather_point_candidates(
        positions,
        geometric_median[:, None, :],
        terrain,
        gather_cfg,
    )

    result = search_optimal_gather_point(
        positions,
        terrain,
        gather_cfg,
        world_xy_limit=12.0,
    )

    assert median_evaluation.feasible.tolist() == [[False]]
    assert result.feasible.tolist() == [True]
    assert not torch.allclose(result.point[..., :2], geometric_median, atol=1.0e-4)
    assert torch.all(result.flatness.height_range <= gather_cfg.max_height_range)
    assert torch.all(result.flatness.max_slope <= gather_cfg.max_slope)
    expected_height = query_height(result.point[..., :2], terrain).squeeze(-1)
    assert torch.allclose(result.point[..., 2], expected_height, atol=1.0e-6)


def test_search_is_agent_permutation_invariant_and_chunking_preserves_results() -> None:
    positions = torch.tensor(
        [
            [[-2.0, -1.0, 0.0], [-1.0, 2.0, 0.0], [1.0, 0.5, 0.0], [3.0, -0.2, 0.0]],
            [[-3.0, 1.0, 0.0], [0.0, -2.0, 0.0], [2.0, 1.5, 0.0], [4.0, 0.0, 0.0]],
            [[-1.5, -2.0, 0.0], [-0.5, 2.5, 0.0], [2.5, 1.0, 0.0], [3.0, -1.0, 0.0]],
        ]
    )
    terrain = TerrainCfg()
    unchunked_cfg = _small_gather_cfg(max_envs_per_batch=32)
    chunked_cfg = replace(unchunked_cfg, max_envs_per_batch=1)

    expected = search_optimal_gather_point(
        positions,
        terrain,
        unchunked_cfg,
        world_xy_limit=12.0,
    )
    chunked = search_optimal_gather_point(
        positions,
        terrain,
        chunked_cfg,
        world_xy_limit=12.0,
    )
    permuted = search_optimal_gather_point(
        positions[:, torch.tensor([2, 0, 3, 1])],
        terrain,
        unchunked_cfg,
        world_xy_limit=12.0,
    )

    assert torch.allclose(chunked.point, expected.point)
    assert torch.allclose(chunked.objective, expected.objective)
    assert torch.equal(chunked.feasible, expected.feasible)
    assert torch.allclose(permuted.point, expected.point, atol=1.0e-6)
    assert torch.allclose(permuted.objective, expected.objective, atol=1.0e-6)


def test_infeasible_search_returns_the_minimum_penalized_fallback() -> None:
    positions = torch.tensor(
        [[[-1.0, -1.0, 0.0], [-1.0, 1.0, 0.0], [1.0, -1.0, 0.0], [1.0, 1.0, 0.0]]]
    )
    terrain = TerrainCfg(
        type="lunar_heightfield_proxy",
        amplitude=1.0,
        wavelength=0.5,
    )
    gather_cfg = _small_gather_cfg(
        refinement_levels=0,
        global_fallback_enabled=False,
        flatness_rings=3,
        flatness_samples_per_ring=12,
        max_height_range=1.0e-5,
        max_slope=1.0e-5,
    )
    candidates = _coarse_search_candidates(
        positions,
        gather_cfg,
        world_xy_limit=12.0,
    )
    evaluation = evaluate_gather_point_candidates(
        positions,
        candidates,
        terrain,
        gather_cfg,
    )
    fallback_score = (
        evaluation.objective
        + gather_cfg.infeasible_penalty * evaluation.violation
    )
    expected_index = fallback_score.argmin(dim=-1)
    expected_xy = torch.gather(
        candidates,
        dim=1,
        index=expected_index[:, None, None].expand(-1, 1, 2),
    ).squeeze(1)

    result = search_optimal_gather_point(
        positions,
        terrain,
        gather_cfg,
        world_xy_limit=12.0,
    )

    assert not evaluation.feasible.any()
    assert result.feasible.tolist() == [False]
    assert torch.allclose(result.point[..., :2], expected_xy)
    assert torch.isfinite(result.point).all()
    assert torch.isfinite(result.objective).all()
    assert torch.all(result.point[..., :2].abs() <= 12.0 - gather_cfg.flatness_radius)
    expected_height = query_height(result.point[..., :2], terrain).squeeze(-1)
    assert torch.allclose(result.point[..., 2], expected_height)


def test_global_fallback_keeps_the_best_penalty_solution_per_env_when_both_fail() -> None:
    positions = torch.tensor(
        [
            [[-4.3, -4.3, 0.0], [-4.3, -3.7, 0.0], [-3.7, -4.3, 0.0], [-3.7, -3.7, 0.0]],
            [[-2.3, -2.3, 0.0], [-2.3, -1.7, 0.0], [-1.7, -2.3, 0.0], [-1.7, -1.7, 0.0]],
        ]
    )
    terrain = TerrainCfg(
        type="lunar_heightfield_proxy",
        amplitude=0.4,
        wavelength=2.7,
    )
    gather_cfg = GatherPointCfg(
        coarse_grid_size=5,
        refinement_grid_size=3,
        refinement_levels=1,
        search_margin=0.8,
        global_fallback_enabled=False,
        global_grid_size=3,
        global_beam_width=2,
        global_refinement_levels=0,
        flatness_radius=0.75,
        flatness_rings=2,
        flatness_samples_per_ring=8,
        max_height_range=0.001,
        max_slope=0.001,
        path_risk_weight=0.0,
        path_height_change_weight=0.0,
        path_samples=2,
    )
    local = search_optimal_gather_point(
        positions,
        terrain,
        gather_cfg,
        world_xy_limit=6.0,
    )
    with_global = search_optimal_gather_point(
        positions,
        terrain,
        replace(gather_cfg, global_fallback_enabled=True),
        world_xy_limit=6.0,
    )

    def penalty_score(result) -> torch.Tensor:
        violation = torch.relu(
            result.flatness.height_range / gather_cfg.max_height_range - 1.0
        ) + torch.relu(result.flatness.max_slope / gather_cfg.max_slope - 1.0)
        return result.objective + gather_cfg.infeasible_penalty * violation

    local_score = penalty_score(local)
    global_score = penalty_score(with_global)

    assert local.feasible.tolist() == [False, False]
    assert with_global.feasible.tolist() == [False, False]
    assert torch.all(global_score <= local_score)
    assert torch.allclose(with_global.point[0], local.point[0])
    assert global_score[1] < local_score[1]


def test_global_fallback_finds_flat_site_outside_infeasible_local_domain() -> None:
    positions = torch.tensor(
        [[[-0.2, -0.2, 0.0], [-0.2, 0.2, 0.0], [0.2, -0.2, 0.0], [0.2, 0.2, 0.0]]]
    )
    terrain = _crater_terrain()
    local_cfg = _small_gather_cfg(
        search_margin=0.0,
        global_fallback_enabled=False,
        global_grid_size=5,
        global_beam_width=8,
        global_refinement_levels=0,
        global_max_envs_per_batch=1,
        flatness_rings=3,
        flatness_samples_per_ring=12,
        path_risk_weight=0.0,
        path_height_change_weight=0.0,
    )

    local_result = search_optimal_gather_point(
        positions,
        terrain,
        local_cfg,
        world_xy_limit=4.0,
    )
    global_result = search_optimal_gather_point(
        positions,
        terrain,
        replace(local_cfg, global_fallback_enabled=True),
        world_xy_limit=4.0,
    )

    assert local_result.feasible.tolist() == [False]
    assert global_result.feasible.tolist() == [True]
    assert global_result.flatness.is_flat.tolist() == [True]
    assert global_result.flatness.height_range.item() <= local_cfg.max_height_range
    assert global_result.flatness.max_slope.item() <= local_cfg.max_slope
    assert global_result.point[..., :2].abs().amax().item() > 0.2


def test_randomized_terrain_chunking_matches_unchunked_search() -> None:
    positions = torch.tensor(
        [
            [[-2.0, -1.0, 0.0], [-1.0, 2.0, 0.0], [1.0, 0.5, 0.0], [3.0, -0.2, 0.0]],
            [[-3.0, 1.0, 0.0], [0.0, -2.0, 0.0], [2.0, 1.5, 0.0], [4.0, 0.0, 0.0]],
            [[-1.5, -2.0, 0.0], [-0.5, 2.5, 0.0], [2.5, 1.0, 0.0], [3.0, -1.0, 0.0]],
        ]
    )
    terrain = TerrainCfg(
        type="lunar_crater_proxy",
        amplitude=0.12,
        wavelength=3.0,
        crater_count=3,
        crater_min_radius=0.6,
        crater_max_radius=1.1,
    )
    runtime = make_terrain_runtime(3, device="cpu")
    runtime.translation_xy.copy_(
        torch.tensor([[0.0, 0.0], [0.7, -0.4], [-0.5, 0.8]])
    )
    runtime.yaw.copy_(torch.tensor([0.0, 0.8, -1.2]))
    runtime.phase.copy_(torch.tensor([0.2, 1.3, 2.4]))
    runtime.amplitude_scale.copy_(torch.tensor([0.8, 1.0, 1.2]))
    runtime.crater_radius_scale.copy_(torch.tensor([0.9, 1.1, 1.0]))
    runtime.crater_depth_scale.copy_(torch.tensor([1.0, 0.8, 1.2]))
    unchunked_cfg = _small_gather_cfg(refinement_levels=1, max_envs_per_batch=32)
    chunked_cfg = replace(unchunked_cfg, max_envs_per_batch=1)

    expected = search_optimal_gather_point(
        positions,
        terrain,
        unchunked_cfg,
        runtime,
        world_xy_limit=12.0,
    )
    chunked = search_optimal_gather_point(
        positions,
        terrain,
        chunked_cfg,
        runtime,
        world_xy_limit=12.0,
    )

    assert torch.allclose(chunked.point, expected.point, atol=1.0e-6)
    assert torch.allclose(chunked.objective, expected.objective, atol=1.0e-6)
    assert torch.equal(chunked.feasible, expected.feasible)
    assert torch.allclose(
        chunked.flatness.height_range,
        expected.flatness.height_range,
        atol=1.0e-6,
    )


def test_environment_success_uses_actual_centroid_flatness_when_dynamics_are_disabled() -> None:
    cfg = make_debug_cfg(num_envs=2, device="cpu")
    cfg.terrain = _crater_terrain()
    cfg.gather_point = _small_gather_cfg(require_flat_for_success=True)
    cfg.success_thresholds.dmax = 1.0
    cfg.success_thresholds.dispersion = 1.0
    cfg.success_thresholds.speed = 0.1
    cfg.success_thresholds.hold_steps = 1
    cfg.success_thresholds.min_pairwise_distance = 0.0
    cfg.safety.collision_distance = 0.05
    env = MultiRoverGatheringCore(cfg)
    offsets = torch.tensor(
        [[-0.2, 0.0], [0.2, 0.0], [0.0, 0.2], [0.0, -0.2]],
        device=env.device,
    )
    env.positions[..., :2] = torch.stack((offsets, offsets + 3.0), dim=0)
    env.positions[..., 2] = 0.0
    env.yaws.zero_()
    env.velocities_xy.zero_()
    env.angular_velocities.zero_()
    env.previous_physical_action.zero_()
    env.step_count.zero_()
    env.success_hold_count.zero_()
    env.metrics = compute_team_metrics(env.positions, env.velocities_xy)
    env.prev_metrics = env.metrics
    env.refresh_oracle_point()
    action = torch.zeros(2, cfg.task.n_agents, cfg.planner.action_dim)
    action[..., 0] = -1.0

    output = env.step(action)

    gates = output.info["success_gates"]
    flatness = output.info["gather_point_flatness"]
    assert cfg.terrain.dynamics_enabled is False
    assert gates.dmax_ok.tolist() == [True, True]
    assert gates.speed_ok.tolist() == [True, True]
    assert flatness.is_flat.tolist() == [False, True]
    assert gates.flatness_ok.tolist() == [False, True]
    assert output.info["done"].success.tolist() == [False, True]


def test_partial_reset_refreshes_only_the_selected_environment_oracle() -> None:
    cfg = make_debug_cfg(num_envs=3, device="cpu")
    cfg.gather_point = _small_gather_cfg()
    env = MultiRoverGatheringCore(cfg)
    tracked_names = (
        "oracle_point",
        "oracle_search_objective",
        "oracle_search_feasible",
        "oracle_search_mean_distance",
        "oracle_search_max_distance",
        "oracle_search_path_risk",
        "oracle_search_path_height_change",
        "oracle_search_height_range",
        "oracle_search_max_slope",
        "prev_mean_oracle_distance",
    )
    snapshots = {name: getattr(env, name).clone() for name in tracked_names}
    positions_before = env.positions.clone()
    env.oracle_point[1].fill_(999.0)
    env.oracle_search_objective[1] = 999.0

    env.reset(torch.tensor([1]))

    unaffected = torch.tensor([0, 2])
    assert torch.allclose(env.positions[unaffected], positions_before[unaffected])
    for name in tracked_names:
        assert torch.equal(
            getattr(env, name)[unaffected],
            snapshots[name][unaffected],
        ), name
    assert torch.isfinite(env.oracle_point[1]).all()
    assert not torch.all(env.oracle_point[1] == 999.0)
    assert env.oracle_search_feasible[1]
    assert torch.isfinite(env.prev_mean_oracle_distance[1])


def test_partial_reset_refreshes_only_selected_centroid_flatness_cache() -> None:
    cfg = make_debug_cfg(num_envs=3, device="cpu")
    cfg.gather_point = _small_gather_cfg()
    env = MultiRoverGatheringCore(cfg)
    env.prev_centroid_flatness_cost.copy_(torch.tensor([0.25, 9.0, 0.75]))

    env.reset(torch.tensor([1]))

    assert env.prev_centroid_flatness_cost[0] == 0.25
    assert env.prev_centroid_flatness_cost[2] == 0.75
    assert torch.isfinite(env.prev_centroid_flatness_cost[1])
    assert env.prev_centroid_flatness_cost[1] != 9.0


def test_terminal_step_info_keeps_pre_reset_oracle_terrain_and_flatness() -> None:
    cfg = make_debug_cfg(num_envs=1, device="cpu")
    cfg.terrain = _crater_terrain()
    cfg.terrain.randomize_per_reset = True
    cfg.terrain.random_translation_m = 1.0
    cfg.terrain.random_yaw_rad = torch.pi
    cfg.gather_point = _small_gather_cfg()
    cfg.simulation.episode_length_s = cfg.simulation.planning_dt
    cfg.success_thresholds.hold_steps = 100
    cfg.safety.collision_distance = 0.01
    env = MultiRoverGatheringCore(cfg)
    offsets = torch.tensor(
        [[-0.2, 0.0], [0.2, 0.0], [0.0, 0.2], [0.0, -0.2]],
        device=env.device,
    )
    env.positions[0, :, :2] = offsets + 3.0
    env.positions[..., 2] = 0.0
    env.yaws.zero_()
    env.velocities_xy.zero_()
    env.angular_velocities.zero_()
    env.success_hold_count.zero_()
    env.refresh_oracle_point()
    before_oracle = env.oracle_point.clone()
    before_runtime = env.terrain_runtime.clone()
    before_flatness = env.evaluate_current_gather_point_flatness()
    before_objective = env.oracle_search_objective.clone()
    action = torch.zeros(1, cfg.task.n_agents, cfg.planner.action_dim)
    action[..., 0] = -1.0

    output = env.step(action)

    assert output.truncated.tolist() == [True]
    assert torch.allclose(output.info["oracle_point"], before_oracle)
    assert torch.allclose(
        output.info["oracle_search"]["objective"],
        before_objective,
    )
    assert torch.allclose(
        output.info["terrain_runtime"].phase,
        before_runtime.phase,
    )
    assert torch.allclose(
        output.info["gather_point_flatness"].height_range,
        before_flatness.height_range,
    )
    assert not torch.allclose(env.terrain_runtime.phase, before_runtime.phase)
