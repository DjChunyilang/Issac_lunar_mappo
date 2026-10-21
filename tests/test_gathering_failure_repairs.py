"""Versioned repairs must preserve direct sensing and conservative boundaries."""
from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.memory import LocalTerrainMemory
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.protocol import TerrainView


def features(count):
    result = np.zeros((count, 5))
    result[:, 4] = 1
    return result


@pytest.mark.parametrize("slope,expected_hold", [(0., True), (.2, False)])
def test_low_relief_capture_requires_own_complete_local_evidence(slope, expected_hold):
    from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.gathering_script import LowReliefCaptureSitePolicy, ScriptObservation, ScriptMessage
    xy = np.stack(np.meshgrid(np.arange(-2, 2.01, .1), np.arange(-2, 2.01, .1)), -1).reshape(-1, 2)
    memory = LocalTerrainMemory(version="direct_batch_v3")
    sample = features(len(xy)); sample[:, 1] = slope
    memory.observe(xy, sample, 0.)
    message = ScriptMessage(0, np.zeros(2), np.zeros(2), 0., True, 0., 1., True)
    obs = ScriptObservation(0, 1, np.array([.95, 0, 0]), np.zeros(2), 0.,
                            TerrainView(memory).conservative(), (message,), ())
    policy = LowReliefCaptureSitePolicy()
    action, records = policy.action([obs], (1, 4))
    assert bool(action.hold[0, 1]) == expected_hold
    if expected_hold:
        assert records[0]["known_nodes"] == 37 and records[0]["local_region_radius"] == pytest.approx(.9)
        saved = policy.state_dict()
        restored = LowReliefCaptureSitePolicy(); restored.load_state_dict(saved)
        np.testing.assert_array_equal(action.hold, restored.action([obs], (1, 4))[0].hold)
        farther = ScriptObservation(0, 1, np.array([1.05, 0, 0]), np.zeros(2), 0., obs.terrain, (message,), ())
        assert restored.action([farther], (1, 4))[0].hold[0, 1]
    empty = LocalTerrainMemory(version="direct_batch_v3")
    unknown = ScriptObservation(0, 1, obs.pose, obs.velocity, 0., TerrainView(empty), (message,), ())
    assert not LowReliefCaptureSitePolicy().action([unknown], (1, 4))[0].hold[0, 1]


def test_low_relief_profile_is_paused_and_does_not_change_original_profiles():
    from _common import cfg_from_experiment
    from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.gathering_script import create_script
    cfg = cfg_from_experiment("configs/experiment/t10_combined_repairs_v10.yaml")
    assert create_script(cfg).version == "leader_local_site_v9"
    assert cfg.local_goal_contract.training_paused
    assert cfg.local_goal_contract.diagnostic_nmpc_adapter == "nmpc_recovery_v4"
    old = cfg_from_experiment("configs/experiment/t10_combined_repairs_v9.yaml")
    assert create_script(old).version == "leader_local_site_v8"


def test_observed_standstill_capture_never_certifies_unknown_common_site():
    from dataclasses import replace
    from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.gathering_script import ObservedStandstillSitePolicy, LowReliefCaptureSitePolicy, ScriptObservation, ScriptMessage
    memory = LocalTerrainMemory(version="direct_batch_v3")
    memory.observe([[.95, 0]], features(1), 0.)
    message = ScriptMessage(0, np.zeros(2), np.zeros(2), 0., True, 0., 1., True)
    obs = ScriptObservation(0, 1, np.array([.95, 0, 0]), np.zeros(2), 0., TerrainView(memory).conservative(), (message,), ())
    assert not LowReliefCaptureSitePolicy().action([obs], (1, 4))[0].hold[0, 1]
    action, records = ObservedStandstillSitePolicy().action([obs], (1, 4))
    assert action.hold[0, 1] and not records[0]["common_site_certified"]
    assert records[0]["known_nodes"] == 9
    assert not obs.terrain.lookup([[0, 0]])[1][0]
    for velocity in ([.06, 0], [0, .11]):
        assert not ObservedStandstillSitePolicy().action([replace(obs, velocity=np.array(velocity))], (1, 4))[0].hold[0, 1]
    bad = features(1); bad[0, 4] = .1
    memory.observe([[.95, 0]], bad, 1.)
    assert not ObservedStandstillSitePolicy().action([replace(obs, terrain=TerrainView(memory).conservative())], (1, 4))[0].hold[0, 1]
    unknown = LocalTerrainMemory(version="direct_batch_v3")
    assert not ObservedStandstillSitePolicy().action([replace(obs, terrain=TerrainView(unknown))], (1, 4))[0].hold[0, 1]


def test_observed_standstill_config_and_state_are_independent():
    from _common import cfg_from_experiment
    from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.gathering_script import create_script, LowReliefCaptureSitePolicy
    cfg = cfg_from_experiment("configs/experiment/t10_combined_repairs_v11.yaml")
    policy = create_script(cfg)
    assert policy.version == "leader_local_site_v10" and cfg.local_goal_contract.training_paused
    with pytest.raises(ValueError, match="version"):
        policy.load_state_dict(LowReliefCaptureSitePolicy().state_dict())


def test_same_cell_direct_observations_survive_and_legacy_remains():
    # Two rotated 0.2 m-spaced samples share a 0.2 m np.rint cell.
    points = np.array([[-.099, -.099], [.042421356, .042421356]])
    legacy = LocalTerrainMemory(resolution=.2)
    fixed = LocalTerrainMemory(resolution=.2, version="direct_batch_v2")
    for memory in (legacy, fixed):
        memory.observe(points, features(2), .2)
    assert not legacy.lookup(points[:1])[1][0]
    assert fixed.lookup(points)[1].all() and fixed.lookup(points)[2].all()
    view = TerrainView(fixed).conservative()
    assert view.lookup(points)[2].all()
    np.testing.assert_array_equal(view.snapshot().xy, fixed.xy)
    assert not fixed.safe([[5, 5]])[0]
    assert fixed.support_radius == legacy.support_radius == .155


def test_direct_batch_duplicates_keep_unsafe_features_and_different_positions():
    memory = LocalTerrainMemory(resolution=.2, version="direct_batch_v2")
    sample = features(3)
    sample[0, 4] = .1
    sample[0, 1] = .9
    memory.observe([[0, 0], [0, 0], [.01, 0]], sample, 1.)
    assert len(memory.samples) == 2
    assert not memory.safe([[0, 0]])[0]
    assert memory.lookup([[.01, 0]])[2][0]
    assert memory._timestamps[:2].tolist() == [1., 1.]


def test_direct_batch_capacity_and_empty_updates_restore():
    memory = LocalTerrainMemory(capacity=3, resolution=.2, version="direct_batch_v2")
    memory.observe([[0, 0], [.01, 0]], features(2), 1.)
    memory.observe([[1, 0], [1.01, 0]], features(2), 2.)
    assert len(memory.samples) == 3
    assert memory.lookup([[1, 0], [1.01, 0]])[2].all()
    saved = deepcopy(memory)
    memory.observe(np.empty((0, 2)), features(0), 3.)
    np.testing.assert_array_equal(memory.xy, saved.xy)
    with pytest.raises(ValueError, match="complete direct"):
        memory.observe([[0, 0], [1, 0], [2, 0], [3, 0]], features(4), 4.)
    np.testing.assert_array_equal(memory.xy, saved.xy)
    with pytest.raises(ValueError, match="Nonfinite"):
        memory.observe([[np.nan, 0]], features(1), 4.)
    memory.reset()
    assert not memory.lookup([[0, 0]])[1].any()


def test_local_site_policy_uses_supported_patch_and_tighter_slots():
    from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.gathering_script import (
        LocalSiteLeaderPolicy, ScriptObservation, ScriptMessage)
    xy = np.stack(np.meshgrid(np.arange(-2, 2.01, .1), np.arange(-2, 2.01, .1)), -1).reshape(-1, 2)
    memory = LocalTerrainMemory(version="direct_batch_v2")
    sample = features(len(xy))
    memory.observe(xy, sample, 0.)
    policy = LocalSiteLeaderPolicy()
    leader = ScriptObservation(0, 0, np.zeros(3), np.zeros(2), 0., TerrainView(memory).conservative(), (), ())
    message = ScriptMessage(0, np.zeros(2), np.zeros(2), 0., True, 0., 1., True)
    follower = ScriptObservation(0, 1, np.array([1.6, 0, 0]), np.zeros(2), 0.,
                                 leader.terrain, (message,), ())
    action, records = policy.action([leader, follower], (1, 4))
    assert action.hold[0, 0] and not action.hold[0, 1]
    assert records[0]["known_nodes"] == 37
    np.testing.assert_allclose(records[1]["slot"], [.6, 0])
    saved = policy.state_dict()
    restored = LocalSiteLeaderPolicy()
    restored.load_state_dict(saved)
    np.testing.assert_array_equal(action.physical, restored.action([leader, follower], (1, 4))[0].physical)
    with pytest.raises(ValueError):
        restored.load_state_dict({**saved, "version": "leader_slot_v1"})
    policy.reset([0])
    assert not policy.states


def test_local_site_rejects_unknown_and_selects_a_different_feasible_center():
    from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.gathering_script import LocalSiteLeaderPolicy, ScriptObservation
    memory = LocalTerrainMemory(version="direct_batch_v2")
    obs = ScriptObservation(0, 0, np.zeros(3), np.zeros(2), 0., TerrainView(memory), (), ())
    assert LocalSiteLeaderPolicy().action([obs], (1, 4))[0].hold[0, 0]
    xy = np.stack(np.meshgrid(np.arange(-2, 2.01, .1), np.arange(-2, 2.01, .1)), -1).reshape(-1, 2)
    sample = features(len(xy))
    sample[:, 1] = np.where(xy[:, 0] > -.2, .35, .05)
    memory.observe(xy, sample, 0.)
    obs = ScriptObservation(0, 0, np.zeros(3), np.zeros(2), 0., TerrainView(memory), (), ())
    policy = LocalSiteLeaderPolicy()
    action, records = policy.action([obs], (1, 4))
    assert not action.hold[0, 0]
    assert records[0]["known_nodes"] == 37 and records[0]["max_slope"] < .25
    assert policy.states[(0, 0)]["site"][0] < -.8


def test_repair_configs_isolate_versions_and_preserve_thresholds():
    from _common import cfg_from_experiment
    from evaluate_gathering_no_learning import config_for_family
    from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.contracts import contract_metadata
    old = cfg_from_experiment("configs/experiment/t10_gathering_no_learning_cuda_v2.yaml")
    new = config_for_family("configs/experiment/t10_combined_repairs_v3.yaml", "npb", "nmpc", True, 180)
    assert old.local_goal_contract.terrain_memory_version == "quantized_latest_v1"
    assert new.local_goal_contract.terrain_memory_version == "direct_batch_v2"
    assert new.local_goal_contract.planner_adapter == "nmpc_recovery_v3"
    assert contract_metadata(new)["script_version"] == "leader_local_site_v2"
    assert new.success_thresholds == old.success_thresholds
    assert new.gather_point.max_slope == old.gather_point.max_slope
    assert new.gather_point.max_height_range == old.gather_point.max_height_range


def test_memory_environment_restore_rejects_old_version_before_mutation(tmp_path):
    from _common import cfg_from_experiment
    from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.environment import LocalGoalEnvironment
    cfg = cfg_from_experiment("configs/experiment/t10_memory_repair_cuda_v3.yaml")
    cfg.local_goal_contract.planner_device = "cpu"
    cfg.local_goal_contract.mppi_samples = 64
    cfg.terrain.type = "flat_proxy"
    cfg.terrain.amplitude = 0
    env = LocalGoalEnvironment(cfg, tmp_path)
    try:
        saved = env.state_dict()
        env.load_state_dict(saved)
        broken = deepcopy(saved)
        broken["local"]["memories"][0].version = "quantized_latest_v1"
        with pytest.raises(ValueError, match="local-memory"):
            env.load_state_dict(broken)
        assert env.memories[0].version == "direct_batch_v2"
        untouched = deepcopy(env.memories[4].xy)
        env.reset([0])
        np.testing.assert_array_equal(untouched, env.memories[4].xy)
    finally:
        env.close()


def test_memory_v3_retains_previous_impassable_points_in_updated_cells():
    memory = LocalTerrainMemory(version="direct_batch_v3", resolution=.2)
    bad = features(1)
    bad[0, 4] = .1
    memory.observe([[0, 0]], bad, 0.)
    memory.observe([[.02, 0]], features(1), 1.)
    assert not memory.safe([[0, 0]])[0]
    assert memory.lookup([[.02, 0]])[2][0]
    memory.observe([[0, 0]], features(1), 2.)
    assert not memory.safe([[0, 0]])[0]
    assert memory.lookup([[0, 0]])[2][0]


def test_local_site_v3_preserves_horizon_clearance_and_version_boundary():
    from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.gathering_script import ConservativeLocalSitePolicy, LocalSiteLeaderPolicy
    new = ConservativeLocalSitePolicy()
    assert new.slot_radius-new.capture_radius > .46+.08*3+.08*.2
    assert new.height_limit-new.height_margin == pytest.approx(.14)
    assert new.slope_limit-new.slope_margin == pytest.approx(.20)
    with pytest.raises(ValueError, match="version"):
        new.load_state_dict(LocalSiteLeaderPolicy().state_dict())


@pytest.mark.parametrize("goal", [(.8, 0), (-.8, 0)])
@pytest.mark.parametrize("version", [3, 4])
def test_real_acados_recovery_public_state_repeat(tmp_path, goal, version):
    import os
    from test_planner_baselines import low_input
    from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.planner import build_solver, NMPCRecoveryPlanner, NMPCAdaptiveRecoveryPlanner
    from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.baselines import NMPCRecoveryAdapter, NMPCAdaptiveRecoveryAdapter
    assert os.environ.get("ACADOS_SOURCE_DIR"), "Real acados required, no skip"
    solver = build_solver(tmp_path, .75, revision=f"merit_recovery_v{version}")
    cls, planner_cls = ((NMPCAdaptiveRecoveryAdapter, NMPCAdaptiveRecoveryPlanner)
                        if version == 4 else (NMPCRecoveryAdapter, NMPCRecoveryPlanner))
    adapter = cls(planner_cls(solver, .75, 10.))
    initial = adapter.state_dict()
    item = low_input(goal)
    first = adapter.plan(item)
    assert first.valid and np.max(abs(first.trajectory.controls[0])) > .01
    adapter.load_state_dict(initial)
    again = adapter.plan(item)
    np.testing.assert_allclose(first.trajectory.controls, again.trajectory.controls, atol=1e-7, rtol=1e-6)
    with pytest.raises(ValueError, match="version"):
        adapter.load_state_dict({**initial, "state_version": 2})


def test_recovery_v4_limits_attempts_and_counts_elapsed_control_time_once():
    from test_local_goal_nmpc import FailingSolver, flat_memory
    from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.planner import NMPCAdaptiveRecoveryPlanner
    from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.contracts import LocalGoalRequest
    planner = NMPCAdaptiveRecoveryPlanner(FailingSolver(), .75, 10.)
    _, feedback = planner.plan(np.zeros(3), LocalGoalRequest(np.array([.8, 0, 0]), 1., 0., 1.),
                               flat_memory(), [], np.zeros(2), 0.)
    assert feedback.emergency_brake and feedback.reason == "solver_failure"
    assert feedback.solver_passes == 4
    assert feedback.blocked_seconds == pytest.approx(.2)
    assert len(planner.last_diagnostics["attempts"]) == 4


def test_site_v4_waits_only_for_legally_published_moving_leader():
    from dataclasses import replace
    from test_stage5_diagnostics import observation
    from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.gathering_script import StagedLocalSitePolicy
    obs = observation()
    policy = StagedLocalSitePolicy()
    moving = replace(obs, messages=(replace(obs.messages[0], stop_intent=False),))
    action, rows = policy.action([moving], (1, 4))
    assert action.hold[0, 1] and rows[0]["reason"] == "leader_site_search_wait"
    action, rows = policy.action([obs], (1, 4))
    assert not action.hold[0, 1] and rows[0]["reason"] == "move_to_slot"


def test_reobserved_hazard_survives_capacity_eviction_with_whole_direct_batch():
    memory = LocalTerrainMemory(capacity=3, version="direct_batch_v3", resolution=.2)
    bad = features(1)
    bad[0, 4] = .1
    memory.observe([[0, 0]], bad, 0.)
    memory.observe([[1, 0], [2, 0]], features(2), 1.)
    memory.observe([[0, 0], [3, 0], [4, 0]], features(3), 2.)
    assert memory.lookup([[0, 0], [3, 0], [4, 0]])[2].all()
    assert not memory.safe([[0, 0]])[0]
    assert len(memory.samples) == 3


def test_final_configs_preserve_task_limits_and_separate_repair_axes():
    from _common import cfg_from_experiment
    from evaluate_gathering_no_learning import config_for_family
    old = cfg_from_experiment("configs/experiment/t10_gathering_no_learning_cuda_v2.yaml")
    memory = config_for_family("configs/experiment/t10_memory_repair_cuda_v4.yaml", "npb", "nmpc", True, 180)
    site = config_for_family("configs/experiment/t10_local_site_repair_v6.yaml", "slope", "nmpc", True, 180)
    recovery = config_for_family("configs/experiment/t10_nmpc_recovery_v5.yaml", "crossing", "nmpc", True, 180)
    combined = config_for_family("configs/experiment/t10_combined_repairs_v7.yaml", "slope", "nmpc", True, 180)
    assert memory.local_goal_contract.planner_adapter == site.local_goal_contract.planner_adapter == "nmpc_polar_v1"
    assert memory.local_goal_contract.no_learning_script == recovery.local_goal_contract.no_learning_script == "leader_slot_v1"
    assert recovery.local_goal_contract.planner_adapter == combined.local_goal_contract.planner_adapter == "nmpc_recovery_v4"
    assert site.local_goal_contract.no_learning_script == combined.local_goal_contract.no_learning_script == "leader_local_site_v6"
    for cfg in (memory, site, recovery, combined):
        assert cfg.local_goal_contract.terrain_memory_version == "direct_batch_v3"
        assert cfg.local_goal_contract.training_paused
        assert cfg.success_thresholds == old.success_thresholds
        assert cfg.gather_point == old.gather_point


def test_site_v6_validates_actual_arrival_patch_before_hold():
    from dataclasses import replace
    from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.gathering_script import ValidatedArrivalSitePolicy, ScriptObservation
    xy = np.stack(np.meshgrid(np.arange(-1, 1.01, .1), np.arange(-1, 1.01, .1)), -1).reshape(-1, 2)
    memory = LocalTerrainMemory(version="direct_batch_v3")
    memory.observe(xy, features(len(xy)), 0.)
    obs = ScriptObservation(0, 0, np.zeros(3), np.zeros(2), 0., TerrainView(memory).conservative(), (), ())
    policy = ValidatedArrivalSitePolicy()
    # A previously selected legal site is nearby, but current actual position
    # still needs its own complete local validation before it becomes an anchor.
    policy.states[(0, 0)] = {"heading": 0., "holding": False, "site": np.array([.15, 0]),
                             "probe": None, "site_evidence": {"source": "local_sensor_memory"}}
    saved = policy.state_dict()
    unknown = LocalTerrainMemory(version="direct_batch_v3")
    action, _ = policy.action([replace(obs, terrain=TerrainView(unknown))], (1, 4))
    assert not action.hold[0, 0] and not policy.states[(0, 0)]["holding"]
    policy.load_state_dict(saved)
    action, rows = policy.action([obs], (1, 4))
    assert action.hold[0, 0] and rows[0]["arrival_validated"]
    np.testing.assert_array_equal(policy.states[(0, 0)]["site"], obs.pose[:2])


def test_current_site_environment_hidden_information_and_restore_isolation(tmp_path):
    from _common import cfg_from_experiment
    from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.environment import LocalGoalEnvironment
    from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.gathering_script import create_script
    cfg = cfg_from_experiment("configs/experiment/t10_local_site_repair_v8.yaml")
    cfg.local_goal_contract.planner_device = "cpu"
    cfg.local_goal_contract.mppi_samples = 64
    cfg.terrain.type = "flat_proxy"
    cfg.terrain.amplitude = 0
    env = LocalGoalEnvironment(cfg, tmp_path)
    try:
        policy = create_script(cfg)
        saved = policy.state_dict()
        before = policy.action(env.script_observations(), (2, 4))[0]
        env.core.oracle_point.add_(1000)
        env.core.positions[1, :, 0].add_(100)
        # A missing peer's uncommunicated state cannot change another car's
        # local site / slot action. No transport or sensor update occurs here.
        env.core.positions[0, 3, 0].add_(50)
        policy.load_state_dict(saved)
        after = policy.action(env.script_observations(), (2, 4))[0]
        np.testing.assert_array_equal(before.physical[0, :3], after.physical[0, :3])
        np.testing.assert_array_equal(before.hold[0, :3], after.hold[0, :3])
        state = env.state_dict()
        env.load_state_dict(state)
        broken = deepcopy(state)
        broken["local"]["memories"][0].version = "quantized_latest_v1"
        with pytest.raises(ValueError, match="local-memory"):
            env.load_state_dict(broken)
        assert env.memories[0].version == "direct_batch_v3"
        policy.action(env.script_observations(), (2, 4))
        second = deepcopy({k: v for k, v in policy.states.items() if k[0] == 1})
        policy.reset([0])
        for key, value in second.items():
            for field in ("site", "probe"):
                np.testing.assert_array_equal(value.get(field), policy.states[key].get(field))
    finally:
        env.close()


@pytest.mark.parametrize("slope,known,expected_hold", [(.219, True, True), (.24, True, False), (.219, False, False)])
def test_site_v7_arrival_keeps_full_support_and_explicit_terminal_reserve(slope, known, expected_hold):
    from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.gathering_script import ConservativeArrivalSitePolicy, ValidatedArrivalSitePolicy, ScriptObservation
    xy = np.stack(np.meshgrid(np.arange(-1, 1.01, .1), np.arange(-1, 1.01, .1)), -1).reshape(-1, 2)
    memory = LocalTerrainMemory(version="direct_batch_v3")
    sample = features(len(xy))
    sample[:, 1] = slope
    if known:
        memory.observe(xy, sample, 0.)
    obs = ScriptObservation(0, 0, np.zeros(3), np.zeros(2), 0., TerrainView(memory).conservative(), (), ())
    old = ValidatedArrivalSitePolicy()
    old.states[(0, 0)] = {"heading": 0., "holding": False, "site": np.array([.15, 0]),
                          "probe": None, "site_evidence": {"source": "local_sensor_memory"}}
    policy = ConservativeArrivalSitePolicy()
    with pytest.raises(ValueError, match="version"):
        policy.load_state_dict(old.state_dict())
    policy.states = deepcopy(old.states)
    action, rows = policy.action([obs], (1, 4))
    assert action.hold[0, 0] == expected_hold
    if expected_hold:
        assert rows[0]["known_nodes"] == 37 and rows[0]["arrival_validated"]
        assert rows[0]["max_slope"] < .25
        restored = ConservativeArrivalSitePolicy()
        restored.load_state_dict(policy.state_dict())
        np.testing.assert_array_equal(restored.action([obs], (1, 4))[0].hold, action.hold)


def test_final_script_factory_config_separates_planner_recovery():
    from evaluate_gathering_no_learning import config_for_family
    from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.gathering_script import create_script
    site = config_for_family("configs/experiment/t10_local_site_repair_v8.yaml", "slope", "mppi512", True, 180)
    combined = config_for_family("configs/experiment/t10_combined_repairs_v9.yaml", "slope", "nmpc", True, 180)
    assert create_script(site).version == create_script(combined).version == "leader_local_site_v8"
    assert site.local_goal_contract.planner_adapter == "mppi_terrain_neighbor_stable_v2"
    assert combined.local_goal_contract.planner_adapter == "nmpc_recovery_v4"


def test_probe_v8_handoff_does_not_certify_a_site_or_relax_unknown_boundaries():
    from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.gathering_script import AdvancingProbeSitePolicy, ConservativeArrivalSitePolicy, ScriptObservation
    xy = np.stack(np.meshgrid(np.arange(-2, 2.01, .1), np.arange(-2, 2.01, .1)), -1).reshape(-1, 2)
    memory = LocalTerrainMemory(version="direct_batch_v3")
    sample = features(len(xy))
    sample[:, 1] = .4  # Passable road, infeasible terminal parking region.
    memory.observe(xy, sample, 0.)
    obs = ScriptObservation(0, 0, np.zeros(3), np.zeros(2), 0., TerrainView(memory).conservative(), (), ())
    old, new = ConservativeArrivalSitePolicy(), AdvancingProbeSitePolicy()
    initial = {"heading": 0., "holding": False, "site": None, "probe": np.array([-.28, 0])}
    for policy in (old, new):
        policy.states[(0, 0)] = deepcopy(initial)
    old.action([obs], (1, 4))
    action, rows = new.action([obs], (1, 4))
    np.testing.assert_array_equal(old.states[(0, 0)]["probe"], initial["probe"])
    assert not np.array_equal(new.states[(0, 0)]["probe"], initial["probe"])
    assert new.states[(0, 0)]["site"] is None and not action.hold[0, 0]
    assert rows[0]["reason"] == "local_site_probe"
    assert obs.terrain.safe(np.linspace(obs.pose[:2], new.states[(0, 0)]["probe"], 9)).all()
    assert not obs.terrain.safe([[10, 10]])[0]
    with pytest.raises(ValueError, match="version"):
        new.load_state_dict(old.state_dict())
