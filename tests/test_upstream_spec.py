"""New physical contracts must not execute through legacy defaults."""

from copy import deepcopy
from dataclasses import FrozenInstanceError
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from _common import ROOT, cfg_from_experiment, load_yaml
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.contracts import contract_metadata, N
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_spec import UpstreamSpec

CONFIG = "configs/experiment/upstream_scale_s00.yaml"
PROFILE = ROOT / "configs/design/upstream_vehicle_proxy_v2.yaml"


def design():
    return yaml.safe_load(PROFILE.read_text())


def spec(data=None):
    return UpstreamSpec.from_design(design() if data is None else data, hashlib.sha256(PROFILE.read_bytes()).hexdigest())


def test_runtime_dispatch_preserves_legacy_and_has_no_hidden_one_meter_cap():
    cfg = cfg_from_experiment(CONFIG)
    assert cfg.spec.vehicle.forward_limit == 10 / 3.6
    assert cfg.spec.vehicle.reverse_limit == .5
    assert cfg.spec.initial_test_speed == 1.
    assert cfg.spec.communication.absolute_velocity_bound == 10 / 3.6
    assert cfg.spec.grid.steps == 40 and cfg.spec.grid.horizon == 8
    assert cfg.spec.vehicle.wheel_limit == 50 / 9
    old = cfg_from_experiment("configs/experiment/t10_combined_repairs_v11.yaml")
    assert old.local_goal_contract.planner_protocol == "planner_protocol_v1"
    assert old.observation.communication_radius == 12.
    assert N == 15
    assert contract_metadata(cfg)["planner_protocol"] == "planner_protocol_v2"


def test_robust_formation_witness_and_sensor_capacity():
    s = spec(); audit = s.geometry_audit()
    slots = np.asarray(s.task.slots)
    delta = slots[:, None] - slots[None]
    distance = np.linalg.norm(delta, axis=-1)
    assert distance[np.triu_indices(4, 1)].min() == 7.2
    assert np.mean(np.sum(slots**2, axis=-1)) == pytest.approx(25.92)
    assert audit["minimum_captured_distance"] > audit["safe_center_distance"]
    assert audit["maximum_captured_distance"] < s.task.dmax
    assert audit["maximum_captured_dispersion"] < s.task.dispersion
    assert audit["captured_outer_radius"] < s.task.worksite_radius
    assert s.perception.frame_shape == (151, 81)
    assert s.perception.sample_count == 12231 < s.perception.capacity
    assert audit["geographic_anchor_buffer"] > 70  # sensor corners, not just 15 m.


@pytest.mark.parametrize("section,key,value", [
    ("motion", "default_forward_limit_mps", 1.),
    ("motion", "linear_deceleration_mps2", 0.),
    ("motion", "wheel_acceleration_limit_radps2", True),
    ("motion", "default_reverse_limit_mps", float("nan")),
    ("planning", "horizon_steps", 40.),
    ("planning", "horizon_s", 3.),
    ("perception", "memory_max_samples", 4096),
    ("terrain", "driving_slope_limit_deg", 100.),
    ("communication", "unknown_sender_forward_speed_bound_mps", 1.15),
    ("communication", "future_prediction_is_control_commitment", True),
    ("task", "square_grid_spacing_m", 1.6),
    ("task", "max_center_distance_m", 2.),
    ("task", "common_worksite_radius_m", .75),
    ("task", "certify_unused_common_area", True),
    ("dynamic_speed_limit", "enabled", False),
])
def test_inconsistent_design_refused_before_execution(section, key, value):
    data = design(); data[section][key] = value
    with pytest.raises(ValueError):
        spec(data)


def test_spec_is_frozen_and_metadata_copies_cannot_change_it():
    cfg = cfg_from_experiment(CONFIG)
    with pytest.raises(FrozenInstanceError):
        cfg.spec.vehicle.forward_limit = 1.
    metadata = cfg.metadata
    metadata["vehicle"]["forward_limit"] = 1.
    assert cfg.metadata["vehicle"]["forward_limit"] == 10 / 3.6
    times = cfg.spec.grid.times(2.)
    with pytest.raises(ValueError):
        times[0] = 3.


@pytest.mark.parametrize("change", ["old_version", "adapter", "device", "seed", "grid"])
def test_state_identity_and_solver_cache_are_isolated(change):
    cfg = cfg_from_experiment(CONFIG)
    state = {"contracts": deepcopy(cfg.metadata)}
    cfg.validate_state_identity(state)
    if change == "old_version":state["contracts"]["planner_protocol"] = "planner_protocol_v1"
    elif change == "adapter":state["contracts"]["planner_adapter"] = "nmpc_polar_v1"
    elif change == "device":state["contracts"]["planner_device"] = "cuda"
    elif change == "seed":state["contracts"]["configuration_fingerprint"] = "other_seed"
    else:state["contracts"]["time_grid"]["steps"] = 15
    with pytest.raises(ValueError, match="mismatch"):
        cfg.validate_state_identity(state)


def test_new_profile_and_runtime_config_refused_by_old_execution_and_training():
    from train_local_goal_nmpc import validate_training_gate
    from lunar_rover_tasks.tasks.multi_rover_gathering.gathering_env import MultiRoverGatheringCore
    from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.environment import LocalGoalEnvironment
    cfg = cfg_from_experiment(CONFIG)
    with pytest.raises(ValueError, match="legacy core refused"):
        MultiRoverGatheringCore(cfg)
    with pytest.raises(ValueError, match="legacy core refused"):
        LocalGoalEnvironment(cfg, "unused_cache")
    for evaluate in (False, True):
        with pytest.raises(ValueError, match="Upstream scale diagnostics only"):
            validate_training_gate(load_yaml(CONFIG), evaluate_only=evaluate)
    with pytest.raises(ValueError, match="not an experiment"):
        cfg_from_experiment(PROFILE)


def test_pinned_design_hash_rejects_changed_file(tmp_path):
    from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_spec import parse_upstream_experiment
    config = load_yaml(CONFIG)
    changed = tmp_path / "profile.yaml"; changed.write_bytes(PROFILE.read_bytes() + b"\n# edit\n")
    config["upstream_scale"]["design_profile"] = str(changed)
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        parse_upstream_experiment(config, ROOT)


def test_profile_fingerprint_includes_placeholder_assumptions():
    data = design(); before = spec(data)
    data["physical_placeholders"]["mass_kg"] += 1
    assert spec(data).fingerprint != before.fingerprint
    assert json.loads(before.profile_json)["physical_placeholders"]["mass_kg"] == 1500


def test_evidence_manifest_binds_upstream_profile_and_rejects_changed_resume(tmp_path):
    from planner_evidence import EvidenceRun, planner_device, planner_metadata
    cfg = cfg_from_experiment(CONFIG)
    run = EvidenceRun(tmp_path / "run", CONFIG, settings={"stage": "S00"})
    assert planner_device(cfg) == "cpu"
    assert planner_metadata(cfg) == cfg.metadata
    assert run.identity["contracts"] == cfg.metadata
    manifest = json.loads((run.path / "run_manifest.json").read_text())
    assert manifest["training_interactions"] == manifest["optimizer_updates"] == 0
    assert manifest["runtime_execution"] == "not_verified_by_manifest"
    EvidenceRun(run.path, CONFIG, resume=True, settings={"stage": "S00"})
    with pytest.raises(ValueError, match="mismatch"):
        EvidenceRun(run.path, CONFIG, resume=True, settings={"stage": "S01"})
