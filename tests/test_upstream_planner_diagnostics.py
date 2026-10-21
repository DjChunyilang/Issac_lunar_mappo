"""Evidence entry contracts without re-running the expensive seven-setting panel."""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import diagnose_upstream_planners as diagnostic
from _common import cfg_from_experiment
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.contracts import PlannerFeedback
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_execution import rollout_numpy
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_planning import UpstreamPlannerOutput
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_baselines import UpstreamPrimitiveInput


@pytest.fixture(scope="module")
def cfg():
    return cfg_from_experiment(diagnostic.DEFAULT_CONFIG)


def test_seven_panel_configurations_preserve_two_iterations_and_native_ids(cfg):
    assert len(diagnostic.SETTINGS) == 7
    samples = []
    for name, adapter, count in diagnostic.SETTINGS:
        panel = diagnostic.panel_config(cfg, adapter, count)
        assert panel.training_paused and panel.mppi_iterations == 2
        assert json.loads(panel.configuration_json)["upstream_scale"]["planner_adapter"] == adapter
        assert panel.metadata["configuration_fingerprint"]
        if count:samples.append(panel.mppi_samples)
        else:assert panel.planner_device == "cpu"
    assert samples == [512, 1024, 2048]
    assert diagnostic.run_settings("cpu", False)["primitive_ids"] == list(range(47))
    gpu = replace(cfg, planner_device="cuda")
    assert diagnostic.panel_config(gpu, "nmpc_upstream_v1", None).planner_device == "cpu"
    assert diagnostic.panel_config(gpu, "mppi_upstream_v1", 1024).planner_device == "cuda"


def test_input_marks_provided_observation_and_explicit_default_information_latency(cfg):
    item = diagnostic.initial_input(cfg)
    assert item.now == pytest.approx(.6) and item.information_time == 0.
    assert item.times[0] == pytest.approx(.6) and item.times[-1] == pytest.approx(8.6)
    assert len(item.terrain.snapshot().xy) == 12231
    assert (item.terrain.snapshot().observed_at == 0.).all()
    assert item.intent.expires_at == 1. and not item.intent.hold
    np.testing.assert_array_equal(item.ego_pose, item.information_pose)
    native = diagnostic.payload(item, 43)
    assert type(native) is UpstreamPrimitiveInput and native.primitive_id == 43
    native.validate()


def test_actual_proxy_applies_delay_once_and_matches_shared_trajectory(cfg):
    item = diagnostic.initial_input(cfg)
    controls = np.zeros((40, 2)); controls[0] = [.04, .03]
    plan = rollout_numpy(item.ego_pose, controls, cfg.spec, item.now)
    output = UpstreamPlannerOutput(plan, PlannerFeedback(), True, "")
    witness = diagnostic.execute_first_control(cfg, item, output)
    assert witness["held_delay_steps"] == 3
    assert witness["actual_step"]["activation_time"][0] == pytest.approx(.6)
    np.testing.assert_array_equal(witness["actual_step"]["applied_control"][0, 0], controls[0])
    reserve = diagnostic.reserve_evidence(cfg, item, output)
    assert reserve["independent_review"]["passed"]
    assert reserve["envelope"]["delay_steps"] == 3 and reserve["envelope"]["fits_horizon"]


def test_valid_output_cannot_hide_bad_actual_control(cfg):
    item = diagnostic.initial_input(cfg)
    controls = np.zeros((40, 2)); controls[0] = [2., 0.]
    plan = rollout_numpy(item.ego_pose, controls, cfg.spec, item.now)
    output = UpstreamPlannerOutput(plan, PlannerFeedback(), True, "")
    with pytest.raises(AssertionError):
        diagnostic.execute_first_control(cfg, item, output)


def test_run_device_is_resolved_and_existing_artifacts_are_not_overwritten(tmp_path):
    cfg, run = diagnostic.create_run(tmp_path / "run", device="cuda")
    assert cfg.planner_device == "cuda"
    assert run.identity["contracts"]["planner_device"] == "cuda"
    before = (run.path / "run_manifest.json").read_bytes()
    with pytest.raises(FileExistsError):diagnostic.create_run(run.path, device="cuda")
    with pytest.raises(ValueError, match="device/configuration mismatch"):
        diagnostic.create_run(run.path, device="cpu", resume=True)
    resumed, other = diagnostic.create_run(run.path, device="cuda", resume=True)
    assert resumed.metadata == cfg.metadata and other.identity == run.identity
    assert (run.path / "run_manifest.json").read_bytes() == before


def test_contract_sources_must_be_frozen_before_execution(tmp_path, monkeypatch):
    _, run = diagnostic.create_run(tmp_path / "run")
    fake_sources = {"tests/actual.py": "before"}
    monkeypatch.setattr(diagnostic, "contract_test_sources", lambda: fake_sources.copy())
    with pytest.raises(ValueError, match="pre-execution"):
        diagnostic.freeze_tests(run)
    source = diagnostic.freeze_tests(run, create=True)
    assert source == fake_sources
    assert diagnostic.freeze_tests(run) == source
    fake_sources["tests/actual.py"] = "after"
    with pytest.raises(ValueError, match="different sources"):
        diagnostic.freeze_tests(run)
    with pytest.raises(ValueError, match="source changed"):
        diagnostic.require_unchanged(run, source)


def test_existing_junit_cannot_be_relabelled_with_new_source(tmp_path, monkeypatch):
    _, run = diagnostic.create_run(tmp_path / "run")
    (run.path / "metrics/contracts.xml").write_text("<testsuites/>")
    monkeypatch.setattr(diagnostic, "contract_test_sources", lambda: {})
    with pytest.raises(ValueError, match="after contract execution"):
        diagnostic.freeze_tests(run, create=True)


def test_strict_record_never_promotes_smoke_to_stage_or_training_gate():
    record = diagnostic.strict_record({"smoke_pass": True}, {"passed": True, "tests": 999})
    assert record["smoke_pass"]
    for name in ("s02_full_acceptance_pass", "s03_perception_communication_task_pass", "s05_gate_b_pass",
                 "t11_training_acceptance_pass", "legacy_23_real_acados_pass", "cpu_cuda_full_matrix_pass",
                 "realtime_pass", "physx_pass"):
        assert record[name] is False
    assert record["training_interactions"] == record["optimizer_updates"] == 0


def test_serial_preserves_nonfinite_failures_without_nonstandard_json():
    data = diagnostic.serial({"cost": float("nan"), "sample": torch.tensor([float("inf")]), "value": np.int64(1)})
    json.dumps(data, allow_nan=False)
    assert data["cost"]["nonfinite"] == "nan" and data["sample"][0]["nonfinite"] == "inf"


def test_construction_failure_is_saved_and_other_settings_continue(cfg, tmp_path, monkeypatch):
    _, run = diagnostic.create_run(tmp_path / "run")
    monkeypatch.setattr(diagnostic, "SETTINGS", (("line", "line_upstream_v1", None),))
    def broken(*args, **kwargs):raise RuntimeError("required backend unavailable")
    monkeypatch.setattr(diagnostic, "create_upstream_adapter", broken)
    result = diagnostic.diagnose(cfg, run)
    assert not result["smoke_pass"] and result["recorded_cases"] == 1
    assert result["records"][0]["error"] == "required backend unavailable"
    assert (run.path / "logs/line_construction_failure.txt").exists()
    assert json.loads((run.path / "metrics/line_construction_failure.json").read_text())["invocation_completed"] is False


def test_native_replay_keeps_all_47_ids_and_reports_invalid_output_as_failure(cfg, tmp_path, monkeypatch):
    _, run = diagnostic.create_run(tmp_path / "run")
    monkeypatch.setattr(diagnostic, "SETTINGS", (("primitive47", "primitive47_upstream_v1", None),))
    seen = []
    class ReplayOnly:
        def reset(self):pass
        def plan(self, request):
            seen.append(request.primitive_id)
            trajectory = rollout_numpy(np.zeros(3), np.zeros((40, 2)), cfg.spec, request.context.now)
            return UpstreamPlannerOutput(trajectory, PlannerFeedback(status="brake"), False, "controlled_failure")
        def state_dict(self):return {"test_double": True}
    monkeypatch.setattr(diagnostic, "create_upstream_adapter", lambda *args, **kwargs: ReplayOnly())
    monkeypatch.setattr(diagnostic, "reserve_evidence", lambda *args: {"independent_review": {"passed": True}})
    monkeypatch.setattr(diagnostic, "execute_first_control", lambda *args: {"test_double": True})
    result = diagnostic.diagnose(cfg, run)
    assert seen == list(range(47)) and result["recorded_cases"] == 47
    assert not result["smoke_pass"]
    assert all(r["invocation_completed"] and not r["valid_execution"] for r in result["records"])
    saved = json.loads((run.path / "samples/primitive47_43_execution.json").read_text())
    assert saved["output"]["fallback_reason"] == "controlled_failure"
    assert saved["goal_completion"] == "not_evaluated"


def test_main_requires_complete_junit_before_planner_calls(tmp_path, monkeypatch):
    path = tmp_path / "run"
    assert diagnostic.main(["--run-dir", str(path), "--prepare"]) == 0
    def failed(*args):raise ValueError("incomplete contract suite")
    monkeypatch.setattr(diagnostic, "require_contract_tests", failed)
    monkeypatch.setattr(diagnostic, "diagnose", lambda *args, **kwargs: pytest.fail("Ran without complete tests"))
    with pytest.raises(ValueError, match="incomplete contract suite"):
        diagnostic.main(["--run-dir", str(path), "--resume"])
    saved = json.loads((path / "metrics/entry_failure.json").read_text())
    assert not saved["s02_full_acceptance_pass"] and not saved["training_interactions"]


def test_both_devices_share_one_evidence_run_without_repeating_nmpc(tmp_path):
    cfg, run = diagnostic.create_run(tmp_path / "run", device="both")
    assert cfg.planner_device == "cpu"
    assert run.identity["settings"]["execution_devices"] == ["cpu", "cuda"]
    panels = diagnostic.requested_panels("both")
    assert len(panels) == 10 and sum(adapter == "nmpc_upstream_v1" for _, adapter, _ in panels) == 1
    extra = [(name, k) for name, _, k in panels if name.endswith("_cuda")]
    assert extra == [("mppi512_cuda", 512), ("mppi1024_cuda", 1024), ("mppi2048_cuda", 2048)]
    actual = diagnostic.panel_config(cfg, "mppi_upstream_v1", 2048, device="cuda")
    assert actual.planner_device == "cuda" and actual.metadata["planner_device"] == "cuda"
    with pytest.raises(ValueError, match="source/config/settings mismatch"):
        diagnostic.create_run(run.path, device="cpu", resume=True)


def test_configuration_bytes_cannot_change_after_producer_freeze(tmp_path):
    _, run = diagnostic.create_run(tmp_path / "run")
    tests = diagnostic.contract_test_sources()
    data = json.loads(Path(run.config).read_text())
    data["experiment"]["seed"] += 1
    Path(run.config).write_text(json.dumps(data))
    with pytest.raises(ValueError, match="source changed"):
        diagnostic.require_unchanged(run, tests)
