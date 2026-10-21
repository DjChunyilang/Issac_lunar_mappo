"""Preparation entry keeps original budgets reserved and runtime disabled."""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import sys

import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from diagnose_t11_actions import DEFAULT_CONFIG, load_preparation, probe_matrix
import diagnose_t11_actions as diagnostic
from _common import cfg_from_experiment, load_yaml


def test_preparation_freezes_original_matrix_and_has_no_runtime_authority():
    value, _, cfg = load_preparation(DEFAULT_CONFIG)
    assert value["representations"] == ["A0", "A1", "A2", "A3"]
    assert value["screening_budgets_reserved"] == {
        "signal_check": 100000, "multi_seed": 500000, "finalist": 2000000, "multi_seed_count": 3,
    }
    assert value["training_paused"] and not value["environment_enabled"]
    assert cfg.spec.rho_max == 4 and cfg.spec.vehicle.forward_limit == pytest.approx(25 / 9)
    with pytest.raises(ValueError, match="not an experiment config"):
        cfg_from_experiment(DEFAULT_CONFIG)


@pytest.mark.parametrize("field,value", [
    ("training_paused", False), ("training_paused", 1), ("environment_enabled", True),
    ("representations", ["A0", "A1", "A2"]), ("seeds", [23]),
    ("schema", "upstream_runtime_v1"), ("base_config", "../ablation/t11_action_contracts.yaml"),
    ("extra", "silently_ignored"),
])
def test_unpaused_or_incomplete_preparation_rejected(tmp_path, field, value):
    data = deepcopy(load_yaml(DEFAULT_CONFIG))
    data["base_config"] = str(ROOT / "configs/experiment/upstream_scale_s01_review_v2.yaml")
    data[field] = value
    path = tmp_path / "mutated.yaml"
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError):
        load_preparation(path)


def test_probe_inspects_gradients_without_constructing_optimizer(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("A preparation diagnostic must not construct an optimizer")

    monkeypatch.setattr(torch.optim, "Adam", forbidden)
    monkeypatch.setattr(torch.optim, "SGD", forbidden)
    torch.set_num_threads(1)
    _, _, cfg = load_preparation(DEFAULT_CONFIG)
    records, payloads = probe_matrix(cfg, [23, 31, 47])
    assert len(records) == len(payloads) == 12
    assert all(r["passed"] and r["weights_unchanged"] and r["restore_exact"] for r in records)
    assert all(r["metadata"]["training_eligible"] is False for r in records)
    assert sum(r["sampled_decisions"] for r in records) == 768
    assert sum(p["recorded_decisions"] for p in payloads) == 768
    assert all(p["sample_record_coverage"] == 1 and len(p["sequence"]["log_prob"]) == 8 for p in payloads)
    assert all(p["executor_feedback"] == "not_connected" for p in payloads)
    native = [p for p in payloads if p["representation"] == "A3"]
    assert all(request["primitive_fingerprint"] and request["goal_xy"] is None
               for p in native for request in p["requests"])


def test_test_body_change_cannot_rebind_old_execution_to_new_source(tmp_path, monkeypatch):
    path = tmp_path / "test_frozen.py"
    path.write_text("def test_same_name():\n    assert False\n")
    monkeypatch.setattr(diagnostic, "TEST_FILES", (str(path),))
    frozen = diagnostic.contract_test_sources()
    diagnostic.require_unchanged_test_sources(frozen)
    path.write_text("def test_same_name():\n    assert True\n")
    with pytest.raises(ValueError, match="test source changed"):
        diagnostic.require_unchanged_test_sources(frozen)


def test_cpu_preparation_rejects_cuda_identity(monkeypatch):
    _, _, cfg = load_preparation(DEFAULT_CONFIG)
    monkeypatch.setattr(diagnostic, "cfg_from_experiment", lambda path: replace(cfg, planner_device="cuda"))
    with pytest.raises(ValueError, match="paused upstream identity"):
        load_preparation(DEFAULT_CONFIG)
