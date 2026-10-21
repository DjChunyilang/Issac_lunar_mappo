"""Replay, scenario selection and separate real-sensor configuration contracts."""

import json
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from diagnose_planners import run, scenario_fixture


def test_request_replay_keeps_native_payload_and_expired_brake(tmp_path):
    requests = tmp_path / "requests.json"
    requests.write_text(
        json.dumps(
            [
                {"goal": [0.8, 0], "now": 0},
                {"pose": [0.1, 0, 0], "previous_control": [0.6, 0], "hold": True, "now": 0.2},
                {"goal": [0.8, 0], "now": 2, "timestamp": 0, "expires_at": 1},
            ]
        )
    )
    result = run(
        "configs/experiment/t06_mppi_base.yaml",
        "mppi_base_v1",
        samples=64,
        requests=str(requests),
        artifact_dir=tmp_path / "samples",
    )
    assert len(result["rows"]) == 3 and result["rows"][0]["valid"]
    assert result["rows"][1]["first_control"][0] == pytest.approx(0.3)
    assert result["rows"][2]["reason"] == "expired_intent" and not result["rows"][2]["valid"]
    assert len(list((tmp_path / "samples").glob("*.npz"))) == 1
    assert result["optimizer_updates"] == 0
    native = tmp_path / "native.json"
    native.write_text(json.dumps([{"goal": [0.8, 0], "primitive_id": 43}]))
    result = run(
        "configs/experiment/t05_planner_baselines.yaml",
        "primitive47_adapter_v1",
        requests=str(native),
    )
    assert result["rows"][0]["first_control"][0] == 0 and result["rows"][0]["first_control"][1] > 0


@pytest.mark.parametrize(
    "scene",
    ["flat", "risk", "blocked", "unknown", "cross", "head_on", "stopped", "missing", "expired"],
)
def test_fixed_scene_produces_legal_local_protocol_input(scene):
    r = scenario_fixture((0.8, 0), scene)
    r.validate()
    assert not hasattr(r.terrain, "core")
    if scene == "unknown":
        assert not r.terrain.safe([[0.8, 0]])[0]
    if scene == "blocked":
        assert not r.terrain.safe([[0.5, 0]])[0]
    if scene in ("cross", "head_on", "stopped", "expired"):
        assert r.neighbors[0].sender_id == 2
