from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
import yaml
from matplotlib.colors import to_rgb

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import render_skrl_proxy_rollout as rollout_render  # noqa: E402


def _frame(
    *,
    centroid_is_flat: bool,
    oracle_feasible: bool | None,
) -> rollout_render.FlatnessFrame:
    return rollout_render.FlatnessFrame(
        centroid_xy=np.asarray([0.25, -0.5], dtype=np.float32),
        centroid_is_flat=centroid_is_flat,
        centroid_height_range=0.17,
        centroid_max_slope=0.24,
        centroid_mean_slope=0.11,
        oracle_feasible=oracle_feasible,
        oracle_height_range=0.12,
        oracle_max_slope=0.20,
    )


def test_flatness_footprints_use_configured_radius_and_status_colors() -> None:
    fig, ax = plt.subplots()
    try:
        centroid_circle, oracle_circle = rollout_render._add_flatness_footprints(
            ax,
            _frame(centroid_is_flat=False, oracle_feasible=True),
            np.asarray([1.5, 2.0], dtype=np.float32),
            radius=1.25,
        )

        assert centroid_circle.radius == pytest.approx(1.25)
        assert oracle_circle.radius == pytest.approx(1.25)
        np.testing.assert_allclose(
            centroid_circle.get_edgecolor()[:3],
            to_rgb(rollout_render.ROUGH_FOOTPRINT_COLOR),
        )
        np.testing.assert_allclose(
            oracle_circle.get_edgecolor()[:3],
            to_rgb(rollout_render.FLAT_FOOTPRINT_COLOR),
        )
        assert centroid_circle.get_label() == "team footprint: rough"
        assert oracle_circle.get_label() == "oracle footprint: feasible"
    finally:
        plt.close(fig)


def test_axis_limits_reserve_the_configured_footprint_radius() -> None:
    positions = np.zeros((4, 3), dtype=np.float32)
    oracle = np.zeros(3, dtype=np.float32)

    xmin, xmax, ymin, ymax = rollout_render._axis_limits(
        [positions],
        [oracle],
        footprint_radius=2.0,
    )

    assert xmin <= -2.2
    assert xmax >= 2.2
    assert ymin <= -2.2
    assert ymax >= 2.2


def test_flatness_frame_extracts_centroid_and_oracle_diagnostics() -> None:
    flatness = SimpleNamespace(
        is_flat=torch.tensor([False]),
        height_range=torch.tensor([0.23]),
        max_slope=torch.tensor([0.31]),
        mean_slope=torch.tensor([0.18]),
    )
    oracle_search = {
        "feasible": torch.tensor([True]),
        "height_range": torch.tensor([0.14]),
        "max_slope": torch.tensor([0.22]),
    }

    frame = rollout_render._flatness_frame(
        torch.tensor([[1.25, -0.75, 0.1]]),
        flatness,
        oracle_search,
    )

    np.testing.assert_allclose(frame.centroid_xy, [1.25, -0.75])
    assert frame.centroid_is_flat is False
    assert frame.centroid_height_range == pytest.approx(0.23)
    assert frame.centroid_max_slope == pytest.approx(0.31)
    assert frame.centroid_mean_slope == pytest.approx(0.18)
    assert frame.oracle_feasible is True
    assert frame.oracle_height_range == pytest.approx(0.14)
    assert frame.oracle_max_slope == pytest.approx(0.22)


def test_render_artifacts_merge_into_existing_manifest_without_losing_fields(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    run_dir = workspace / "outputs" / "runs" / "exp_test" / "run_test"
    gif_path = run_dir / "videos" / "proxy_eval_rollout.gif"
    terrain_path = run_dir / "figures" / "terrain_height_map.png"
    metrics_path = run_dir / "metrics" / "proxy_rollout_render.json"
    for path in (gif_path, terrain_path, metrics_path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"artifact")
    manifest_path = run_dir / "run_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "producer": "existing-producer",
                "custom": {"keep": ["all", "unknown", "fields"]},
                "artifacts": {
                    "checkpoint_best": "checkpoints/best.pt",
                },
            }
        ),
        encoding="utf-8",
    )

    with patch.object(rollout_render, "ROOT", workspace):
        updated = rollout_render._merge_render_artifacts_into_manifest(
            run_dir,
            gif_path=gif_path,
            terrain_height_path=terrain_path,
            metrics_path=metrics_path,
        )

    assert updated == manifest_path
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["producer"] == "existing-producer"
    assert manifest["custom"] == {"keep": ["all", "unknown", "fields"]}
    assert manifest["artifacts"]["checkpoint_best"] == "checkpoints/best.pt"
    assert manifest["artifacts"]["metrics_proxy_rollout_render"] == str(
        metrics_path.relative_to(workspace)
    )
    assert manifest["artifacts"]["figures_terrain_height"] == str(
        terrain_path.relative_to(workspace)
    )
    assert manifest["artifacts"]["videos_proxy_rollout"] == str(
        gif_path.relative_to(workspace)
    )


def test_render_artifact_manifest_merge_skips_missing_manifest_or_artifact(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    gif_path = run_dir / "videos" / "proxy_eval_rollout.gif"
    terrain_path = run_dir / "figures" / "terrain_height_map.png"
    metrics_path = run_dir / "metrics" / "proxy_rollout_render.json"
    for path in (gif_path, metrics_path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"artifact")

    assert (
        rollout_render._merge_render_artifacts_into_manifest(
            run_dir,
            gif_path=gif_path,
            terrain_height_path=terrain_path,
            metrics_path=metrics_path,
        )
        is None
    )
    manifest_path = run_dir / "run_manifest.json"
    assert not manifest_path.exists()

    original = {
        "schema_version": 1,
        "artifacts": {"checkpoint_best": "checkpoints/best.pt"},
    }
    manifest_path.write_text(json.dumps(original), encoding="utf-8")
    assert (
        rollout_render._merge_render_artifacts_into_manifest(
            run_dir,
            gif_path=gif_path,
            terrain_height_path=terrain_path,
            metrics_path=metrics_path,
        )
        is None
    )
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == original


def test_render_rollout_legacy_config_writes_final_flatness_metrics(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "legacy_experiment.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "experiment": {
                    "name": "legacy_render",
                    "num_envs": 1,
                    "device": "cpu",
                },
                "simulation": {
                    "episode_length_s": 2.0,
                    "physics_dt": 0.05,
                    "control_decimation": 4,
                },
                "terrain": {
                    "type": "flat",
                    "dynamics_enabled": False,
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save({}, checkpoint_path)
    run_dir = tmp_path / "legacy_run"
    run_dir.mkdir()
    manifest_path = run_dir / "run_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "custom": {"preserved": True},
                "artifacts": {"checkpoint_best": "checkpoints/best.pt"},
            }
        ),
        encoding="utf-8",
    )
    gif_path = run_dir / "videos" / "proxy_eval_rollout.gif"
    metrics_path = run_dir / "metrics" / "proxy_rollout_render.json"

    def act(actor_obs: torch.Tensor) -> torch.Tensor:
        return torch.zeros(
            actor_obs.shape[0],
            actor_obs.shape[1],
            2,
            dtype=actor_obs.dtype,
            device=actor_obs.device,
        )

    def fake_height_map(*args, **kwargs) -> None:
        del kwargs
        path = Path(args[3])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"png")

    def fake_save_gif(*args, **kwargs) -> int:
        del kwargs
        path = Path(args[3])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"gif")
        return 2

    with (
        patch.object(
            rollout_render,
            "_load_policy_players",
            return_value=(act, "test-policy"),
        ),
        patch.object(
            rollout_render,
            "save_height_map",
            side_effect=fake_height_map,
        ) as save_height_map,
        patch.object(
            rollout_render,
            "_save_gif",
            side_effect=fake_save_gif,
        ) as save_gif,
    ):
        result = rollout_render.render_rollout(
            config_path,
            checkpoint_path,
            device="cpu",
            steps=1,
            run_dir=run_dir,
        )

    assert save_height_map.call_count == 1
    assert save_gif.call_count == 1
    assert save_gif.call_args.kwargs["footprint_radius"] == pytest.approx(0.75)
    assert len(save_gif.call_args.args[2]) == len(save_gif.call_args.args[0])

    stored = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert result["status"] == "ok"
    assert stored["flatness_footprint"]["radius"] == pytest.approx(0.75)
    assert stored["flatness_footprint"]["require_flat_for_success"] is True
    assert stored["final_flatness_ok"] is True
    assert stored["final_gather_point_is_flat"] is True
    assert stored["final_gather_point_height_range"] == pytest.approx(0.0)
    assert stored["final_gather_point_max_slope"] == pytest.approx(0.0)
    assert stored["final_flatness"]["centroid_xy"]
    assert stored["oracle_search_feasible"] is True
    assert stored["oracle_search"]["feasible"] is True
    assert stored["oracle_search"]["point"]
    assert result["run_manifest"] == str(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["custom"] == {"preserved": True}
    assert manifest["artifacts"]["checkpoint_best"] == "checkpoints/best.pt"
    assert manifest["artifacts"]["metrics_proxy_rollout_render"] == str(
        metrics_path
    )
    assert manifest["artifacts"]["figures_terrain_height"] == str(
        run_dir / "figures" / "terrain_height_map.png"
    )
    assert manifest["artifacts"]["videos_proxy_rollout"] == str(gif_path)


def test_render_rollout_uses_terminal_flatness_snapshot_before_auto_reset(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "terminal_experiment.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "experiment": {
                    "name": "terminal_render",
                    "num_envs": 1,
                    "device": "cpu",
                },
                "simulation": {
                    "episode_length_s": 0.2,
                    "physics_dt": 0.05,
                    "control_decimation": 4,
                },
                "terrain": {
                    "type": "flat",
                    "dynamics_enabled": False,
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save({}, checkpoint_path)
    metrics_path = tmp_path / "terminal_render.json"
    real_env_class = rollout_render.MultiRoverGatheringCore

    def env_factory(cfg):
        env = real_env_class(cfg)
        original_step = env.step

        def terminal_step(action):
            output = original_step(action)
            assert bool(output.info["done"].done[0])
            output.info["gather_point_flatness"] = SimpleNamespace(
                is_flat=torch.tensor([False]),
                height_range=torch.tensor([0.42]),
                max_slope=torch.tensor([0.51]),
                mean_slope=torch.tensor([0.29]),
            )
            output.info["oracle_search"] = {
                "feasible": torch.tensor([False]),
                "height_range": torch.tensor([0.33]),
                "max_slope": torch.tensor([0.44]),
            }
            return output

        env.step = terminal_step
        return env

    def act(actor_obs: torch.Tensor) -> torch.Tensor:
        return torch.zeros(
            actor_obs.shape[0],
            actor_obs.shape[1],
            2,
            dtype=actor_obs.dtype,
            device=actor_obs.device,
        )

    with (
        patch.object(
            rollout_render,
            "MultiRoverGatheringCore",
            side_effect=env_factory,
        ),
        patch.object(
            rollout_render,
            "_load_policy_players",
            return_value=(act, "test-policy"),
        ),
        patch.object(rollout_render, "save_height_map"),
        patch.object(rollout_render, "_save_gif", return_value=1),
    ):
        result = rollout_render.render_rollout(
            config_path,
            checkpoint_path,
            device="cpu",
            steps=2,
            output=tmp_path / "terminal.gif",
            metrics_output=metrics_path,
        )

    assert result["done_reason"] == "timeout"
    assert result["final_flatness_ok"] is False
    assert result["final_gather_point_is_flat"] is False
    assert result["final_gather_point_height_range"] == pytest.approx(0.42)
    assert result["final_gather_point_max_slope"] == pytest.approx(0.51)
    assert result["oracle_search_feasible"] is False
    assert result["oracle_gather_point_height_range"] == pytest.approx(0.33)
    assert result["oracle_gather_point_max_slope"] == pytest.approx(0.44)
    stored = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert stored["final_flatness"] == result["final_flatness"]
    assert stored["oracle_search"] == result["oracle_search"]
