from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from benchmark_cuda_short_training import run_cuda_short_training  # noqa: E402


def test_cuda_short_training_benchmark_when_available(tmp_path: Path) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not visible to this Python process")

    result = run_cuda_short_training(
        config="configs/experiment/exp_001_minimal.yaml",
        device="cuda",
        timesteps=32,
        output_root=tmp_path,
    )

    assert result["status"] == "ok"
    assert result["cuda_available"] is True
    assert result["wall_time_s"] > 0.0
    assert result["env_steps_per_s"] > 0.0
    assert result["agent_steps_per_s"] > 0.0
    assert result["estimated_seconds_per_1m_env_steps"] > 0.0

