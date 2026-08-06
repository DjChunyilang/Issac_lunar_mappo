from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from analyze_failed_episode_repeated_conflicts import (  # noqa: E402
    summarize_failed_episode_repeats,
)


def test_failed_episode_summary_counts_pair_onsets_and_includes_zero_failures() -> None:
    repeated = torch.zeros((6, 2, 2, 2), dtype=torch.bool)
    repeated[1, 0, 0, 1] = True
    repeated[3, 0, 0, 1] = True
    done = torch.zeros((6, 2), dtype=torch.bool)
    success = torch.zeros_like(done)
    collision = torch.zeros_like(done)
    out_of_bounds = torch.zeros_like(done)
    timeout = torch.zeros_like(done)
    done[3, 0] = True
    collision[3, 0] = True
    done[2, 1] = True
    success[2, 1] = True
    done[5, 1] = True
    timeout[5, 1] = True

    result = summarize_failed_episode_repeats(
        repeated_pairs=repeated,
        done=done,
        success=success,
        collision_done=collision,
        out_of_bounds=out_of_bounds,
        timeout=timeout,
    )
    assert result["completed_episodes"] == 3
    assert result["success_episodes"] == 1
    assert result["failed_episodes"] == 2
    assert result["failed_with_repeated_fraction"] == pytest.approx(0.5)
    # Counts are [2, 0]; torch.median intentionally retains the lower median.
    assert result["failed_repeated_event_count"]["median"] == pytest.approx(0.0)
    assert result["by_reason"]["collision"]["event_count"]["mean"] == pytest.approx(
        2.0
    )
    assert result["by_reason"]["timeout"]["event_count"]["mean"] == pytest.approx(
        0.0
    )


def test_incomplete_episode_is_excluded() -> None:
    repeated = torch.ones((3, 1, 2, 2), dtype=torch.bool)
    zeros = torch.zeros((3, 1), dtype=torch.bool)
    result = summarize_failed_episode_repeats(
        repeated_pairs=repeated,
        done=zeros,
        success=zeros,
        collision_done=zeros,
        out_of_bounds=zeros,
        timeout=zeros,
    )
    assert result["completed_episodes"] == 0
    assert result["failed_episodes"] == 0
