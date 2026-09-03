from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from audit_exp161_all_routes import _flood_sources, _hpp_goal_belief  # noqa: E402
from lunar_rover_tasks.tasks.multi_rover_gathering.site_commitment import SiteProposal  # noqa: E402


def _proposal(source: int, proposal_id: str, center: tuple[float, float]) -> SiteProposal:
    return SiteProposal(
        epoch=0,
        source_id=source,
        proposal_id=proposal_id,
        local_center_xy=center,
        center_xy=center,
        verification_radius_m=1.25,
        required_radius_m=0.75,
        terrain_cost=0.1,
        height_range_m=0.02,
        max_slope=0.03,
    )


def test_bounded_flooding_reaches_all_sources_on_path_graph() -> None:
    graph = torch.tensor(
        [
            [1, 1, 0, 0],
            [1, 1, 1, 0],
            [0, 1, 1, 1],
            [0, 0, 1, 1],
        ],
        dtype=torch.bool,
    )
    knowledge, rounds = _flood_sources(graph, max_rounds=3)
    assert rounds == 3
    assert all(items == {0, 1, 2, 3} for items in knowledge)


def test_bounded_flooding_fails_closed_on_disconnected_graph() -> None:
    graph = torch.tensor(
        [
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [0, 0, 1, 1],
            [0, 0, 1, 1],
        ],
        dtype=torch.bool,
    )
    knowledge, rounds = _flood_sources(graph, max_rounds=3)
    assert rounds is None
    assert all(len(items) == 2 for items in knowledge)


def test_hpp_private_candidates_do_not_fake_common_goal() -> None:
    groups = [
        [_proposal(0, "a", (-3.0, 0.0))],
        [_proposal(1, "b", (3.0, 0.0))],
        [_proposal(2, "c", (0.0, -3.0))],
        [_proposal(3, "d", (0.0, 3.0))],
    ]
    result = _hpp_goal_belief(
        groups,
        torch.tensor([[-2.0, 0.0], [2.0, 0.0], [0.0, -2.0], [0.0, 2.0]]),
        rounds=4,
        response_step_m=0.8,
        agreement_radius_m=0.5,
    )
    assert result["complete"]
    assert not result["agreed"]
    assert float(result["goal_spread_m"]) > 0.5
