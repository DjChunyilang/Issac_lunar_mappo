from __future__ import annotations

import torch

from lunar_rover_tasks.tasks.multi_rover_gathering.mapf_diagnostics import (
    TrajectoryConflictTracker,
    analyze_traversability_topology,
    conflict_based_search,
    trajectory_pairwise_min_distance,
)


def test_open_map_is_not_misclassified_by_center_centrality() -> None:
    analysis = analyze_traversability_topology(torch.ones(10, 10, dtype=torch.bool))
    assert analysis.topology_class == "Open"
    assert analysis.blocked_ratio == 0.0


def test_mixed_map_remains_mixed() -> None:
    traversable = torch.ones(12, 12, dtype=torch.bool)
    traversable[3:9, 5:7] = False
    analysis = analyze_traversability_topology(traversable)
    assert analysis.topology_class == "Mixed"


def test_concentrated_narrow_component_is_bottleneck() -> None:
    traversable = torch.zeros(30, 30, dtype=torch.bool)
    traversable[15, 2:11] = True
    # Disconnected single cells carry zero BC; the only routed component is a
    # deliberately narrow corridor whose central nodes dominate S10.
    for row in range(1, 20, 2):
        for col in range(12, 30, 2):
            traversable[row, col] = True
    analysis = analyze_traversability_topology(traversable)
    assert analysis.bc_high_region_ratio >= 0.60
    assert analysis.topology_class == "Bottleneck"


def test_matched_timestamp_conflict_distinguishes_crossing_paths() -> None:
    points = torch.tensor(
        [[
            [[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.0, -1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        ]]
    )
    minimum = trajectory_pairwise_min_distance(points)
    assert minimum[0, 0, 1] == 0.0


def test_physical_time_alignment_rejects_normalized_progress_false_conflict() -> None:
    points = torch.tensor(
        [[
            [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.0, 0.0]],
        ]]
    )
    unequal_timing = torch.tensor([[[0.0, 0.5, 1.0], [0.0, 1.0, 2.0]]])
    equal_timing = torch.tensor([[[0.0, 0.5, 1.0], [0.0, 0.5, 1.0]]])

    normalized_progress = trajectory_pairwise_min_distance(points)
    physical_time = trajectory_pairwise_min_distance(points, unequal_timing)
    actual_crossing = trajectory_pairwise_min_distance(points, equal_timing)

    assert normalized_progress[0, 0, 1] == 0.0
    assert physical_time[0, 0, 1] == 0.5
    assert actual_crossing[0, 0, 1] == 0.0


def test_pairwise_time_alignment_is_invariant_to_third_rover_duration() -> None:
    points = torch.tensor(
        [[
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            [[2.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[5.0, 5.0, 0.0], [6.0, 5.0, 0.0], [7.0, 5.0, 0.0]],
        ]]
    )
    long_third = torch.tensor(
        [[[0.0, 0.5, 1.0], [0.0, 0.5, 1.0], [0.0, 5.0, 10.0]]]
    )
    short_third = long_third.clone()
    short_third[0, 2] = torch.tensor([0.0, 0.5, 1.0])

    long_result = trajectory_pairwise_min_distance(points, long_third)
    short_result = trajectory_pairwise_min_distance(points, short_third)

    assert long_result[0, 0, 1] == 0.0
    assert short_result[0, 0, 1] == 0.0
    assert long_result[0, 0, 1] == short_result[0, 0, 1]


def test_repeated_conflicts_and_resolution_are_tracked() -> None:
    tracker = TrajectoryConflictTracker(1, 2, "cpu")
    crossing = torch.tensor(
        [[
            [[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        ]]
    )
    separated = crossing.clone()
    separated[:, 1, :, 1] = 5.0
    first = tracker.update(crossing, 0.42)
    second = tracker.update(crossing, 0.42)
    resolved = tracker.update(separated, 0.42)
    assert first["predicted_conflict_count"].item() == 1
    assert first["repeated_pair_conflict_count"].item() == 0
    assert second["repeated_pair_conflict_count"].item() == 1
    assert resolved["resolved"].sum().item() == 1
    assert resolved["resolved_steps"].sum().item() == 2


def test_offline_cbs_resolves_swap_without_entering_policy_path() -> None:
    adjacency = [[1, 2], [0, 2], [0, 1]]
    result = conflict_based_search(adjacency, [0, 1], [1, 0], max_time=8)
    assert result is not None
    assert result.paths[0][0] == 0 and result.paths[0][-1] == 1
    assert result.paths[1][0] == 1 and result.paths[1][-1] == 0
    for time in range(max(len(path) for path in result.paths)):
        occupied = [path[min(time, len(path) - 1)] for path in result.paths]
        assert len(occupied) == len(set(occupied))
