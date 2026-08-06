"""Offline MAPF topology analysis and non-intervening trajectory diagnostics."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import heapq
import itertools
import random

import torch


@dataclass(slots=True)
class TopologyAnalysis:
    topology_class: str
    blocked_ratio: float
    bc_mean: float
    bc_variance: float
    bc_high_region_ratio: float
    betweenness: torch.Tensor


@dataclass(frozen=True, slots=True)
class VertexConstraint:
    agent: int
    node: int
    time: int


@dataclass(frozen=True, slots=True)
class EdgeConstraint:
    agent: int
    source: int
    target: int
    time: int


@dataclass(slots=True)
class CBSResult:
    paths: list[list[int]]
    sum_of_costs: int
    makespan: int
    expanded_nodes: int


def grid_adjacency(traversable: torch.Tensor) -> tuple[list[list[int]], torch.Tensor]:
    """Build a four-neighbor graph over a prevalidated executable terrain mask."""
    if traversable.ndim != 2 or traversable.dtype != torch.bool:
        raise ValueError("traversable must be a 2-D bool tensor.")
    height, width = traversable.shape
    flat_ids = torch.full(
        (height, width),
        -1,
        dtype=torch.long,
        device=traversable.device,
    )
    coordinates = torch.nonzero(traversable, as_tuple=False)
    if coordinates.numel() == 0:
        return [], flat_ids
    flat_ids[coordinates[:, 0], coordinates[:, 1]] = torch.arange(
        coordinates.shape[0],
        device=traversable.device,
    )
    adjacency: list[list[int]] = [[] for _ in range(coordinates.shape[0])]
    for row, col in coordinates.cpu().tolist():
        node = int(flat_ids[row, col])
        for delta_row, delta_col in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            neighbor_row = row + delta_row
            neighbor_col = col + delta_col
            if not (0 <= neighbor_row < height and 0 <= neighbor_col < width):
                continue
            neighbor = int(flat_ids[neighbor_row, neighbor_col])
            if neighbor >= 0:
                adjacency[node].append(neighbor)
    return adjacency, flat_ids


def approximate_betweenness_centrality(
    adjacency: list[list[int]],
    *,
    max_sources: int = 256,
    seed: int = 0,
) -> torch.Tensor:
    """Compute deterministic sampled Brandes centrality for an unweighted graph."""
    node_count = len(adjacency)
    centrality = torch.zeros(node_count, dtype=torch.float64)
    if node_count == 0:
        return centrality
    source_count = min(max(int(max_sources), 1), node_count)
    if source_count == node_count:
        sources = list(range(node_count))
    else:
        sources = random.Random(seed).sample(range(node_count), source_count)
    for source in sources:
        stack: list[int] = []
        predecessors: list[list[int]] = [[] for _ in range(node_count)]
        paths = [0.0] * node_count
        paths[source] = 1.0
        distance = [-1] * node_count
        distance[source] = 0
        queue: deque[int] = deque([source])
        while queue:
            vertex = queue.popleft()
            stack.append(vertex)
            for neighbor in adjacency[vertex]:
                if distance[neighbor] < 0:
                    queue.append(neighbor)
                    distance[neighbor] = distance[vertex] + 1
                if distance[neighbor] == distance[vertex] + 1:
                    paths[neighbor] += paths[vertex]
                    predecessors[neighbor].append(vertex)
        dependency = [0.0] * node_count
        while stack:
            target = stack.pop()
            if paths[target] > 0.0:
                scale = (1.0 + dependency[target]) / paths[target]
                for predecessor in predecessors[target]:
                    dependency[predecessor] += paths[predecessor] * scale
            if target != source:
                centrality[target] += dependency[target]
    # Undirected paths are counted in both directions. Sampling is rescaled so
    # values remain comparable across graph sizes and source budgets.
    centrality *= 0.5 * (node_count / float(source_count))
    return centrality


def analyze_traversability_topology(
    traversable: torch.Tensor,
    *,
    max_sources: int = 256,
    seed: int = 0,
) -> TopologyAnalysis:
    adjacency, _ = grid_adjacency(traversable)
    blocked_ratio = 1.0 - float(traversable.float().mean())
    centrality = approximate_betweenness_centrality(
        adjacency,
        max_sources=max_sources,
        seed=seed,
    )
    if centrality.numel() == 0:
        mean = variance = high_share = 0.0
    else:
        mean = float(centrality.mean())
        variance = float(centrality.var(unbiased=False))
        total = float(centrality.sum())
        high_count = max(1, (centrality.numel() + 9) // 10)
        high_share = (
            float(torch.topk(centrality, k=high_count).values.sum()) / total
            if total > 0.0
            else 0.0
        )
    if blocked_ratio < 0.05:
        topology_class = "Open"
    elif high_share >= 0.60:
        topology_class = "Bottleneck"
    else:
        topology_class = "Mixed"
    return TopologyAnalysis(
        topology_class=topology_class,
        blocked_ratio=blocked_ratio,
        bc_mean=mean,
        bc_variance=variance,
        bc_high_region_ratio=high_share,
        betweenness=centrality,
    )


def resample_trajectory_points_common_time(
    points: torch.Tensor,
    timestamps: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Interpolate per-rover trajectories onto one physical-time grid per env."""

    if points.ndim != 4 or points.shape[-1] < 2:
        raise ValueError("trajectory points must have shape [env, agent, time, xyz].")
    if timestamps.shape != points.shape[:-1]:
        raise ValueError("trajectory timestamps must match points without xyz dimension.")
    if points.shape[-2] < 2:
        raise ValueError("trajectory resampling requires at least two time points.")
    if not torch.isfinite(points).all() or not torch.isfinite(timestamps).all():
        raise ValueError("trajectory points and timestamps must be finite.")
    if (timestamps[..., 1:] < timestamps[..., :-1]).any():
        raise ValueError("trajectory timestamps must be monotonic.")

    n_samples = points.shape[-2]
    fractions = torch.linspace(
        0.0,
        1.0,
        n_samples,
        device=points.device,
        dtype=points.dtype,
    )
    common_horizon = timestamps[..., -1].amax(dim=1)
    common_times = common_horizon[:, None] * fractions[None, :]
    upper = (
        timestamps[:, :, None, :] < common_times[:, None, :, None]
    ).sum(dim=-1).clamp(1, n_samples - 1)
    lower = upper - 1
    point_dim = points.shape[-1]
    lower_points = torch.gather(
        points,
        2,
        lower[..., None].expand(-1, -1, -1, point_dim),
    )
    upper_points = torch.gather(
        points,
        2,
        upper[..., None].expand(-1, -1, -1, point_dim),
    )
    lower_time = torch.gather(timestamps, 2, lower)
    upper_time = torch.gather(timestamps, 2, upper)
    query = common_times[:, None, :]
    alpha = ((query - lower_time) / (upper_time - lower_time).clamp_min(1.0e-8)).clamp(
        0.0, 1.0
    )
    return torch.lerp(lower_points, upper_points, alpha[..., None]), common_times


def trajectory_pairwise_min_distance(
    points: torch.Tensor,
    timestamps: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return matched-timestamp minimum XY distance for every rover pair."""
    if points.ndim != 4 or points.shape[-1] < 2:
        raise ValueError("trajectory points must have shape [env, agent, time, xyz].")
    if timestamps is None:
        delta = points[:, :, None, :, :2] - points[:, None, :, :, :2]
        return torch.linalg.norm(delta, dim=-1).amin(dim=-1)
    if timestamps.shape != points.shape[:-1]:
        raise ValueError("trajectory timestamps must match points without xyz dimension.")
    if not torch.isfinite(points).all() or not torch.isfinite(timestamps).all():
        raise ValueError("trajectory points and timestamps must be finite.")
    if (timestamps[..., 1:] < timestamps[..., :-1]).any():
        raise ValueError("trajectory timestamps must be monotonic.")

    # Each pair needs its own common physical-time grid. Using the longest
    # trajectory in the whole team makes the measured distance of pair (i, j)
    # change when an unrelated rover changes only its path duration.
    n_envs, n_agents, n_samples, point_dim = points.shape
    fractions = torch.linspace(
        0.0,
        1.0,
        n_samples,
        device=points.device,
        dtype=points.dtype,
    )
    pair_horizon = torch.maximum(
        timestamps[..., -1, None],
        timestamps[:, None, :, -1],
    )
    query = pair_horizon[..., None] * fractions

    first_times = timestamps[:, :, None, :].expand(-1, -1, n_agents, -1)
    second_times = timestamps[:, None, :, :].expand(-1, n_agents, -1, -1)
    first_points = points[:, :, None, :, :].expand(
        -1, -1, n_agents, -1, -1
    )
    second_points = points[:, None, :, :, :].expand(
        -1, n_agents, -1, -1, -1
    )

    def interpolate(
        source_points: torch.Tensor,
        source_times: torch.Tensor,
    ) -> torch.Tensor:
        upper = (source_times[..., None, :] < query[..., :, None]).sum(dim=-1)
        upper = upper.clamp(1, n_samples - 1)
        lower = upper - 1
        gather_index = lower[..., None].expand(
            n_envs, n_agents, n_agents, n_samples, point_dim
        )
        lower_points = torch.gather(source_points, 3, gather_index)
        upper_points = torch.gather(
            source_points,
            3,
            upper[..., None].expand_as(gather_index),
        )
        lower_time = torch.gather(source_times, 3, lower)
        upper_time = torch.gather(source_times, 3, upper)
        alpha = (
            (query - lower_time) / (upper_time - lower_time).clamp_min(1.0e-8)
        ).clamp(0.0, 1.0)
        return torch.lerp(lower_points, upper_points, alpha[..., None])

    first_aligned = interpolate(first_points, first_times)
    second_aligned = interpolate(second_points, second_times)
    delta = first_aligned[..., :2] - second_aligned[..., :2]
    return torch.linalg.norm(delta, dim=-1).amin(dim=-1)


class TrajectoryConflictTracker:
    """Track predicted conflicts without changing actions or controls."""

    def __init__(self, num_envs: int, n_agents: int, device: torch.device | str) -> None:
        shape = (int(num_envs), int(n_agents), int(n_agents))
        self.device = torch.device(device)
        self.consecutive_steps = torch.zeros(shape, dtype=torch.long, device=self.device)
        self.active_steps = torch.zeros(shape, dtype=torch.long, device=self.device)
        self._upper = torch.triu(
            torch.ones(n_agents, n_agents, dtype=torch.bool, device=self.device),
            diagonal=1,
        ).unsqueeze(0)

    def reset(self, env_ids: torch.Tensor) -> None:
        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        self.consecutive_steps[env_ids] = 0
        self.active_steps[env_ids] = 0

    def update(
        self,
        points: torch.Tensor,
        safe_distance: float,
        *,
        timestamps: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if safe_distance <= 0.0:
            raise ValueError("safe_distance must be positive.")
        min_distance = trajectory_pairwise_min_distance(points, timestamps)
        active = (min_distance < float(safe_distance)) & self._upper
        previous_active = self.consecutive_steps > 0
        repeated = active & previous_active
        resolved = ~active & previous_active & self._upper
        resolved_steps = torch.where(
            resolved,
            self.active_steps,
            torch.zeros_like(self.active_steps),
        )
        self.consecutive_steps = torch.where(
            active,
            self.consecutive_steps + 1,
            torch.zeros_like(self.consecutive_steps),
        )
        self.active_steps = torch.where(
            active,
            self.active_steps + 1,
            torch.zeros_like(self.active_steps),
        )
        return {
            "min_pair_distance": min_distance,
            "active": active,
            "repeated": repeated,
            "resolved": resolved,
            "resolved_steps": resolved_steps,
            "predicted_conflict_count": active.sum(dim=(1, 2)),
            "repeated_pair_conflict_count": repeated.sum(dim=(1, 2)),
        }


def _constrained_shortest_path(
    adjacency: list[list[int]],
    start: int,
    goal: int,
    agent: int,
    constraints: tuple[VertexConstraint | EdgeConstraint, ...],
    max_time: int,
) -> list[int] | None:
    vertex_constraints = {
        (constraint.node, constraint.time)
        for constraint in constraints
        if isinstance(constraint, VertexConstraint) and constraint.agent == agent
    }
    edge_constraints = {
        (constraint.source, constraint.target, constraint.time)
        for constraint in constraints
        if isinstance(constraint, EdgeConstraint) and constraint.agent == agent
    }
    if (start, 0) in vertex_constraints:
        return None
    required_time = max(
        (
            constraint.time
            for constraint in constraints
            if constraint.agent == agent
        ),
        default=0,
    )
    queue: deque[tuple[int, int]] = deque([(start, 0)])
    parent: dict[tuple[int, int], tuple[int, int] | None] = {(start, 0): None}
    terminal: tuple[int, int] | None = None
    while queue:
        node, time = queue.popleft()
        if node == goal and time >= required_time:
            terminal = (node, time)
            break
        if time >= max_time:
            continue
        next_time = time + 1
        for neighbor in (*adjacency[node], node):
            state = (neighbor, next_time)
            if state in parent:
                continue
            if (neighbor, next_time) in vertex_constraints:
                continue
            if (node, neighbor, time) in edge_constraints:
                continue
            parent[state] = (node, time)
            queue.append(state)
    if terminal is None:
        return None
    path: list[int] = []
    state: tuple[int, int] | None = terminal
    while state is not None:
        path.append(state[0])
        state = parent[state]
    return list(reversed(path))


def _path_node(path: list[int], time: int) -> int:
    return path[min(time, len(path) - 1)]


def _first_conflict(paths: list[list[int]]) -> dict[str, int] | None:
    horizon = max(len(path) for path in paths)
    for time in range(horizon):
        for first in range(len(paths)):
            first_node = _path_node(paths[first], time)
            for second in range(first + 1, len(paths)):
                second_node = _path_node(paths[second], time)
                if first_node == second_node:
                    return {
                        "kind": 0,
                        "first": first,
                        "second": second,
                        "node": first_node,
                        "time": time,
                    }
                if time + 1 >= horizon:
                    continue
                first_next = _path_node(paths[first], time + 1)
                second_next = _path_node(paths[second], time + 1)
                if first_node == second_next and second_node == first_next:
                    return {
                        "kind": 1,
                        "first": first,
                        "second": second,
                        "first_source": first_node,
                        "first_target": first_next,
                        "second_source": second_node,
                        "second_target": second_next,
                        "time": time,
                    }
    return None


def conflict_based_search(
    adjacency: list[list[int]],
    starts: list[int],
    goals: list[int],
    *,
    max_time: int = 256,
    max_expansions: int = 10000,
) -> CBSResult | None:
    """Small offline vanilla-CBS baseline; never used by policy execution."""
    if len(starts) != len(goals) or not starts:
        raise ValueError("starts and goals must be non-empty and have equal length.")
    node_count = len(adjacency)
    if any(node < 0 or node >= node_count for node in (*starts, *goals)):
        raise ValueError("starts and goals must reference valid graph nodes.")
    root_constraints: tuple[VertexConstraint | EdgeConstraint, ...] = ()
    root_paths: list[list[int]] = []
    for agent, (start, goal) in enumerate(zip(starts, goals, strict=True)):
        path = _constrained_shortest_path(
            adjacency,
            start,
            goal,
            agent,
            root_constraints,
            max_time,
        )
        if path is None:
            return None
        root_paths.append(path)
    counter = itertools.count()
    queue: list[
        tuple[
            int,
            int,
            tuple[VertexConstraint | EdgeConstraint, ...],
            list[list[int]],
        ]
    ] = []
    heapq.heappush(
        queue,
        (sum(len(path) - 1 for path in root_paths), next(counter), root_constraints, root_paths),
    )
    expanded = 0
    while queue and expanded < max_expansions:
        _, _, constraints, paths = heapq.heappop(queue)
        expanded += 1
        conflict = _first_conflict(paths)
        if conflict is None:
            return CBSResult(
                paths=paths,
                sum_of_costs=sum(len(path) - 1 for path in paths),
                makespan=max(len(path) - 1 for path in paths),
                expanded_nodes=expanded,
            )
        for agent_key in ("first", "second"):
            agent = conflict[agent_key]
            if conflict["kind"] == 0:
                new_constraint: VertexConstraint | EdgeConstraint = VertexConstraint(
                    agent=agent,
                    node=conflict["node"],
                    time=conflict["time"],
                )
            elif agent_key == "first":
                new_constraint = EdgeConstraint(
                    agent=agent,
                    source=conflict["first_source"],
                    target=conflict["first_target"],
                    time=conflict["time"],
                )
            else:
                new_constraint = EdgeConstraint(
                    agent=agent,
                    source=conflict["second_source"],
                    target=conflict["second_target"],
                    time=conflict["time"],
                )
            child_constraints = (*constraints, new_constraint)
            child_paths = [path.copy() for path in paths]
            replanned = _constrained_shortest_path(
                adjacency,
                starts[agent],
                goals[agent],
                agent,
                child_constraints,
                max_time,
            )
            if replanned is None:
                continue
            child_paths[agent] = replanned
            heapq.heappush(
                queue,
                (
                    sum(len(path) - 1 for path in child_paths),
                    next(counter),
                    child_constraints,
                    child_paths,
                ),
            )
    return None
