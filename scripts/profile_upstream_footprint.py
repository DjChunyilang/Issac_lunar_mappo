#!/usr/bin/env python3
"""Read-only runtime diagnosis; batching prototype is never runtime-enabled."""

import argparse
from contextlib import contextmanager
import cProfile
from dataclasses import asdict
from pathlib import Path
import pstats
import time

import numpy as np
import torch

from _common import cfg_from_experiment
from diagnose_upstream_execution import serial, synthetic_local_frame
from planner_evidence import EvidenceRun, json_write, source_snapshot
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.protocol import TerrainSnapshot
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_execution import rollout_numpy
from lunar_rover_tasks.tasks.multi_rover_gathering.local_goal.upstream_footprint import (
    FootprintTerrain, FootprintReviewer, FootprintNeighbor, admit_control,
)


class CountedTree:
    """Process-local instrumentation; no change to stored runtime sources."""
    def __init__(self, tree):
        self.tree = tree
        self.query_calls = 0; self.ball_calls = 0; self.ball_points = 0
        self.query_seconds = 0.; self.ball_seconds = 0.

    def query(self, points, *args, **kwargs):
        start = time.perf_counter(); result = self.tree.query(points, *args, **kwargs)
        self.query_seconds += time.perf_counter() - start; self.query_calls += 1
        return result

    def query_ball_point(self, points, *args, **kwargs):
        start = time.perf_counter(); result = self.tree.query_ball_point(points, *args, **kwargs)
        self.ball_seconds += time.perf_counter() - start; self.ball_calls += 1
        self.ball_points += 1 if np.asarray(points).ndim == 1 else len(points)
        return result

    def metrics(self):
        return {key: value for key, value in vars(self).items() if key != "tree"}


def batch_ties_prototype(self, xy):
    """Same queries/tie order/thresholds, batched ball query and segmented merge.

    Keep both k=1/k=2 queries to preserve all five source-channel semantics.
    return_sorted=False matches scalar cKDTree query_ball_point defaults.
    """
    points = np.atleast_2d(np.asarray(xy, dtype=float))
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError("Footprint query requires finite [M,2]")
    if not len(self._xy):return np.zeros((len(points), 5)), np.zeros(len(points), dtype=bool)
    distance, index = self._tree.query(points)
    features = self._features[index].copy(); known = distance <= self.support_radius + 1e-10
    if len(self._xy) > 1:
        nearest, _ = self._tree.query(points, k=2)
        ties = np.flatnonzero(np.isclose(nearest[:, 0], nearest[:, 1], atol=1e-8, rtol=0) & known)
        if len(ties):
            peers = self._tree.query_ball_point(points[ties], distance[ties] + 1e-8, return_sorted=False)
            sizes = np.fromiter((len(ids) for ids in peers), dtype=np.int64, count=len(peers))
            flat = np.concatenate(peers).astype(np.int64)
            offsets = np.r_[0, np.cumsum(sizes[:-1])]
            parent = np.repeat(np.arange(len(peers)), sizes)
            values = self._features[flat]
            features[ties, 0] = np.maximum.reduceat(values[:, 0], offsets)
            features[ties, 4] = np.minimum.reduceat(values[:, 4], offsets)
            norms = np.linalg.norm(values[:, 1:3], axis=-1)
            maxima = np.maximum.reduceat(norms, offsets)
            picks = np.minimum.reduceat(np.where(norms == maxima[parent], np.arange(len(norms)), len(norms)), offsets)
            features[ties, 1:3] = values[picks, 1:3]
    return features, known


@contextmanager
def isolated_lookup(function):
    original = FootprintTerrain.lookup
    try:
        FootprintTerrain.lookup = function
        yield
    finally:
        FootprintTerrain.lookup = original


def pair_queries(spec):
    rng = np.random.default_rng(23)
    xy = np.array([[x, y] for x in np.arange(-1., 1.01, .2) for y in np.arange(-1., 1.01, .2)])
    features = rng.uniform(-.5, .5, (len(xy), 5)); features[:, 4] = rng.uniform(0., 1., len(xy))
    # Equal-magnitude slopes with different directions exercise scalar tie order.
    features[:12, 1:3] = np.tile([[.3, .4], [-.3, .4]], (6, 1))
    query = np.vstack([rng.uniform(-1.5, 1.5, (3000, 2)), xy,
                       xy[:-12] + [.1, .1], xy[:-12] + [.1, 0.], [[2., 2.]]])
    counts = []
    for selection in (np.arange(len(xy)), np.arange(len(xy))[::3], np.array([0]), np.array([], dtype=int)):
        terrain = FootprintTerrain(TerrainSnapshot(xy[selection], features[selection],
                        np.zeros(len(selection)), spec.perception.support_radius), spec, query_version="scalar_ties_v1")
        reference = terrain.lookup(query); other = batch_ties_prototype(terrain, query)
        for expected, actual in zip(reference, other):np.testing.assert_array_equal(expected, actual)
        counts.append(len(query))
    return {"datasets": 4, "query_rows": sum(counts), "all_five_channels_exact": True,
            "known_masks_exact": True, "tie_order_preserved": True, "runtime_enabled": False}


def reproductions(spec):
    terrain = synthetic_local_frame(np.zeros(3), spec, 0.)
    reviewer = FootprintReviewer(spec)
    stopped = rollout_numpy([0., 0., 0.], [[0., 0.]], spec)
    peer = FootprintNeighbor(1, np.array([0., .2]), np.array([[1., 0.], [1., 0.]]), np.array([.1, .1]))
    radius_review = reviewer.trajectory(stopped, [0., 0.], terrain, neighbors=[peer])
    invented = rollout_numpy([0., 0., 0.], [[1., 0.]], spec, gain=.25)
    gain_review = reviewer.trajectory(invented, [1., 0.], terrain)
    return {
        "undersized_neighbor_radius": {"report": asdict(radius_review),
            "input_center_distance": 1., "input_radius": .1,
            "minimum_physical_center_distance": 2 * spec.vehicle.physical_radius,
            "minimum_static_safe_center_distance": 2 * spec.vehicle.effective_radius,
            "expected": "reject_invalid_prediction_or_collision", "reproduced": radius_review.passed},
        "invented_terrain_gain": {"report": asdict(gain_review), "input_control": [1., 0.],
            "input_actual_velocity": invented.velocities[0].tolist(), "declared_end_position": invented.states[-1].tolist(),
            "flat_model_expected_actual_velocity": [1., 0.], "flat_model_expected_end_position": [.2, 0., 0.],
            "expected": "reject_velocity_inconsistent_with_local_terrain_model", "reproduced": gain_review.passed},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment/upstream_scale_s01.yaml")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.repeats <= 5:raise ValueError("Repeats must be in [1,5]")
    torch.set_num_threads(1); cfg = cfg_from_experiment(args.config); spec = cfg.spec
    if cfg.planner_device != "cpu" or cfg.clock_mode != "offline_simulated_latency":
        raise ValueError("Profiling requires explicit CPU/offline configuration")
    run = EvidenceRun(args.run_dir, args.config, settings={"purpose": "S01_issue_diagnosis",
                        "repeats": args.repeats, "torch_threads": 1, "runtime_changes": False})
    run.resolved(cfg)
    paired = pair_queries(spec); findings = reproductions(spec)
    json_write(run.path / "metrics/reproductions.json", findings)
    print({"counterexamples": {key: value["reproduced"] for key, value in findings.items()}}, flush=True)
    reviewer = FootprintReviewer(spec); original = FootprintTerrain.lookup
    cap = spec.vehicle.forward_limit
    results = {}; selected = {}
    for name, lookup in (("reference", original), ("batch_ties_prototype", batch_ties_prototype)):
        durations = []; counters = []
        for repeat in range(args.repeats):
            terrain = synthetic_local_frame(np.zeros(3), spec, 0.)
            terrain._tree = CountedTree(terrain._tree)
            with isolated_lookup(lookup):
                start = time.perf_counter()
                decision = admit_control([cap, 0.], [cap, 0.], np.zeros(3), terrain, spec, latency=cfg.simulated_latency)
                durations.append(time.perf_counter() - start)
            counters.append(terrain._tree.metrics())
            if not decision.verified:raise ValueError("Reference profiling scene rejected")
            selected[name] = decision
            print({"path": name, "repeat": repeat, "wall_s": durations[-1]}, flush=True)
        results[name] = {"wall_s": durations, "median_s": float(np.median(durations)), "tree_calls": counters}
    for attribute in ("states", "controls", "velocities", "times"):
        np.testing.assert_array_equal(getattr(selected['reference'].envelope.trajectory, attribute),
                                     getattr(selected['batch_ties_prototype'].envelope.trajectory, attribute))
    assert selected['reference'].verified == selected['batch_ties_prototype'].verified
    # Profiling adds overhead; its elapsed time is separate from the comparisons.
    profile = cProfile.Profile(); terrain = synthetic_local_frame(np.zeros(3), spec, 0.)
    start = time.perf_counter()
    profile.runcall(admit_control, [cap, 0.], [cap, 0.], np.zeros(3), terrain, spec, latency=cfg.simulated_latency)
    profile_seconds = time.perf_counter() - start
    profile.dump_stats(str(run.path / "logs/reference.cprofile"))
    stats = pstats.Stats(profile)
    rows = [{"file": key[0], "line": key[1], "function": key[2], "primitive_calls": value[0],
             "calls": value[1], "self_s": value[2], "cumulative_s": value[3]}
            for key, value in stats.stats.items()]
    json_write(run.path / "metrics/profile_functions.json", sorted(rows, key=lambda row: row['cumulative_s'], reverse=True))
    if source_snapshot() != run.identity['source_snapshot']:raise ValueError("Source changed during diagnosis")
    summary = {"purpose": "diagnosis_only_no_new_acceptance", "runtime_sources_unchanged": True,
               "input": {"sensor_samples": spec.perception.sample_count, "footprint_points_per_pose": len(reviewer.points),
                         "stop_reserve_steps": len(selected['reference'].envelope.trajectory.controls), "speed": cap},
               "results": results, "prototype_pairing": paired, "full_selected_trajectory_exact": True,
               "prototype_median_speedup": results['reference']['median_s'] / results['batch_ties_prototype']['median_s'],
               "profiled_reference_wall_s": profile_seconds, "reproductions": findings,
               "realtime_pass": False, "runtime_optimization_enabled": False, "formal_training": "paused"}
    json_write(run.path / "metrics/summary.json", summary)
    print({"diagnosis_complete": True, "reference_median_s": results['reference']['median_s'],
           "prototype_median_s": results['batch_ties_prototype']['median_s'], "speedup": summary['prototype_median_speedup']})


if __name__ == "__main__":main()
