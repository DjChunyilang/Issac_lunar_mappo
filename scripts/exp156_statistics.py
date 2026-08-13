#!/usr/bin/env python3
"""Confidence-aware paired statistics for exp156 frozen evaluations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.stats import beta


def clopper_pearson_upper(
    events: int,
    total: int,
    *,
    confidence: float = 0.95,
) -> float:
    """One-sided exact upper bound for a binomial event probability."""

    if total <= 0 or not 0 <= events <= total:
        raise ValueError("Require total > 0 and 0 <= events <= total.")
    if events == total:
        return 1.0
    return float(beta.ppf(confidence, events + 1, total - events))


def clopper_pearson_lower(
    events: int,
    total: int,
    *,
    confidence: float = 0.95,
) -> float:
    """One-sided exact lower bound for a binomial event probability."""

    if total <= 0 or not 0 <= events <= total:
        raise ValueError("Require total > 0 and 0 <= events <= total.")
    if events == 0:
        return 0.0
    return float(beta.ppf(1.0 - confidence, events, total - events + 1))


def interquartile_mean(values: Iterable[float]) -> float:
    data = np.sort(np.asarray(tuple(values), dtype=np.float64))
    if data.size == 0:
        raise ValueError("IQM requires at least one value.")
    lower = int(np.floor(0.25 * data.size))
    upper = int(np.ceil(0.75 * data.size))
    return float(data[lower:upper].mean())


def bootstrap_upper_mean(
    values: Iterable[float],
    *,
    confidence: float = 0.95,
    samples: int = 10_000,
    seed: int = 156,
) -> float:
    data = np.asarray(tuple(values), dtype=np.float64)
    if data.size == 0 or samples <= 0:
        raise ValueError("Bootstrap requires non-empty values and positive samples.")
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, data.size, size=(samples, data.size))
    means = data[indices].mean(axis=1)
    return float(np.quantile(means, confidence, method="higher"))


def paired_bootstrap_difference(
    first: Iterable[float],
    second: Iterable[float],
    *,
    confidence: float = 0.95,
    samples: int = 10_000,
    seed: int = 156,
) -> dict[str, float]:
    left = np.asarray(tuple(first), dtype=np.float64)
    right = np.asarray(tuple(second), dtype=np.float64)
    if left.shape != right.shape or left.size == 0:
        raise ValueError("Paired arrays must be non-empty and have equal shape.")
    delta = left - right
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, delta.size, size=(samples, delta.size))
    estimates = delta[indices].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    return {
        "mean_difference": float(delta.mean()),
        "lower": float(np.quantile(estimates, tail, method="lower")),
        "upper": float(np.quantile(estimates, 1.0 - tail, method="higher")),
    }


def strict_cell_acceptance(
    *,
    success_count: int,
    collision_count: int,
    timeout_count: int,
    dmax_ratios: Iterable[float],
    confidence: float = 0.95,
    bootstrap_samples: int = 10_000,
    seed: int = 156,
) -> dict:
    ratios = tuple(float(value) for value in dmax_ratios)
    total = len(ratios)
    if total == 0:
        raise ValueError("A strict cell cannot be empty.")
    success_lower = clopper_pearson_lower(
        success_count, total, confidence=confidence
    )
    collision_upper = clopper_pearson_upper(
        collision_count, total, confidence=confidence
    )
    timeout_upper = clopper_pearson_upper(
        timeout_count, total, confidence=confidence
    )
    dmax_mean = float(np.mean(ratios))
    dmax_upper = bootstrap_upper_mean(
        ratios,
        confidence=confidence,
        samples=bootstrap_samples,
        seed=seed,
    )
    checks = {
        "success_lower_ge_0_90": success_lower >= 0.90,
        "collision_upper_le_0_02": collision_upper <= 0.02,
        "timeout_upper_lt_0_10": timeout_upper < 0.10,
        "dmax_point_le_0_20": dmax_mean <= 0.20,
        "dmax_upper_le_0_20": dmax_upper <= 0.20,
    }
    return {
        "episodes": total,
        "counts": {
            "success": int(success_count),
            "collision": int(collision_count),
            "timeout": int(timeout_count),
        },
        "point_estimates": {
            "success": success_count / total,
            "collision": collision_count / total,
            "timeout": timeout_count / total,
            "dmax_ratio": dmax_mean,
            "dmax_ratio_iqm": interquartile_mean(ratios),
        },
        "one_sided_95": {
            "success_lower": success_lower,
            "collision_upper": collision_upper,
            "timeout_upper": timeout_upper,
            "dmax_ratio_upper": dmax_upper,
        },
        "checks": checks,
        "passed": all(checks.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="JSON with an episodes list")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    episodes = payload["episodes"]
    report = strict_cell_acceptance(
        success_count=sum(bool(item["success"]) for item in episodes),
        collision_count=sum(bool(item["collision"]) for item in episodes),
        timeout_count=sum(bool(item["timeout"]) for item in episodes),
        dmax_ratios=[float(item["dmax_ratio"]) for item in episodes],
        bootstrap_samples=args.bootstrap_samples,
    )
    text = json.dumps(report, indent=2)
    if args.output is None:
        print(text)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
