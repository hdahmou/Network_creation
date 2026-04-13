"""Benchmark ADD benefit computations used by BestResponsePolicy.

Run with, for example:

    uv run python scripts/benchmark_add_benefit.py
"""

from __future__ import annotations

import argparse
import statistics
import time
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray


BenefitFn = Callable[
    [NDArray[np.float64], NDArray[np.float64], float], float
]


def current_two_line(
    baseline_dist: NDArray[np.float64],
    target_dists: NDArray[np.float64],
    distance_discount: float,
) -> float:
    total = 0.0
    for target_dist in target_dists:
        new_dist_add = np.minimum(baseline_dist, 1.0 + target_dist)
        new_benefit_add = np.sum(distance_discount**new_dist_add)
        total += float(new_benefit_add)
    return total


def scratch_min_power(
    baseline_dist: NDArray[np.float64],
    target_dists: NDArray[np.float64],
    distance_discount: float,
) -> float:
    total = 0.0
    work = np.empty_like(baseline_dist)
    for target_dist in target_dists:
        np.add(target_dist, 1.0, out=work)
        np.minimum(baseline_dist, work, out=work)
        np.power(distance_discount, work, out=work)
        total += float(np.sum(work))
    return total


def discount_max_identity(
    baseline_dist: NDArray[np.float64],
    target_dists: NDArray[np.float64],
    distance_discount: float,
) -> float:
    total = 0.0
    work = np.empty_like(baseline_dist)
    baseline_benefits = np.power(distance_discount, baseline_dist)

    for target_dist in target_dists:
        np.power(distance_discount, target_dist, out=work)
        work *= distance_discount
        np.maximum(baseline_benefits, work, out=work)
        total += float(np.sum(work))
    return total


def make_distances(
    nodes: int,
    candidates: int,
    unreachable_fraction: float,
    seed: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    rng = np.random.default_rng(seed)
    baseline_dist = rng.integers(0, 20, size=nodes).astype(np.float64)
    target_dists = rng.integers(0, 20, size=(candidates, nodes)).astype(np.float64)

    baseline_dist[0] = 0.0
    target_dists[:, 0] = rng.integers(0, 20, size=candidates).astype(np.float64)

    if unreachable_fraction > 0.0:
        baseline_unreachable = rng.random(nodes) < unreachable_fraction
        target_unreachable = rng.random((candidates, nodes)) < unreachable_fraction
        baseline_dist[baseline_unreachable] = np.inf
        target_dists[target_unreachable] = np.inf
        baseline_dist[0] = 0.0

    return baseline_dist, target_dists


def time_function(
    name: str,
    fn: BenefitFn,
    baseline_dist: NDArray[np.float64],
    target_dists: NDArray[np.float64],
    distance_discount: float,
    repeat: int,
) -> tuple[str, list[float], float]:
    result = fn(baseline_dist, target_dists, distance_discount)
    timings = []
    for _ in range(repeat):
        start = time.perf_counter()
        checked_result = fn(baseline_dist, target_dists, distance_discount)
        timings.append(time.perf_counter() - start)
        if not np.isclose(result, checked_result, rtol=1e-12, atol=1e-9):
            raise AssertionError(
                f"{name} produced unstable results: {result} != {checked_result}"
            )
    return name, timings, result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", type=int, default=50)
    parser.add_argument("--candidates", type=int, default=50)
    parser.add_argument("--repeat", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--distance-discount", type=float, default=0.9)
    parser.add_argument("--unreachable-fraction", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline_dist, target_dists = make_distances(
        nodes=args.nodes,
        candidates=args.candidates,
        unreachable_fraction=args.unreachable_fraction,
        seed=args.seed,
    )

    functions: tuple[tuple[str, BenefitFn], ...] = (
        ("current_two_line", current_two_line),
        ("scratch_min_power", scratch_min_power),
        ("discount_max_identity", discount_max_identity),
    )

    results = [
        time_function(
            name,
            fn,
            baseline_dist,
            target_dists,
            args.distance_discount,
            args.repeat,
        )
        for name, fn in functions
    ]

    baseline_result = results[0][2]
    for name, _, result in results[1:]:
        if not np.isclose(result, baseline_result, rtol=1e-12, atol=1e-9):
            raise AssertionError(
                f"{name} != current_two_line: {result} != {baseline_result}"
            )

    print(
        "nodes={nodes:,} candidates={candidates:,} repeat={repeat:,} "
        "distance_discount={discount} unreachable_fraction={unreachable}".format(
            nodes=args.nodes,
            candidates=args.candidates,
            repeat=args.repeat,
            discount=args.distance_discount,
            unreachable=args.unreachable_fraction,
        )
    )
    print(f"{'function':<24} {'best':>12} {'mean':>12} {'stdev':>12}")
    for name, timings, _ in results:
        best = min(timings)
        mean = statistics.mean(timings)
        stdev = statistics.stdev(timings) if len(timings) > 1 else 0.0
        print(f"{name:<24} {best:>12.6f} {mean:>12.6f} {stdev:>12.6f}")


if __name__ == "__main__":
    main()
