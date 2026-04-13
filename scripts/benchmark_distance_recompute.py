"""Benchmark shortest-path recomputation under temporary edge updates.

This script builds a single Erdos-Renyi graph, then compares three workflows:

1. Convert the graph to a SciPy sparse adjacency matrix and recompute the
   distance vector from a source vertex after temporary edge additions and
   deletions.
2. Perform the same temporary updates directly on an ``igraph.Graph`` and
   recompute distances after each change.
3. Convert the graph to a ``networkit.Graph`` and repeat the same temporary
   update and recomputation sequence there.

Run with, for example:

    uv run python scripts/benchmark_distance_recompute.py
"""

from __future__ import annotations

import argparse
import random
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import igraph as ig
import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from scipy.sparse.csgraph import shortest_path


type Edge = tuple[int, int]
type TimedFn = Callable[[], float]


@dataclass(frozen=True)
class WorkflowResult:
    baseline: NDArray[np.float64]
    additions: tuple[NDArray[np.float64], ...]
    deletions: tuple[NDArray[np.float64], ...]
    checksum: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", type=int, default=50)
    parser.add_argument("--probability", type=float, default=0.1)
    parser.add_argument("--source", type=int, default=0)
    parser.add_argument("--additions", type=int, default=20)
    parser.add_argument("--deletions", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def build_graph(nodes: int, probability: float, seed: int) -> ig.Graph:
    random.seed(seed)
    return ig.Graph.Erdos_Renyi(
        n=nodes,
        p=probability,
        directed=False,
        loops=False,
    )


def canonical_edge(edge: Edge) -> Edge:
    source, target = edge
    if source < target:
        return source, target
    return target, source


def sample_edges(
    graph: ig.Graph,
    additions: int,
    deletions: int,
    seed: int,
) -> tuple[tuple[Edge, ...], tuple[Edge, ...]]:
    nodes = graph.vcount()
    existing_edges = {
        canonical_edge((int(source), int(target)))
        for source, target in graph.get_edgelist()
    }
    all_pairs = [
        (source, target)
        for source in range(nodes)
        for target in range(source + 1, nodes)
    ]
    missing_edges = [edge for edge in all_pairs if edge not in existing_edges]
    sorted_existing = sorted(existing_edges)

    if len(missing_edges) < additions:
        raise ValueError(
            f"graph only has {len(missing_edges)} missing edges, need {additions}"
        )
    if len(sorted_existing) < deletions:
        raise ValueError(
            f"graph only has {len(sorted_existing)} edges, need {deletions}"
        )

    rng = np.random.default_rng(seed)
    addition_indices = rng.choice(len(missing_edges), size=additions, replace=False)
    deletion_indices = rng.choice(len(sorted_existing), size=deletions, replace=False)

    return (
        tuple(missing_edges[int(index)] for index in addition_indices),
        tuple(sorted_existing[int(index)] for index in deletion_indices),
    )


def distances_from_graph(graph: ig.Graph, source_idx: int) -> NDArray[np.float64]:
    return np.asarray(
        graph.distances(source=[source_idx], mode="all")[0],
        dtype=np.float64,
    )


def require_networkit() -> Any:
    try:
        import networkit as nk
    except ImportError as exc:
        raise ImportError(
            "networkit is required for this benchmark. "
            "Run it via `uv run python scripts/benchmark_distance_recompute.py`."
        ) from exc
    return nk


def to_networkit_graph(graph: ig.Graph, nk: Any) -> Any:
    networkit_graph = nk.Graph(graph.vcount(), False, graph.is_directed())
    for source, target in graph.get_edgelist():
        networkit_graph.addEdge(int(source), int(target))
    return networkit_graph


def distances_from_networkit(
    graph: Any, source_idx: int, nk: Any
) -> NDArray[np.float64]:
    distances = np.asarray(
        nk.distance.SPSP(graph, [source_idx]).run().getDistances()[0],
        dtype=np.float64,
    )
    distances[distances > graph.numberOfNodes() - 1] = np.inf
    return distances


def to_sparse_adjacency(graph: ig.Graph) -> sparse.lil_matrix:
    nodes = graph.vcount()
    edges = graph.get_edgelist()

    rows: list[int] = []
    cols: list[int] = []
    for source, target in edges:
        rows.extend((int(source), int(target)))
        cols.extend((int(target), int(source)))

    data = np.ones(len(rows), dtype=np.int8)
    adjacency = sparse.csr_matrix((data, (rows, cols)), shape=(nodes, nodes))
    return adjacency.tolil()


def distances_from_matrix(
    adjacency: sparse.spmatrix, source_idx: int
) -> NDArray[np.float64]:
    return np.asarray(
        shortest_path(
            csgraph=adjacency,
            directed=False,
            unweighted=True,
            indices=source_idx,
        ),
        dtype=np.float64,
    )


def checksum_distance_vector(distances: NDArray[np.float64]) -> float:
    finite_distances = np.where(np.isfinite(distances), distances, distances.size)
    return float(np.sum(finite_distances))


def run_matrix_workflow(
    graph: ig.Graph,
    source_idx: int,
    addition_edges: tuple[Edge, ...],
    deletion_edges: tuple[Edge, ...],
) -> WorkflowResult:
    adjacency = to_sparse_adjacency(graph)
    baseline = distances_from_matrix(adjacency, source_idx)

    checksum = checksum_distance_vector(baseline)
    addition_results: list[NDArray[np.float64]] = []
    deletion_results: list[NDArray[np.float64]] = []

    for source, target in addition_edges:
        adjacency[source, target] = 1
        adjacency[target, source] = 1
        distances = distances_from_matrix(adjacency, source_idx)
        addition_results.append(distances)
        checksum += checksum_distance_vector(distances)
        adjacency[source, target] = 0
        adjacency[target, source] = 0

    for source, target in deletion_edges:
        adjacency[source, target] = 0
        adjacency[target, source] = 0
        distances = distances_from_matrix(adjacency, source_idx)
        deletion_results.append(distances)
        checksum += checksum_distance_vector(distances)
        adjacency[source, target] = 1
        adjacency[target, source] = 1

    return WorkflowResult(
        baseline=baseline,
        additions=tuple(addition_results),
        deletions=tuple(deletion_results),
        checksum=checksum,
    )


def run_igraph_workflow(
    graph: ig.Graph,
    source_idx: int,
    addition_edges: tuple[Edge, ...],
    deletion_edges: tuple[Edge, ...],
) -> WorkflowResult:
    mutable_graph = graph.copy()
    baseline = distances_from_graph(mutable_graph, source_idx)

    checksum = checksum_distance_vector(baseline)
    addition_results: list[NDArray[np.float64]] = []
    deletion_results: list[NDArray[np.float64]] = []

    for source, target in addition_edges:
        mutable_graph.add_edge(source, target)
        distances = distances_from_graph(mutable_graph, source_idx)
        addition_results.append(distances)
        checksum += checksum_distance_vector(distances)
        mutable_graph.delete_edges([(source, target)])

    for source, target in deletion_edges:
        mutable_graph.delete_edges([(source, target)])
        distances = distances_from_graph(mutable_graph, source_idx)
        deletion_results.append(distances)
        checksum += checksum_distance_vector(distances)
        mutable_graph.add_edge(source, target)

    return WorkflowResult(
        baseline=baseline,
        additions=tuple(addition_results),
        deletions=tuple(deletion_results),
        checksum=checksum,
    )


def run_networkit_workflow(
    graph: ig.Graph,
    source_idx: int,
    addition_edges: tuple[Edge, ...],
    deletion_edges: tuple[Edge, ...],
) -> WorkflowResult:
    nk = require_networkit()
    mutable_graph = to_networkit_graph(graph, nk)
    baseline = distances_from_networkit(mutable_graph, source_idx, nk)

    checksum = checksum_distance_vector(baseline)
    addition_results: list[NDArray[np.float64]] = []
    deletion_results: list[NDArray[np.float64]] = []

    for source, target in addition_edges:
        mutable_graph.addEdge(source, target)
        distances = distances_from_networkit(mutable_graph, source_idx, nk)
        addition_results.append(distances)
        checksum += checksum_distance_vector(distances)
        mutable_graph.removeEdge(source, target)

    for source, target in deletion_edges:
        mutable_graph.removeEdge(source, target)
        distances = distances_from_networkit(mutable_graph, source_idx, nk)
        deletion_results.append(distances)
        checksum += checksum_distance_vector(distances)
        mutable_graph.addEdge(source, target)

    return WorkflowResult(
        baseline=baseline,
        additions=tuple(addition_results),
        deletions=tuple(deletion_results),
        checksum=checksum,
    )


def assert_results_match(
    reference_result: WorkflowResult,
    candidate_result: WorkflowResult,
    candidate_name: str,
) -> None:
    if not np.array_equal(reference_result.baseline, candidate_result.baseline):
        raise AssertionError(f"{candidate_name}: baseline distance vectors differ")

    for index, (matrix_distances, graph_distances) in enumerate(
        zip(reference_result.additions, candidate_result.additions, strict=True)
    ):
        if not np.array_equal(matrix_distances, graph_distances):
            raise AssertionError(
                f"{candidate_name}: addition #{index} distance vectors differ"
            )

    for index, (matrix_distances, graph_distances) in enumerate(
        zip(reference_result.deletions, candidate_result.deletions, strict=True)
    ):
        if not np.array_equal(matrix_distances, graph_distances):
            raise AssertionError(
                f"{candidate_name}: deletion #{index} distance vectors differ"
            )


def time_function(
    name: str, fn: TimedFn, repeat: int
) -> tuple[str, list[float], float]:
    reference_checksum = fn()
    timings: list[float] = []

    for _ in range(repeat):
        start = time.perf_counter()
        checksum = fn()
        timings.append(time.perf_counter() - start)
        if checksum != reference_checksum:
            raise AssertionError(
                f"{name} produced inconsistent checksums: "
                f"{checksum} != {reference_checksum}"
            )

    return name, timings, reference_checksum


def main() -> None:
    args = parse_args()
    graph = build_graph(
        nodes=args.nodes,
        probability=args.probability,
        seed=args.seed,
    )
    addition_edges, deletion_edges = sample_edges(
        graph,
        additions=args.additions,
        deletions=args.deletions,
        seed=args.seed,
    )

    matrix_result = run_matrix_workflow(
        graph,
        source_idx=args.source,
        addition_edges=addition_edges,
        deletion_edges=deletion_edges,
    )
    graph_result = run_igraph_workflow(
        graph,
        source_idx=args.source,
        addition_edges=addition_edges,
        deletion_edges=deletion_edges,
    )
    networkit_result = run_networkit_workflow(
        graph,
        source_idx=args.source,
        addition_edges=addition_edges,
        deletion_edges=deletion_edges,
    )
    assert_results_match(matrix_result, graph_result, "igraph")
    assert_results_match(matrix_result, networkit_result, "networkit")

    results = [
        time_function(
            "scipy_matrix",
            lambda: run_matrix_workflow(
                graph,
                source_idx=args.source,
                addition_edges=addition_edges,
                deletion_edges=deletion_edges,
            ).checksum,
            args.repeat,
        ),
        time_function(
            "igraph",
            lambda: run_igraph_workflow(
                graph,
                source_idx=args.source,
                addition_edges=addition_edges,
                deletion_edges=deletion_edges,
            ).checksum,
            args.repeat,
        ),
        time_function(
            "networkit",
            lambda: run_networkit_workflow(
                graph,
                source_idx=args.source,
                addition_edges=addition_edges,
                deletion_edges=deletion_edges,
            ).checksum,
            args.repeat,
        ),
    ]

    density = 0.0
    if args.nodes > 1:
        density = 2.0 * graph.ecount() / (args.nodes * (args.nodes - 1))

    print(
        "graph=erdos_renyi nodes={nodes} probability={probability} "
        "edges={edges} density={density:.4f}".format(
            nodes=args.nodes,
            probability=args.probability,
            edges=graph.ecount(),
            density=density,
        )
    )
    print(
        "source={source} additions={additions} deletions={deletions} "
        "repeat={repeat} seed={seed}".format(
            source=args.source,
            additions=args.additions,
            deletions=args.deletions,
            repeat=args.repeat,
            seed=args.seed,
        )
    )
    print(
        "verified: scipy_matrix, igraph, and networkit produced identical "
        "distance vectors"
    )
    print(f"{'function':<16} {'best':>12} {'mean':>12} {'stdev':>12}")
    for name, timings, _ in results:
        best = min(timings)
        mean = statistics.mean(timings)
        stdev = statistics.stdev(timings) if len(timings) > 1 else 0.0
        print(f"{name:<16} {best:>12.6f} {mean:>12.6f} {stdev:>12.6f}")


if __name__ == "__main__":
    main()
