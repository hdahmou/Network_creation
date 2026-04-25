from __future__ import annotations

from collections.abc import Sequence

import igraph as ig


def evaluate_best_response_terms(
    graph: ig.Graph,
    source: int,
    delta: float,
    add_targets: Sequence[int],
    delete_targets: Sequence[int],
) -> tuple[float, list[float], list[float], list[float], list[int]]:
    """Return score deltas for source-local delete, add, and swap candidates.

    The score is `sum(delta ** distance(source, v))` over reachable vertices
    other than `source`. Distances use direct edges touching `source`; all other
    non-loop edges are treated as undirected shadow-graph edges.

    Returns `(base_score, delete_values, add_only_values, best_swap_values,
    best_swap_delete)`, where each list matches the order of the corresponding
    target sequence. `best_swap_delete[i]` is the delete target used for
    `add_targets[i]`, or `-1` when no delete is used.

    Raises `ValueError` for invalid vertices, invalid `delta`, invalid `source`,
    invalid graph layout, or a delete target without an existing
    `source -> target` arc.
    """
    ...
