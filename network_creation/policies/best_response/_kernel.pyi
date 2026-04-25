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
    """Return absolute scores for source-local delete, add, and swap candidates.

    Let `s` be `source`. The kernel builds an undirected shadow graph `H` from
    every non-loop edge that does not touch `s`; directed edge orientation is
    ignored inside `H`. Let

        T = {t != s : edge s -> t or t -> s exists in graph}

    be the current source-neighbor set, using source-touching edges only for the
    first hop. For any candidate source-neighbor set `U`, define

        h_U(x) = min_{t in U} d_H(t, x)
        Phi(U) = sum_{x != s, h_U(x) < inf} delta ** (1 + h_U(x))

    where `d_H` is the shortest-path distance in `H`. The returned
    `base_score` is `Phi(T)`.

    For each delete target `v`, `delete_scores[j]` is the score after deleting
    one outgoing arc `s -> v`:

        Phi(T without v)

    when that deleted arc was the last source-touching edge incident to `v`, and
    `Phi(T)` otherwise. Duplicate source arcs and incoming `v -> s` arcs therefore
    preserve `v` as a first-hop source neighbor.

    For each add target `u`, `add_only_scores[i]` is

        Phi(T union {u})

    when `u` is not already source-adjacent, and `Phi(T)` otherwise.

    `best_swap_scores[i]` is the best score from simultaneously adding
    `s -> add_targets[i]` and deleting one target from `delete_targets`:

        max_v Phi((T after deleting s -> v) union {add_targets[i]})

    `best_swap_delete[i]` is the maximizing delete target, or `-1` when no
    delete is used. In that no-delete case, `best_swap_scores[i]` is `Phi(T)`.

    Callers that need score changes should subtract `base_score` from the
    candidate score.

    Returns `(base_score, delete_scores, add_only_scores, best_swap_scores,
    best_swap_delete)`, where each list matches the order of the corresponding
    target sequence. `best_swap_delete[i]` is the delete target used for
    `add_targets[i]`, or `-1` when no delete is used.

    Raises `ValueError` for invalid vertices, invalid `delta`, invalid `source`,
    invalid graph layout, or a delete target without an existing
    `source -> target` arc.
    """
    ...
