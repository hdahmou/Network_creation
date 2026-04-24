from __future__ import annotations

from collections.abc import Sequence


def evaluate_best_response_terms_from_edges(
    n: int,
    directed_edges: Sequence[tuple[int, int]],
    source: int,
    delta: float,
    add_targets: Sequence[int],
    delete_targets: Sequence[int],
) -> tuple[float, list[float], list[float], list[float], list[int]]: ...
