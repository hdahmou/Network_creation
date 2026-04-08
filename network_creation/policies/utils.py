import igraph as ig
import numpy as np
from numpy.typing import NDArray


def _distances_from_index(graph: ig.Graph, vertex_idx: int) -> NDArray[np.float64]:
    return np.asarray(
        graph.distances(source=[vertex_idx], mode="all")[0],
        dtype=np.float64,
    )


def _sample_argmax(values: list[float], rng: np.random.Generator) -> int:
    """Return the index of a maximal value, breaking ties uniformly at random."""
    max_value = max(values)
    best_indices = [idx for idx, value in enumerate(values) if value == max_value]
    return int(best_indices[int(rng.integers(len(best_indices)))])
