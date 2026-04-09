import igraph as ig
import numpy as np
from numpy.typing import NDArray


def _distances_from_index(graph: ig.Graph, vertex_idx: int) -> NDArray[np.float64]:
    return np.asarray(
        graph.distances(source=[vertex_idx], mode="all")[0],
        dtype=np.float64,
    )


def _sample_argmax(values: NDArray[np.float32], rng: np.random.Generator) -> int:
    """Return the index of a maximal value, breaking ties uniformly at random."""
    best_indices = np.flatnonzero(values == values.max())
    return int(best_indices[int(rng.integers(best_indices.size))])
