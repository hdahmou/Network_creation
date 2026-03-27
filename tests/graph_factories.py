"""Test-only graph construction helpers."""

from __future__ import annotations

import igraph as ig
import numpy as np
from numpy.typing import NDArray


def graph_from_adjacency_matrix(adjacency_matrix: NDArray[np.int8]) -> ig.Graph:
    """Build a directed graph from a square 0/1 adjacency matrix."""

    if adjacency_matrix.ndim != 2 or adjacency_matrix.shape[0] != adjacency_matrix.shape[1]:
        raise ValueError("adjacency_matrix must be square")

    graph = ig.Graph.Adjacency(adjacency_matrix.tolist(), mode="directed")
    graph.vs["name"] = list(range(graph.vcount()))
    return graph
