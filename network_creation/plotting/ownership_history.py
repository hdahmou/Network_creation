"""Plot helpers for ownership-matrix history as directed graphs."""

from __future__ import annotations

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray

_NODE_COLOR = "#f4d35e"
_EDGE_COLOR = "#1f4e79"
_AXIS_WIDTH = 4
_AXIS_HEIGHT = 4


def build_circle_layout(n_nodes: int) -> dict[int, tuple[float, float]]:
    """Place nodes uniformly on the unit circle."""

    angles = np.linspace(0, 2 * np.pi, n_nodes, endpoint=False)
    return {
        node_id: (float(np.cos(angle)), float(np.sin(angle)))
        for node_id, angle in enumerate(angles)
    }


def plot_ownership_history(
    steps: NDArray[np.int_],
    ownership_matrices: NDArray[np.int8],
    layout: dict[int, tuple[float, float]],
    figure_title: str,
    max_cols: int,
) -> tuple[Figure, NDArray[Axes]]:
    """Plot one ownership graph per sampled step."""

    if len(steps) != len(ownership_matrices):
        raise ValueError("steps and ownership_matrices must have the same length")

    n_rows, n_cols = _compute_subplot_shape(
        n_plots=len(steps),
        max_cols=max_cols,
    )
    figure, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(_AXIS_WIDTH * n_cols, _AXIS_HEIGHT * n_rows),
    )
    axes_grid = np.atleast_1d(axes).reshape(n_rows, n_cols)

    for axis in axes_grid.ravel():
        axis.axis("off")

    for axis, step, ownership_matrix in zip(
        axes_grid.ravel(),
        steps,
        ownership_matrices,
        strict=False,
    ):
        nx.draw_networkx(
            _build_ownership_graph(ownership_matrix),
            pos=layout,
            ax=axis,
            node_color=_NODE_COLOR,
            edge_color=_EDGE_COLOR,
            arrows=True,
            arrowsize=16,
            node_size=700,
            width=2,
            connectionstyle="arc3,rad=0.08",
            font_size=10,
        )
        axis.set_title(f"step {int(step)}")
        axis.set_axis_off()

    figure.suptitle(figure_title, fontsize=14)
    figure.tight_layout()
    return figure, axes_grid


def _compute_subplot_shape(n_plots: int, max_cols: int) -> tuple[int, int]:
    n_cols = min(max_cols, max(1, int(np.ceil(np.sqrt(n_plots)))))
    n_rows = int(np.ceil(n_plots / n_cols))
    return n_rows, n_cols


def _build_ownership_graph(ownership_matrix: NDArray[np.int8]) -> nx.DiGraph:
    graph = nx.DiGraph()
    n_nodes = ownership_matrix.shape[0]
    graph.add_nodes_from(range(n_nodes))
    source_ids, target_ids = np.nonzero(ownership_matrix)
    for source_id, target_id in zip(source_ids, target_ids, strict=False):
        graph.add_edge(int(source_id), int(target_id))
    return graph
