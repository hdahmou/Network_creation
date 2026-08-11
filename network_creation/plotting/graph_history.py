"""Plot helpers for directed graph history."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any

import igraph as ig
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
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


def plot_graph_history(
    steps: NDArray[np.int_],
    graphs: Iterable[ig.Graph],
    layout: dict[int, tuple[float, float]] | Callable[[nx.Graph], Any],
    figure_title: str | None,
    max_cols: int,
    *,
    draw_kwargs: Mapping[str, Any] | None = None,
) -> tuple[Figure, np.ndarray[Any, Any]]:
    """Plot one directed graph per sampled step."""

    graph_history = list(graphs)

    if len(steps) != len(graph_history):
        raise ValueError("steps and graphs must have the same length")

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

    draw_options: dict[str, Any] = {
        "node_color": _NODE_COLOR,
        "edge_color": _EDGE_COLOR,
        "arrows": True,
        "arrowsize": 16,
        "node_size": 700,
        "node_shape": "o",
        "width": 2,
        "connectionstyle": "arc3,rad=0.08",
        "font_size": 10,
        "with_labels": True,
    }
    if draw_kwargs is not None:
        draw_options.update(draw_kwargs)

    for axis, step, graph in zip(axes_grid.ravel(), steps, graph_history, strict=False):
        graph_nx = graph.to_networkx(vertex_attr_hashable="name")
        if isinstance(layout, dict):
            layout_nx = layout
        else:
            layout_nx = layout(graph_nx)
        nx.draw_networkx(
            graph_nx,
            pos=layout_nx,
            ax=axis,
            **draw_options,
        )
        axis.set_aspect("equal", adjustable="box")
        axis.set_title(f"step {int(step)}")
        axis.set_axis_off()

    if figure_title is not None:
        figure.suptitle(figure_title, fontsize=14)
    figure.tight_layout()
    return figure, axes_grid


def _compute_subplot_shape(n_plots: int, max_cols: int) -> tuple[int, int]:
    n_cols = min(max_cols, max(1, n_plots))
    n_rows = int(np.ceil(n_plots / n_cols))
    return n_rows, n_cols
