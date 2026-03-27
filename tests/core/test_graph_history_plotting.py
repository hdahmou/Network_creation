"""Tests for graph-history plotting helpers."""

from __future__ import annotations

import unittest

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from network_creation.core.graph import (
    create_empty_graph,
)
from network_creation.plotting.graph_history import (
    build_circle_layout,
    plot_graph_history,
)
from tests.graph_factories import graph_from_adjacency_matrix


class GraphHistoryPlottingTest(unittest.TestCase):
    def test_plot_graph_history_returns_figure_and_axes_grid(self) -> None:
        steps = np.array([0, 1], dtype=np.int_)
        graphs = (
            create_empty_graph(2),
            graph_from_adjacency_matrix(
                np.array(
                    [
                        [0, 1],
                        [0, 0],
                    ],
                    dtype=np.int8,
                )
            ),
        )

        figure, axes = plot_graph_history(
            steps=steps,
            graphs=graphs,
            layout=build_circle_layout(n_nodes=2),
            figure_title="Graph history",
            max_cols=4,
        )

        self.assertEqual(axes.shape, (1, 2))
        self.assertEqual(axes[0, 0].get_title(), "step 0")
        self.assertEqual(axes[0, 1].get_title(), "step 1")
        assert figure._suptitle is not None
        self.assertEqual(figure._suptitle.get_text(), "Graph history")
        plt.close(figure)

    def test_plot_graph_history_accepts_numpy_array_of_graphs(self) -> None:
        steps = np.array([0, 1], dtype=np.int_)
        graphs = np.array(
            [
                create_empty_graph(2),
                graph_from_adjacency_matrix(
                    np.array(
                        [
                            [0, 1],
                            [0, 0],
                        ],
                        dtype=np.int8,
                    )
                ),
            ],
            dtype=object,
        )

        figure, axes = plot_graph_history(
            steps=steps,
            graphs=graphs,
            layout=build_circle_layout(n_nodes=2),
            figure_title="Graph history",
            max_cols=4,
        )

        self.assertEqual(axes.shape, (1, 2))
        self.assertEqual(axes[0, 0].get_title(), "step 0")
        self.assertEqual(axes[0, 1].get_title(), "step 1")
        plt.close(figure)

    def test_plot_graph_history_rejects_mismatched_lengths(self) -> None:
        with self.assertRaises(ValueError):
            plot_graph_history(
                steps=np.array([0], dtype=np.int_),
                graphs=(create_empty_graph(2), create_empty_graph(2)),
                layout=build_circle_layout(n_nodes=2),
                figure_title="Graph history",
                max_cols=4,
            )


if __name__ == "__main__":
    unittest.main()
