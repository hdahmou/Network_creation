"""Tests for graph-history plotting helpers."""

from __future__ import annotations

import unittest
from unittest import mock

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
        self.assertEqual(axes[0, 0].get_aspect(), 1.0)
        self.assertEqual(axes[0, 1].get_aspect(), 1.0)
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
            figure_title=None,
            max_cols=4,
        )

        self.assertEqual(axes.shape, (1, 2))
        self.assertEqual(axes[0, 0].get_title(), "step 0")
        self.assertEqual(axes[0, 1].get_title(), "step 1")
        self.assertIsNone(figure._suptitle)
        plt.close(figure)

    @mock.patch("network_creation.plotting.graph_history.nx.draw_networkx")
    def test_plot_graph_history_uses_available_columns_and_draw_options(
        self,
        draw_networkx: mock.Mock,
    ) -> None:
        graphs = tuple(create_empty_graph(3) for _ in range(4))

        figure, axes = plot_graph_history(
            steps=np.arange(4, dtype=np.int_),
            graphs=graphs,
            layout=build_circle_layout(n_nodes=3),
            figure_title="Graph history",
            max_cols=4,
            draw_kwargs={
                "with_labels": False,
                "node_shape": "o",
                "node_size": 45,
            },
        )

        self.assertEqual(axes.shape, (1, 4))
        self.assertEqual(draw_networkx.call_count, 4)
        for call in draw_networkx.call_args_list:
            self.assertFalse(call.kwargs["with_labels"])
            self.assertEqual(call.kwargs["node_shape"], "o")
            self.assertEqual(call.kwargs["node_size"], 45)
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
