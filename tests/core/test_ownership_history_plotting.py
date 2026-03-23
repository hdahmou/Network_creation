"""Tests for ownership-history plotting helpers."""

from __future__ import annotations

import unittest

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from network_creation.plotting.ownership_history import (
    build_circle_layout,
    plot_ownership_history,
)


class OwnershipHistoryPlottingTest(unittest.TestCase):
    def test_plot_ownership_history_returns_figure_and_axes_grid(self) -> None:
        steps = np.array([0, 1], dtype=np.int_)
        ownership_matrices = np.array(
            [
                [[0, 0], [0, 0]],
                [[0, 1], [0, 0]],
            ],
            dtype=np.int8,
        )

        figure, axes = plot_ownership_history(
            steps=steps,
            ownership_matrices=ownership_matrices,
            layout=build_circle_layout(n_nodes=2),
            figure_title="Ownership history",
            max_cols=4,
        )

        self.assertEqual(axes.shape, (1, 2))
        self.assertEqual(axes[0, 0].get_title(), "step 0")
        self.assertEqual(axes[0, 1].get_title(), "step 1")
        assert figure._suptitle is not None
        self.assertEqual(figure._suptitle.get_text(), "Ownership history")
        plt.close(figure)

    def test_plot_ownership_history_rejects_mismatched_lengths(self) -> None:
        with self.assertRaises(ValueError):
            plot_ownership_history(
                steps=np.array([0], dtype=np.int_),
                ownership_matrices=np.zeros((2, 2, 2), dtype=np.int8),
                layout=build_circle_layout(n_nodes=2),
                figure_title="Ownership history",
                max_cols=4,
            )


if __name__ == "__main__":
    unittest.main()
