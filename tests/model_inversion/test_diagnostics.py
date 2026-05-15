from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
import torch

from network_creation.model_inversion.diagnostics import (
    DiagnosticData,
    load_diagnostic_data,
    plot_expected_coverage,
    run_expected_coverage,
)
from network_creation.model_inversion.recovery import (
    plot_parameter_marginals,
    plot_parameter_pairs,
    plot_posterior_ridgelines,
)


@dataclass(frozen=True)
class FakeArtifact:
    theta_order: list[str]
    categories_by_name: dict[str, list[int]]
    metadata: dict[str, object] | None = None
    artifact_dir: Path | None = None
    posterior: object | None = None

    def is_discrete_theta(self, name: str) -> bool:
        return name in self.categories_by_name

    def category_values(self, name: str) -> NDArray[np.int64]:
        return np.asarray(self.categories_by_name.get(name, []), dtype=np.int64)

    def to_device(self, device: str | torch.device) -> torch.device:
        torch_device = torch.device(device)
        move = getattr(self.posterior, "to", None)
        if callable(move):
            move(torch_device)
        return torch_device


class SbiDiagnosticTests(unittest.TestCase):
    def test_load_diagnostic_data_reads_artifact_holdout(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            artifact_dir = Path(temp_dir)
            np.savez_compressed(
                artifact_dir / "diagnostic_data.npz",
                theta=np.asarray([[0.1], [0.2]], dtype=np.float32),
                x=np.asarray([[1.0], [2.0]], dtype=np.float32),
                source_sample=np.asarray([3, 5], dtype=np.int64),
            )
            artifact = FakeArtifact(
                theta_order=["alpha"],
                categories_by_name={},
                metadata={"diagnostics": {"file": "diagnostic_data.npz"}},
                artifact_dir=artifact_dir,
            )

            data = load_diagnostic_data(artifact)

        np.testing.assert_allclose(data.theta, [[0.1], [0.2]])
        np.testing.assert_allclose(data.x, [[1.0], [2.0]])
        np.testing.assert_array_equal(data.source_sample, [3, 5])

    def test_run_expected_coverage_delegates_to_sbi_with_tensors(self) -> None:
        class DeviceRecordingPosterior:
            def __init__(self) -> None:
                self.device: torch.device | None = None

            def to(self, device):  # noqa: ANN001
                self.device = torch.device(device)

        posterior = DeviceRecordingPosterior()
        artifact = FakeArtifact(
            theta_order=["alpha", "beta"],
            categories_by_name={},
            posterior=posterior,
        )
        data = DiagnosticData(
            theta=np.asarray([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
            x=np.asarray([[1.0], [2.0]], dtype=np.float32),
            source_sample=np.asarray([0, 1], dtype=np.int64),
        )
        ranks = torch.tensor([[0, 3], [2, 1]])
        dap_samples = torch.ones((2, 2))

        with patch("sbi.diagnostics.run_sbc", return_value=(ranks, dap_samples)) as run_sbc:
            result = run_expected_coverage(
                artifact,
                data,
                num_posterior_samples=17,
                num_workers=2,
                show_progress_bar=False,
            )

        call_args = run_sbc.call_args
        torch.testing.assert_close(
            call_args.args[0],
            torch.tensor([[0.1, 0.2], [0.3, 0.4]], dtype=torch.float32),
        )
        torch.testing.assert_close(
            call_args.args[1],
            torch.tensor([[1.0], [2.0]], dtype=torch.float32),
        )
        self.assertIs(call_args.args[2], posterior)
        self.assertEqual(call_args.kwargs["num_posterior_samples"], 17)
        self.assertEqual(call_args.kwargs["num_workers"], 2)
        self.assertFalse(call_args.kwargs["show_progress_bar"])
        self.assertEqual(posterior.device, torch.device("cpu"))
        self.assertIs(result["ranks"], ranks)
        self.assertIs(result["dap_samples"], dap_samples)
        self.assertEqual(result["n_diagnostic_simulations"], 2)

    def test_plot_expected_coverage_delegates_to_sbi_plot(self) -> None:
        artifact = FakeArtifact(theta_order=["alpha"], categories_by_name={})
        coverage = {
            "ranks": torch.tensor([[0], [1]]),
            "num_posterior_samples": 5,
        }

        with patch(
            "sbi.analysis.plot.sbc_rank_plot",
            return_value=("figure", "axis"),
        ) as rank_plot:
            result = plot_expected_coverage(
                artifact,
                coverage,
            )

        self.assertEqual(result, ("figure", "axis"))
        self.assertIs(rank_plot.call_args.args[0], coverage["ranks"])
        self.assertEqual(rank_plot.call_args.args[1], 5)
        self.assertEqual(rank_plot.call_args.kwargs["parameter_labels"], ["alpha"])

class PlotParameterPairsTests(unittest.TestCase):
    def three_parameter_plot(self):
        artifact = FakeArtifact(
            theta_order=["alpha", "n_agents", "beta"],
            categories_by_name={"n_agents": [10, 20]},
        )
        prior = np.asarray(
            [
                [0.10, 10.0, 0.20],
                [0.40, 20.0, 0.30],
                [0.80, 10.0, 0.10],
                [0.90, 20.0, 0.40],
                [0.20, 10.0, 0.35],
            ],
            dtype=np.float32,
        )
        posterior = np.asarray(
            [
                [0.20, 10.0, 0.22],
                [0.25, 10.0, 0.24],
                [0.30, 10.0, 0.28],
                [0.35, 20.0, 0.30],
                [0.28, 10.0, 0.26],
            ],
            dtype=np.float32,
        )
        figure = plot_parameter_pairs(
            artifact,
            prior_samples=prior,
            posterior_samples=posterior,
            true_theta=np.asarray([0.25, 10.0, 0.27], dtype=np.float32),
            seed=7,
        )
        self.addCleanup(plt.close, figure)
        figure.canvas.draw()
        return np.asarray(figure.axes, dtype=object).reshape(2, 2)

    def test_uses_shifted_pair_grid_with_legend_in_top_right(self) -> None:
        axes = self.three_parameter_plot()

        legend = axes[0, 1].get_legend()
        self.assertIsNotNone(legend)
        self.assertFalse(axes[0, 1].axison)
        self.assertEqual(
            [text.get_text() for text in legend.get_texts()],
            ["prior", "posterior", "true"],
        )

        self.assertEqual(axes.size, 4)
        self.assertEqual(axes[0, 0].get_ylabel(), "n_agents")
        self.assertEqual(axes[1, 0].get_ylabel(), "beta")
        self.assertEqual(axes[1, 0].get_xlabel(), "alpha")
        self.assertEqual(axes[1, 1].get_xlabel(), "n_agents")

    def test_lower_triangle_panels_show_pair_points(self) -> None:
        axes = self.three_parameter_plot()

        for axis in (axes[0, 0], axes[1, 0], axes[1, 1]):
            self.assertTrue(axis.axison)
            self.assertEqual(len(axis.collections), 2)
            self.assertEqual(len(axis.lines), 2)

    def test_visible_pair_panels_are_square(self) -> None:
        axes = self.three_parameter_plot()

        for axis in (axes[0, 0], axes[1, 0], axes[1, 1]):
            bounds = axis.get_position().bounds
            self.assertAlmostEqual(bounds[2], bounds[3], places=6)

    def test_categorical_axes_use_only_category_ticks(self) -> None:
        axes = self.three_parameter_plot()

        np.testing.assert_allclose(axes[1, 1].get_xticks(), [0.0, 1.0])
        self.assertEqual(
            [label.get_text() for label in axes[1, 1].get_xticklabels()],
            ["10", "20"],
        )
        np.testing.assert_allclose(axes[0, 0].get_yticks(), [0.0, 1.0])
        self.assertEqual(
            [label.get_text() for label in axes[0, 0].get_yticklabels()],
            ["10", "20"],
        )
        np.testing.assert_allclose(axes[0, 0].get_ylim(), [-0.5, 1.5])
        np.testing.assert_allclose(axes[1, 1].get_xlim(), [-0.5, 1.5])

    def test_two_parameter_plot_uses_single_pair_panel(self) -> None:
        artifact = FakeArtifact(
            theta_order=["alpha", "n_agents"],
            categories_by_name={"n_agents": [10, 20]},
        )
        prior = np.asarray(
            [[0.10, 10.0], [0.40, 20.0], [0.80, 10.0], [0.90, 20.0]],
            dtype=np.float32,
        )
        posterior = np.asarray(
            [[0.20, 10.0], [0.25, 10.0], [0.30, 10.0], [0.35, 20.0]],
            dtype=np.float32,
        )
        figure = plot_parameter_pairs(
            artifact,
            prior_samples=prior,
            posterior_samples=posterior,
            true_theta=np.asarray([0.25, 10.0], dtype=np.float32),
            seed=7,
        )
        self.addCleanup(plt.close, figure)
        figure.canvas.draw()
        axes = np.asarray(figure.axes, dtype=object).reshape(1, 1)

        self.assertTrue(axes[0, 0].axison)
        self.assertEqual(len(axes[0, 0].collections), 2)
        self.assertEqual(len(axes[0, 0].lines), 2)
        self.assertIsNotNone(axes[0, 0].get_legend())


class PlotParameterMarginalsTests(unittest.TestCase):
    def test_marginals_use_simple_titles_labels_and_category_ticks(self) -> None:
        artifact = FakeArtifact(
            theta_order=["alpha", "n_agents", "beta"],
            categories_by_name={"n_agents": [10, 20]},
        )
        prior = np.asarray(
            [
                [0.10, 10.0, 0.20],
                [0.40, 20.0, 0.30],
                [0.80, 10.0, 0.10],
                [0.90, 20.0, 0.40],
                [0.20, 10.0, 0.35],
            ],
            dtype=np.float32,
        )
        posterior = np.asarray(
            [
                [0.20, 10.0, 0.22],
                [0.25, 10.0, 0.24],
                [0.30, 10.0, 0.28],
                [0.35, 20.0, 0.30],
                [0.28, 10.0, 0.26],
            ],
            dtype=np.float32,
        )

        figure = plot_parameter_marginals(
            artifact,
            prior_samples=prior,
            posterior_samples=posterior,
            true_theta=np.asarray([0.25, 10.0, 0.27], dtype=np.float32),
        )
        self.addCleanup(plt.close, figure)
        figure.canvas.draw()
        axes = np.asarray(figure.axes, dtype=object).reshape(1, 3)

        self.assertEqual([axis.get_title() for axis in axes.ravel()], ["alpha", "n_agents", "beta"])
        self.assertEqual(axes[0, 0].get_ylabel(), "density")
        self.assertEqual(axes[0, 1].get_ylabel(), "probability")
        self.assertEqual(axes[0, 2].get_ylabel(), "density")
        self.assertIsNotNone(axes[0, 0].get_legend())
        np.testing.assert_allclose(axes[0, 1].get_xticks(), [0.0, 1.0])
        self.assertEqual(
            [label.get_text() for label in axes[0, 1].get_xticklabels()],
            ["10", "20"],
        )
        np.testing.assert_allclose(axes[0, 1].get_xlim(), [-0.5, 1.5])

    def test_marginals_hide_unused_grid_cells(self) -> None:
        artifact = FakeArtifact(
            theta_order=["alpha", "beta", "gamma", "n_agents"],
            categories_by_name={"n_agents": [10, 20]},
        )
        prior = np.asarray(
            [
                [0.10, 0.20, 0.30, 10.0],
                [0.40, 0.30, 0.20, 20.0],
                [0.80, 0.10, 0.60, 10.0],
                [0.90, 0.40, 0.50, 20.0],
                [0.20, 0.35, 0.70, 10.0],
            ],
            dtype=np.float32,
        )
        posterior = np.asarray(
            [
                [0.20, 0.22, 0.35, 10.0],
                [0.25, 0.24, 0.40, 10.0],
                [0.30, 0.28, 0.45, 10.0],
                [0.35, 0.30, 0.50, 20.0],
                [0.28, 0.26, 0.42, 10.0],
            ],
            dtype=np.float32,
        )

        figure = plot_parameter_marginals(
            artifact,
            prior_samples=prior,
            posterior_samples=posterior,
            true_theta=np.asarray([0.25, 0.27, 0.43, 10.0], dtype=np.float32),
        )
        self.addCleanup(plt.close, figure)
        axes = np.asarray(figure.axes, dtype=object).reshape(2, 3)

        self.assertTrue(axes[1, 0].axison)
        self.assertFalse(axes[1, 1].axison)
        self.assertFalse(axes[1, 2].axison)


class PlotPosteriorRidgelinesTests(unittest.TestCase):
    def test_ridgelines_draw_one_panel_per_parameter(self) -> None:
        artifact = FakeArtifact(
            theta_order=["alpha", "beta"],
            categories_by_name={},
            metadata={"dataset": {"prior_low": [0.0, 0.0], "prior_high": [1.0, 1.0]}},
        )
        prior = np.linspace(0.0, 1.0, 40, dtype=np.float32)
        posterior = np.stack(
            [
                np.column_stack(
                    [
                        np.linspace(0.10, 0.30, 30),
                        np.linspace(0.20, 0.40, 30),
                    ]
                ),
                np.column_stack(
                    [
                        np.linspace(0.55, 0.75, 30),
                        np.linspace(0.60, 0.80, 30),
                    ]
                ),
            ],
            axis=0,
        ).astype(np.float32)
        truth = np.asarray([[0.2, 0.3], [0.65, 0.7]], dtype=np.float32)

        figure = plot_posterior_ridgelines(
            artifact,
            prior_samples=np.column_stack([prior, prior]),
            posterior_samples=posterior,
            true_theta=truth,
        )
        self.addCleanup(plt.close, figure)
        figure.canvas.draw()
        axes = np.asarray(figure.axes, dtype=object).reshape(1, 2)

        self.assertEqual([axis.get_title() for axis in axes.ravel()], ["alpha", "beta"])
        self.assertEqual(axes[0, 0].get_xlabel(), "parameter value")
        self.assertIsNotNone(figure.legends[0])
        self.assertGreater(len(axes[0, 0].collections), 0)

    def test_ridgelines_use_category_ticks_for_discrete_parameters(self) -> None:
        artifact = FakeArtifact(
            theta_order=["n_agents"],
            categories_by_name={"n_agents": [10, 20, 30]},
            metadata={"dataset": {"prior_low": [10.0], "prior_high": [30.0]}},
        )
        posterior = np.asarray(
            [
                [[10.0], [10.0], [20.0]],
                [[20.0], [30.0], [30.0]],
            ],
            dtype=np.float32,
        )

        figure = plot_posterior_ridgelines(
            artifact,
            prior_samples=np.asarray([[10.0], [20.0], [30.0]], dtype=np.float32),
            posterior_samples=posterior,
            true_theta=np.asarray([[10.0], [30.0]], dtype=np.float32),
        )
        self.addCleanup(plt.close, figure)
        axis = figure.axes[0]

        np.testing.assert_allclose(axis.get_xticks(), [10.0, 20.0, 30.0])
        self.assertEqual(axis.get_xticks(minor=True).tolist(), [])
        self.assertEqual(
            [label.get_text() for label in axis.get_xticklabels()],
            ["10", "20", "30"],
        )
        np.testing.assert_allclose(axis.get_xlim(), [7.8, 32.2])
        self.assertTrue(all(patch.get_width() <= 2.2 for patch in axis.patches))
        vertical_x = []
        for collection in axis.collections:
            get_segments = getattr(collection, "get_segments", None)
            if get_segments is None:
                continue
            for segment in get_segments():
                if len(segment) == 2 and np.allclose(segment[:, 0], segment[0, 0]):
                    vertical_x.append(float(segment[0, 0]))
        self.assertIn(10.0, vertical_x)
        self.assertIn(30.0, vertical_x)
        self.assertNotIn(0.0, vertical_x)

    def test_ridgelines_accept_category_index_samples(self) -> None:
        artifact = FakeArtifact(
            theta_order=["n_agents"],
            categories_by_name={"n_agents": [10, 20, 30]},
            metadata={"dataset": {"prior_low": [10.0], "prior_high": [30.0]}},
        )

        figure = plot_posterior_ridgelines(
            artifact,
            prior_samples=np.asarray([[0.0], [1.0], [2.0]], dtype=np.float32),
            posterior_samples=np.asarray(
                [
                    [[0.0], [0.0], [1.0]],
                    [[1.0], [2.0], [2.0]],
                ],
                dtype=np.float32,
            ),
            true_theta=np.asarray([[0.0], [2.0]], dtype=np.float32),
        )
        self.addCleanup(plt.close, figure)
        axis = figure.axes[0]

        patch_centers = {
            round(float(patch.get_x() + patch.get_width() / 2.0), 6)
            for patch in axis.patches
        }
        self.assertTrue({10.0, 20.0, 30.0}.issubset(patch_centers))


if __name__ == "__main__":
    unittest.main()
