"""Tests for plotting helpers."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PathCollection, QuadMesh
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.ticker import FormatStrFormatter, NullLocator

from network_creation.core.agent import Agent
from network_creation.core.environment import Environment
from network_creation.core.graph import create_empty_graph
from network_creation.core.model import (
    Action,
    ActionType,
    ActorScheduler,
)
from network_creation.core.observability import ObservationSampling
from network_creation.core.simulator import Simulator
from network_creation.data.choi_2020_references import (
    ReferenceRequest,
    reference_key,
    save_reference_cache,
    sigmoid_reference_key,
)
from network_creation.observers.density_observer import DensityObserver
from network_creation.observers.statistics.episode_tensor_series import (
    EpisodeSeriesTensorStatisticObserver,
)
from network_creation.observers.triad_census_observer import TriadCensusObserver
from network_creation.plotting.episode_series import plot_episode_series_observer
from network_creation.plotting.plot_pair import plot_pair
from network_creation.plotting.reference_values import (
    annotate_colorbar_reference_values,
    get_observer_reference_style,
    get_sigmoid_fit_reference_values,
    positive_reference_norm,
)
from network_creation.policies.random import RandomPolicy
from tests.core.simulation_support import ConstantPolicy


class MatrixObserver(EpisodeSeriesTensorStatisticObserver):
    label = "Matrix Statistic"
    x_label = "step"
    y_label = "matrix value"
    y_min = None
    y_max = None
    dimension_names = ("a", "b", "c", "d")

    def compute_initial_value(self, event) -> np.ndarray:
        del event
        return np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)

    def compute_step_value(self, event) -> np.ndarray:
        base = 10.0 * (event.step + 1)
        return np.array(
            [[base, base + 1.0], [base + 2.0, base + 3.0]],
            dtype=np.float32,
        )


class PlotEpisodeSeriesObserverTest(unittest.TestCase):
    def test_plot_raises_when_observer_has_no_episodes(self) -> None:
        observer = DensityObserver(
            n_agents=0,
            observation_sampling=ObservationSampling.periodic(1),
            aggregations=(),
        )
        figure, axis = plt.subplots()

        with self.assertRaises(ValueError):
            plot_episode_series_observer(observer=observer, ax=axis)

        plt.close(figure)

    def test_single_episode_plot_returns_axis(self) -> None:
        observer = DensityObserver(
            n_agents=2,
            observation_sampling=ObservationSampling.periodic(1),
            aggregations=(),
        )
        simulator = Simulator(
            env=Environment(create_empty_graph(2), seed=3),
            agents=(
                Agent(
                    id=0,
                    visibility_distance=10,
                    policy=ConstantPolicy(Action(ActionType.BRIDGE, source=0, target=1)),
                    seed=5,
                ),
                Agent(
                    id=1,
                    visibility_distance=10,
                    policy=ConstantPolicy(Action(ActionType.PASS, source=1)),
                    seed=7,
                ),
            ),
            actor_scheduler=ActorScheduler(n_agents=1, n_actors=1, seed=21),
            max_steps=2,
            rng=np.random.default_rng(11),
            observers=(observer,),
        )

        simulator.run(n_episodes=1, reset_observers=False)

        figure, axis = plt.subplots()
        returned_axis = plot_episode_series_observer(observer=observer, ax=axis)

        self.assertIs(returned_axis, axis)
        self.assertEqual(len(axis.lines), 1)
        self.assertEqual(axis.get_xlabel(), "step")
        self.assertEqual(axis.get_ylabel(), "density")
        self.assertEqual(axis.get_title(), "Density over one episode")
        plt.close(figure)

    def test_multi_episode_plot_draws_mean_and_band_without_individual_runs(
        self,
    ) -> None:
        observer = DensityObserver(
            n_agents=4,
            observation_sampling=ObservationSampling.periodic(1),
            aggregations=("mean", "std", "runs"),
        )
        simulator = Simulator(
            env=Environment(create_empty_graph(4), seed=13),
            agents=tuple(
                Agent(
                    id=node_id,
                    visibility_distance=10,
                    policy=RandomPolicy(seed=17 + node_id),
                    seed=17 + node_id,
                )
                for node_id in range(4)
            ),
            actor_scheduler=ActorScheduler(n_agents=1, n_actors=1, seed=21),
            max_steps=8,
            rng=np.random.default_rng(29),
            observers=(observer,),
        )

        simulator.run(n_episodes=4, reset_observers=False)

        figure, axis = plt.subplots()
        returned_axis = plot_episode_series_observer(observer=observer, ax=axis)

        self.assertIs(returned_axis, axis)
        self.assertEqual(len(axis.lines), 1)
        self.assertEqual(len(axis.collections), 1)
        self.assertEqual(axis.get_title(), "Density across multiple episodes")
        plt.close(figure)

    def test_runs_only_plot_uses_single_episode_title_when_needed(self) -> None:
        observer = DensityObserver(
            n_agents=2,
            observation_sampling=ObservationSampling.periodic(1),
            aggregations=("runs",),
        )
        simulator = Simulator(
            env=Environment(create_empty_graph(2), seed=31),
            agents=(
                Agent(
                    id=0,
                    visibility_distance=10,
                    policy=ConstantPolicy(Action(ActionType.BRIDGE, source=0, target=1)),
                    seed=37,
                ),
                Agent(
                    id=1,
                    visibility_distance=10,
                    policy=ConstantPolicy(Action(ActionType.PASS, source=1)),
                    seed=41,
                ),
            ),
            actor_scheduler=ActorScheduler(n_agents=1, n_actors=1, seed=21),
            max_steps=2,
            rng=np.random.default_rng(43),
            observers=(observer,),
        )

        simulator.run(n_episodes=1, reset_observers=False)

        figure, axis = plt.subplots()
        returned_axis = plot_episode_series_observer(observer=observer, ax=axis)

        self.assertIs(returned_axis, axis)
        self.assertEqual(len(axis.lines), 0)
        self.assertEqual(axis.get_title(), "Density over one episode")
        plt.close(figure)

    def test_multi_episode_plot_can_overlay_individual_runs_when_requested(
        self,
    ) -> None:
        observer = DensityObserver(
            n_agents=4,
            observation_sampling=ObservationSampling.periodic(1),
            aggregations=("mean", "std", "runs"),
        )
        simulator = Simulator(
            env=Environment(create_empty_graph(4), seed=47),
            agents=tuple(
                Agent(
                    id=node_id,
                    visibility_distance=10,
                    policy=RandomPolicy(seed=53 + node_id),
                    seed=53 + node_id,
                )
                for node_id in range(4)
            ),
            actor_scheduler=ActorScheduler(n_agents=1, n_actors=1, seed=21),
            max_steps=8,
            rng=np.random.default_rng(61),
            observers=(observer,),
        )

        simulator.run(n_episodes=4, reset_observers=False)

        figure, axis = plt.subplots()
        returned_axis = plot_episode_series_observer(
            observer=observer,
            ax=axis,
            show_runs=True,
        )

        self.assertIs(returned_axis, axis)
        self.assertEqual(len(axis.lines), 5)
        self.assertEqual(len(axis.collections), 1)
        self.assertEqual(axis.get_title(), "Density across multiple episodes")
        plt.close(figure)

    def test_multi_episode_plot_accepts_custom_label_color_and_title_control(
        self,
    ) -> None:
        observer = DensityObserver(
            n_agents=4,
            observation_sampling=ObservationSampling.periodic(1),
            aggregations=("mean", "std"),
        )
        simulator = Simulator(
            env=Environment(create_empty_graph(4), seed=71),
            agents=tuple(
                Agent(
                    id=node_id,
                    visibility_distance=10,
                    policy=RandomPolicy(seed=73 + node_id),
                    seed=73 + node_id,
                )
                for node_id in range(4)
            ),
            actor_scheduler=ActorScheduler(n_agents=1, n_actors=1, seed=21),
            max_steps=8,
            rng=np.random.default_rng(89),
            observers=(observer,),
        )

        simulator.run(n_episodes=3, reset_observers=False)

        figure, axis = plt.subplots()
        axis.set_title("Keep this title")
        returned_axis = plot_episode_series_observer(
            observer=observer,
            ax=axis,
            color="#123456",
            band_color="#abcdef",
            label="group size 10",
            set_title=False,
        )

        self.assertIs(returned_axis, axis)
        self.assertEqual(axis.get_title(), "Keep this title")
        self.assertEqual(axis.lines[0].get_label(), "group size 10")
        self.assertEqual(axis.lines[0].get_color(), "#123456")
        self.assertEqual(len(axis.collections), 1)
        plt.close(figure)

    def test_scalar_plot_rejects_dimension_argument(self) -> None:
        observer = DensityObserver(
            n_agents=0,
            observation_sampling=ObservationSampling.periodic(1),
            aggregations=(),
        )
        simulator = Simulator(
            env=Environment(create_empty_graph(2), seed=97),
            agents=(),
            actor_scheduler=ActorScheduler(n_agents=1, n_actors=1, seed=21),
            max_steps=0,
            rng=np.random.default_rng(101),
            observers=(observer,),
        )

        simulator.run(n_episodes=1, reset_observers=False)

        figure, axis = plt.subplots()
        with self.assertRaisesRegex(
            ValueError,
            "dimension_name is only supported",
        ):
            plot_episode_series_observer(
                observer=observer,
                ax=axis,
                dimension_name="density",
            )

        plt.close(figure)

    def test_tensor_plot_raises_without_dimension(self) -> None:
        observer = TriadCensusObserver(
            n_agents=0,
            observation_sampling=ObservationSampling.periodic(1),
            aggregations=(),
        )
        simulator = Simulator(
            env=Environment(create_empty_graph(3), seed=97),
            agents=(),
            actor_scheduler=ActorScheduler(n_agents=1, n_actors=1, seed=21),
            max_steps=0,
            rng=np.random.default_rng(101),
            observers=(observer,),
        )

        simulator.run(n_episodes=1, reset_observers=False)

        figure, axis = plt.subplots()
        with self.assertRaisesRegex(
            ValueError,
            "dimension_name must be provided",
        ):
            plot_episode_series_observer(observer=observer, ax=axis)

        plt.close(figure)

    def test_tensor_plot_selects_dimension_by_name(self) -> None:
        observer = MatrixObserver(
            n_agents=1,
            observation_sampling=ObservationSampling.periodic(1),
            aggregations=(),
        )
        simulator = Simulator(
            env=Environment(create_empty_graph(2), seed=103),
            agents=(
                Agent(
                    id=0,
                    visibility_distance=10,
                    policy=ConstantPolicy(Action(ActionType.PASS, source=0)),
                    seed=107,
                ),
            ),
            actor_scheduler=ActorScheduler(n_agents=1, n_actors=1, seed=21),
            max_steps=1,
            rng=np.random.default_rng(109),
            observers=(observer,),
        )

        simulator.run(n_episodes=1, reset_observers=False)

        figure, axis = plt.subplots()
        returned_axis = plot_episode_series_observer(
            observer=observer,
            ax=axis,
            dimension_name="c",
        )

        self.assertIs(returned_axis, axis)
        self.assertEqual(axis.get_title(), "Matrix Statistic over one episode")
        np.testing.assert_allclose(axis.lines[0].get_ydata(), np.array([2.0, 12.0]))
        plt.close(figure)

    def test_tensor_plot_selects_aggregated_dimension_by_name(self) -> None:
        observer = MatrixObserver(
            n_agents=1,
            observation_sampling=ObservationSampling.periodic(1),
            aggregations=("mean", "std", "runs"),
        )
        simulator = Simulator(
            env=Environment(create_empty_graph(2), seed=113),
            agents=(
                Agent(
                    id=0,
                    visibility_distance=10,
                    policy=ConstantPolicy(Action(ActionType.PASS, source=0)),
                    seed=127,
                ),
            ),
            actor_scheduler=ActorScheduler(n_agents=1, n_actors=1, seed=21),
            max_steps=1,
            rng=np.random.default_rng(131),
            observers=(observer,),
        )

        simulator.run(n_episodes=2, reset_observers=False)

        figure, axis = plt.subplots()
        returned_axis = plot_episode_series_observer(
            observer=observer,
            ax=axis,
            dimension_name="c",
            show_runs=True,
        )

        self.assertIs(returned_axis, axis)
        self.assertEqual(axis.get_title(), "Matrix Statistic across multiple episodes")
        self.assertEqual(len(axis.lines), 3)
        self.assertEqual(len(axis.collections), 1)
        np.testing.assert_allclose(axis.lines[-1].get_ydata(), np.array([2.0, 12.0]))
        plt.close(figure)

    def test_tensor_plot_rejects_unknown_dimension_name(self) -> None:
        observer = MatrixObserver(
            n_agents=1,
            observation_sampling=ObservationSampling.periodic(1),
            aggregations=(),
        )
        simulator = Simulator(
            env=Environment(create_empty_graph(2), seed=137),
            agents=(
                Agent(
                    id=0,
                    visibility_distance=10,
                    policy=ConstantPolicy(Action(ActionType.PASS, source=0)),
                    seed=139,
                ),
            ),
            actor_scheduler=ActorScheduler(n_agents=1, n_actors=1, seed=21),
            max_steps=1,
            rng=np.random.default_rng(149),
            observers=(observer,),
        )

        simulator.run(n_episodes=1, reset_observers=False)

        figure, axis = plt.subplots()
        with self.assertRaisesRegex(
            ValueError,
            "unknown dimension name",
        ):
            plot_episode_series_observer(
                observer=observer,
                ax=axis,
                dimension_name="missing",
            )

        plt.close(figure)


class PlotPairTest(unittest.TestCase):
    def setUp(self) -> None:
        self.X = np.array(
            [
                [0.0, 1.0, 2.0],
                [1.0, 2.0, 3.0],
                [2.0, 3.0, 4.0],
                [3.0, 4.0, 5.0],
            ],
            dtype=float,
        )
        self.y = np.array([10.0, 20.0, 30.0, 40.0], dtype=float)

    def pair_layout(self, X, *, atol=0.0):
        n = int(np.count_nonzero(np.var(X, axis=0) > atol) - 1)
        figure, axes = plt.subplots(
            n,
            n,
            figsize=(2.4 * n, 2.4 * n),
            layout="constrained",
            squeeze=False,
        )
        return figure, axes

    def plot_pair_in_layout(self, X, y=None, **kwargs):
        figure, axes = self.pair_layout(X, atol=kwargs.get("atol", 0.0))
        returned_figure, returned_axes = plot_pair(
            X,
            self.y if y is None else y,
            fig=figure,
            axes=axes,
            **kwargs,
        )
        self.assertIs(returned_figure, figure)
        return figure, returned_axes

    def fixed_reference_layout(self):
        figure = plt.figure(figsize=(12.9, 10.0), layout="constrained")
        grid = figure.add_gridspec(
            2,
            4,
            width_ratios=[5.0, 5.0, 0.3, 2.6],
        )
        axes = np.empty((2, 2), dtype=object)
        for i in range(2):
            for j in range(2):
                axes[i, j] = figure.add_subplot(grid[i, j])
        colorbar_axis = figure.add_subplot(grid[:, 2])
        label_axis = figure.add_subplot(grid[:, 3], sharey=colorbar_axis)
        label_axis.axis("off")
        return figure, axes, colorbar_axis, label_axis

    def add_external_colorbar(
        self,
        figure,
        colorbar_axis,
        *,
        cmap="Greys",
        norm=None,
        label="y",
    ):
        norm = norm or Normalize(vmin=float(self.y.min()), vmax=float(self.y.max()))
        colorbar = figure.colorbar(
            plt.cm.ScalarMappable(cmap=cmap, norm=norm),
            cax=colorbar_axis,
        )
        colorbar.set_label(label)
        colorbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        return colorbar

    def test_plot_pair_uses_data_limits_when_no_overrides_are_provided(self) -> None:
        figure, axes = self.plot_pair_in_layout(self.X, method="points")

        self.assertEqual(axes[0, 0].collections[0].get_clim(), (10.0, 40.0))
        self.assertEqual(set(figure.axes), set(axes.ravel()))
        plt.close(figure)

    def test_plot_pair_does_not_add_default_colorbar_for_single_pair_axis(self) -> None:
        figure, axes = self.plot_pair_in_layout(self.X[:, :2], method="points")

        self.assertEqual(axes.shape, (1, 1))
        self.assertEqual(figure.axes, [axes[0, 0]])
        plt.close(figure)

    def test_plot_pair_can_draw_into_subfigure_panels(self) -> None:
        parent_figure = plt.figure(figsize=(9, 3), layout="constrained")
        panels = parent_figure.subfigures(1, 3)
        figure_numbers = set(plt.get_fignums())

        try:
            for panel in panels:
                axes = panel.subplots(2, 2, squeeze=False)
                returned_panel, axes = plot_pair(
                    self.X,
                    self.y,
                    method="points",
                    fig=panel,
                    axes=axes,
                )

                self.assertIs(returned_panel, panel)
                self.assertEqual(axes.shape, (2, 2))
                self.assertGreater(len(axes[0, 0].collections), 0)

                grid_axes = list(axes.ravel())
                extra_axes = [
                    axis
                    for axis in panel.axes
                    if all(axis is not grid_axis for grid_axis in grid_axes)
                ]
                self.assertEqual(extra_axes, [])

            self.assertEqual(set(plt.get_fignums()), figure_numbers)
        finally:
            plt.close(parent_figure)

    def test_plot_pair_accepts_explicit_axes_and_leaves_colorbar_external(self) -> None:
        figure, axes, colorbar_axis, label_axis = self.fixed_reference_layout()

        returned_figure, returned_axes = plot_pair(
            self.X,
            self.y,
            method="points",
            fig=figure,
            axes=axes,
        )

        self.assertIs(returned_figure, figure)
        self.assertIs(returned_axes[0, 0], axes[0, 0])
        self.assertGreater(len(axes[0, 0].collections), 0)
        self.assertEqual(len(colorbar_axis.collections), 0)

        self.add_external_colorbar(figure, colorbar_axis)
        np.testing.assert_allclose(colorbar_axis.get_ybound(), (10.0, 40.0))
        self.assertEqual(
            set(figure.axes),
            set(axes.ravel()) | {colorbar_axis, label_axis},
        )
        plt.close(figure)

    def test_plot_pair_uses_caller_provided_norm_for_points(self) -> None:
        custom_norm = TwoSlopeNorm(vmin=0.0, vcenter=25.0, vmax=50.0)
        figure, axes = self.plot_pair_in_layout(
            self.X,
            method="points",
            cmap="RdBu_r",
            norm=custom_norm,
        )

        collection = axes[0, 0].collections[0]
        self.assertEqual(collection.get_clim(), (0.0, 50.0))
        self.assertIs(collection.norm, custom_norm)
        self.assertEqual(set(figure.axes), set(axes.ravel()))
        plt.close(figure)

    def test_colorbar_reference_annotation_is_external_to_plot_pair(self) -> None:
        figure, axes, colorbar_axis, label_axis = self.fixed_reference_layout()
        plot_pair(
            self.X,
            self.y,
            method="points",
            fig=figure,
            axes=axes,
        )
        self.add_external_colorbar(figure, colorbar_axis)

        annotate_colorbar_reference_values(
            colorbar_axis,
            label_axis,
            {"Low": 12.0, "Experimental": 38.0},
        )

        self.assertEqual(
            [text.get_text() for text in colorbar_axis.texts],
            ["12.00", "Low", "38.00", "Experimental"],
        )
        self.assertEqual(len(label_axis.texts), 0)
        self.assertEqual(len(colorbar_axis.lines), 0)
        self.assertEqual(len(label_axis.lines), 0)
        np.testing.assert_allclose(label_axis.get_ylim(), colorbar_axis.get_ylim())

        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        figure_bounds = figure.bbox
        for text in colorbar_axis.texts:
            self.assertLessEqual(text.get_window_extent(renderer).x1, figure_bounds.x1)
        plt.close(figure)

    def test_reference_annotation_does_not_change_pair_plot_axes_positions(self) -> None:
        def rendered_positions(references):
            figure, axes, colorbar_axis, label_axis = self.fixed_reference_layout()
            plot_pair(
                self.X,
                self.y,
                method="points",
                fig=figure,
                axes=axes,
            )
            self.add_external_colorbar(figure, colorbar_axis)
            annotate_colorbar_reference_values(
                colorbar_axis,
                label_axis,
                references,
            )
            figure.canvas.draw()
            return figure, np.array(
                [axis.get_position().bounds for axis in axes.ravel()],
                dtype=float,
            )

        empty_figure, empty_positions = rendered_positions({})
        reference_figure, reference_positions = rendered_positions(
            {"Low": 12.0, "Middle": 25.0, "High": 38.0}
        )

        try:
            np.testing.assert_allclose(empty_positions, reference_positions, atol=1e-6)
        finally:
            plt.close(empty_figure)
            plt.close(reference_figure)

    def test_reference_style_returns_welfare_references_and_bounded_norm(self) -> None:
        theta = np.array(
            [
                [10.0, 0.5, 0.9],
                [10.0, 0.5, 0.9],
            ],
            dtype=float,
        )

        references, _ = get_observer_reference_style(
            "centered_mean_welfare.mean",
            np.array([0.1, 0.2], dtype=float),
            n_agents=10,
            theta_labels=("n_agents", "benefit_to_cost_ratio", "distance_discount"),
            theta_values=theta,
            cache_path=Path("/tmp/missing-choi-reference-cache.pkl"),
        )

        self.assertEqual(tuple(references), ("empty", "random", "star"))
        self.assertEqual(references["empty"], 0.0)
        self.assertTrue(np.isfinite(references["random"]))
        self.assertTrue(np.isfinite(references["star"]))

        _, norm = get_observer_reference_style(
            "central_point_of_dominance.mean",
            np.array([0.2, 0.8], dtype=float),
            n_agents=10,
            theta_labels=(),
            theta_values=np.empty((1, 0), dtype=float),
            cache_path=Path("/tmp/missing-choi-reference-cache.pkl"),
        )

        self.assertIsNotNone(norm)
        self.assertEqual((norm.vmin, norm.vmax), (0.0, 1.0))

    def test_reference_style_looks_up_welfare_cache_with_choi_parameters(self) -> None:
        request = ReferenceRequest("centered_mean_welfare", 10, 0.5, 0.9)

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "choi-reference.pkl"
            save_reference_cache(
                {"values": {reference_key(request): 0.42}, "missing_group_sizes": []},
                cache_path,
            )

            references, _ = get_observer_reference_style(
                "centered_mean_welfare.mean",
                np.array([0.1, 0.2], dtype=float),
                n_agents=10,
                theta_labels=(
                    "n_agents",
                    "benefit_to_cost_ratio",
                    "distance_discount",
                ),
                theta_values=np.array([[10.0, 0.25, 0.9]], dtype=float),
                cache_path=cache_path,
            )

        self.assertEqual(references["choi2020"], 0.42)

    def test_positive_reference_norm_includes_reference_values(self) -> None:
        norm = positive_reference_norm(
            np.array([0.1, 0.2], dtype=float),
            {"choi2020": 0.8},
        )

        self.assertEqual((norm.vmin, norm.vmax), (0.0, 0.8))

    def test_default_reference_values_preserve_caller_references(self) -> None:
        self.assertEqual(
            DensityObserver.reference_values({}, {"choi2020": 0.8}),
            {"choi2020": 0.8},
        )

    def test_reference_style_filters_unavailable_reference_values(self) -> None:
        references, _ = get_observer_reference_style(
            "density.mean",
            np.array([0.1, 0.2], dtype=float),
            n_agents=10,
            theta_labels=(),
            theta_values=np.empty((1, 0), dtype=float),
            cache_path=Path("/tmp/missing-choi-reference-cache.pkl"),
            extra_references={
                "ok": 0.7,
                "missing": None,
                "nan": float("nan"),
                "infinite": float("inf"),
            },
        )

        self.assertEqual(references, {"ok": 0.7, "empty": 0.0, "star": 0.2})

    def test_sigmoid_reference_lookup_returns_cached_fit_parameters_without_rescaling(
        self,
    ) -> None:
        request = ReferenceRequest("central_point_of_dominance", 10)

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "choi-reference.pkl"
            save_reference_cache(
                {
                    "values": {},
                    "sigmoid_values": {
                        sigmoid_reference_key(request, "A"): 0.75,
                        sigmoid_reference_key(request, "k"): 0.2,
                        sigmoid_reference_key(request, "t0"): 20.0,
                    },
                    "missing_group_sizes": [],
                },
                cache_path,
            )

            references = get_sigmoid_fit_reference_values(
                "central_point_of_dominance.mean",
                n_agents=10,
                cache_path=cache_path,
            )

        self.assertEqual(references["A"]["choi2020"], 0.75)
        self.assertEqual(references["k"]["choi2020"], 0.2)
        self.assertEqual(references["t0"]["choi2020"], 20.0)

    def test_sigmoid_reference_lookup_ignores_non_sigmoid_observers(self) -> None:
        request = ReferenceRequest("density", 10)

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "choi-reference.pkl"
            save_reference_cache(
                {
                    "values": {},
                    "sigmoid_values": {
                        sigmoid_reference_key(request, "A"): 0.75,
                        sigmoid_reference_key(request, "k"): 0.2,
                        sigmoid_reference_key(request, "t0"): 20.0,
                    },
                    "missing_group_sizes": [],
                },
                cache_path,
            )

            references = get_sigmoid_fit_reference_values(
                "density.mean",
                n_agents=10,
                cache_path=cache_path,
            )

        self.assertEqual(references, {})

    def test_plot_pair_uses_caller_provided_norm_for_gaussian_mode(self) -> None:
        custom_norm = TwoSlopeNorm(vmin=0.0, vcenter=25.0, vmax=50.0)
        figure, axes = self.plot_pair_in_layout(
            self.X,
            method="gaussian",
            cmap="RdBu_r",
            norm=custom_norm,
        )

        image = axes[0, 0].images[0]
        self.assertEqual(image.get_clim(), (0.0, 50.0))
        self.assertIs(image.norm, custom_norm)
        self.assertEqual(set(figure.axes), set(axes.ravel()))
        plt.close(figure)

    def test_plot_pair_preserves_points_path_for_ordered_discrete_axis(self) -> None:
        X = np.array(
            [
                [1.0, 0.2, 3.0],
                [1.0, 0.5, 4.0],
                [2.0, 0.8, 5.0],
                [3.0, 1.1, 6.0],
            ],
            dtype=float,
        )
        custom_norm = TwoSlopeNorm(vmin=0.0, vcenter=25.0, vmax=50.0)
        figure, axes = self.plot_pair_in_layout(
            X,
            labels=["stage", "weight", "bias"],
            ordered_discrete_keys=["stage"],
            method="points",
            norm=custom_norm,
        )

        collection = axes[0, 0].collections[0]
        self.assertIsInstance(collection, PathCollection)
        self.assertIs(collection.norm, custom_norm)
        self.assertEqual(len(axes[0, 0].images), 0)
        np.testing.assert_allclose(axes[0, 0].get_xticks(), [1.0, 2.0, 3.0])
        self.assertIsInstance(axes[0, 0].xaxis.get_minor_locator(), NullLocator)
        self.assertEqual(len(axes[0, 0].lines), 2)
        np.testing.assert_allclose(
            [line.get_xdata()[0] for line in axes[0, 0].lines],
            [1.5, 2.5],
        )
        plt.close(figure)

    def test_plot_pair_uses_quadmesh_for_mixed_ordered_discrete_gaussian_mode(
        self,
    ) -> None:
        X = np.array(
            [
                [1.0, 0.2, 3.0],
                [1.0, 0.5, 4.0],
                [2.0, 0.8, 5.0],
                [3.0, 1.1, 6.0],
            ],
            dtype=float,
        )
        custom_norm = TwoSlopeNorm(vmin=0.0, vcenter=25.0, vmax=50.0)
        figure, axes = self.plot_pair_in_layout(
            X,
            labels=["stage", "weight", "bias"],
            ordered_discrete_keys=["stage"],
            method="gaussian",
            norm=custom_norm,
        )

        collection = axes[0, 0].collections[0]
        self.assertIsInstance(collection, QuadMesh)
        self.assertIs(collection.norm, custom_norm)
        np.testing.assert_allclose(axes[0, 0].get_xticks(), [1.0, 2.0, 3.0])
        self.assertIsInstance(axes[0, 0].xaxis.get_minor_locator(), NullLocator)
        self.assertEqual(len(axes[0, 0].lines), 2)
        np.testing.assert_allclose(
            [line.get_xdata()[0] for line in axes[0, 0].lines],
            [1.5, 2.5],
        )
        plt.close(figure)

    def test_plot_pair_uses_discrete_heatmap_for_two_ordered_discrete_axes(
        self,
    ) -> None:
        X = np.array(
            [
                [1.0, 10.0, 0.1],
                [1.0, 20.0, 0.2],
                [2.0, 10.0, 0.3],
                [3.0, 20.0, 0.4],
            ],
            dtype=float,
        )
        custom_norm = TwoSlopeNorm(vmin=0.0, vcenter=25.0, vmax=50.0)
        figure, axes = self.plot_pair_in_layout(
            X,
            labels=["stage", "rank", "bias"],
            ordered_discrete_keys=["stage", "rank"],
            method="gaussian",
            norm=custom_norm,
        )

        collection = axes[0, 0].collections[0]
        self.assertIsInstance(collection, QuadMesh)
        self.assertIs(collection.norm, custom_norm)
        np.testing.assert_allclose(axes[0, 0].get_xticks(), [1.0, 2.0, 3.0])
        np.testing.assert_allclose(axes[0, 0].get_yticks(), [10.0, 20.0])
        self.assertIsInstance(axes[0, 0].xaxis.get_minor_locator(), NullLocator)
        self.assertIsInstance(axes[0, 0].yaxis.get_minor_locator(), NullLocator)
        self.assertEqual(len(axes[0, 0].lines), 3)
        np.testing.assert_allclose(
            [line.get_xdata()[0] for line in axes[0, 0].lines[:2]],
            [1.5, 2.5],
        )
        np.testing.assert_allclose(
            [line.get_ydata()[0] for line in axes[0, 0].lines[2:]],
            [15.0],
        )
        plt.close(figure)

    def test_plot_pair_rejects_unknown_ordered_discrete_keys(self) -> None:
        figure, axes = self.pair_layout(self.X)
        try:
            with self.assertRaisesRegex(
                ValueError,
                "unknown labels: missing",
            ):
                plot_pair(
                    self.X,
                    self.y,
                    fig=figure,
                    axes=axes,
                    labels=["a", "b", "c"],
                    ordered_discrete_keys=["missing"],
                )
        finally:
            plt.close(figure)


if __name__ == "__main__":
    unittest.main()
