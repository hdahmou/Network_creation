"""Tests for plotting helpers."""

from __future__ import annotations

import unittest

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from network_creation.core.agent import Agent
from network_creation.core.environment import Environment
from network_creation.core.graph import create_empty_graph
from network_creation.core.model import Action, ActionSpace, ActionType, Observation, SchedulingMode
from network_creation.core.simulator import Simulator
from network_creation.observers.density_observer import DensityObserver
from network_creation.observers.statistics.episode_tensor_series import (
    EpisodeSeriesTensorStatisticObserver,
)
from network_creation.observers.triad_census_observer import TriadCensusObserver
from network_creation.plotting.episode_series import plot_episode_series_observer
from network_creation.policies.base import Policy
from network_creation.policies.random import RandomPolicy


class ConstantPolicy(Policy):
    def __init__(self, action: Action) -> None:
        self.action = action

    def reset(self, seed: int) -> None:
        del seed

    def get_actions(
        self,
        observation: Observation,
        action_space: ActionSpace,
    ) -> list[Action]:
        del observation
        del action_space
        return [self.action]


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
            observation_frequency=1,
            aggregations=(),
        )
        figure, axis = plt.subplots()

        with self.assertRaises(ValueError):
            plot_episode_series_observer(observer=observer, ax=axis)

        plt.close(figure)

    def test_single_episode_plot_returns_axis(self) -> None:
        observer = DensityObserver(
            observation_frequency=1,
            aggregations=(),
        )
        simulator = Simulator(
            env=Environment(create_empty_graph(2), seed=3),
            agents=(
                Agent(
                    id=0,
                    radius=1,
                    policy=ConstantPolicy(Action(ActionType.BRIDGE, source=0, target=1)),
                    seed=5,
                ),
                Agent(
                    id=1,
                    radius=1,
                    policy=ConstantPolicy(Action(ActionType.PASS, source=1)),
                    seed=7,
                ),
            ),
            mode=SchedulingMode.SEQUENTIAL,
            max_steps=2,
            rng=np.random.default_rng(11),
            observers=(observer,),
        )

        simulator.run(n_episodes=1)

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
            observation_frequency=1,
            aggregations=("mean", "std", "runs"),
        )
        simulator = Simulator(
            env=Environment(create_empty_graph(4), seed=13),
            agents=tuple(
                Agent(
                    id=node_id,
                    radius=2,
                    policy=RandomPolicy(seed=17 + node_id),
                    seed=17 + node_id,
                )
                for node_id in range(4)
            ),
            mode=SchedulingMode.SEQUENTIAL,
            max_steps=8,
            rng=np.random.default_rng(29),
            observers=(observer,),
        )

        simulator.run(n_episodes=4)

        figure, axis = plt.subplots()
        returned_axis = plot_episode_series_observer(observer=observer, ax=axis)

        self.assertIs(returned_axis, axis)
        self.assertEqual(len(axis.lines), 1)
        self.assertEqual(len(axis.collections), 1)
        self.assertEqual(axis.get_title(), "Density across multiple episodes")
        plt.close(figure)

    def test_runs_only_plot_uses_single_episode_title_when_needed(self) -> None:
        observer = DensityObserver(
            observation_frequency=1,
            aggregations=("runs",),
        )
        simulator = Simulator(
            env=Environment(create_empty_graph(2), seed=31),
            agents=(
                Agent(
                    id=0,
                    radius=1,
                    policy=ConstantPolicy(Action(ActionType.BRIDGE, source=0, target=1)),
                    seed=37,
                ),
                Agent(
                    id=1,
                    radius=1,
                    policy=ConstantPolicy(Action(ActionType.PASS, source=1)),
                    seed=41,
                ),
            ),
            mode=SchedulingMode.SEQUENTIAL,
            max_steps=2,
            rng=np.random.default_rng(43),
            observers=(observer,),
        )

        simulator.run(n_episodes=1)

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
            observation_frequency=1,
            aggregations=("mean", "std", "runs"),
        )
        simulator = Simulator(
            env=Environment(create_empty_graph(4), seed=47),
            agents=tuple(
                Agent(
                    id=node_id,
                    radius=2,
                    policy=RandomPolicy(seed=53 + node_id),
                    seed=53 + node_id,
                )
                for node_id in range(4)
            ),
            mode=SchedulingMode.SEQUENTIAL,
            max_steps=8,
            rng=np.random.default_rng(61),
            observers=(observer,),
        )

        simulator.run(n_episodes=4)

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
            observation_frequency=1,
            aggregations=("mean", "std"),
        )
        simulator = Simulator(
            env=Environment(create_empty_graph(4), seed=71),
            agents=tuple(
                Agent(
                    id=node_id,
                    radius=2,
                    policy=RandomPolicy(seed=73 + node_id),
                    seed=73 + node_id,
                )
                for node_id in range(4)
            ),
            mode=SchedulingMode.SEQUENTIAL,
            max_steps=8,
            rng=np.random.default_rng(89),
            observers=(observer,),
        )

        simulator.run(n_episodes=3)

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
            observation_frequency=1,
            aggregations=(),
        )
        simulator = Simulator(
            env=Environment(create_empty_graph(2), seed=97),
            agents=(),
            mode=SchedulingMode.SEQUENTIAL,
            max_steps=0,
            rng=np.random.default_rng(101),
            observers=(observer,),
        )

        simulator.run(n_episodes=1)

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
            observation_frequency=1,
            aggregations=(),
        )
        simulator = Simulator(
            env=Environment(create_empty_graph(3), seed=97),
            agents=(),
            mode=SchedulingMode.SEQUENTIAL,
            max_steps=0,
            rng=np.random.default_rng(101),
            observers=(observer,),
        )

        simulator.run(n_episodes=1)

        figure, axis = plt.subplots()
        with self.assertRaisesRegex(
            ValueError,
            "dimension_name must be provided",
        ):
            plot_episode_series_observer(observer=observer, ax=axis)

        plt.close(figure)

    def test_tensor_plot_selects_dimension_by_name(self) -> None:
        observer = MatrixObserver(
            observation_frequency=1,
            aggregations=(),
        )
        simulator = Simulator(
            env=Environment(create_empty_graph(2), seed=103),
            agents=(
                Agent(
                    id=0,
                    radius=1,
                    policy=ConstantPolicy(Action(ActionType.PASS, source=0)),
                    seed=107,
                ),
            ),
            mode=SchedulingMode.SEQUENTIAL,
            max_steps=1,
            rng=np.random.default_rng(109),
            observers=(observer,),
        )

        simulator.run(n_episodes=1)

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
            observation_frequency=1,
            aggregations=("mean", "std", "runs"),
        )
        simulator = Simulator(
            env=Environment(create_empty_graph(2), seed=113),
            agents=(
                Agent(
                    id=0,
                    radius=1,
                    policy=ConstantPolicy(Action(ActionType.PASS, source=0)),
                    seed=127,
                ),
            ),
            mode=SchedulingMode.SEQUENTIAL,
            max_steps=1,
            rng=np.random.default_rng(131),
            observers=(observer,),
        )

        simulator.run(n_episodes=2)

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
            observation_frequency=1,
            aggregations=(),
        )
        simulator = Simulator(
            env=Environment(create_empty_graph(2), seed=137),
            agents=(
                Agent(
                    id=0,
                    radius=1,
                    policy=ConstantPolicy(Action(ActionType.PASS, source=0)),
                    seed=139,
                ),
            ),
            mode=SchedulingMode.SEQUENTIAL,
            max_steps=1,
            rng=np.random.default_rng(149),
            observers=(observer,),
        )

        simulator.run(n_episodes=1)

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


if __name__ == "__main__":
    unittest.main()
