"""Tests for plotting helpers."""

from __future__ import annotations

import unittest

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from network_creation.core.agent import Agent
from network_creation.core.environment import Environment
from network_creation.core.model import Action, ActionSpace, ActionType, Observation, SchedulingMode
from network_creation.core.simulator import Simulator
from network_creation.observers.density_observer import DensityObserver
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
            env=Environment(np.zeros((2, 2), dtype=np.int8), seed=3),
            agents=(
                Agent(
                    id=0,
                    radius=0,
                    policy=ConstantPolicy(Action(ActionType.BRIDGE, source=0, target=1)),
                    seed=5,
                ),
                Agent(
                    id=1,
                    radius=0,
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

    def test_multi_episode_plot_draws_runs_mean_and_band(self) -> None:
        observer = DensityObserver(
            observation_frequency=1,
            aggregations=("mean", "std", "runs"),
        )
        simulator = Simulator(
            env=Environment(np.zeros((4, 4), dtype=np.int8), seed=13),
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
        self.assertEqual(len(axis.lines), 5)
        self.assertEqual(len(axis.collections), 1)
        self.assertEqual(axis.get_title(), "Density across multiple episodes")
        plt.close(figure)

    def test_runs_only_plot_uses_single_episode_title_when_needed(self) -> None:
        observer = DensityObserver(
            observation_frequency=1,
            aggregations=("runs",),
        )
        simulator = Simulator(
            env=Environment(np.zeros((2, 2), dtype=np.int8), seed=31),
            agents=(
                Agent(
                    id=0,
                    radius=0,
                    policy=ConstantPolicy(Action(ActionType.BRIDGE, source=0, target=1)),
                    seed=37,
                ),
                Agent(
                    id=1,
                    radius=0,
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
        self.assertEqual(len(axis.lines), 1)
        self.assertEqual(axis.get_title(), "Density over one episode")
        plt.close(figure)


if __name__ == "__main__":
    unittest.main()
