"""Tests for concrete simulator observers."""

from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout

import numpy as np

from network_creation.core.agent import Agent
from network_creation.core.environment import Environment
from network_creation.core.model import Action, ActionSpace, ActionType, Observation, SchedulingMode
from network_creation.core.simulator import Simulator
from network_creation.observers.action_print_observer import ActionPrintObserver
from network_creation.observers.density_observer import DensityObserver
from network_creation.observers.state_history_observer import StateHistoryObserver
from network_creation.policies.base import Policy


class ConstantPolicy(Policy):
    def __init__(self, action: Action) -> None:
        self.action = action

    def reset(self, seed: int) -> None:
        del seed
        return None

    def get_actions(
        self, observation: Observation, action_space: ActionSpace
    ) -> list[Action]:
        del observation
        del action_space
        return [self.action]


class ObserverTest(unittest.TestCase):
    def test_action_print_observer_can_print_initial_empty_step(self) -> None:
        observer = ActionPrintObserver(1, True)
        stdout = io.StringIO()

        with redirect_stdout(stdout):
            observer.on_episode_start(event=None)

        self.assertIn("step 0", stdout.getvalue())

    def test_action_print_observer_groups_actions_by_source_node(self) -> None:
        env = Environment(np.zeros((3, 3), dtype=np.int8), seed=3)
        simulator = Simulator(
            env=env,
            agents=(
                Agent(id=0, radius=0, policy=ConstantPolicy(Action(ActionType.BRIDGE, source=0, target=1)), seed=5),
                Agent(id=1, radius=0, policy=ConstantPolicy(Action(ActionType.PASS, source=1)), seed=7),
                Agent(id=2, radius=0, policy=ConstantPolicy(Action(ActionType.BRIDGE, source=2, target=0)), seed=11),
            ),
            mode=SchedulingMode.SIMULTANEOUS,
            max_steps=1,
            rng=np.random.default_rng(13),
            observers=(ActionPrintObserver(1, False),),
        )

        stdout = io.StringIO()
        with redirect_stdout(stdout):
            simulator.run(n_episodes=1)

        output = stdout.getvalue()
        self.assertIn("step 1", output)
        self.assertIn("  0: BRIDGE->1", output)
        self.assertIn("  1: PASS", output)
        self.assertIn("  2: BRIDGE->0", output)

    def test_density_observer_tracks_density_only(self) -> None:
        density_observer = DensityObserver(
            observation_frequency=1,
            aggregations=(),
        )
        env = Environment(np.zeros((2, 2), dtype=np.int8), seed=13)
        simulator = Simulator(
            env=env,
            agents=(
                Agent(id=0, radius=0, policy=ConstantPolicy(Action(ActionType.BRIDGE, source=0, target=1)), seed=17),
                Agent(id=1, radius=0, policy=ConstantPolicy(Action(ActionType.PASS, source=1)), seed=19),
            ),
            mode=SchedulingMode.SEQUENTIAL,
            max_steps=2,
            rng=np.random.default_rng(23),
            observers=(density_observer,),
        )

        simulator.run(n_episodes=1)

        np.testing.assert_array_equal(density_observer.steps, np.array([0, 1, 2]))
        np.testing.assert_allclose(
            density_observer.densities,
            np.array([0.0, 1.0, 1.0], dtype=np.float32),
        )

    def test_state_history_observer_tracks_graphs_and_ownership_matrices(self) -> None:
        state_history_observer = StateHistoryObserver(
            observation_frequency=1,
            keep_runs=False,
        )
        env = Environment(np.zeros((2, 2), dtype=np.int8), seed=29)
        simulator = Simulator(
            env=env,
            agents=(
                Agent(id=0, radius=0, policy=ConstantPolicy(Action(ActionType.BRIDGE, source=0, target=1)), seed=31),
                Agent(id=1, radius=0, policy=ConstantPolicy(Action(ActionType.PASS, source=1)), seed=37),
            ),
            mode=SchedulingMode.SEQUENTIAL,
            max_steps=2,
            rng=np.random.default_rng(41),
            observers=(state_history_observer,),
        )

        simulator.run(n_episodes=1)

        np.testing.assert_array_equal(state_history_observer.steps, np.array([0, 1, 2]))
        self.assertEqual(len(state_history_observer.graphs), 3)
        self.assertEqual(len(state_history_observer.ownership_matrices), 3)
        self.assertFalse(state_history_observer.graphs[0].are_adjacent(0, 1))
        self.assertTrue(state_history_observer.graphs[1].are_adjacent(0, 1))
        np.testing.assert_array_equal(
            state_history_observer.ownership_matrices[0],
            np.array([[0, 0], [0, 0]], dtype=np.int8),
        )
        np.testing.assert_array_equal(
            state_history_observer.ownership_matrices[1],
            np.array([[0, 1], [0, 0]], dtype=np.int8),
        )

    def test_observers_record_sampled_step_ids_with_observation_frequency(self) -> None:
        density_observer = DensityObserver(
            observation_frequency=2,
            aggregations=(),
        )
        state_history_observer = StateHistoryObserver(
            observation_frequency=2,
            keep_runs=False,
        )
        env = Environment(np.zeros((2, 2), dtype=np.int8), seed=43)
        simulator = Simulator(
            env=env,
            agents=(
                Agent(id=0, radius=0, policy=ConstantPolicy(Action(ActionType.PASS, source=0)), seed=47),
                Agent(id=1, radius=0, policy=ConstantPolicy(Action(ActionType.PASS, source=1)), seed=53),
            ),
            mode=SchedulingMode.SEQUENTIAL,
            max_steps=5,
            rng=np.random.default_rng(59),
            observers=(density_observer, state_history_observer),
        )

        simulator.run(n_episodes=1)

        np.testing.assert_array_equal(density_observer.steps, np.array([0, 2, 4]))
        np.testing.assert_allclose(
            density_observer.densities,
            np.array([0.0, 0.0, 0.0], dtype=np.float32),
        )
        np.testing.assert_array_equal(state_history_observer.steps, np.array([0, 2, 4]))
        self.assertEqual(state_history_observer.ownership_matrices.shape, (3, 2, 2))

    def test_density_observer_aggregates_selected_outputs_and_runs(self) -> None:
        density_observer = DensityObserver(
            observation_frequency=1,
            aggregations=("mean", "runs"),
        )
        env = Environment(np.zeros((2, 2), dtype=np.int8), seed=61)
        simulator = Simulator(
            env=env,
            agents=(
                Agent(id=0, radius=0, policy=ConstantPolicy(Action(ActionType.BRIDGE, source=0, target=1)), seed=67),
                Agent(id=1, radius=0, policy=ConstantPolicy(Action(ActionType.PASS, source=1)), seed=71),
            ),
            mode=SchedulingMode.SEQUENTIAL,
            max_steps=2,
            rng=np.random.default_rng(73),
            observers=(density_observer,),
        )

        simulator.run(n_episodes=2)

        result = density_observer.get()

        self.assertEqual(result.n_episodes, 2)
        np.testing.assert_array_equal(result.steps, np.array([0, 1, 2]))
        self.assertEqual(set(result.outputs), {"mean", "runs"})
        np.testing.assert_allclose(
            result.outputs["mean"],
            np.array([0.0, 1.0, 1.0], dtype=np.float32),
        )
        runs = result.outputs["runs"]
        self.assertEqual(len(runs), 2)
        np.testing.assert_allclose(
            runs[0],
            np.array([0.0, 1.0, 1.0], dtype=np.float32),
        )

    def test_density_observer_get_keeps_sampled_steps_for_aggregated_series(self) -> None:
        density_observer = DensityObserver(
            observation_frequency=2,
            aggregations=("mean", "runs"),
        )
        env = Environment(np.zeros((2, 2), dtype=np.int8), seed=101)
        simulator = Simulator(
            env=env,
            agents=(
                Agent(id=0, radius=0, policy=ConstantPolicy(Action(ActionType.PASS, source=0)), seed=103),
                Agent(id=1, radius=0, policy=ConstantPolicy(Action(ActionType.PASS, source=1)), seed=107),
            ),
            mode=SchedulingMode.SEQUENTIAL,
            max_steps=5,
            rng=np.random.default_rng(109),
            observers=(density_observer,),
        )

        simulator.run(n_episodes=2)

        result = density_observer.get()

        np.testing.assert_array_equal(result.steps, np.array([0, 2, 4]))
        np.testing.assert_allclose(
            result.outputs["mean"],
            np.array([0.0, 0.0, 0.0], dtype=np.float32),
        )

    def test_density_observer_can_request_std_without_mean(self) -> None:
        density_observer = DensityObserver(
            observation_frequency=1,
            aggregations=("std",),
        )
        env = Environment(np.zeros((2, 2), dtype=np.int8), seed=79)
        simulator = Simulator(
            env=env,
            agents=(
                Agent(id=0, radius=0, policy=ConstantPolicy(Action(ActionType.BRIDGE, source=0, target=1)), seed=83),
                Agent(id=1, radius=0, policy=ConstantPolicy(Action(ActionType.PASS, source=1)), seed=89),
            ),
            mode=SchedulingMode.SEQUENTIAL,
            max_steps=2,
            rng=np.random.default_rng(97),
            observers=(density_observer,),
        )

        simulator.run(n_episodes=2)

        result = density_observer.get()

        self.assertEqual(set(result.outputs), {"std"})
        np.testing.assert_allclose(
            result.outputs["std"],
            np.array([0.0, 0.0, 0.0], dtype=np.float32),
        )

    def test_observers_keep_sampling_tail_steps_after_early_stopping(self) -> None:
        density_observer = DensityObserver(
            observation_frequency=2,
            aggregations=(),
        )
        state_history_observer = StateHistoryObserver(
            observation_frequency=2,
            keep_runs=False,
        )
        env = Environment(np.zeros((2, 2), dtype=np.int8), seed=113)
        simulator = Simulator(
            env=env,
            agents=(
                Agent(id=0, radius=0, policy=ConstantPolicy(Action(ActionType.PASS, source=0)), seed=127),
                Agent(id=1, radius=0, policy=ConstantPolicy(Action(ActionType.PASS, source=1)), seed=131),
            ),
            mode=SchedulingMode.SEQUENTIAL,
            max_steps=5,
            rng=np.random.default_rng(137),
            observers=(density_observer, state_history_observer),
            stagnation_patience=1,
        )

        simulator.run(n_episodes=1)

        np.testing.assert_array_equal(density_observer.steps, np.array([0, 2, 4]))
        np.testing.assert_allclose(
            density_observer.densities,
            np.array([0.0, 0.0, 0.0], dtype=np.float32),
        )
        np.testing.assert_array_equal(state_history_observer.steps, np.array([0, 2, 4]))
        self.assertEqual(state_history_observer.ownership_matrices.shape, (3, 2, 2))


if __name__ == "__main__":
    unittest.main()
