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

    def reset(self) -> None:
        return None

    def get_action(self, observation: Observation, action_space: ActionSpace) -> Action:
        del observation
        del action_space
        return self.action

    def get_params(self) -> dict[str, object]:
        return {}


class ObserverTest(unittest.TestCase):
    def test_action_print_observer_prints_one_line_per_step(self) -> None:
        env = Environment(np.zeros((2, 2), dtype=np.int8), seed=3)
        simulator = Simulator(
            env=env,
            agents={
                0: Agent(id=0, radius=0, policy=ConstantPolicy(Action(ActionType.BRIDGE, source=0, target=1)), seed=5),
                1: Agent(id=1, radius=0, policy=ConstantPolicy(Action(ActionType.PASS, source=1)), seed=7),
            },
            mode=SchedulingMode.SEQUENTIAL,
            max_steps=2,
            seed=11,
            observers=[ActionPrintObserver(1)],
        )

        stdout = io.StringIO()
        with redirect_stdout(stdout):
            simulator.run_episode()

        output = stdout.getvalue()
        self.assertIn("step 0: 0 BRIDGE 1", output)
        self.assertIn("step 1: 1 PASS", output)

    def test_density_observer_tracks_density_only(self) -> None:
        density_observer = DensityObserver(1)
        env = Environment(np.zeros((2, 2), dtype=np.int8), seed=13)
        simulator = Simulator(
            env=env,
            agents={
                0: Agent(id=0, radius=0, policy=ConstantPolicy(Action(ActionType.BRIDGE, source=0, target=1)), seed=17),
                1: Agent(id=1, radius=0, policy=ConstantPolicy(Action(ActionType.PASS, source=1)), seed=19),
            },
            mode=SchedulingMode.SEQUENTIAL,
            max_steps=2,
            seed=23,
            observers=[density_observer],
        )

        simulator.run_episode()

        np.testing.assert_array_equal(density_observer.steps, np.array([0, 1, 2]))
        np.testing.assert_allclose(
            density_observer.densities,
            np.array([0.0, 1.0, 1.0], dtype=np.float32),
        )

    def test_state_history_observer_tracks_graphs_and_ownership_matrices(self) -> None:
        state_history_observer = StateHistoryObserver(1)
        env = Environment(np.zeros((2, 2), dtype=np.int8), seed=29)
        simulator = Simulator(
            env=env,
            agents={
                0: Agent(id=0, radius=0, policy=ConstantPolicy(Action(ActionType.BRIDGE, source=0, target=1)), seed=31),
                1: Agent(id=1, radius=0, policy=ConstantPolicy(Action(ActionType.PASS, source=1)), seed=37),
            },
            mode=SchedulingMode.SEQUENTIAL,
            max_steps=2,
            seed=41,
            observers=[state_history_observer],
        )

        simulator.run_episode()

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
        density_observer = DensityObserver(observation_frequency=2)
        state_history_observer = StateHistoryObserver(observation_frequency=2)
        env = Environment(np.zeros((2, 2), dtype=np.int8), seed=43)
        simulator = Simulator(
            env=env,
            agents={
                0: Agent(id=0, radius=0, policy=ConstantPolicy(Action(ActionType.PASS, source=0)), seed=47),
                1: Agent(id=1, radius=0, policy=ConstantPolicy(Action(ActionType.PASS, source=1)), seed=53),
            },
            mode=SchedulingMode.SEQUENTIAL,
            max_steps=5,
            seed=59,
            observers=[density_observer, state_history_observer],
        )

        simulator.run_episode()

        np.testing.assert_array_equal(density_observer.steps, np.array([0, 2, 4]))
        np.testing.assert_allclose(
            density_observer.densities,
            np.array([0.0, 0.0, 0.0], dtype=np.float32),
        )
        np.testing.assert_array_equal(state_history_observer.steps, np.array([0, 2, 4]))
        self.assertEqual(state_history_observer.ownership_matrices.shape, (3, 2, 2))


if __name__ == "__main__":
    unittest.main()
