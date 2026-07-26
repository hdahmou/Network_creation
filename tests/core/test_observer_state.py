"""Tests for action/state observers and sampling behavior."""

from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout

import numpy as np

from network_creation.core.agent import Agent
from network_creation.core.environment import Environment
from network_creation.core.graph import create_empty_graph
from network_creation.core.model import Action, ActionType, ActorScheduler
from network_creation.core.observability import ObservationSampling
from network_creation.core.simulator import Simulator
from network_creation.observers.action_count_observer import ActionCountObserver
from network_creation.observers.action_print_observer import ActionPrintObserver
from network_creation.observers.density_observer import DensityObserver
from network_creation.observers.state_history_observer import StateHistoryObserver
from tests.core.simulation_support import ActionListPolicy, ConstantPolicy


class StateAndActionObserverTest(unittest.TestCase):
    def test_action_print_observer_can_print_initial_empty_step(self) -> None:
        observer = ActionPrintObserver(ObservationSampling.periodic(1), True)
        stdout = io.StringIO()

        with redirect_stdout(stdout):
            observer.on_episode_start(event=None)

        self.assertIn("step 0", stdout.getvalue())

    def test_action_print_observer_groups_actions_by_source_node(self) -> None:
        env = Environment(create_empty_graph(3), seed=3)
        simulator = Simulator(
            env=env,
            agents=(
                Agent(
                    activation_weight=1.0,
                    id=0,
                    visibility_distance=10,
                    policy=ConstantPolicy(Action(ActionType.BRIDGE, source=0, target=1)),
                    seed=5,
                ),
                Agent(
                    activation_weight=1.0,
                    id=1,
                    visibility_distance=10,
                    policy=ConstantPolicy(Action(ActionType.PASS, source=1)),
                    seed=7,
                ),
                Agent(
                    activation_weight=1.0,
                    id=2,
                    visibility_distance=10,
                    policy=ConstantPolicy(Action(ActionType.BRIDGE, source=2, target=0)),
                    seed=11,
                ),
            ),
            actor_scheduler=ActorScheduler(n_agents=3, n_actors=3, activation_weights=(1.0, 1.0, 1.0), seed=21),
            max_steps=1,
            rng=np.random.default_rng(13),
            observers=(ActionPrintObserver(ObservationSampling.periodic(1), False),),
        )

        stdout = io.StringIO()
        with redirect_stdout(stdout):
            simulator.run(n_episodes=1, reset_observers=False)

        output = stdout.getvalue()
        self.assertIn("step 1", output)
        self.assertIn("  0: BRIDGE->1", output)
        self.assertIn("  1: PASS", output)
        self.assertIn("  2: BRIDGE->0", output)

    def test_action_print_observer_shows_decision_value_on_same_line(self) -> None:
        env = Environment(create_empty_graph(2), seed=3)
        simulator = Simulator(
            env=env,
            agents=(
                Agent(
                    activation_weight=1.0,
                    id=0,
                    visibility_distance=10,
                    policy=ConstantPolicy(
                        Action(ActionType.BRIDGE, source=0, target=1),
                        value={"score": 1.25},
                    ),
                    seed=5,
                ),
                Agent(
                    activation_weight=1.0,
                    id=1,
                    visibility_distance=10,
                    policy=ConstantPolicy(Action(ActionType.PASS, source=1)),
                    seed=7,
                ),
            ),
            actor_scheduler=ActorScheduler(n_agents=2, n_actors=2, activation_weights=(1.0, 1.0), seed=21),
            max_steps=1,
            rng=np.random.default_rng(13),
            observers=(ActionPrintObserver(ObservationSampling.periodic(1), False),),
        )

        stdout = io.StringIO()
        with redirect_stdout(stdout):
            simulator.run(n_episodes=1, reset_observers=False)

        output = stdout.getvalue()
        self.assertIn("  0: BRIDGE->1 (value={score=1.25})", output)
        self.assertIn("  1: PASS", output)

    def test_action_count_observer_normalizes_by_total_agents(self) -> None:
        action_count_observer = ActionCountObserver(
            n_agents=2,
            observation_sampling=ObservationSampling.periodic(1),
            aggregations=(),
        )
        env = Environment(create_empty_graph(2), seed=41)
        simulator = Simulator(
            env=env,
            agents=(
                Agent(
                    activation_weight=1.0,
                    id=0,
                    visibility_distance=10,
                    policy=ActionListPolicy(
                        [Action(ActionType.BRIDGE, source=0, target=1)]
                    ),
                    seed=43,
                ),
                Agent(
                    activation_weight=1.0,
                    id=1,
                    visibility_distance=10,
                    policy=ActionListPolicy([Action(ActionType.PASS, source=1)]),
                    seed=47,
                ),
            ),
            actor_scheduler=ActorScheduler(n_agents=1, n_actors=1, activation_weights=(1.0,), seed=21),
            max_steps=1,
            rng=np.random.default_rng(53),
            observers=(action_count_observer,),
        )

        simulator.run(n_episodes=1, reset_observers=False)

        np.testing.assert_array_equal(action_count_observer.steps, np.array([0, 1]))
        np.testing.assert_allclose(
            action_count_observer.values,
            np.array([0.0, 0.5], dtype=np.float32),
        )

    def test_action_count_observer_excludes_pass_and_supports_multi_action(
        self,
    ) -> None:
        action_count_observer = ActionCountObserver(
            n_agents=3,
            observation_sampling=ObservationSampling.periodic(1),
            aggregations=(),
        )
        env = Environment(create_empty_graph(3), seed=59)
        simulator = Simulator(
            env=env,
            agents=(
                Agent(
                    activation_weight=1.0,
                    id=0,
                    visibility_distance=10,
                    policy=ActionListPolicy(
                        [
                            Action(ActionType.BRIDGE, source=0, target=1),
                            Action(ActionType.BRIDGE, source=0, target=2),
                        ]
                    ),
                    seed=61,
                ),
                Agent(
                    activation_weight=1.0,
                    id=1,
                    visibility_distance=10,
                    policy=ActionListPolicy([Action(ActionType.PASS, source=1)]),
                    seed=67,
                ),
                Agent(
                    activation_weight=1.0,
                    id=2,
                    visibility_distance=10,
                    policy=ActionListPolicy(
                        [Action(ActionType.BRIDGE, source=2, target=1)]
                    ),
                    seed=71,
                ),
            ),
            actor_scheduler=ActorScheduler(n_agents=3, n_actors=3, activation_weights=(1.0, 1.0, 1.0), seed=21),
            max_steps=1,
            rng=np.random.default_rng(73),
            observers=(action_count_observer,),
        )

        simulator.run(n_episodes=1, reset_observers=False)

        np.testing.assert_array_equal(action_count_observer.steps, np.array([0, 1]))
        np.testing.assert_allclose(
            action_count_observer.values,
            np.array([0.0, 1.0], dtype=np.float32),
        )

    def test_state_history_observer_tracks_graphs(self) -> None:
        state_history_observer = StateHistoryObserver(
            observation_sampling=ObservationSampling.periodic(1),
            keep_runs=False,
        )
        env = Environment(create_empty_graph(2), seed=29)
        simulator = Simulator(
            env=env,
            agents=(
                Agent(
                    activation_weight=1.0,
                    id=0,
                    visibility_distance=10,
                    policy=ConstantPolicy(Action(ActionType.BRIDGE, source=0, target=1)),
                    seed=31,
                ),
                Agent(
                    activation_weight=1.0,
                    id=1,
                    visibility_distance=10,
                    policy=ConstantPolicy(Action(ActionType.PASS, source=1)),
                    seed=37,
                ),
            ),
            actor_scheduler=ActorScheduler(n_agents=1, n_actors=1, activation_weights=(1.0,), seed=21),
            max_steps=2,
            rng=np.random.default_rng(41),
            observers=(state_history_observer,),
        )

        simulator.run(n_episodes=1, reset_observers=False)

        np.testing.assert_array_equal(state_history_observer.steps, np.array([0, 1, 2]))
        self.assertEqual(len(state_history_observer.graphs), 3)
        self.assertFalse(state_history_observer.graphs[0].are_adjacent(0, 1))
        self.assertTrue(state_history_observer.graphs[1].are_adjacent(0, 1))
        self.assertEqual(state_history_observer.graphs[0].ecount(), 0)
        self.assertEqual(state_history_observer.graphs[1].get_edgelist(), [(0, 1)])

    def test_state_history_observer_keeps_persistent_copies_of_graph_history(self) -> None:
        state_history_observer = StateHistoryObserver(
            observation_sampling=ObservationSampling.periodic(1),
            keep_runs=False,
        )
        env = Environment(create_empty_graph(2), seed=29)
        simulator = Simulator(
            env=env,
            agents=(
                Agent(
                    activation_weight=1.0,
                    id=0,
                    visibility_distance=10,
                    policy=ConstantPolicy(Action(ActionType.BRIDGE, source=0, target=1)),
                    seed=31,
                ),
                Agent(
                    activation_weight=1.0,
                    id=1,
                    visibility_distance=10,
                    policy=ConstantPolicy(Action(ActionType.PASS, source=1)),
                    seed=37,
                ),
            ),
            actor_scheduler=ActorScheduler(n_agents=1, n_actors=1, activation_weights=(1.0,), seed=21),
            max_steps=2,
            rng=np.random.default_rng(41),
            observers=(state_history_observer,),
        )

        simulator.run(n_episodes=1, reset_observers=False)

        exported_graphs = state_history_observer.graphs
        simulator.env.state.graph.delete_edges([(0, 1)])
        self.assertTrue(state_history_observer.graphs[1].are_adjacent(0, 1))

        exported_graphs[1].delete_edges([(0, 1)])
        self.assertTrue(state_history_observer.graphs[1].are_adjacent(0, 1))

    def test_observers_record_sampled_step_ids_with_observation_sampling(self) -> None:
        density_observer = DensityObserver(
            n_agents=2,
            observation_sampling=ObservationSampling.periodic(2),
            aggregations=(),
        )
        state_history_observer = StateHistoryObserver(
            observation_sampling=ObservationSampling.periodic(2),
            keep_runs=False,
        )
        env = Environment(create_empty_graph(2), seed=43)
        simulator = Simulator(
            env=env,
            agents=(
                Agent(
                    activation_weight=1.0,
                    id=0,
                    visibility_distance=10,
                    policy=ConstantPolicy(Action(ActionType.PASS, source=0)),
                    seed=47,
                ),
                Agent(
                    activation_weight=1.0,
                    id=1,
                    visibility_distance=10,
                    policy=ConstantPolicy(Action(ActionType.PASS, source=1)),
                    seed=53,
                ),
            ),
            actor_scheduler=ActorScheduler(n_agents=1, n_actors=1, activation_weights=(1.0,), seed=21),
            max_steps=5,
            rng=np.random.default_rng(59),
            observers=(density_observer, state_history_observer),
        )

        simulator.run(n_episodes=1, reset_observers=False)

        np.testing.assert_array_equal(density_observer.steps, np.array([0, 2, 4]))
        np.testing.assert_allclose(
            density_observer.values,
            np.array([0.0, 0.0, 0.0], dtype=np.float32),
        )
        np.testing.assert_array_equal(state_history_observer.steps, np.array([0, 2, 4]))
        self.assertEqual(len(state_history_observer.graphs), 3)

    def test_action_count_observer_terminal_sampling_uses_last_step_decisions(self) -> None:
        action_count_observer = ActionCountObserver(
            n_agents=2,
            observation_sampling=ObservationSampling.terminal(),
            aggregations=(),
        )
        env = Environment(create_empty_graph(2), seed=41)
        simulator = Simulator(
            env=env,
            agents=(
                Agent(
                    activation_weight=1.0,
                    id=0,
                    visibility_distance=10,
                    policy=ActionListPolicy(
                        [Action(ActionType.BRIDGE, source=0, target=1)]
                    ),
                    seed=43,
                ),
                Agent(
                    activation_weight=1.0,
                    id=1,
                    visibility_distance=10,
                    policy=ActionListPolicy([Action(ActionType.PASS, source=1)]),
                    seed=47,
                ),
            ),
            actor_scheduler=ActorScheduler(n_agents=1, n_actors=1, activation_weights=(1.0,), seed=21),
            max_steps=1,
            rng=np.random.default_rng(53),
            observers=(action_count_observer,),
        )

        simulator.run(n_episodes=1, reset_observers=False)

        np.testing.assert_array_equal(action_count_observer.steps, np.array([1]))
        np.testing.assert_allclose(
            action_count_observer.values,
            np.array([0.5], dtype=np.float32),
        )

    def test_state_history_observer_terminal_sampling_keeps_final_graph_only(self) -> None:
        state_history_observer = StateHistoryObserver(
            observation_sampling=ObservationSampling.terminal(),
            keep_runs=False,
        )
        env = Environment(create_empty_graph(2), seed=29)
        simulator = Simulator(
            env=env,
            agents=(
                Agent(
                    activation_weight=1.0,
                    id=0,
                    visibility_distance=10,
                    policy=ConstantPolicy(Action(ActionType.BRIDGE, source=0, target=1)),
                    seed=31,
                ),
                Agent(
                    activation_weight=1.0,
                    id=1,
                    visibility_distance=10,
                    policy=ConstantPolicy(Action(ActionType.PASS, source=1)),
                    seed=37,
                ),
            ),
            actor_scheduler=ActorScheduler(n_agents=1, n_actors=1, activation_weights=(1.0,), seed=21),
            max_steps=2,
            rng=np.random.default_rng(41),
            observers=(state_history_observer,),
        )

        simulator.run(n_episodes=1, reset_observers=False)

        np.testing.assert_array_equal(state_history_observer.steps, np.array([2]))
        self.assertEqual(len(state_history_observer.graphs), 1)
        self.assertTrue(state_history_observer.graphs[0].are_adjacent(0, 1))

    def test_action_print_observer_terminal_sampling_prints_only_last_completed_step(
        self,
    ) -> None:
        env = Environment(create_empty_graph(2), seed=3)
        simulator = Simulator(
            env=env,
            agents=(
                Agent(
                    activation_weight=1.0,
                    id=0,
                    visibility_distance=10,
                    policy=ConstantPolicy(Action(ActionType.BRIDGE, source=0, target=1)),
                    seed=5,
                ),
                Agent(
                    activation_weight=1.0,
                    id=1,
                    visibility_distance=10,
                    policy=ConstantPolicy(Action(ActionType.PASS, source=1)),
                    seed=7,
                ),
            ),
            actor_scheduler=ActorScheduler(n_agents=1, n_actors=1, activation_weights=(1.0,), seed=21),
            max_steps=2,
            rng=np.random.default_rng(13),
            observers=(ActionPrintObserver(ObservationSampling.terminal(), True),),
        )

        stdout = io.StringIO()
        with redirect_stdout(stdout):
            simulator.run(n_episodes=1, reset_observers=False)

        output = stdout.getvalue()
        self.assertNotIn("step 0", output)
        self.assertIn("step 2", output)

    def test_action_print_observer_terminal_sampling_prints_step_zero_once_when_no_step_executes(
        self,
    ) -> None:
        observer = ActionPrintObserver(ObservationSampling.terminal(), True)
        env = Environment(create_empty_graph(1), seed=3)
        simulator = Simulator(
            env=env,
            agents=(),
            actor_scheduler=ActorScheduler(n_agents=1, n_actors=1, activation_weights=(1.0,), seed=21),
            max_steps=0,
            rng=np.random.default_rng(5),
            observers=(observer,),
        )

        stdout = io.StringIO()
        with redirect_stdout(stdout):
            simulator.run(n_episodes=1, reset_observers=False)

        self.assertEqual(stdout.getvalue().strip(), "step 0")

    def test_observers_keep_sampling_tail_steps_after_early_stopping(self) -> None:
        density_observer = DensityObserver(
            n_agents=2,
            observation_sampling=ObservationSampling.periodic(2),
            aggregations=(),
        )
        state_history_observer = StateHistoryObserver(
            observation_sampling=ObservationSampling.periodic(2),
            keep_runs=False,
        )
        env = Environment(create_empty_graph(2), seed=113)
        simulator = Simulator(
            env=env,
            agents=(
                Agent(
                    activation_weight=1.0,
                    id=0,
                    visibility_distance=10,
                    policy=ConstantPolicy(Action(ActionType.PASS, source=0)),
                    seed=127,
                ),
                Agent(
                    activation_weight=1.0,
                    id=1,
                    visibility_distance=10,
                    policy=ConstantPolicy(Action(ActionType.PASS, source=1)),
                    seed=131,
                ),
            ),
            actor_scheduler=ActorScheduler(n_agents=1, n_actors=1, activation_weights=(1.0,), seed=21),
            max_steps=5,
            rng=np.random.default_rng(137),
            observers=(density_observer, state_history_observer),
            stagnation_patience=1,
        )

        simulator.run(n_episodes=1, reset_observers=False)

        np.testing.assert_array_equal(density_observer.steps, np.array([0, 2, 4]))
        np.testing.assert_allclose(
            density_observer.values,
            np.array([0.0, 0.0, 0.0], dtype=np.float32),
        )
        np.testing.assert_array_equal(state_history_observer.steps, np.array([0, 2, 4]))
        self.assertEqual(len(state_history_observer.graphs), 3)


if __name__ == "__main__":
    unittest.main()
