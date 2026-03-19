"""Tests for simulator scheduling and episode execution."""

from __future__ import annotations

import unittest

import numpy as np

from network_creation.core.agent import Agent
from network_creation.core.environment import Environment
from network_creation.core.model import Action, ActionSpace, ActionType, Observation, SchedulingMode
from network_creation.core.observability import EpisodeEndEvent, EpisodeStartEvent, StepEvent
from network_creation.core.simulator import Simulator
from network_creation.policies.base import Policy


class ConstantPolicy(Policy):
    def __init__(self, action: Action) -> None:
        self.action = action
        self.reset_calls = 0

    def reset(self) -> None:
        self.reset_calls += 1

    def get_actions(
        self, observation: Observation, action_space: ActionSpace
    ) -> list[Action]:
        del observation
        del action_space
        return [self.action]

    def get_params(self) -> dict[str, object]:
        return {}


class PreferAddElsePassPolicy(Policy):
    def __init__(self) -> None:
        self.reset_calls = 0
        self.observed_node_names: list[list[int]] = []

    def reset(self) -> None:
        self.reset_calls += 1

    def get_actions(
        self, observation: Observation, action_space: ActionSpace
    ) -> list[Action]:
        self.observed_node_names.append(list(observation.graph.vs["name"]))
        if action_space[ActionType.ADD]:
            return [action_space[ActionType.ADD][0]]
        return [action_space[ActionType.PASS][0]]

    def get_params(self) -> dict[str, object]:
        return {}


class RecordingObserver:
    def __init__(self) -> None:
        self.observation_frequency = 1
        self.episode_start_event: EpisodeStartEvent | None = None
        self.step_events: list[StepEvent] = []
        self.episode_end_event: EpisodeEndEvent | None = None

    def on_episode_start(self, event: EpisodeStartEvent) -> None:
        self.episode_start_event = event

    def on_step_end(self, event: StepEvent) -> None:
        self.step_events.append(event)

    def on_episode_end(self, event: EpisodeEndEvent) -> None:
        self.episode_end_event = event


class MultiActionPolicy(Policy):
    def __init__(self, actions: list[Action]) -> None:
        self.actions = actions
        self.reset_calls = 0

    def reset(self) -> None:
        self.reset_calls += 1

    def get_actions(
        self, observation: Observation, action_space: ActionSpace
    ) -> list[Action]:
        del observation
        del action_space
        return list(self.actions)

    def get_params(self) -> dict[str, object]:
        return {}


class SimulatorTest(unittest.TestCase):
    def test_get_acting_agents_sequential_cycles_by_sorted_node_id(self) -> None:
        env = Environment(np.zeros((3, 3), dtype=np.int8), seed=3)
        simulator = Simulator(
            env=env,
            agents={
                2: Agent(id=2, radius=0, policy=ConstantPolicy(Action(ActionType.PASS, source=2)), seed=13),
                0: Agent(id=0, radius=0, policy=ConstantPolicy(Action(ActionType.PASS, source=0)), seed=11),
                1: Agent(id=1, radius=0, policy=ConstantPolicy(Action(ActionType.PASS, source=1)), seed=12),
            },
            mode=SchedulingMode.SEQUENTIAL,
            max_steps=4,
            seed=17,
        )

        self.assertEqual([agent.id for agent in simulator.get_acting_agents()], [0])
        simulator.current_step = 1
        self.assertEqual([agent.id for agent in simulator.get_acting_agents()], [1])
        simulator.current_step = 2
        self.assertEqual([agent.id for agent in simulator.get_acting_agents()], [2])
        simulator.current_step = 3
        self.assertEqual([agent.id for agent in simulator.get_acting_agents()], [0])

    def test_get_acting_agents_simultaneous_returns_all_agents(self) -> None:
        env = Environment(np.zeros((2, 2), dtype=np.int8), seed=19)
        simulator = Simulator(
            env=env,
            agents={
                1: Agent(id=1, radius=0, policy=ConstantPolicy(Action(ActionType.PASS, source=1)), seed=23),
                0: Agent(id=0, radius=0, policy=ConstantPolicy(Action(ActionType.PASS, source=0)), seed=29),
            },
            mode=SchedulingMode.SIMULTANEOUS,
            max_steps=1,
            seed=31,
        )

        self.assertEqual([agent.id for agent in simulator.get_acting_agents()], [0, 1])

    def test_run_episode_sequential_tracks_steps_and_last_modification(self) -> None:
        env = Environment(np.zeros((2, 2), dtype=np.int8), seed=37)
        simulator = Simulator(
            env=env,
            agents={
                0: Agent(id=0, radius=0, policy=ConstantPolicy(Action(ActionType.BRIDGE, source=0, target=1)), seed=41),
                1: Agent(id=1, radius=0, policy=ConstantPolicy(Action(ActionType.PASS, source=1)), seed=43),
            },
            mode=SchedulingMode.SEQUENTIAL,
            max_steps=2,
            seed=47,
        )

        simulator.run_episode()

        self.assertEqual(simulator.current_step, 2)
        self.assertEqual(simulator.last_modified_step, 0)
        self.assertEqual(simulator.env.state.ownership_matrix[0, 1], 1)
        self.assertTrue(simulator.env.state.graph.are_adjacent(0, 1))

    def test_run_episode_simultaneous_uses_same_pre_step_state(self) -> None:
        env = Environment(
            np.array(
                [
                    [0, 1, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                ],
                dtype=np.int8,
            ),
            seed=53,
        )
        add_or_pass_policy = PreferAddElsePassPolicy()
        simulator = Simulator(
            env=env,
            agents={
                0: Agent(id=0, radius=2, policy=add_or_pass_policy, seed=59),
                1: Agent(id=1, radius=1, policy=ConstantPolicy(Action(ActionType.BRIDGE, source=1, target=2)), seed=61),
                2: Agent(id=2, radius=0, policy=ConstantPolicy(Action(ActionType.PASS, source=2)), seed=67),
            },
            mode=SchedulingMode.SIMULTANEOUS,
            max_steps=1,
            seed=71,
        )

        simulator.run_episode()

        self.assertEqual(add_or_pass_policy.observed_node_names, [[0, 1]])
        self.assertEqual(simulator.env.state.ownership_matrix[1, 2], 1)
        self.assertEqual(simulator.env.state.ownership_matrix[0, 2], 0)
        self.assertTrue(simulator.env.state.graph.are_adjacent(1, 2))

    def test_run_episode_applies_multiple_actions_from_one_agent(self) -> None:
        env = Environment(np.zeros((3, 3), dtype=np.int8), seed=72)
        simulator = Simulator(
            env=env,
            agents={
                0: Agent(
                    id=0,
                    radius=2,
                    policy=MultiActionPolicy(
                        [
                            Action(ActionType.BRIDGE, source=0, target=1),
                            Action(ActionType.BRIDGE, source=0, target=2),
                        ]
                    ),
                    seed=73,
                ),
                1: Agent(id=1, radius=0, policy=ConstantPolicy(Action(ActionType.PASS, source=1)), seed=74),
                2: Agent(id=2, radius=0, policy=ConstantPolicy(Action(ActionType.PASS, source=2)), seed=75),
            },
            mode=SchedulingMode.SEQUENTIAL,
            max_steps=1,
            seed=76,
        )

        simulator.run_episode()

        self.assertEqual(simulator.current_step, 1)
        self.assertEqual(simulator.last_modified_step, 0)
        self.assertEqual(simulator.env.state.ownership_matrix[0, 1], 1)
        self.assertEqual(simulator.env.state.ownership_matrix[0, 2], 1)

    def test_reset_restores_environment_agents_and_counters(self) -> None:
        left_policy = ConstantPolicy(Action(ActionType.BRIDGE, source=0, target=1))
        right_policy = ConstantPolicy(Action(ActionType.PASS, source=1))
        env = Environment(np.zeros((2, 2), dtype=np.int8), seed=73)
        simulator = Simulator(
            env=env,
            agents={
                0: Agent(id=0, radius=0, policy=left_policy, seed=79),
                1: Agent(id=1, radius=0, policy=right_policy, seed=83),
            },
            mode=SchedulingMode.SEQUENTIAL,
            max_steps=1,
            seed=89,
        )

        simulator.run_episode()
        simulator.reset()

        self.assertEqual(simulator.current_step, 0)
        self.assertEqual(simulator.last_modified_step, -1)
        self.assertEqual(left_policy.reset_calls, 2)
        self.assertEqual(right_policy.reset_calls, 2)
        np.testing.assert_array_equal(
            simulator.env.state.ownership_matrix,
            np.zeros((2, 2), dtype=np.int8),
        )

    def test_run_episode_starts_from_a_fresh_reset_state(self) -> None:
        observer = RecordingObserver()
        env = Environment(np.zeros((2, 2), dtype=np.int8), seed=90)
        simulator = Simulator(
            env=env,
            agents={
                0: Agent(id=0, radius=0, policy=ConstantPolicy(Action(ActionType.PASS, source=0)), seed=91),
                1: Agent(id=1, radius=0, policy=ConstantPolicy(Action(ActionType.PASS, source=1)), seed=92),
            },
            mode=SchedulingMode.SEQUENTIAL,
            max_steps=1,
            seed=93,
            observers=[observer],
        )

        env.step([Action(ActionType.BRIDGE, source=0, target=1)])
        simulator.current_step = 7
        simulator.last_modified_step = 4

        simulator.run_episode()

        assert observer.episode_start_event is not None
        np.testing.assert_array_equal(
            observer.episode_start_event.initial_state.ownership_matrix,
            np.zeros((2, 2), dtype=np.int8),
        )
        np.testing.assert_array_equal(
            simulator.env.state.ownership_matrix,
            np.zeros((2, 2), dtype=np.int8),
        )
        self.assertEqual(simulator.current_step, 1)
        self.assertEqual(simulator.last_modified_step, -1)

    def test_observer_receives_episode_and_step_events(self) -> None:
        observer = RecordingObserver()
        env = Environment(np.zeros((2, 2), dtype=np.int8), seed=97)
        simulator = Simulator(
            env=env,
            agents={
                0: Agent(id=0, radius=0, policy=ConstantPolicy(Action(ActionType.BRIDGE, source=0, target=1)), seed=101),
                1: Agent(id=1, radius=0, policy=ConstantPolicy(Action(ActionType.PASS, source=1)), seed=103),
            },
            mode=SchedulingMode.SEQUENTIAL,
            max_steps=1,
            seed=107,
            observers=[observer],
        )

        simulator.run_episode()

        self.assertIsNotNone(observer.episode_start_event)
        self.assertIsNotNone(observer.episode_end_event)
        self.assertEqual(len(observer.step_events), 1)

        start_event = observer.episode_start_event
        assert start_event is not None
        self.assertEqual(start_event.seed, 107)
        self.assertEqual(start_event.n_agents, 2)
        np.testing.assert_array_equal(
            start_event.initial_state.ownership_matrix,
            np.zeros((2, 2), dtype=np.int8),
        )

        step_event = observer.step_events[0]
        self.assertEqual(step_event.step, 0)
        self.assertEqual(step_event.acting_agent_ids, (0,))
        self.assertEqual(
            step_event.actions,
            (Action(ActionType.BRIDGE, source=0, target=1),),
        )
        self.assertTrue(step_event.modified)
        self.assertTrue(step_event.post_state.graph.are_adjacent(0, 1))

        end_event = observer.episode_end_event
        assert end_event is not None
        self.assertEqual(end_event.current_step, 1)
        self.assertEqual(end_event.last_modified_step, 0)
        self.assertTrue(end_event.final_state.graph.are_adjacent(0, 1))

    def test_observer_cannot_mutate_live_state_through_event_snapshots(self) -> None:
        observer = RecordingObserver()
        env = Environment(np.zeros((2, 2), dtype=np.int8), seed=109)
        simulator = Simulator(
            env=env,
            agents={
                0: Agent(id=0, radius=0, policy=ConstantPolicy(Action(ActionType.BRIDGE, source=0, target=1)), seed=113),
                1: Agent(id=1, radius=0, policy=ConstantPolicy(Action(ActionType.PASS, source=1)), seed=127),
            },
            mode=SchedulingMode.SEQUENTIAL,
            max_steps=1,
            seed=131,
            observers=[observer],
        )

        simulator.run_episode()

        step_event = observer.step_events[0]

        with self.assertRaises(ValueError):
            step_event.post_state.ownership_matrix[0, 1] = 0

        snapshot_graph = step_event.post_state.graph
        snapshot_graph.delete_edges([(0, 1)])

        self.assertTrue(simulator.env.state.graph.are_adjacent(0, 1))


if __name__ == "__main__":
    unittest.main()
