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
        self.reset_seeds: list[int] = []

    def reset(self, seed: int) -> None:
        self.reset_calls += 1
        self.reset_seeds.append(seed)

    def get_actions(
        self, observation: Observation, action_space: ActionSpace
    ) -> list[Action]:
        del observation
        del action_space
        return [self.action]


class PreferAddElsePassPolicy(Policy):
    def __init__(self) -> None:
        self.reset_calls = 0
        self.reset_seeds: list[int] = []
        self.observed_node_names: list[list[int]] = []

    def reset(self, seed: int) -> None:
        self.reset_calls += 1
        self.reset_seeds.append(seed)

    def get_actions(
        self, observation: Observation, action_space: ActionSpace
    ) -> list[Action]:
        self.observed_node_names.append(list(observation.graph.vs["name"]))
        if action_space[ActionType.ADD]:
            return [action_space[ActionType.ADD][0]]
        return [action_space[ActionType.PASS][0]]


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


class BatchRecordingObserver:
    def __init__(self) -> None:
        self.observation_frequency = 1
        self.episode_start_events: list[EpisodeStartEvent] = []
        self.episode_end_events: list[EpisodeEndEvent] = []

    def reset(self) -> None:
        self.episode_start_events = []
        self.episode_end_events = []

    def on_episode_start(self, event: EpisodeStartEvent) -> None:
        self.episode_start_events.append(event)

    def on_step_end(self, event: StepEvent) -> None:
        del event

    def on_episode_end(self, event: EpisodeEndEvent) -> None:
        self.episode_end_events.append(event)


class MultiActionPolicy(Policy):
    def __init__(self, actions: list[Action]) -> None:
        self.actions = actions
        self.reset_calls = 0
        self.reset_seeds: list[int] = []

    def reset(self, seed: int) -> None:
        self.reset_calls += 1
        self.reset_seeds.append(seed)

    def get_actions(
        self, observation: Observation, action_space: ActionSpace
    ) -> list[Action]:
        del observation
        del action_space
        return list(self.actions)


class SimulatorTest(unittest.TestCase):
    def test_get_acting_agents_sequential_cycles_by_sequence_order(self) -> None:
        env = Environment(np.zeros((3, 3), dtype=np.int8), seed=3)
        simulator = Simulator(
            env=env,
            agents=(
                Agent(id=2, radius=0, policy=ConstantPolicy(Action(ActionType.PASS, source=2)), seed=13),
                Agent(id=0, radius=0, policy=ConstantPolicy(Action(ActionType.PASS, source=0)), seed=11),
                Agent(id=1, radius=0, policy=ConstantPolicy(Action(ActionType.PASS, source=1)), seed=12),
            ),
            mode=SchedulingMode.SEQUENTIAL,
            max_steps=4,
            rng=np.random.default_rng(17),
            observers=(),
        )

        self.assertEqual([agent.id for agent in simulator.get_acting_agents(step=0)], [2])
        self.assertEqual([agent.id for agent in simulator.get_acting_agents(step=1)], [0])
        self.assertEqual([agent.id for agent in simulator.get_acting_agents(step=2)], [1])
        self.assertEqual([agent.id for agent in simulator.get_acting_agents(step=3)], [2])

    def test_get_acting_agents_simultaneous_returns_all_agents_in_sequence_order(self) -> None:
        env = Environment(np.zeros((2, 2), dtype=np.int8), seed=19)
        simulator = Simulator(
            env=env,
            agents=(
                Agent(id=1, radius=0, policy=ConstantPolicy(Action(ActionType.PASS, source=1)), seed=23),
                Agent(id=0, radius=0, policy=ConstantPolicy(Action(ActionType.PASS, source=0)), seed=29),
            ),
            mode=SchedulingMode.SIMULTANEOUS,
            max_steps=1,
            rng=np.random.default_rng(31),
            observers=(),
        )

        self.assertEqual([agent.id for agent in simulator.get_acting_agents(step=0)], [1, 0])

    def test_run_sequential_tracks_final_state(self) -> None:
        env = Environment(np.zeros((2, 2), dtype=np.int8), seed=37)
        observer = RecordingObserver()
        simulator = Simulator(
            env=env,
            agents=(
                Agent(id=0, radius=0, policy=ConstantPolicy(Action(ActionType.BRIDGE, source=0, target=1)), seed=41),
                Agent(id=1, radius=0, policy=ConstantPolicy(Action(ActionType.PASS, source=1)), seed=43),
            ),
            mode=SchedulingMode.SEQUENTIAL,
            max_steps=2,
            rng=np.random.default_rng(47),
            observers=(observer,),
        )

        simulator.run(n_episodes=1)

        assert observer.episode_end_event is not None
        self.assertEqual(observer.episode_end_event.current_step, 2)
        self.assertEqual(observer.episode_end_event.last_modified_step, 0)
        self.assertEqual(simulator.env.state.ownership_matrix[0, 1], 1)
        self.assertTrue(simulator.env.state.graph.are_adjacent(0, 1))

    def test_run_simultaneous_uses_same_pre_step_state(self) -> None:
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
            agents=(
                Agent(id=0, radius=2, policy=add_or_pass_policy, seed=59),
                Agent(id=1, radius=1, policy=ConstantPolicy(Action(ActionType.BRIDGE, source=1, target=2)), seed=61),
                Agent(id=2, radius=0, policy=ConstantPolicy(Action(ActionType.PASS, source=2)), seed=67),
            ),
            mode=SchedulingMode.SIMULTANEOUS,
            max_steps=1,
            rng=np.random.default_rng(71),
            observers=(),
        )

        simulator.run(n_episodes=1)

        self.assertEqual(add_or_pass_policy.observed_node_names, [[0, 1]])
        self.assertEqual(simulator.env.state.ownership_matrix[1, 2], 1)
        self.assertEqual(simulator.env.state.ownership_matrix[0, 2], 0)
        self.assertTrue(simulator.env.state.graph.are_adjacent(1, 2))

    def test_run_applies_multiple_actions_from_one_agent(self) -> None:
        env = Environment(np.zeros((3, 3), dtype=np.int8), seed=72)
        simulator = Simulator(
            env=env,
            agents=(
                Agent(
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
                Agent(id=1, radius=0, policy=ConstantPolicy(Action(ActionType.PASS, source=1)), seed=74),
                Agent(id=2, radius=0, policy=ConstantPolicy(Action(ActionType.PASS, source=2)), seed=75),
            ),
            mode=SchedulingMode.SEQUENTIAL,
            max_steps=1,
            rng=np.random.default_rng(76),
            observers=(),
        )

        simulator.run(n_episodes=1)

        self.assertEqual(simulator.env.state.ownership_matrix[0, 1], 1)
        self.assertEqual(simulator.env.state.ownership_matrix[0, 2], 1)

    def test_reset_restores_environment_agents_and_state(self) -> None:
        left_policy = ConstantPolicy(Action(ActionType.BRIDGE, source=0, target=1))
        right_policy = ConstantPolicy(Action(ActionType.PASS, source=1))
        env = Environment(np.zeros((2, 2), dtype=np.int8), seed=73)
        simulator = Simulator(
            env=env,
            agents=(
                Agent(id=0, radius=0, policy=left_policy, seed=79),
                Agent(id=1, radius=0, policy=right_policy, seed=83),
            ),
            mode=SchedulingMode.SEQUENTIAL,
            max_steps=1,
            rng=np.random.default_rng(89),
            observers=(),
        )

        simulator.run(n_episodes=1)
        simulator.reset()

        self.assertEqual(left_policy.reset_calls, 2)
        self.assertEqual(right_policy.reset_calls, 2)
        self.assertEqual(left_policy.reset_seeds[-1], 79)
        self.assertEqual(right_policy.reset_seeds[-1], 83)
        np.testing.assert_array_equal(
            simulator.env.state.ownership_matrix,
            np.zeros((2, 2), dtype=np.int8),
        )

    def test_run_uses_distinct_policy_seeds_across_episodes_and_run_calls(self) -> None:
        left_policy = ConstantPolicy(Action(ActionType.PASS, source=0))
        right_policy = ConstantPolicy(Action(ActionType.PASS, source=1))
        observer = BatchRecordingObserver()
        env = Environment(np.zeros((2, 2), dtype=np.int8), seed=137)
        simulator = Simulator(
            env=env,
            agents=(
                Agent(id=0, radius=0, policy=left_policy, seed=139),
                Agent(id=1, radius=0, policy=right_policy, seed=149),
            ),
            mode=SchedulingMode.SEQUENTIAL,
            max_steps=1,
            rng=np.random.default_rng(151),
            observers=(observer,),
        )

        simulator.run(n_episodes=2)
        simulator.run(n_episodes=1)

        self.assertEqual(simulator.n_completed_episodes, 3)
        self.assertEqual(len(left_policy.reset_seeds), 3)
        self.assertEqual(len(right_policy.reset_seeds), 3)
        self.assertEqual(len(set(left_policy.reset_seeds)), 3)
        self.assertEqual(len(set(right_policy.reset_seeds)), 3)
        self.assertNotEqual(left_policy.reset_seeds[0], right_policy.reset_seeds[0])
        self.assertEqual(
            [event.episode_index for event in observer.episode_start_events],
            [0, 1, 2],
        )
        self.assertEqual(
            len({event.seed for event in observer.episode_start_events}),
            3,
        )

    def test_run_starts_from_a_fresh_reset_state(self) -> None:
        observer = RecordingObserver()
        env = Environment(np.zeros((2, 2), dtype=np.int8), seed=90)
        simulator = Simulator(
            env=env,
            agents=(
                Agent(id=0, radius=0, policy=ConstantPolicy(Action(ActionType.PASS, source=0)), seed=91),
                Agent(id=1, radius=0, policy=ConstantPolicy(Action(ActionType.PASS, source=1)), seed=92),
            ),
            mode=SchedulingMode.SEQUENTIAL,
            max_steps=1,
            rng=np.random.default_rng(93),
            observers=(observer,),
        )

        env.step([Action(ActionType.BRIDGE, source=0, target=1)])
        simulator.run(n_episodes=1)

        assert observer.episode_start_event is not None
        np.testing.assert_array_equal(
            observer.episode_start_event.initial_state.ownership_matrix,
            np.zeros((2, 2), dtype=np.int8),
        )
        np.testing.assert_array_equal(
            simulator.env.state.ownership_matrix,
            np.zeros((2, 2), dtype=np.int8),
        )
        end_event = observer.episode_end_event
        assert end_event is not None
        self.assertEqual(end_event.current_step, 1)
        self.assertEqual(end_event.last_modified_step, -1)

    def test_observer_receives_episode_and_step_events(self) -> None:
        observer = RecordingObserver()
        env = Environment(np.zeros((2, 2), dtype=np.int8), seed=97)
        simulator = Simulator(
            env=env,
            agents=(
                Agent(id=0, radius=0, policy=ConstantPolicy(Action(ActionType.BRIDGE, source=0, target=1)), seed=101),
                Agent(id=1, radius=0, policy=ConstantPolicy(Action(ActionType.PASS, source=1)), seed=103),
            ),
            mode=SchedulingMode.SEQUENTIAL,
            max_steps=1,
            rng=np.random.default_rng(107),
            observers=(observer,),
        )

        simulator.run(n_episodes=1)

        self.assertIsNotNone(observer.episode_start_event)
        self.assertIsNotNone(observer.episode_end_event)
        self.assertEqual(len(observer.step_events), 1)

        start_event = observer.episode_start_event
        assert start_event is not None
        expected_episode_seed = int(np.random.default_rng(107).bit_generator.random_raw())
        self.assertEqual(start_event.seed, expected_episode_seed)
        self.assertEqual(start_event.episode_index, 0)
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
        self.assertEqual(end_event.seed, expected_episode_seed)
        self.assertEqual(end_event.current_step, 1)
        self.assertEqual(end_event.last_modified_step, 0)
        self.assertTrue(end_event.final_state.graph.are_adjacent(0, 1))

    def test_observer_cannot_mutate_live_state_through_event_snapshots(self) -> None:
        observer = RecordingObserver()
        env = Environment(np.zeros((2, 2), dtype=np.int8), seed=109)
        simulator = Simulator(
            env=env,
            agents=(
                Agent(id=0, radius=0, policy=ConstantPolicy(Action(ActionType.BRIDGE, source=0, target=1)), seed=113),
                Agent(id=1, radius=0, policy=ConstantPolicy(Action(ActionType.PASS, source=1)), seed=127),
            ),
            mode=SchedulingMode.SEQUENTIAL,
            max_steps=1,
            rng=np.random.default_rng(131),
            observers=(observer,),
        )

        simulator.run(n_episodes=1)

        step_event = observer.step_events[0]

        with self.assertRaises(ValueError):
            step_event.post_state.ownership_matrix[0, 1] = 0

        snapshot_graph = step_event.post_state.graph
        snapshot_graph.delete_edges([(0, 1)])

        self.assertTrue(simulator.env.state.graph.are_adjacent(0, 1))


if __name__ == "__main__":
    unittest.main()
