"""Tests for simulator scheduling and episode execution."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from network_creation.core.agent import Agent
from network_creation.core.environment import Environment
from network_creation.core.graph import build_observation_from_state, create_empty_graph
from network_creation.core.model import (
    Action,
    ActionSpace,
    ActionType,
    Observation,
    PolicyDecision,
    ActorScheduler,
)
from network_creation.core.observability import (
    EpisodeEndEvent,
    EpisodeStartEvent,
    ObservationSampling,
    StateView,
    StepEvent,
)
from network_creation.core.simulator import Simulator
from network_creation.policies.base import Policy
from tests.graph_factories import graph_from_adjacency_matrix


def _flatten_decisions(
    decisions: tuple[PolicyDecision, ...],
) -> tuple[Action, ...]:
    return tuple(action for decision in decisions for action in decision.actions)


class ConstantPolicy(Policy):
    def __init__(self, action: Action) -> None:
        self.action = action
        self.reset_calls = 0
        self.reset_seeds: list[int] = []
        self.get_decision_calls = 0

    def reset(self, seed: int) -> None:
        self.reset_calls += 1
        self.reset_seeds.append(seed)

    def get_decision(
        self, observation: Observation, action_space: ActionSpace
    ) -> PolicyDecision:
        del observation
        del action_space
        self.get_decision_calls += 1
        return PolicyDecision(actions=(self.action,), value=None)


class PreferAddElsePassPolicy(Policy):
    def __init__(self) -> None:
        self.reset_calls = 0
        self.reset_seeds: list[int] = []
        self.observed_node_names: list[list[int]] = []

    def reset(self, seed: int) -> None:
        self.reset_calls += 1
        self.reset_seeds.append(seed)

    def get_decision(
        self, observation: Observation, action_space: ActionSpace
    ) -> PolicyDecision:
        self.observed_node_names.append(list(observation.graph.vs["name"]))
        if action_space[ActionType.ADD]:
            return PolicyDecision(
                actions=(action_space[ActionType.ADD][0],),
                value=None,
            )
        return PolicyDecision(
            actions=(action_space[ActionType.PASS][0],),
            value=None,
        )


class RecordingObserver:
    def __init__(self) -> None:
        self.observation_sampling = ObservationSampling.periodic(1)
        self.episode_start_event: EpisodeStartEvent | None = None
        self.step_events: list[StepEvent] = []
        self.episode_end_event: EpisodeEndEvent | None = None

    def on_episode_start(self, event: EpisodeStartEvent) -> None:
        self.episode_start_event = event

    def observes_step(self, step_index: int, max_steps: int) -> bool:
        return self.observation_sampling.samples_step(step_index, max_steps)

    def on_step_end(self, event: StepEvent) -> None:
        self.step_events.append(event)

    def on_episode_end(self, event: EpisodeEndEvent) -> None:
        self.episode_end_event = event


class BatchRecordingObserver:
    def __init__(self) -> None:
        self.observation_sampling = ObservationSampling.periodic(1)
        self.episode_start_events: list[EpisodeStartEvent] = []
        self.episode_end_events: list[EpisodeEndEvent] = []

    def reset(self) -> None:
        self.episode_start_events = []
        self.episode_end_events = []

    def on_episode_start(self, event: EpisodeStartEvent) -> None:
        self.episode_start_events.append(event)

    def observes_step(self, step_index: int, max_steps: int) -> bool:
        return self.observation_sampling.samples_step(step_index, max_steps)

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

    def get_decision(
        self, observation: Observation, action_space: ActionSpace
    ) -> PolicyDecision:
        del observation
        del action_space
        return PolicyDecision(actions=tuple(self.actions), value=None)


class ScriptedPolicy(Policy):
    def __init__(self, actions_by_call: list[list[Action]]) -> None:
        self.actions_by_call = [list(actions) for actions in actions_by_call]
        self.reset_calls = 0
        self.reset_seeds: list[int] = []
        self.get_decision_calls = 0
        self._next_call_idx = 0

    def reset(self, seed: int) -> None:
        self.reset_calls += 1
        self.reset_seeds.append(seed)
        self._next_call_idx = 0

    def get_decision(
        self, observation: Observation, action_space: ActionSpace
    ) -> PolicyDecision:
        del observation
        del action_space
        if self._next_call_idx >= len(self.actions_by_call):
            raise AssertionError("policy was queried more times than expected")

        self.get_decision_calls += 1
        actions = list(self.actions_by_call[self._next_call_idx])
        self._next_call_idx += 1
        return PolicyDecision(actions=tuple(actions), value=None)


class SimulatorTest(unittest.TestCase):
    def test_run_sequential_tracks_final_state(self) -> None:
        env = Environment(create_empty_graph(2), seed=37)
        observer = RecordingObserver()
        simulator = Simulator(
            env=env,
            agents=(
                Agent(id=0, visibility_fraction=1, policy=ConstantPolicy(Action(ActionType.BRIDGE, source=0, target=1)), seed=41),
                Agent(id=1, visibility_fraction=1, policy=ConstantPolicy(Action(ActionType.PASS, source=1)), seed=43),
            ),
            actor_scheduler=ActorScheduler(n_agents=2, fraction=0.5, seed=21),
            max_steps=2,
            rng=np.random.default_rng(47),
            observers=(observer,),
        )

        simulator.run(n_episodes=1, reset_observers=False)

        assert observer.episode_end_event is not None
        self.assertEqual(observer.episode_end_event.current_step, 2)
        self.assertEqual(observer.episode_end_event.last_modified_step, 0)
        self.assertEqual(simulator.env.state.graph.get_edgelist(), [(0, 1)])
        self.assertTrue(simulator.env.state.graph.are_adjacent(0, 1))

    def test_step_state_views_are_skipped_when_no_observer_samples_step(self) -> None:
        env = Environment(create_empty_graph(2), seed=45)
        observer = RecordingObserver()
        observer.observation_sampling = ObservationSampling.periodic(2)
        simulator = Simulator(
            env=env,
            agents=(
                Agent(
                    id=0,
                    visibility_fraction=1,
                    policy=ConstantPolicy(Action(ActionType.BRIDGE, source=0, target=1)),
                    seed=47,
                ),
                Agent(
                    id=1,
                    visibility_fraction=1,
                    policy=ConstantPolicy(Action(ActionType.PASS, source=1)),
                    seed=49,
                ),
            ),
            actor_scheduler=ActorScheduler(n_agents=2, fraction=0.5, seed=21),
            max_steps=1,
            rng=np.random.default_rng(51),
            observers=(observer,),
        )

        with patch(
            "network_creation.core.simulator.StateView.from_state",
            wraps=StateView.from_state,
        ) as from_state:
            simulator.run(n_episodes=1, reset_observers=False)

        self.assertEqual(from_state.call_count, 2)
        self.assertEqual(observer.step_events, [])

    def test_terminal_observer_only_receives_last_step_event(self) -> None:
        env = Environment(create_empty_graph(2), seed=45)
        observer = RecordingObserver()
        observer.observation_sampling = ObservationSampling.terminal()
        simulator = Simulator(
            env=env,
            agents=(
                Agent(
                    id=0,
                    visibility_fraction=1,
                    policy=ConstantPolicy(Action(ActionType.BRIDGE, source=0, target=1)),
                    seed=47,
                ),
                Agent(
                    id=1,
                    visibility_fraction=1,
                    policy=ConstantPolicy(Action(ActionType.PASS, source=1)),
                    seed=49,
                ),
            ),
            actor_scheduler=ActorScheduler(n_agents=2, fraction=0.5, seed=21),
            max_steps=2,
            rng=np.random.default_rng(51),
            observers=(observer,),
        )

        simulator.run(n_episodes=1, reset_observers=False)

        self.assertEqual([event.step for event in observer.step_events], [1])

    def test_periodic_and_terminal_observers_share_step_sampling_fast_path(self) -> None:
        env = Environment(create_empty_graph(2), seed=45)
        periodic_observer = RecordingObserver()
        periodic_observer.observation_sampling = ObservationSampling.periodic(2)
        terminal_observer = RecordingObserver()
        terminal_observer.observation_sampling = ObservationSampling.terminal()
        simulator = Simulator(
            env=env,
            agents=(
                Agent(
                    id=0,
                    visibility_fraction=1,
                    policy=ConstantPolicy(Action(ActionType.BRIDGE, source=0, target=1)),
                    seed=47,
                ),
                Agent(
                    id=1,
                    visibility_fraction=1,
                    policy=ConstantPolicy(Action(ActionType.PASS, source=1)),
                    seed=49,
                ),
            ),
            actor_scheduler=ActorScheduler(n_agents=2, fraction=0.5, seed=21),
            max_steps=3,
            rng=np.random.default_rng(51),
            observers=(periodic_observer, terminal_observer),
        )

        with patch(
            "network_creation.core.simulator.StateView.from_state",
            wraps=StateView.from_state,
        ) as from_state:
            simulator.run(n_episodes=1, reset_observers=False)

        self.assertEqual(from_state.call_count, 6)
        self.assertEqual([event.step for event in periodic_observer.step_events], [1])
        self.assertEqual([event.step for event in terminal_observer.step_events], [2])

    def test_run_builds_observations_from_agent_visibility_fractions(self) -> None:
        env = Environment(
            graph_from_adjacency_matrix(
                np.array(
                    [
                        [0, 1],
                        [0, 0],
                    ],
                    dtype=np.int8,
                )
            ),
            seed=52,
        )
        simulator = Simulator(
            env=env,
            agents=(
                Agent(
                    id=0,
                    visibility_fraction=1.0,
                    policy=ConstantPolicy(Action(ActionType.PASS, source=0)),
                    seed=53,
                ),
                Agent(
                    id=1,
                    visibility_fraction=0.25,
                    policy=ConstantPolicy(Action(ActionType.PASS, source=1)),
                    seed=54,
                ),
            ),
            actor_scheduler=ActorScheduler(n_agents=2, fraction=1.0, seed=21),
            max_steps=1,
            rng=np.random.default_rng(55),
            observers=(),
        )

        with patch(
            "network_creation.core.environment.build_observation_from_state",
            wraps=build_observation_from_state,
        ) as build_observation:
            simulator.run(n_episodes=1, reset_observers=False)

        self.assertEqual(
            [call.kwargs["radius"] for call in build_observation.call_args_list],
            [2, 1],
        )

    def test_run_simultaneous_uses_same_pre_step_state(self) -> None:
        env = Environment(
            graph_from_adjacency_matrix(
                np.array(
                    [
                        [0, 1, 0],
                        [0, 0, 0],
                        [0, 0, 0],
                    ],
                    dtype=np.int8,
                )
            ),
            seed=53,
        )
        add_or_pass_policy = PreferAddElsePassPolicy()
        simulator = Simulator(
            env=env,
            agents=(
                Agent(id=0, visibility_fraction=1.0, policy=add_or_pass_policy, seed=59),
                Agent(id=1, visibility_fraction=1, policy=ConstantPolicy(Action(ActionType.BRIDGE, source=1, target=2)), seed=61),
                Agent(id=2, visibility_fraction=1, policy=ConstantPolicy(Action(ActionType.PASS, source=2)), seed=67),
            ),
            actor_scheduler=ActorScheduler(n_agents=3, fraction=1.0, seed=21),
            max_steps=1,
            rng=np.random.default_rng(71),
            observers=(),
        )

        simulator.run(n_episodes=1, reset_observers=False)

        self.assertEqual(add_or_pass_policy.observed_node_names, [[0, 1]])
        self.assertEqual(sorted(simulator.env.state.graph.get_edgelist()), [(0, 1), (1, 2)])
        self.assertTrue(simulator.env.state.graph.are_adjacent(1, 2))

    def test_run_applies_multiple_actions_from_one_agent(self) -> None:
        env = Environment(create_empty_graph(3), seed=72)
        simulator = Simulator(
            env=env,
            agents=(
                Agent(
                    id=0,
                    visibility_fraction=1.0,
                    policy=MultiActionPolicy(
                        [
                            Action(ActionType.BRIDGE, source=0, target=1),
                            Action(ActionType.BRIDGE, source=0, target=2),
                        ]
                    ),
                    seed=73,
                ),
                Agent(id=1, visibility_fraction=1, policy=ConstantPolicy(Action(ActionType.PASS, source=1)), seed=74),
                Agent(id=2, visibility_fraction=1, policy=ConstantPolicy(Action(ActionType.PASS, source=2)), seed=75),
            ),
            actor_scheduler=ActorScheduler(n_agents=2, fraction=0.5, seed=21),
            max_steps=1,
            rng=np.random.default_rng(76),
            observers=(),
        )

        simulator.run(n_episodes=1, reset_observers=False)

        self.assertEqual(sorted(simulator.env.state.graph.get_edgelist()), [(0, 1), (0, 2)])

    def test_reset_restores_environment_agents_and_state(self) -> None:
        left_policy = ConstantPolicy(Action(ActionType.BRIDGE, source=0, target=1))
        right_policy = ConstantPolicy(Action(ActionType.PASS, source=1))
        env = Environment(create_empty_graph(2), seed=73)
        simulator = Simulator(
            env=env,
            agents=(
                Agent(id=0, visibility_fraction=1, policy=left_policy, seed=79),
                Agent(id=1, visibility_fraction=1, policy=right_policy, seed=83),
            ),
            actor_scheduler=ActorScheduler(n_agents=2, fraction=0.5, seed=21),
            max_steps=1,
            rng=np.random.default_rng(89),
            observers=(),
        )

        simulator.run(n_episodes=1, reset_observers=False)
        simulator.reset()

        self.assertEqual(left_policy.reset_calls, 2)
        self.assertEqual(right_policy.reset_calls, 2)
        self.assertEqual(left_policy.reset_seeds[-1], 79)
        self.assertEqual(right_policy.reset_seeds[-1], 83)
        self.assertEqual(simulator.env.state.graph.ecount(), 0)

    def test_run_uses_distinct_policy_seeds_across_episodes_and_run_calls(self) -> None:
        left_policy = ConstantPolicy(Action(ActionType.PASS, source=0))
        right_policy = ConstantPolicy(Action(ActionType.PASS, source=1))
        observer = BatchRecordingObserver()
        env = Environment(create_empty_graph(2), seed=137)
        simulator = Simulator(
            env=env,
            agents=(
                Agent(id=0, visibility_fraction=1, policy=left_policy, seed=139),
                Agent(id=1, visibility_fraction=1, policy=right_policy, seed=149),
            ),
            actor_scheduler=ActorScheduler(n_agents=2, fraction=0.5, seed=21),
            max_steps=1,
            rng=np.random.default_rng(151),
            observers=(observer,),
        )

        simulator.run(n_episodes=2, reset_observers=False)
        simulator.run(n_episodes=1, reset_observers=False)

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

    def test_run_can_reset_observers_before_collecting_new_batch(self) -> None:
        observer = BatchRecordingObserver()
        env = Environment(create_empty_graph(2), seed=152)
        simulator = Simulator(
            env=env,
            agents=(
                Agent(id=0, visibility_fraction=1, policy=ConstantPolicy(Action(ActionType.PASS, source=0)), seed=153),
                Agent(id=1, visibility_fraction=1, policy=ConstantPolicy(Action(ActionType.PASS, source=1)), seed=154),
            ),
            actor_scheduler=ActorScheduler(n_agents=2, fraction=0.5, seed=21),
            max_steps=1,
            rng=np.random.default_rng(155),
            observers=(observer,),
        )

        simulator.run(n_episodes=2, reset_observers=False)
        simulator.run(n_episodes=1, reset_observers=True)

        self.assertEqual(simulator.n_completed_episodes, 3)
        self.assertEqual(
            [event.episode_index for event in observer.episode_start_events],
            [2],
        )
        self.assertEqual(
            [event.episode_index for event in observer.episode_end_events],
            [2],
        )

    def test_run_starts_from_a_fresh_reset_state(self) -> None:
        observer = RecordingObserver()
        env = Environment(create_empty_graph(2), seed=90)
        simulator = Simulator(
            env=env,
            agents=(
                Agent(id=0, visibility_fraction=1, policy=ConstantPolicy(Action(ActionType.PASS, source=0)), seed=91),
                Agent(id=1, visibility_fraction=1, policy=ConstantPolicy(Action(ActionType.PASS, source=1)), seed=92),
            ),
            actor_scheduler=ActorScheduler(n_agents=2, fraction=0.5, seed=21),
            max_steps=1,
            rng=np.random.default_rng(93),
            observers=(observer,),
        )

        env.step([Action(ActionType.BRIDGE, source=0, target=1)])
        simulator.run(n_episodes=1, reset_observers=False)

        assert observer.episode_start_event is not None
        self.assertEqual(observer.episode_start_event.initial_state.graph.ecount(), 0)
        self.assertEqual(simulator.env.state.graph.ecount(), 0)
        end_event = observer.episode_end_event
        assert end_event is not None
        self.assertEqual(end_event.current_step, 1)
        self.assertEqual(end_event.last_modified_step, -1)

    def test_observer_receives_episode_and_step_events(self) -> None:
        observer = RecordingObserver()
        env = Environment(create_empty_graph(2), seed=97)
        simulator = Simulator(
            env=env,
            agents=(
                Agent(id=0, visibility_fraction=1, policy=ConstantPolicy(Action(ActionType.BRIDGE, source=0, target=1)), seed=101),
                Agent(id=1, visibility_fraction=1, policy=ConstantPolicy(Action(ActionType.PASS, source=1)), seed=103),
            ),
            actor_scheduler=ActorScheduler(n_agents=2, fraction=0.5, seed=21),
            max_steps=1,
            rng=np.random.default_rng(107),
            observers=(observer,),
        )

        simulator.run(n_episodes=1, reset_observers=False)

        self.assertIsNotNone(observer.episode_start_event)
        self.assertIsNotNone(observer.episode_end_event)
        self.assertEqual(len(observer.step_events), 1)

        start_event = observer.episode_start_event
        assert start_event is not None
        expected_episode_seed = int(np.random.default_rng(107).bit_generator.random_raw())
        self.assertEqual(start_event.seed, expected_episode_seed)
        self.assertEqual(start_event.episode_index, 0)
        self.assertEqual(start_event.n_agents, 2)
        self.assertEqual(start_event.initial_state.graph.ecount(), 0)

        step_event = observer.step_events[0]
        self.assertEqual(step_event.step, 0)
        self.assertEqual(
            step_event.decisions,
            (
                PolicyDecision(
                    actions=(Action(ActionType.BRIDGE, source=0, target=1),),
                    value=None,
                ),
            ),
        )
        self.assertEqual(
            _flatten_decisions(step_event.decisions),
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

    def test_observer_event_graphs_are_live_references(self) -> None:
        observer = RecordingObserver()
        env = Environment(create_empty_graph(2), seed=109)
        simulator = Simulator(
            env=env,
            agents=(
                Agent(id=0, visibility_fraction=1, policy=ConstantPolicy(Action(ActionType.BRIDGE, source=0, target=1)), seed=113),
                Agent(id=1, visibility_fraction=1, policy=ConstantPolicy(Action(ActionType.PASS, source=1)), seed=127),
            ),
            actor_scheduler=ActorScheduler(n_agents=2, fraction=0.5, seed=21),
            max_steps=1,
            rng=np.random.default_rng(131),
            observers=(observer,),
        )

        simulator.run(n_episodes=1, reset_observers=False)

        start_event = observer.episode_start_event
        step_event = observer.step_events[0]
        end_event = observer.episode_end_event
        assert start_event is not None
        assert end_event is not None
        self.assertIs(start_event.initial_state.graph, step_event.pre_state.graph)
        self.assertIs(step_event.post_state.graph, simulator.env.state.graph)
        self.assertIs(end_event.final_state.graph, simulator.env.state.graph)

        step_event.post_state.graph.delete_edges([(0, 1)])
        self.assertFalse(simulator.env.state.graph.are_adjacent(0, 1))

    def test_stagnation_patience_must_be_positive_when_provided(self) -> None:
        env = Environment(create_empty_graph(1), seed=173)

        with self.assertRaises(ValueError):
            Simulator(
                env=env,
                agents=(
                    Agent(
                        id=0,
                        visibility_fraction=1,
                        policy=ConstantPolicy(Action(ActionType.PASS, source=0)),
                        seed=179,
                    ),
                ),
                actor_scheduler=ActorScheduler(n_agents=1, fraction=1.0, seed=21),
                max_steps=1,
                rng=np.random.default_rng(181),
                observers=(),
                stagnation_patience=0,
            )

    def test_stagnation_skips_policy_queries_but_keeps_step_events_to_max_steps(self) -> None:
        left_policy = ConstantPolicy(Action(ActionType.PASS, source=0))
        right_policy = ConstantPolicy(Action(ActionType.PASS, source=1))
        observer = RecordingObserver()
        env = Environment(create_empty_graph(2), seed=183)
        simulator = Simulator(
            env=env,
            agents=(
                Agent(id=0, visibility_fraction=1, policy=left_policy, seed=191),
                Agent(id=1, visibility_fraction=1, policy=right_policy, seed=193),
            ),
            actor_scheduler=ActorScheduler(n_agents=2, fraction=0.5, seed=21),
            max_steps=5,
            rng=np.random.default_rng(197),
            observers=(observer,),
            stagnation_patience=2,
        )

        simulator.run(n_episodes=1, reset_observers=False)

        self.assertEqual(left_policy.get_decision_calls, 1)
        self.assertEqual(right_policy.get_decision_calls, 1)
        self.assertEqual(
            [event.decisions for event in observer.step_events],
            [
                (
                    PolicyDecision(
                        actions=(Action(ActionType.PASS, source=0),),
                        value=None,
                    ),
                ),
                (
                    PolicyDecision(
                        actions=(Action(ActionType.PASS, source=1),),
                        value=None,
                    ),
                ),
                (),
                (),
                (),
            ],
        )
        self.assertTrue(all(not event.modified for event in observer.step_events))
        self.assertTrue(
            all(
                sorted(event.pre_state.graph.get_edgelist())
                == sorted(event.post_state.graph.get_edgelist())
                for event in observer.step_events
            )
        )
        end_event = observer.episode_end_event
        assert end_event is not None
        self.assertEqual(end_event.current_step, 5)
        self.assertEqual(end_event.last_modified_step, -1)

    def test_stagnation_counter_resets_after_modifying_step(self) -> None:
        left_policy = ScriptedPolicy(
            [
                [Action(ActionType.BRIDGE, source=0, target=1)],
                [Action(ActionType.DELETE, source=0, target=1)],
                [Action(ActionType.PASS, source=0)],
            ]
        )
        right_policy = ConstantPolicy(Action(ActionType.PASS, source=1))
        observer = RecordingObserver()
        env = Environment(create_empty_graph(2), seed=199)
        simulator = Simulator(
            env=env,
            agents=(
                Agent(id=0, visibility_fraction=1, policy=left_policy, seed=211),
                Agent(id=1, visibility_fraction=1, policy=right_policy, seed=223),
            ),
            actor_scheduler=ActorScheduler(n_agents=2, fraction=0.5, seed=21),
            max_steps=6,
            rng=np.random.default_rng(227),
            observers=(observer,),
            stagnation_patience=2,
        )

        simulator.run(n_episodes=1, reset_observers=False)

        self.assertEqual(left_policy.get_decision_calls, 3)
        self.assertEqual(right_policy.get_decision_calls, 2)
        self.assertEqual(
            [event.decisions for event in observer.step_events],
            [
                (
                    PolicyDecision(
                        actions=(Action(ActionType.BRIDGE, source=0, target=1),),
                        value=None,
                    ),
                ),
                (
                    PolicyDecision(
                        actions=(Action(ActionType.PASS, source=1),),
                        value=None,
                    ),
                ),
                (
                    PolicyDecision(
                        actions=(Action(ActionType.DELETE, source=0, target=1),),
                        value=None,
                    ),
                ),
                (
                    PolicyDecision(
                        actions=(Action(ActionType.PASS, source=1),),
                        value=None,
                    ),
                ),
                (
                    PolicyDecision(
                        actions=(Action(ActionType.PASS, source=0),),
                        value=None,
                    ),
                ),
                (),
            ],
        )
        self.assertEqual(
            [event.modified for event in observer.step_events],
            [True, False, True, False, False, False],
        )
        end_event = observer.episode_end_event
        assert end_event is not None
        self.assertEqual(end_event.current_step, 6)
        self.assertEqual(end_event.last_modified_step, 2)


if __name__ == "__main__":
    unittest.main()
