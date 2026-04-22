"""Tests for simulation runtime assembly helpers."""

from __future__ import annotations

import unittest

import numpy as np

from network_creation.core.agent import Agent
from network_creation.core.model import (
    Action,
    ActionSpace,
    ActionType,
    Observation,
    PolicyDecision,
)
from network_creation.core.observability import ObservationSampling
from network_creation.core.runtime import (
    SimulationSettings,
    create_agents,
    create_environment,
    create_simulation_runtime,
    create_simulator,
)
from network_creation.policies.base import Policy
from tests.core.simulation_support import ConstantPolicy, RecordingObserver


class SeedRecordingPolicy(Policy):
    def __init__(self) -> None:
        self.reset_seeds: list[int] = []

    def reset(self, seed: int) -> None:
        self.reset_seeds.append(seed)

    def get_decision(
        self, observation: Observation, action_space: ActionSpace
    ) -> PolicyDecision:
        del observation, action_space
        return PolicyDecision(actions=(Action(ActionType.PASS, source=0),), value=None)


class RuntimeAssemblyTest(unittest.TestCase):
    def test_agent_rejects_bool_visibility_distance(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive integer"):
            Agent(
                id=0,
                visibility_distance=True,
                policy=ConstantPolicy(Action(ActionType.PASS, source=0)),
                seed=11,
            )

    def test_create_environment_rejects_unsupported_initial_graph_kind(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported initial graph kind"):
            create_environment(
                simulation_settings=SimulationSettings(
                    n_agents=2,
                    n_actors=1,
                    seed=7,
                    max_steps=3,
                    n_episodes=1,
                    stagnation_patience=None,
                    initial_graph_kind="cycle",
                ),
                seed=13,
            )

    def test_create_agents_builds_agents_in_node_id_order(self) -> None:
        build_calls: list[int] = []

        def build_agent(node_id: int) -> Agent:
            build_calls.append(node_id)
            return Agent(
                id=node_id,
                visibility_distance=2,
                policy=ConstantPolicy(Action(ActionType.PASS, source=node_id)),
                seed=17 + node_id,
            )

        agents = create_agents(
            simulation_settings=SimulationSettings(
                n_agents=3,
                n_actors=1,
                seed=5,
                max_steps=2,
                n_episodes=1,
                stagnation_patience=None,
            ),
            build_agent=build_agent,
        )

        self.assertEqual(build_calls, [0, 1, 2])
        self.assertEqual(tuple(agent.id for agent in agents), (0, 1, 2))

    def test_create_simulator_preserves_observers_and_stagnation_settings(self) -> None:
        settings = SimulationSettings(
            n_agents=2,
            n_actors=1,
            seed=19,
            max_steps=4,
            n_episodes=2,
            stagnation_patience=3,
        )
        environment = create_environment(simulation_settings=settings, seed=23)
        agents = tuple(
            Agent(
                id=node_id,
                visibility_distance=2,
                policy=ConstantPolicy(Action(ActionType.PASS, source=node_id)),
                seed=29 + node_id,
            )
            for node_id in range(settings.n_agents)
        )
        observer = RecordingObserver()

        simulator = create_simulator(
            simulation_settings=settings,
            environment=environment,
            agents=agents,
            observers=(observer,),
            seed=31,
        )

        self.assertIs(simulator.env, environment)
        self.assertIs(simulator.agents, agents)
        self.assertEqual(simulator.observers, (observer,))
        self.assertEqual(simulator.n_agents, 2)
        self.assertEqual(simulator.max_steps, 4)
        self.assertEqual(simulator.stagnation_patience, 3)

    def test_create_simulation_runtime_propagates_seed_to_environment_and_agents(
        self,
    ) -> None:
        settings = SimulationSettings(
            n_agents=2,
            n_actors=2,
            seed=7,
            max_steps=0,
            n_episodes=1,
            stagnation_patience=None,
        )
        policies: dict[int, SeedRecordingPolicy] = {}
        observer = RecordingObserver()

        def build_agent(node_id: int) -> Agent:
            policy = SeedRecordingPolicy()
            policies[node_id] = policy
            return Agent(
                id=node_id,
                visibility_distance=2,
                policy=policy,
                seed=41 + node_id,
            )

        environment, agents, simulator = create_simulation_runtime(
            simulation_settings=settings,
            build_agent=build_agent,
            observers=(observer,),
            seed=37,
        )

        simulator.run(n_episodes=1, reset_observers=False)

        expected_rng = np.random.default_rng(37)
        expected_episode_seed = int(expected_rng.bit_generator.random_raw())
        expected_agent_seeds = [
            int(expected_rng.bit_generator.random_raw()) for _ in range(len(agents))
        ]

        self.assertEqual(environment.seed, 37)
        self.assertIs(simulator.env, environment)
        self.assertIs(simulator.agents, agents)
        self.assertEqual(simulator.observers, (observer,))
        self.assertIsNotNone(observer.episode_start_event)
        self.assertEqual(observer.episode_start_event.seed, expected_episode_seed)
        self.assertEqual(
            [policies[node_id].reset_seeds for node_id in range(len(agents))],
            [[expected_agent_seeds[0]], [expected_agent_seeds[1]]],
        )


if __name__ == "__main__":
    unittest.main()
