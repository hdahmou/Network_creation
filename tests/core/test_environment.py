"""Tests for Environment, Agent, and simple policy integration."""

from __future__ import annotations

import unittest

import numpy as np

from network_creation.core.agent import Agent
from network_creation.core.environment import Environment
from network_creation.core.model import Action, ActionSpace, ActionType, Observation
from network_creation.policies.base import Policy
from network_creation.policies.random import RandomPolicy


class StubPolicy(Policy):
    def __init__(self, action: Action) -> None:
        self._action = action
        self.reset_calls = 0
        self.reset_seeds: list[int] = []
        self.calls: list[tuple[Observation, ActionSpace]] = []

    def reset(self, seed: int) -> None:
        self.reset_calls += 1
        self.reset_seeds.append(seed)

    def get_actions(
        self, observation: Observation, action_space: ActionSpace
    ) -> list[Action]:
        self.calls.append((observation, action_space))
        return [self._action]


class EnvironmentTest(unittest.TestCase):
    @staticmethod
    def _chain_ownership() -> np.ndarray:
        return np.array(
            [
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            dtype=np.int8,
        )

    def test_get_observation_uses_agent_radius(self) -> None:
        env = Environment(self._chain_ownership(), seed=11)
        agent = Agent(id=0, radius=2, policy=RandomPolicy(seed=17), seed=19)

        observation = env.get_observation(agent)

        self.assertEqual(observation.node_id, 0)
        self.assertEqual(observation.graph.vs["name"], [0, 1, 2])
        np.testing.assert_array_equal(observation.outside_node_ids, np.array([3]))
        np.testing.assert_array_equal(observation.paid_node_ids, np.array([1]))

    def test_get_action_space_composes_environment_observation(self) -> None:
        env = Environment(self._chain_ownership(), seed=23)
        agent = Agent(id=0, radius=2, policy=RandomPolicy(seed=29), seed=31)

        action_space = env.get_action_space(agent)

        self.assertEqual(
            action_space[ActionType.ADD],
            [Action(ActionType.ADD, source=0, target=2)],
        )
        self.assertEqual(
            action_space[ActionType.BRIDGE],
            [Action(ActionType.BRIDGE, source=0, target=3)],
        )
        self.assertEqual(
            action_space[ActionType.DELETE],
            [Action(ActionType.DELETE, source=0, target=1)],
        )
        self.assertEqual(
            action_space[ActionType.PASS],
            [Action(ActionType.PASS, source=0)],
        )

    def test_step_updates_state_and_reset_restores_initial_state(self) -> None:
        env = Environment(np.zeros((3, 3), dtype=np.int8), seed=37)

        env.step([Action(ActionType.BRIDGE, source=0, target=2)])

        self.assertEqual(env.node_ids, [0, 1, 2])
        self.assertEqual(env.state.ownership_matrix[0, 2], 1)
        self.assertTrue(env.state.graph.are_adjacent(0, 2))

        reset_state = env.reset()

        np.testing.assert_array_equal(
            reset_state.ownership_matrix,
            np.zeros((3, 3), dtype=np.int8),
        )
        self.assertFalse(reset_state.graph.are_adjacent(0, 2))


class AgentPolicyIntegrationTest(unittest.TestCase):
    def test_agent_delegates_act_and_reset_to_policy(self) -> None:
        expected_action = Action(ActionType.PASS, source=0)
        policy = StubPolicy(expected_action)
        agent = Agent(id=0, radius=1, policy=policy, seed=41)
        observation = Observation(
            node_id=0,
            graph=Environment(np.zeros((1, 1), dtype=np.int8), seed=43).state.graph,
            outside_node_ids=np.array([], dtype=np.int_),
            paid_node_ids=np.array([], dtype=np.int_),
        )
        action_space = ActionSpace(
            actions={
                ActionType.ADD: [],
                ActionType.BRIDGE: [],
                ActionType.DELETE: [],
                ActionType.PASS: [expected_action],
            }
        )

        actions = agent.act(observation, action_space)
        agent.reset(seed=47)

        self.assertEqual(actions, [expected_action])
        self.assertEqual(policy.calls, [(observation, action_space)])
        self.assertEqual(policy.reset_calls, 1)

    def test_random_policy_returns_available_actions_and_reset_is_deterministic(self) -> None:
        env = Environment(
            np.array(
                [
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                ],
                dtype=np.int8,
            ),
            seed=47,
        )
        agent = Agent(id=0, radius=2, policy=RandomPolicy(seed=7), seed=53)
        observation = env.get_observation(agent)
        action_space = env.get_action_space(agent)
        policy = RandomPolicy(seed=7)
        available_actions = {
            action
            for actions in action_space.actions.values()
            for action in actions
        }

        first_run = [
            policy.get_actions(observation, action_space)[0]
            for _ in range(5)
        ]
        policy.reset(seed=7)
        second_run = [
            policy.get_actions(observation, action_space)[0]
            for _ in range(5)
        ]

        self.assertEqual(first_run, second_run)
        self.assertTrue(all(action in available_actions for action in first_run))


if __name__ == "__main__":
    unittest.main()
