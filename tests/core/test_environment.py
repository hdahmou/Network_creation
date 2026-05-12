"""Tests for Environment, Agent, and simple policy integration."""

from __future__ import annotations

import unittest

import numpy as np

from network_creation.core.agent import Agent
from network_creation.core.environment import Environment
from network_creation.core.graph import (
    build_observation_from_state,
    build_state_from_graph,
    create_empty_graph,
    get_action_space,
)
from network_creation.core.model import (
    Action,
    ActionSpace,
    ActionType,
    Observation,
    PolicyDecision,
)
from network_creation.policies.base import Policy
from network_creation.policies.random import RandomPolicy
from network_creation.policies.threshold_link import ThresholdLinkPolicy
from tests.graph_factories import graph_from_adjacency_matrix


class StubPolicy(Policy):
    def __init__(self, action: Action) -> None:
        self._action = action
        self.reset_calls = 0
        self.reset_seeds: list[int] = []
        self.calls: list[tuple[Observation, ActionSpace]] = []

    def reset(self, seed: int) -> None:
        self.reset_calls += 1
        self.reset_seeds.append(seed)

    def get_decision(
        self, observation: Observation, action_space: ActionSpace
    ) -> PolicyDecision:
        self.calls.append((observation, action_space))
        return PolicyDecision(actions=(self._action,), value=None)


class EnvironmentTest(unittest.TestCase):
    @staticmethod
    def _chain_adjacency() -> np.ndarray:
        return np.array(
            [
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            dtype=np.int8,
        )

    def test_get_observation_uses_supplied_visibility_distance(self) -> None:
        env = Environment(graph_from_adjacency_matrix(self._chain_adjacency()), seed=11)
        agent = Agent(id=0, visibility_distance=2, policy=RandomPolicy(seed=17), seed=19)

        observation = env.get_observation(agent)

        self.assertEqual(observation.node_idx, 0)
        self.assertEqual(observation.graph.vs["name"], [0, 1, 2])
        np.testing.assert_array_equal(observation.outside_node_idxs, np.array([3]))

    def test_get_observation_uses_one_hop_visibility_distance(self) -> None:
        env = Environment(graph_from_adjacency_matrix(self._chain_adjacency()), seed=11)
        agent = Agent(id=0, visibility_distance=1, policy=RandomPolicy(seed=17), seed=19)

        observation = env.get_observation(agent)

        self.assertEqual(observation.graph.vs["name"], [0, 1])
        np.testing.assert_array_equal(observation.outside_node_idxs, np.array([2, 3]))

    def test_get_observation_supports_distance_larger_than_graph_diameter(self) -> None:
        env = Environment(graph_from_adjacency_matrix(self._chain_adjacency()), seed=11)
        agent = Agent(
            id=0, visibility_distance=10, policy=RandomPolicy(seed=17), seed=19
        )

        observation = env.get_observation(agent)

        self.assertEqual(observation.graph.vs["name"], [0, 1, 2])
        np.testing.assert_array_equal(observation.outside_node_idxs, np.array([3]))

    def test_get_action_space_composes_environment_observation(self) -> None:
        env = Environment(graph_from_adjacency_matrix(self._chain_adjacency()), seed=23)
        agent = Agent(id=0, visibility_distance=2, policy=RandomPolicy(seed=29), seed=31)

        action_space = env.get_action_space(agent)

        self.assertEqual(
            action_space[ActionType.ADD],
            [Action(ActionType.ADD, source=0, target=2)],
        )
        self.assertEqual(
            action_space[ActionType.BRIDGE],
            [Action(ActionType.BRIDGE, source=0)],
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
        env = Environment(create_empty_graph(3), seed=37)

        env.step([Action(ActionType.BRIDGE, source=0, target=2)])

        self.assertEqual(env.node_ids, [0, 1, 2])
        self.assertEqual(sorted(env.state.graph.get_edgelist()), [(0, 2)])
        self.assertTrue(env.state.graph.are_adjacent(0, 2))

        reset_state = env.reset()

        self.assertEqual(reset_state.graph.ecount(), 0)
        self.assertFalse(reset_state.graph.are_adjacent(0, 2))


class AgentPolicyIntegrationTest(unittest.TestCase):
    @staticmethod
    def _empty_observation(n_agents: int = 3) -> Observation:
        graph = create_empty_graph(n_agents)
        return Observation(
            node_idx=0,
            graph=graph,
            outside_node_idxs=np.array([], dtype=np.int_),
            local_global=Observation.create_local_global_map(
                graph,
                np.array([], dtype=np.int_),
            ),
        )

    @staticmethod
    def _random_policy_action_space() -> ActionSpace:
        return ActionSpace(
            actions={
                ActionType.ADD: [Action(ActionType.ADD, source=0, target=1)],
                ActionType.BRIDGE: [Action(ActionType.BRIDGE, source=0)],
                ActionType.DELETE: [Action(ActionType.DELETE, source=0, target=2)],
                ActionType.PASS: [Action(ActionType.PASS, source=0)],
            }
        )

    def test_agent_rejects_non_integer_visibility_distance(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive integer"):
            Agent(
                id=0,
                visibility_distance=0.5,
                policy=StubPolicy(Action(ActionType.PASS, source=0)),
                seed=41,
            )

    def test_agent_rejects_non_positive_visibility_distance(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive integer"):
            Agent(
                id=0,
                visibility_distance=0,
                policy=StubPolicy(Action(ActionType.PASS, source=0)),
                seed=41,
            )

    def test_agent_get_visibility_distance_returns_stored_visibility_distance(self) -> None:
        agent = Agent(
            id=0,
            visibility_distance=2,
            policy=StubPolicy(Action(ActionType.PASS, source=0)),
            seed=41,
        )

        self.assertEqual(agent.get_visibility_distance(), 2)

    def test_agent_delegates_act_and_reset_to_policy(self) -> None:
        expected_action = Action(ActionType.PASS, source=0)
        policy = StubPolicy(expected_action)
        agent = Agent(id=0, visibility_distance=1, policy=policy, seed=41)
        graph = Environment(create_empty_graph(1), seed=43).state.graph
        observation = Observation(
            node_idx=0,
            graph=graph,
            outside_node_idxs=np.array([], dtype=np.int_),
            local_global=Observation.create_local_global_map(
                graph,
                np.array([], dtype=np.int_),
            ),
        )
        action_space = ActionSpace(
            actions={
                ActionType.ADD: [],
                ActionType.BRIDGE: [],
                ActionType.DELETE: [],
                ActionType.PASS: [expected_action],
            }
        )

        decision = agent.act(observation, action_space)
        agent.reset(seed=47)

        self.assertEqual(
            decision,
            PolicyDecision(actions=(expected_action,), value=None),
        )
        self.assertEqual(policy.calls, [(observation, action_space)])
        self.assertEqual(policy.reset_calls, 1)

    def test_random_policy_returns_available_actions_and_reset_is_deterministic(
        self,
    ) -> None:
        env = Environment(
            graph_from_adjacency_matrix(
                np.array(
                    [
                        [0, 1, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                    ],
                    dtype=np.int8,
                )
            ),
            seed=47,
        )
        agent = Agent(
            id=0, visibility_distance=2, policy=RandomPolicy(seed=7), seed=53
        )
        observation = env.get_observation(agent)
        action_space = env.get_action_space(agent)
        policy = RandomPolicy(seed=7)
        local_actions = {
            action for actions in action_space.actions.values() for action in actions
        }
        available_actions = {
            Action(
                action.type,
                observation.global_from_local(action.source),
                None
                if action.target is None
                else observation.global_from_local(action.target),
            )
            for action in local_actions
        }

        first_run = [
            policy.get_decision(observation, action_space).actions[0] for _ in range(5)
        ]
        policy.reset(seed=7)
        second_run = [
            policy.get_decision(observation, action_space).actions[0] for _ in range(5)
        ]

        self.assertEqual(first_run, second_run)
        self.assertTrue(all(action in available_actions for action in first_run))

    def test_random_policy_returns_local_actions_from_local_action_space(self) -> None:
        state = build_state_from_graph(
            graph_from_adjacency_matrix(
                np.array(
                    [
                        [0, 1, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0, 1],
                        [0, 0, 1, 0, 1, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                    ],
                    dtype=np.int8,
                )
            )
        )
        observation = build_observation_from_state(state, node_id=4, radius=2)
        action_space = get_action_space(observation)
        local_actions = {
            action for actions in action_space.actions.values() for action in actions
        }
        global_actions = {
            Action(
                action.type,
                observation.global_from_local(action.source),
                None
                if action.target is None
                else observation.global_from_local(action.target),
            )
            for action in local_actions
        }
        policy = RandomPolicy(seed=7)

        samples = [
            policy.get_decision(observation, action_space).actions[0]
            for _ in range(20)
        ]

        self.assertNotEqual(local_actions, global_actions)
        self.assertTrue(all(action in local_actions for action in samples))

    def test_random_policy_without_p_act_preserves_legacy_uniform_action_sampling(
        self,
    ) -> None:
        action_space = self._random_policy_action_space()
        policy = RandomPolicy(seed=7)

        samples = [
            policy.get_decision(
                self._empty_observation(),
                action_space,
            ).actions[0]
            for _ in range(8)
        ]

        self.assertIsNone(policy.p_act)
        self.assertEqual(
            samples,
            [
                Action(ActionType.BRIDGE, source=0),
                Action(ActionType.PASS, source=0),
                Action(ActionType.PASS, source=0),
                Action(ActionType.BRIDGE, source=0),
                Action(ActionType.PASS, source=0),
                Action(ActionType.BRIDGE, source=0),
                Action(ActionType.BRIDGE, source=0),
                Action(ActionType.ADD, source=0, target=1),
            ],
        )

    def test_random_policy_rejects_invalid_p_act(self) -> None:
        for invalid_p_act in (-0.1, 1.1, True):
            with self.subTest(p_act=invalid_p_act):
                with self.assertRaisesRegex(ValueError, "p_act"):
                    RandomPolicy(seed=7, p_act=invalid_p_act)

    def test_random_policy_p_act_zero_always_passes(self) -> None:
        action_space = self._random_policy_action_space()
        policy = RandomPolicy(seed=7, p_act=0.0)

        samples = [
            policy.get_decision(
                self._empty_observation(),
                action_space,
            ).actions[0]
            for _ in range(8)
        ]

        self.assertEqual(policy.p_act, 0.0)
        self.assertEqual(samples, [Action(ActionType.PASS, source=0)] * 8)

    def test_random_policy_p_act_one_samples_non_pass_actions(self) -> None:
        action_space = self._random_policy_action_space()
        policy = RandomPolicy(seed=7, p_act=1.0)

        samples = [
            policy.get_decision(
                self._empty_observation(),
                action_space,
            ).actions[0]
            for _ in range(20)
        ]

        self.assertTrue(all(action.type is not ActionType.PASS for action in samples))

    def test_random_policy_p_act_one_can_sample_bridge_action(self) -> None:
        bridge_action = Action(ActionType.BRIDGE, source=0)
        action_space = ActionSpace(
            actions={
                ActionType.ADD: [],
                ActionType.BRIDGE: [bridge_action],
                ActionType.DELETE: [],
                ActionType.PASS: [Action(ActionType.PASS, source=0)],
            }
        )
        policy = RandomPolicy(seed=7, p_act=1.0)

        decision = policy.get_decision(
            self._empty_observation(),
            action_space,
        )

        self.assertEqual(decision.actions, (bridge_action,))

    def test_random_policy_p_act_falls_back_to_pass_when_no_non_pass_actions(
        self,
    ) -> None:
        pass_action = Action(ActionType.PASS, source=0)
        action_space = ActionSpace(
            actions={
                ActionType.ADD: [],
                ActionType.BRIDGE: [],
                ActionType.DELETE: [],
                ActionType.PASS: [pass_action],
            }
        )
        policy = RandomPolicy(seed=7, p_act=1.0)

        decision = policy.get_decision(
            self._empty_observation(n_agents=1),
            action_space,
        )

        self.assertEqual(decision.actions, (pass_action,))

    def test_random_policy_with_p_act_reset_is_deterministic(self) -> None:
        action_space = self._random_policy_action_space()
        policy = RandomPolicy(seed=7, p_act=0.5)
        observation = self._empty_observation()

        first_run = [
            policy.get_decision(observation, action_space).actions[0] for _ in range(8)
        ]
        policy.reset(seed=7)
        second_run = [
            policy.get_decision(observation, action_space).actions[0] for _ in range(8)
        ]

        self.assertEqual(first_run, second_run)

    def test_threshold_link_policy_rejects_invalid_parameters(self) -> None:
        for invalid_threshold in (-0.1, 1.1, True):
            with self.subTest(link_threshold=invalid_threshold):
                with self.assertRaisesRegex(ValueError, "link_threshold"):
                    ThresholdLinkPolicy(seed=7, link_threshold=invalid_threshold)

        for invalid_p_act in (-0.1, 1.1, True):
            with self.subTest(p_act=invalid_p_act):
                with self.assertRaisesRegex(ValueError, "p_act"):
                    ThresholdLinkPolicy(
                        seed=7,
                        link_threshold=0.5,
                        p_act=invalid_p_act,
                    )

    def test_threshold_link_policy_p_act_zero_always_passes(self) -> None:
        pass_action = Action(ActionType.PASS, source=0)
        action_space = ActionSpace(
            actions={
                ActionType.ADD: [Action(ActionType.ADD, source=0, target=1)],
                ActionType.BRIDGE: [Action(ActionType.BRIDGE, source=0)],
                ActionType.DELETE: [],
                ActionType.PASS: [pass_action],
            }
        )
        policy = ThresholdLinkPolicy(seed=7, link_threshold=1.0, p_act=0.0)

        samples = [
            policy.get_decision(
                self._empty_observation(),
                action_space,
            ).actions[0]
            for _ in range(8)
        ]

        self.assertEqual(samples, [pass_action] * 8)

    def test_threshold_link_policy_creates_below_threshold(self) -> None:
        add_action = Action(ActionType.ADD, source=0, target=2)
        action_space = ActionSpace(
            actions={
                ActionType.ADD: [add_action],
                ActionType.BRIDGE: [],
                ActionType.DELETE: [Action(ActionType.DELETE, source=0, target=1)],
                ActionType.PASS: [Action(ActionType.PASS, source=0)],
            }
        )
        policy = ThresholdLinkPolicy(seed=7, link_threshold=0.5)

        decision = policy.get_decision(
            self._empty_observation(n_agents=5),
            action_space,
        )

        self.assertEqual(decision.actions, (add_action,))

    def test_threshold_link_policy_deletes_at_threshold(self) -> None:
        delete_action = Action(ActionType.DELETE, source=0, target=1)
        action_space = ActionSpace(
            actions={
                ActionType.ADD: [Action(ActionType.ADD, source=0, target=2)],
                ActionType.BRIDGE: [],
                ActionType.DELETE: [delete_action],
                ActionType.PASS: [Action(ActionType.PASS, source=0)],
            }
        )
        policy = ThresholdLinkPolicy(seed=7, link_threshold=0.5)

        decision = policy.get_decision(
            self._empty_observation(n_agents=3),
            action_space,
        )

        self.assertEqual(decision.actions, (delete_action,))

    def test_threshold_link_policy_stays_in_delete_phase_below_threshold(self) -> None:
        first_delete_action = Action(ActionType.DELETE, source=0, target=1)
        at_threshold_action_space = ActionSpace(
            actions={
                ActionType.ADD: [Action(ActionType.ADD, source=0, target=3)],
                ActionType.BRIDGE: [],
                ActionType.DELETE: [
                    first_delete_action,
                    Action(ActionType.DELETE, source=0, target=2),
                ],
                ActionType.PASS: [Action(ActionType.PASS, source=0)],
            }
        )
        second_delete_action = Action(ActionType.DELETE, source=0, target=1)
        below_threshold_action_space = ActionSpace(
            actions={
                ActionType.ADD: [Action(ActionType.ADD, source=0, target=2)],
                ActionType.BRIDGE: [],
                ActionType.DELETE: [second_delete_action],
                ActionType.PASS: [Action(ActionType.PASS, source=0)],
            }
        )
        policy = ThresholdLinkPolicy(seed=7, link_threshold=0.5)
        observation = self._empty_observation(n_agents=5)

        first_decision = policy.get_decision(observation, at_threshold_action_space)
        second_decision = policy.get_decision(observation, below_threshold_action_space)

        self.assertEqual(first_decision.actions[0].type, ActionType.DELETE)
        self.assertEqual(second_decision.actions, (second_delete_action,))

    def test_threshold_link_policy_reset_restores_create_phase(self) -> None:
        at_threshold_action_space = ActionSpace(
            actions={
                ActionType.ADD: [Action(ActionType.ADD, source=0, target=3)],
                ActionType.BRIDGE: [],
                ActionType.DELETE: [
                    Action(ActionType.DELETE, source=0, target=1),
                    Action(ActionType.DELETE, source=0, target=2),
                ],
                ActionType.PASS: [Action(ActionType.PASS, source=0)],
            }
        )
        add_action = Action(ActionType.ADD, source=0, target=2)
        below_threshold_action_space = ActionSpace(
            actions={
                ActionType.ADD: [add_action],
                ActionType.BRIDGE: [],
                ActionType.DELETE: [Action(ActionType.DELETE, source=0, target=1)],
                ActionType.PASS: [Action(ActionType.PASS, source=0)],
            }
        )
        policy = ThresholdLinkPolicy(seed=7, link_threshold=0.5)
        observation = self._empty_observation(n_agents=5)

        policy.get_decision(observation, at_threshold_action_space)
        policy.reset(seed=7)
        decision = policy.get_decision(observation, below_threshold_action_space)

        self.assertEqual(decision.actions, (add_action,))

    def test_threshold_link_policy_passes_when_preferred_action_set_is_empty(
        self,
    ) -> None:
        pass_action = Action(ActionType.PASS, source=0)
        action_space = ActionSpace(
            actions={
                ActionType.ADD: [],
                ActionType.BRIDGE: [],
                ActionType.DELETE: [],
                ActionType.PASS: [pass_action],
            }
        )
        policy = ThresholdLinkPolicy(seed=7, link_threshold=1.0)

        decision = policy.get_decision(
            self._empty_observation(n_agents=3),
            action_space,
        )

        self.assertEqual(decision.actions, (pass_action,))

    def test_threshold_link_policy_reset_is_deterministic(self) -> None:
        action_space = ActionSpace(
            actions={
                ActionType.ADD: [
                    Action(ActionType.ADD, source=0, target=1),
                    Action(ActionType.ADD, source=0, target=2),
                ],
                ActionType.BRIDGE: [Action(ActionType.BRIDGE, source=0)],
                ActionType.DELETE: [],
                ActionType.PASS: [Action(ActionType.PASS, source=0)],
            }
        )
        observation = self._empty_observation(n_agents=4)
        policy = ThresholdLinkPolicy(seed=7, link_threshold=1.0, p_act=0.5)

        first_run = [
            policy.get_decision(observation, action_space).actions[0] for _ in range(8)
        ]
        policy.reset(seed=7)
        second_run = [
            policy.get_decision(observation, action_space).actions[0] for _ in range(8)
        ]

        self.assertEqual(first_run, second_run)


if __name__ == "__main__":
    unittest.main()
