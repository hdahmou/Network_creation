"""Tests for the myopic payoff policy."""

from __future__ import annotations

import unittest

import numpy as np

from network_creation.core.agent import Agent
from network_creation.core.environment import Environment
from network_creation.core.graph import (
    build_observation_from_state,
    build_state_from_graph,
    get_action_space,
)
from network_creation.core.model import Action, ActionType, PolicyDecision, SchedulingMode
from network_creation.core.simulator import Simulator
from network_creation.policies.myopic_payoff import MyopicPayoffPolicy
from tests.graph_factories import graph_from_adjacency_matrix


class MyopicPayoffPolicyTest(unittest.TestCase):
    @staticmethod
    def _build_observation_and_action_space(
        adjacency: np.ndarray,
        node_id: int,
        radius: int,
    ) -> tuple:
        state = build_state_from_graph(graph_from_adjacency_matrix(adjacency))
        observation = build_observation_from_state(state, node_id=node_id, radius=radius)
        action_space = get_action_space(observation)
        return observation, action_space

    def test_constructor_rejects_invalid_ratio_discount_and_bridge_probability(self) -> None:
        invalid_cases = (
            (
                "benefit_to_cost_ratio must satisfy 0 <= "
                "benefit_to_cost_ratio <= 1",
                {"benefit_to_cost_ratio": -0.1},
            ),
            (
                "benefit_to_cost_ratio must satisfy 0 <= "
                "benefit_to_cost_ratio <= 1",
                {"benefit_to_cost_ratio": 1.1},
            ),
            (
                "distance_discount must satisfy 0 <= distance_discount < 1",
                {"distance_discount": -0.1},
            ),
            (
                "distance_discount must satisfy 0 <= distance_discount < 1",
                {"distance_discount": 1.0},
            ),
            (
                "bridge_probability must satisfy 0 <= bridge_probability <= 1",
                {"bridge_probability": -0.1},
            ),
            (
                "bridge_probability must satisfy 0 <= bridge_probability <= 1",
                {"bridge_probability": 1.1},
            ),
        )

        for expected_message, overrides in invalid_cases:
            with self.subTest(overrides=overrides):
                kwargs = {
                    "seed": 7,
                    "benefit_to_cost_ratio": 0.1,
                    "distance_discount": 0.5,
                    "bridge_probability": 0.5,
                }
                kwargs.update(overrides)

                with self.assertRaisesRegex(ValueError, expected_message):
                    MyopicPayoffPolicy(**kwargs)

    def test_reset_is_deterministic_from_seed(self) -> None:
        observation, action_space = self._build_observation_and_action_space(
            adjacency=np.array(
                [
                    [0, 0, 0, 0],
                    [1, 0, 1, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                ],
                dtype=np.int8,
            ),
            node_id=0,
            radius=2,
        )
        policy = MyopicPayoffPolicy(
            seed=7,
            benefit_to_cost_ratio=0.1,
            distance_discount=0.5,
            bridge_probability=0.5,
        )

        first_run = [
            policy.get_decision(observation, action_space).actions[0]
            for _ in range(10)
        ]
        policy.reset(seed=7)
        second_run = [
            policy.get_decision(observation, action_space).actions[0]
            for _ in range(10)
        ]

        self.assertEqual(first_run, second_run)

    def test_bridge_branch_selects_only_from_bridge_actions(self) -> None:
        observation, action_space = self._build_observation_and_action_space(
            adjacency=np.zeros((3, 3), dtype=np.int8),
            node_id=0,
            radius=2,
        )
        policy = MyopicPayoffPolicy(
            seed=11,
            benefit_to_cost_ratio=0.1,
            distance_discount=0.5,
            bridge_probability=1.0,
        )
        available_bridge_actions = set(action_space[ActionType.BRIDGE])

        samples = [
            policy.get_decision(observation, action_space).actions[0]
            for _ in range(20)
        ]

        self.assertTrue(all(action.type is ActionType.BRIDGE for action in samples))
        self.assertTrue(all(action in available_bridge_actions for action in samples))

    def test_bridge_branch_returns_pass_when_no_bridge_action_exists(self) -> None:
        observation, action_space = self._build_observation_and_action_space(
            adjacency=np.array(
                [
                    [0, 1, 0],
                    [0, 0, 1],
                    [0, 0, 0],
                ],
                dtype=np.int8,
            ),
            node_id=0,
            radius=2,
        )
        policy = MyopicPayoffPolicy(
            seed=13,
            benefit_to_cost_ratio=0.1,
            distance_discount=0.5,
            bridge_probability=1.0,
        )

        self.assertEqual(
            policy.get_decision(observation, action_space),
            PolicyDecision(actions=(Action(ActionType.PASS, source=0),), value=None),
        )

    def test_non_bridge_branch_returns_pass_when_add_and_delete_are_empty(self) -> None:
        observation, action_space = self._build_observation_and_action_space(
            adjacency=np.zeros((3, 3), dtype=np.int8),
            node_id=0,
            radius=2,
        )
        policy = MyopicPayoffPolicy(
            seed=17,
            benefit_to_cost_ratio=0.1,
            distance_discount=0.5,
            bridge_probability=0.0,
        )

        self.assertEqual(
            policy.get_decision(observation, action_space),
            PolicyDecision(actions=(Action(ActionType.PASS, source=0),), value=None),
        )

    def test_positive_visible_add_is_accepted(self) -> None:
        observation, action_space = self._build_observation_and_action_space(
            adjacency=np.array(
                [
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 1, 0, 1, 1, 1, 1, 1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                ],
                dtype=np.int8,
            ),
            node_id=0,
            radius=3,
        )
        policy = MyopicPayoffPolicy(
            seed=2,
            benefit_to_cost_ratio=1.0,
            distance_discount=0.5,
            bridge_probability=0.0,
        )

        decision = policy.get_decision(observation, action_space)

        self.assertEqual(
            decision.actions,
            (Action(ActionType.ADD, source=0, target=1),),
        )
        assert decision.value is not None
        self.assertEqual(
            decision.value["considered_action"],
            Action(ActionType.ADD, source=0, target=1),
        )
        self.assertAlmostEqual(float(decision.value["delta"]), 0.125)

    def test_reciprocal_visible_delete_is_accepted(self) -> None:
        observation, action_space = self._build_observation_and_action_space(
            adjacency=np.array(
                [
                    [0, 1, 0],
                    [1, 0, 0],
                    [0, 1, 0],
                ],
                dtype=np.int8,
            ),
            node_id=0,
            radius=1,
        )
        policy = MyopicPayoffPolicy(
            seed=23,
            benefit_to_cost_ratio=0.1,
            distance_discount=0.5,
            bridge_probability=0.0,
        )

        decision = policy.get_decision(observation, action_space)

        self.assertEqual(
            decision.actions,
            (Action(ActionType.DELETE, source=0, target=1),),
        )
        assert decision.value is not None
        self.assertEqual(
            decision.value["considered_action"],
            Action(ActionType.DELETE, source=0, target=1),
        )
        self.assertAlmostEqual(float(decision.value["delta"]), 1.0)

    def test_non_positive_delete_returns_pass(self) -> None:
        observation, action_space = self._build_observation_and_action_space(
            adjacency=np.array(
                [
                    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                ],
                dtype=np.int8,
            ),
            node_id=0,
            radius=2,
        )
        policy = MyopicPayoffPolicy(
            seed=9,
            benefit_to_cost_ratio=1.0,
            distance_discount=0.5,
            bridge_probability=0.0,
        )

        decision = policy.get_decision(observation, action_space)

        self.assertEqual(
            decision.actions,
            (Action(ActionType.PASS, source=0),),
        )
        assert decision.value is not None
        self.assertEqual(
            decision.value["considered_action"],
            Action(ActionType.DELETE, source=0, target=1),
        )
        self.assertAlmostEqual(float(decision.value["delta"]), -1.5)

    def test_policy_runs_end_to_end_in_simulator(self) -> None:
        adjacency = np.array(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 0, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
            dtype=np.int8,
        )
        simulator = Simulator(
            env=Environment(graph_from_adjacency_matrix(adjacency), seed=31),
            agents=(
                Agent(
                    id=0,
                    visibility_fraction=1.0,
                    policy=MyopicPayoffPolicy(
                        seed=2,
                        benefit_to_cost_ratio=1.0,
                        distance_discount=0.5,
                        bridge_probability=0.0,
                    ),
                    seed=2,
                ),
            ),
            mode=SchedulingMode.SEQUENTIAL,
            max_steps=1,
            rng=np.random.default_rng(7),
            observers=(),
        )

        simulator.run(n_episodes=1, reset_observers=False)

        self.assertIn((0, 1), simulator.env.state.graph.get_edgelist())


if __name__ == "__main__":
    unittest.main()
