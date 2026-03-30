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

    def test_constructor_rejects_invalid_decay_and_bridge_probability(self) -> None:
        invalid_cases = (
            ("decay must satisfy 0 <= decay < 1", {"decay": -0.1}),
            ("decay must satisfy 0 <= decay < 1", {"decay": 1.0}),
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
                    "link_cost": 1.0,
                    "decay": 0.5,
                    "per_link_benefit": 10.0,
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
            link_cost=1.0,
            decay=0.5,
            per_link_benefit=10.0,
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
            link_cost=1.0,
            decay=0.5,
            per_link_benefit=10.0,
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
            link_cost=1.0,
            decay=0.5,
            per_link_benefit=10.0,
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
            link_cost=1.0,
            decay=0.5,
            per_link_benefit=10.0,
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
                    [0, 0, 0],
                    [1, 0, 1],
                    [0, 0, 0],
                ],
                dtype=np.int8,
            ),
            node_id=0,
            radius=2,
        )
        policy = MyopicPayoffPolicy(
            seed=19,
            link_cost=1.0,
            decay=0.5,
            per_link_benefit=10.0,
            bridge_probability=0.0,
        )

        decision = policy.get_decision(observation, action_space)

        self.assertEqual(
            decision.actions,
            (Action(ActionType.ADD, source=0, target=2),),
        )
        assert decision.value is not None
        self.assertEqual(
            decision.value["considered_action"],
            Action(ActionType.ADD, source=0, target=2),
        )
        self.assertAlmostEqual(float(decision.value["delta"]), 1.5)

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
            link_cost=1.0,
            decay=0.5,
            per_link_benefit=10.0,
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
                    [0, 1, 0, 0],
                    [0, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 0, 0],
                ],
                dtype=np.int8,
            ),
            node_id=0,
            radius=1,
        )
        policy = MyopicPayoffPolicy(
            seed=29,
            link_cost=0.1,
            decay=0.5,
            per_link_benefit=10.0,
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
        self.assertAlmostEqual(float(decision.value["delta"]), -4.9)

    def test_policy_runs_end_to_end_in_simulator(self) -> None:
        adjacency = np.array(
            [
                [0, 0, 0],
                [1, 0, 1],
                [0, 0, 0],
            ],
            dtype=np.int8,
        )
        simulator = Simulator(
            env=Environment(graph_from_adjacency_matrix(adjacency), seed=31),
            agents=(
                Agent(
                    id=0,
                    radius=2,
                    policy=MyopicPayoffPolicy(
                        seed=37,
                        link_cost=1.0,
                        decay=0.5,
                        per_link_benefit=10.0,
                        bridge_probability=0.0,
                    ),
                    seed=37,
                ),
            ),
            mode=SchedulingMode.SEQUENTIAL,
            max_steps=1,
            rng=np.random.default_rng(41),
            observers=(),
        )

        simulator.run(n_episodes=1, reset_observers=False)

        self.assertIn((0, 2), simulator.env.state.graph.get_edgelist())


if __name__ == "__main__":
    unittest.main()
