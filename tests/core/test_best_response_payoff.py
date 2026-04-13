"""Tests for the best-response payoff policy."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from network_creation.core.graph import (
    build_observation_from_state,
    build_state_from_graph,
    get_action_space,
)
from network_creation.core.model import Action, ActionType, PolicyDecision
from network_creation.policies.best_response import (
    BestResponsePolicy,
    stable_softmax_boltzmann,
)
from tests.graph_factories import graph_from_adjacency_matrix


class BestResponsePolicyTest(unittest.TestCase):
    temperature = 1e-6

    @staticmethod
    def _build_observation_and_action_space(
        adjacency: np.ndarray,
        node_id: int,
        radius: int,
    ) -> tuple:
        state = build_state_from_graph(graph_from_adjacency_matrix(adjacency))
        observation = build_observation_from_state(
            state,
            node_id=node_id,
            radius=radius,
        )
        action_space = get_action_space(observation)
        return observation, action_space

    def test_constructor_rejects_inertia_below_lower_bound(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            r"inertia_term must be in \[-0\.5,0\.5\]",
        ):
            BestResponsePolicy(
                n_nodes=3,
                seed=7,
                benefit_to_cost_ratio=0.5,
                distance_discount=0.5,
                bridge_value=0.0,
                inertia_term=-0.6,
                temperature=self.temperature,
            )

    def test_constructor_rejects_non_positive_benefit_ratio(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            r"benefit_to_cost_ratio must be in \(0,1\]",
        ):
            BestResponsePolicy(
                n_nodes=3,
                seed=7,
                benefit_to_cost_ratio=0.0,
                distance_discount=0.5,
                bridge_value=0.0,
                inertia_term=0.0,
                temperature=self.temperature,
            )

    def test_constructor_rejects_non_positive_distance_discount(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            r"distance_discount must be in \(0,1\)",
        ):
            BestResponsePolicy(
                n_nodes=3,
                seed=7,
                benefit_to_cost_ratio=0.5,
                distance_discount=0.0,
                bridge_value=0.0,
                inertia_term=0.0,
                temperature=self.temperature,
            )

    def test_constructor_rejects_non_positive_temperature(self) -> None:
        with self.assertRaisesRegex(ValueError, r"temperature must be in \(0,1\]"):
            BestResponsePolicy(
                n_nodes=3,
                seed=7,
                benefit_to_cost_ratio=0.5,
                distance_discount=0.5,
                bridge_value=0.0,
                inertia_term=0.0,
                temperature=0.0,
            )

    def test_stable_softmax_boltzmann_returns_probabilities(self) -> None:
        probabilities = stable_softmax_boltzmann(
            np.array([0.0, 1.0, 1.0], dtype=np.float32),
            temperature=1.0,
        )

        expected = np.exp(np.array([-1.0, 0.0, 0.0], dtype=np.float64))
        expected /= expected.sum()
        np.testing.assert_allclose(probabilities, expected)
        self.assertAlmostEqual(float(probabilities.sum()), 1.0)

    def test_stable_softmax_boltzmann_rejects_non_positive_temperature(self) -> None:
        with self.assertRaisesRegex(ValueError, "temperature must be > 0"):
            stable_softmax_boltzmann(
                np.array([0.0, 1.0], dtype=np.float32),
                temperature=0.0,
            )

    def test_isolated_node_can_still_choose_bridge_action(self) -> None:
        observation, action_space = self._build_observation_and_action_space(
            adjacency=np.zeros((3, 3), dtype=np.int8),
            node_id=0,
            radius=1,
        )
        policy = BestResponsePolicy(
            n_nodes=3,
            seed=11,
            benefit_to_cost_ratio=0.5,
            distance_discount=0.5,
            bridge_value=0.5,
            inertia_term=0.0,
            temperature=self.temperature,
        )

        decision = policy.get_decision(observation, action_space)

        self.assertEqual(decision.actions[0].type, ActionType.BRIDGE)
        self.assertIn(decision.actions[0], action_space[ActionType.BRIDGE])
        assert decision.value is not None
        self.assertAlmostEqual(float(decision.value["value"]), 0.5)

    def test_bridge_value_offsets_median_reference_value(self) -> None:
        observation, action_space = self._build_observation_and_action_space(
            adjacency=np.array(
                [
                    [0, 0, 0, 0],
                    [1, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                ],
                dtype=np.int8,
            ),
            node_id=0,
            radius=1,
        )
        policy = BestResponsePolicy(
            n_nodes=4,
            seed=13,
            benefit_to_cost_ratio=0.5,
            distance_discount=0.5,
            bridge_value=0.5,
            inertia_term=0.0,
            temperature=self.temperature,
        )

        decision = policy.get_decision(observation, action_space)

        self.assertEqual(decision.actions[0].type, ActionType.BRIDGE)
        assert decision.value is not None
        self.assertAlmostEqual(float(decision.value["value"]), 5.0 / 14.0)

    def test_inertia_term_biases_selection_toward_pass(self) -> None:
        observation, action_space = self._build_observation_and_action_space(
            adjacency=np.array(
                [
                    [0, 1, 0],
                    [1, 0, 0],
                    [0, 0, 0],
                ],
                dtype=np.int8,
            ),
            node_id=0,
            radius=1,
        )
        policy = BestResponsePolicy(
            n_nodes=3,
            seed=17,
            benefit_to_cost_ratio=0.5,
            distance_discount=0.5,
            bridge_value=0.0,
            inertia_term=0.5,
            temperature=self.temperature,
        )

        decision = policy.get_decision(observation, action_space)

        self.assertEqual(
            decision.actions,
            (Action(ActionType.PASS, source=0),),
        )
        self.assertIsNotNone(decision.value)
        self.assertAlmostEqual(float(decision.value["value"]), 0.7)

    def test_bridge_value_below_inertia_returns_pass(self) -> None:
        observation, action_space = self._build_observation_and_action_space(
            adjacency=np.zeros((3, 3), dtype=np.int8),
            node_id=0,
            radius=1,
        )
        policy = BestResponsePolicy(
            n_nodes=3,
            seed=19,
            benefit_to_cost_ratio=0.5,
            distance_discount=0.5,
            bridge_value=0.25,
            inertia_term=0.5,
            temperature=self.temperature,
        )

        decision = policy.get_decision(observation, action_space)

        self.assertEqual(
            decision.actions,
            (Action(ActionType.PASS, source=0),),
        )
        self.assertIsNotNone(decision.value)
        self.assertAlmostEqual(float(decision.value["value"]), 0.5)

    def test_equal_value_bridge_ties_with_pass_and_can_be_selected(self) -> None:
        observation, action_space = self._build_observation_and_action_space(
            adjacency=np.zeros((3, 3), dtype=np.int8),
            node_id=0,
            radius=1,
        )
        policy = BestResponsePolicy(
            n_nodes=3,
            seed=1,
            benefit_to_cost_ratio=0.5,
            distance_discount=0.5,
            bridge_value=0.0,
            inertia_term=0.0,
            temperature=self.temperature,
        )
        with patch(
            "network_creation.policies.best_response._sample_boltzmann",
            return_value=1,
        ) as sample_boltzmann:
            decision = policy.get_decision(observation, action_space)

        sample_boltzmann.assert_called_once()
        self.assertEqual(decision.actions[0].type, ActionType.BRIDGE)
        self.assertIn(decision.actions[0], action_space[ActionType.BRIDGE])
        assert decision.value is not None
        self.assertAlmostEqual(float(decision.value["value"]), 0.0)

    def test_non_contiguous_delete_action_returns_local_action(self) -> None:
        adjacency = np.zeros((6, 6), dtype=np.int8)
        adjacency[4, 5] = 1
        observation, action_space = self._build_observation_and_action_space(
            adjacency=adjacency,
            node_id=4,
            radius=1,
        )
        self.assertEqual(observation.graph.vs["name"], [4, 5])
        self.assertEqual(
            action_space[ActionType.DELETE],
            [Action(ActionType.DELETE, source=0, target=1)],
        )

        policy = BestResponsePolicy(
            n_nodes=6,
            seed=23,
            benefit_to_cost_ratio=0.5,
            distance_discount=0.5,
            bridge_value=0.0,
            inertia_term=0.0,
            temperature=self.temperature,
        )

        with patch(
            "network_creation.policies.best_response._sample_boltzmann",
            return_value=0,
        ):
            decision = policy.get_decision(observation, action_space)

        self.assertEqual(
            decision.actions,
            (Action(ActionType.DELETE, source=0, target=1),),
        )
        assert decision.value is not None
        self.assertAlmostEqual(float(decision.value["value"]), 3.0 / 44.0)

    def test_non_contiguous_bridge_action_returns_local_action(self) -> None:
        observation, action_space = self._build_observation_and_action_space(
            adjacency=np.zeros((6, 6), dtype=np.int8),
            node_id=4,
            radius=1,
        )
        self.assertEqual(observation.graph.vs["name"], [4])
        self.assertEqual(
            action_space[ActionType.PASS],
            [Action(ActionType.PASS, source=0)],
        )
        self.assertEqual(
            action_space[ActionType.BRIDGE],
            [
                Action(ActionType.BRIDGE, source=0, target=1),
                Action(ActionType.BRIDGE, source=0, target=2),
                Action(ActionType.BRIDGE, source=0, target=3),
                Action(ActionType.BRIDGE, source=0, target=4),
                Action(ActionType.BRIDGE, source=0, target=5),
            ],
        )
        policy = BestResponsePolicy(
            n_nodes=6,
            seed=29,
            benefit_to_cost_ratio=0.5,
            distance_discount=0.5,
            bridge_value=0.5,
            inertia_term=0.0,
            temperature=self.temperature,
        )

        decision = policy.get_decision(observation, action_space)

        self.assertEqual(decision.actions[0].type, ActionType.BRIDGE)
        self.assertEqual(decision.actions[0].source, 0)
        self.assertIn(decision.actions[0].target, {1, 2, 3, 4, 5})
        self.assertIsNotNone(decision.value)
        self.assertAlmostEqual(float(decision.value["value"]), 0.5)

    def test_bridge_candidates_are_duplicated_in_boltzmann_input(self) -> None:
        observation, action_space = self._build_observation_and_action_space(
            adjacency=np.zeros((3, 3), dtype=np.int8),
            node_id=0,
            radius=1,
        )
        policy = BestResponsePolicy(
            n_nodes=3,
            seed=37,
            benefit_to_cost_ratio=0.5,
            distance_discount=0.5,
            bridge_value=0.5,
            inertia_term=0.0,
            temperature=1.0,
        )

        def capture(
            values: np.ndarray,
            temperature: float,
            _rng: np.random.Generator,
        ) -> int:
            self.assertIsInstance(values, np.ndarray)
            np.testing.assert_allclose(values, np.array([0.0, 0.5, 0.5]))
            self.assertEqual(temperature, 1.0)
            return 2

        with patch(
            "network_creation.policies.best_response._sample_boltzmann",
            side_effect=capture,
        ) as sample_boltzmann:
            decision = policy.get_decision(observation, action_space)

        sample_boltzmann.assert_called_once()
        self.assertEqual(
            decision.actions,
            (Action(ActionType.BRIDGE, source=0, target=2),),
        )
        self.assertIsNotNone(decision.value)
        self.assertAlmostEqual(float(decision.value["value"]), 0.5)

    def test_policy_uses_boltzmann_sampling(self) -> None:
        observation, action_space = self._build_observation_and_action_space(
            adjacency=np.zeros((3, 3), dtype=np.int8),
            node_id=0,
            radius=1,
        )
        policy = BestResponsePolicy(
            n_nodes=3,
            seed=41,
            benefit_to_cost_ratio=0.5,
            distance_discount=0.5,
            bridge_value=0.5,
            inertia_term=0.0,
            temperature=1.0,
        )

        def capture(
            values: np.ndarray,
            temperature: float,
            _rng: np.random.Generator,
        ) -> int:
            self.assertIsInstance(values, np.ndarray)
            np.testing.assert_allclose(values, np.array([0.0, 0.5, 0.5]))
            self.assertEqual(temperature, 1.0)
            return 1

        with patch(
            "network_creation.policies.best_response._sample_boltzmann",
            side_effect=capture,
        ) as sample_boltzmann:
            decision = policy.get_decision(observation, action_space)

        sample_boltzmann.assert_called_once()
        self.assertEqual(
            decision.actions,
            (Action(ActionType.BRIDGE, source=0, target=1),),
        )
        self.assertIsNotNone(decision.value)
        self.assertAlmostEqual(float(decision.value["value"]), 0.5)

    def test_add_action_can_use_best_add_then_delete_combo(self) -> None:
        observation, action_space = self._build_observation_and_action_space(
            adjacency=np.array(
                [
                    [0, 0, 1, 0, 0],
                    [0, 0, 1, 1, 1],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                ],
                dtype=np.int8,
            ),
            node_id=0,
            radius=5,
        )
        self.assertEqual(
            action_space[ActionType.DELETE],
            [Action(ActionType.DELETE, source=0, target=2)],
        )
        self.assertEqual(
            action_space[ActionType.ADD],
            [
                Action(ActionType.ADD, source=0, target=1),
                Action(ActionType.ADD, source=0, target=3),
                Action(ActionType.ADD, source=0, target=4),
            ],
        )

        policy = BestResponsePolicy(
            n_nodes=5,
            seed=31,
            benefit_to_cost_ratio=1.0,
            distance_discount=0.5,
            bridge_value=0.0,
            inertia_term=0.0,
            temperature=self.temperature,
        )

        decision = policy.get_decision(observation, action_space)

        self.assertEqual(
            decision.actions,
            (Action(ActionType.ADD, source=0, target=1),),
        )
        assert decision.value is not None
        self.assertAlmostEqual(float(decision.value["value"]), 1.0 / 36.0)


if __name__ == "__main__":
    unittest.main()
