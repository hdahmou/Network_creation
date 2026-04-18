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
    tau = 1e-6

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

    @staticmethod
    def _expected_bridge_score(
        action_space,
        policy: BestResponsePolicy,
    ) -> float:
        return float(
            2.0
            * (
                policy.bridge_multiplier
                * (len(action_space[ActionType.BRIDGE]) / policy.n_nodes)
            )
            - 1.0
        )

    def test_constructor_accepts_unit_interval_bridge_multiplier(self) -> None:
        policy = BestResponsePolicy(
            n_nodes=3,
            seed=7,
            benefit_to_cost_ratio=0.5,
            distance_discount=0.5,
            bridge_multiplier=1.0,
            inertia=-1.75,
            tau=2.0,
        )

        self.assertEqual(policy.bridge_multiplier, 1.0)
        self.assertEqual(policy.inertia, -1.75)
        self.assertEqual(policy.tau, 2.0)

    def test_constructor_rejects_bridge_multiplier_outside_unit_interval(self) -> None:
        for bridge_multiplier in (-0.1, 1.1):
            with self.subTest(bridge_multiplier=bridge_multiplier):
                with self.assertRaisesRegex(
                    ValueError,
                    r"bridge_multiplier must be in \[0,1\]",
                ):
                    BestResponsePolicy(
                        n_nodes=3,
                        seed=7,
                        benefit_to_cost_ratio=0.5,
                        distance_discount=0.5,
                        bridge_multiplier=bridge_multiplier,
                        inertia=0.0,
                        tau=self.tau,
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
                bridge_multiplier=0.0,
                inertia=0.0,
                tau=self.tau,
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
                bridge_multiplier=0.0,
                inertia=0.0,
                tau=self.tau,
            )

    def test_stable_softmax_boltzmann_returns_probabilities(self) -> None:
        values = np.array([0.0, 1.0, 1.0], dtype=np.float32)
        probabilities = stable_softmax_boltzmann(values, tau=1.0)

        expected = np.exp(np.array([-1.0, 0.0, 0.0], dtype=np.float64))
        expected /= expected.sum()
        np.testing.assert_allclose(probabilities, expected)
        self.assertAlmostEqual(float(probabilities.sum()), 1.0)

    def test_stable_softmax_boltzmann_rejects_non_positive_tau(self) -> None:
        with self.assertRaisesRegex(ValueError, "tau must be > 0"):
            stable_softmax_boltzmann(
                np.array([0.0, 1.0], dtype=np.float32),
                tau=0.0,
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
            bridge_multiplier=1.0,
            inertia=0.0,
            tau=self.tau,
        )

        decision = policy.get_decision(observation, action_space)
        expected_bridge_score = self._expected_bridge_score(action_space, policy)

        self.assertEqual(decision.actions[0].type, ActionType.BRIDGE)
        self.assertIn(decision.actions[0], action_space[ActionType.BRIDGE])
        assert decision.value is not None
        self.assertAlmostEqual(float(decision.value["value"]), expected_bridge_score)

    def test_bridge_score_uses_bridge_multiplier_and_available_ratio(self) -> None:
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
            bridge_multiplier=1.0,
            inertia=-0.1,
            tau=self.tau,
        )

        decision = policy.get_decision(observation, action_space)
        expected_bridge_score = self._expected_bridge_score(action_space, policy)

        self.assertEqual(decision.actions[0].type, ActionType.BRIDGE)
        assert decision.value is not None
        self.assertAlmostEqual(float(decision.value["value"]), expected_bridge_score)
        self.assertEqual(expected_bridge_score, 0.0)

    def test_inertia_biases_selection_toward_pass(self) -> None:
        non_pass_values = np.array([0.2, -0.1], dtype=np.float32)
        low_inertia_policy = BestResponsePolicy(
            n_nodes=3,
            seed=17,
            benefit_to_cost_ratio=0.5,
            distance_discount=0.5,
            bridge_multiplier=0.0,
            inertia=0.0,
            tau=self.tau,
        )
        high_inertia_policy = BestResponsePolicy(
            n_nodes=3,
            seed=17,
            benefit_to_cost_ratio=0.5,
            distance_discount=0.5,
            bridge_multiplier=0.0,
            inertia=1.25,
            tau=self.tau,
        )

        low_pass_probability = low_inertia_policy._pass_probability(non_pass_values)
        high_pass_probability = high_inertia_policy._pass_probability(non_pass_values)
        expected_high_pass_probability = float(
            1.0 / (1.0 + np.exp(-(high_inertia_policy.inertia - 0.2)))
        )

        self.assertLess(low_pass_probability, high_pass_probability)
        self.assertAlmostEqual(high_pass_probability, expected_high_pass_probability)

    def test_pass_gate_can_force_pass_action(self) -> None:
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
            bridge_multiplier=0.75,
            inertia=1.25,
            tau=self.tau,
        )

        with patch.object(policy, "_pass_probability", return_value=1.0):
            decision = policy.get_decision(observation, action_space)

        self.assertEqual(
            decision.actions,
            (Action(ActionType.PASS, source=0),),
        )
        self.assertIsNotNone(decision.value)
        self.assertAlmostEqual(float(decision.value["value"]), 1.0)

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
            bridge_multiplier=0.75,
            inertia=0.0,
            tau=self.tau,
        )
        bridge_score = self._expected_bridge_score(action_space, policy)
        self.assertAlmostEqual(bridge_score, 0.0)
        self.assertAlmostEqual(
            policy._pass_probability(np.array([bridge_score], dtype=np.float32)),
            0.5,
        )

        with patch.object(policy, "_pass_probability", return_value=0.0):
            with patch(
                "network_creation.policies.best_response._sample_boltzmann",
                return_value=0,
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
            bridge_multiplier=0.0,
            inertia=0.0,
            tau=self.tau,
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
            bridge_multiplier=1.0,
            inertia=0.0,
            tau=self.tau,
        )

        decision = policy.get_decision(observation, action_space)
        expected_bridge_score = self._expected_bridge_score(action_space, policy)

        self.assertEqual(decision.actions[0].type, ActionType.BRIDGE)
        self.assertEqual(decision.actions[0].source, 0)
        self.assertIn(decision.actions[0].target, {1, 2, 3, 4, 5})
        self.assertIsNotNone(decision.value)
        self.assertAlmostEqual(float(decision.value["value"]), expected_bridge_score)

    def test_bridge_candidates_use_one_random_target_in_boltzmann_input(self) -> None:
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
            bridge_multiplier=1.0,
            inertia=0.0,
            tau=1.0,
        )
        expected_bridge_score = self._expected_bridge_score(action_space, policy)

        def capture(
            values: np.ndarray,
            tau: float,
            _rng: np.random.Generator,
        ) -> int:
            self.assertIsInstance(values, np.ndarray)
            np.testing.assert_allclose(values, np.array([expected_bridge_score]))
            self.assertEqual(tau, 1.0)
            return 0

        with patch.object(policy, "_pass_probability", return_value=0.0):
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
        self.assertAlmostEqual(float(decision.value["value"]), expected_bridge_score)

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
            bridge_multiplier=1.0,
            inertia=0.0,
            tau=1.0,
        )
        expected_bridge_score = self._expected_bridge_score(action_space, policy)

        def capture(
            values: np.ndarray,
            tau: float,
            _rng: np.random.Generator,
        ) -> int:
            self.assertIsInstance(values, np.ndarray)
            np.testing.assert_allclose(values, np.array([expected_bridge_score]))
            self.assertEqual(tau, 1.0)
            return 0

        with patch.object(policy, "_pass_probability", return_value=0.0):
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
        self.assertAlmostEqual(float(decision.value["value"]), expected_bridge_score)

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
            bridge_multiplier=0.0,
            inertia=0.0,
            tau=self.tau,
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
