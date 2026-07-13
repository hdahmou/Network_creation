"""Tests for the rewritten best-response policy contract."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from network_creation.core.graph import (
    build_observation_from_state,
    build_state_from_graph,
    get_action_space,
)
from network_creation.core.model import Action, ActionType
from network_creation.policies.best_response import (
    BestResponsePolicy,
    stable_softmax_boltzmann,
)
from tests.graph_factories import graph_from_adjacency_matrix


class BestResponsePolicyTest(unittest.TestCase):
    @staticmethod
    def _discounted_benefit(
        policy: BestResponsePolicy,
        graph,
        *,
        source_idx: int = 0,
    ) -> float:
        distances = np.asarray(
            graph.distances(source=[source_idx], mode="all")[0],
            dtype=np.float32,
        )
        return float(np.power(policy.distance_discount, distances).sum())

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
    def _build_policy(**overrides: float | int) -> BestResponsePolicy:
        kwargs: dict[str, float | int] = {
            "n_nodes": 3,
            "seed": 7,
            "benefit_to_cost_ratio": 0.5,
            "distance_discount": 0.5,
            "exploration": 1e-3,
            "outside_optimism": 0.0,
        }
        kwargs.update(overrides)
        return BestResponsePolicy(**kwargs)

    @staticmethod
    def _expected_bridge_value(
        policy: BestResponsePolicy,
        observation,
    ) -> float:
        spillover_targets = max(len(observation.outside_node_idxs) - 1, 0)
        expected_bridge_gain = (
            policy.distance_discount
            + policy.outside_optimism
            * spillover_targets
            * policy.distance_discount**2
        )
        return float(
            policy.benefit_to_cost_ratio * expected_bridge_gain - 1.0
        )

    @staticmethod
    def _expected_delete_value(
        policy: BestResponsePolicy,
        observation,
        *,
        delete_target: int,
    ) -> float:
        graph = observation.graph.copy()
        base_benefit = BestResponsePolicyTest._discounted_benefit(policy, graph)
        eid = graph.get_eid(0, delete_target, directed=True, error=True)
        graph.delete_edges([eid])
        deleted_benefit = BestResponsePolicyTest._discounted_benefit(policy, graph)
        return float(
            policy.benefit_to_cost_ratio * (deleted_benefit - base_benefit) + 1.0
        )

    @staticmethod
    def _expected_add_value(
        policy: BestResponsePolicy,
        observation,
        *,
        add_target: int,
    ) -> float:
        graph = observation.graph
        base_benefit = BestResponsePolicyTest._discounted_benefit(policy, graph)

        add_only_graph = graph.copy()
        add_only_graph.add_edge(0, add_target)
        add_only_benefit = BestResponsePolicyTest._discounted_benefit(
            policy,
            add_only_graph,
        )
        best_add_value = (
            policy.benefit_to_cost_ratio * (add_only_benefit - base_benefit) - 1.0
        )

        for edge in graph.es.select(_source=0):
            swap_graph = graph.copy()
            eid = swap_graph.get_eid(0, edge.target, directed=True, error=True)
            swap_graph.delete_edges([eid])
            swap_graph.add_edge(0, add_target)
            swap_benefit = BestResponsePolicyTest._discounted_benefit(
                policy,
                swap_graph,
            )
            best_add_value = max(
                best_add_value,
                policy.benefit_to_cost_ratio * (swap_benefit - base_benefit),
            )

        return float(best_add_value)

    def test_constructor_stores_parameters(self) -> None:
        policy = self._build_policy(
            n_nodes=5,
            seed=11,
            benefit_to_cost_ratio=1.0,
            exploration=2.5,
            outside_optimism=0.75,
        )

        self.assertEqual(policy.n_nodes, 5)
        self.assertEqual(policy.benefit_to_cost_ratio, 1.0)
        self.assertEqual(policy.exploration, 2.5)
        self.assertEqual(policy.outside_optimism, 0.75)

    def test_constructor_rejects_invalid_benefit_ratio(self) -> None:
        invalid_values = (
            (0.0, r"benefit_to_cost_ratio must be in \(0,1\]"),
            (1.1, r"benefit_to_cost_ratio must be in \(0,1\]"),
        )

        for invalid_value, error_message in invalid_values:
            with self.subTest(value=invalid_value):
                with self.assertRaisesRegex(ValueError, error_message):
                    self._build_policy(benefit_to_cost_ratio=invalid_value)

    def test_constructor_rejects_invalid_distance_discount(self) -> None:
        invalid_values = (0.0, 1.0)

        for invalid_value in invalid_values:
            with self.subTest(value=invalid_value):
                with self.assertRaisesRegex(
                    ValueError,
                    r"distance_discount must be in \(0,1\)",
                ):
                    self._build_policy(distance_discount=invalid_value)

    def test_constructor_rejects_non_positive_exploration(self) -> None:
        invalid_values = (0.0, -0.1)

        for invalid_value in invalid_values:
            with self.subTest(value=invalid_value):
                with self.assertRaisesRegex(
                    ValueError,
                    r"exploration must be positive",
                ):
                    self._build_policy(exploration=invalid_value)

    def test_constructor_rejects_outside_optimism_outside_unit_interval(self) -> None:
        invalid_values = (-0.1, 1.1)

        for invalid_value in invalid_values:
            with self.subTest(value=invalid_value):
                with self.assertRaisesRegex(
                    ValueError,
                    r"outside_optimism must be in \[0,1\]",
                ):
                    self._build_policy(outside_optimism=invalid_value)

    def test_stable_softmax_boltzmann_returns_probabilities(self) -> None:
        values = np.array([0.0, 1.0, 1.0], dtype=np.float32)
        probabilities = stable_softmax_boltzmann(values, tau=1.0)

        expected = np.exp(np.array([-1.0, 0.0, 0.0], dtype=np.float64))
        expected /= expected.sum()
        np.testing.assert_allclose(probabilities, expected)
        self.assertAlmostEqual(float(probabilities.sum()), 1.0)

    def test_stable_softmax_boltzmann_rejects_invalid_inputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "tau must be > 0"):
            stable_softmax_boltzmann(np.array([0.0, 1.0], dtype=np.float32), tau=0.0)

        with self.assertRaisesRegex(ValueError, "values must be non-empty"):
            stable_softmax_boltzmann(np.array([], dtype=np.float32), tau=0.5)

    def test_policy_passes_when_bridge_value_is_negative(self) -> None:
        observation, action_space = self._build_observation_and_action_space(
            adjacency=np.zeros((3, 3), dtype=np.int8),
            node_id=0,
            radius=1,
        )
        policy = self._build_policy(
            n_nodes=3,
            seed=19,
            benefit_to_cost_ratio=0.5,
            distance_discount=0.5,
            exploration=1e-3,
            outside_optimism=0.0,
        )

        decision = policy.get_decision(observation, action_space)

        self.assertEqual(decision.actions, (Action(ActionType.PASS, source=0),))
        self.assertIsNotNone(decision.value)
        assert decision.value is not None
        self.assertIn("value", decision.value)
        self.assertAlmostEqual(float(decision.value["value"]), 0.0)

    def test_add_action_is_returned_as_one_local_action_without_bridge(self) -> None:
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
            radius=3,
        )
        self.assertEqual(action_space[ActionType.BRIDGE], [])
        self.assertEqual(
            action_space[ActionType.ADD],
            [Action(ActionType.ADD, source=0, target=2)],
        )

        policy = self._build_policy(
            n_nodes=3,
            seed=31,
            benefit_to_cost_ratio=1.0,
            distance_discount=0.5,
            exploration=0.8,
            outside_optimism=0.0,
        )

        with patch(
            "network_creation.policies.best_response._sample_boltzmann",
            return_value=1,
        ):
            decision = policy.get_decision(observation, action_space)

        self.assertEqual(len(decision.actions), 1)
        self.assertEqual(
            decision.actions,
            (Action(ActionType.ADD, source=0, target=2),),
        )
        self.assertIsNotNone(decision.value)
        assert decision.value is not None
        self.assertIn("value", decision.value)
        self.assertAlmostEqual(
            float(decision.value["value"]),
            self._expected_add_value(policy, observation, add_target=2),
        )

    def test_delete_action_uses_local_indices_for_non_contiguous_observations(self) -> None:
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

        policy = self._build_policy(
            n_nodes=6,
            seed=23,
            benefit_to_cost_ratio=0.5,
            distance_discount=0.5,
            exploration=0.8,
            outside_optimism=0.0,
        )

        with patch(
            "network_creation.policies.best_response._sample_boltzmann",
            return_value=0,
        ):
            decision = policy.get_decision(observation, action_space)

        self.assertEqual(len(decision.actions), 1)
        self.assertEqual(
            decision.actions,
            (Action(ActionType.DELETE, source=0, target=1),),
        )
        self.assertIsNotNone(decision.value)
        assert decision.value is not None
        self.assertAlmostEqual(
            float(decision.value["value"]),
            self._expected_delete_value(policy, observation, delete_target=1),
        )

    def test_bridge_action_returns_concrete_local_target(self) -> None:
        observation, action_space = self._build_observation_and_action_space(
            adjacency=np.zeros((10, 10), dtype=np.int8),
            node_id=4,
            radius=3,
        )
        self.assertEqual(observation.graph.vs["name"], [4])
        self.assertEqual(
            action_space[ActionType.BRIDGE],
            [Action(ActionType.BRIDGE, source=0)],
        )

        policy = self._build_policy(
            n_nodes=10,
            seed=29,
            benefit_to_cost_ratio=1.0,
            distance_discount=0.9,
            exploration=0.8,
            outside_optimism=1.0,
        )

        with patch(
            "network_creation.policies.best_response._sample_boltzmann",
            return_value=0,
        ):
            decision = policy.get_decision(observation, action_space)

        self.assertEqual(len(decision.actions), 1)
        self.assertEqual(decision.actions[0].type, ActionType.BRIDGE)
        self.assertEqual(decision.actions[0].source, 0)
        self.assertIsNotNone(decision.actions[0].target)
        assert decision.actions[0].target is not None
        self.assertIn(int(decision.actions[0].target), observation.outside_node_idxs)
        self.assertIsNotNone(decision.value)
        assert decision.value is not None
        self.assertAlmostEqual(
            float(decision.value["value"]),
            self._expected_bridge_value(policy, observation),
            places=6,
        )


if __name__ == "__main__":
    unittest.main()
