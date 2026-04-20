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
from network_creation.core.model import Action, ActionType
from network_creation.policies.best_response import (
    BestResponsePolicy,
    stable_log_mean_exp,
    stable_softmax_boltzmann,
)
from tests.graph_factories import graph_from_adjacency_matrix


class BestResponsePolicyTest(unittest.TestCase):
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
            "exploration": 0.0,
            "outside_optimism": 0.0,
        }
        kwargs.update(overrides)
        return BestResponsePolicy(**kwargs)

    @staticmethod
    def _expected_bridge_score(
        policy: BestResponsePolicy,
        n_outside_nodes: int,
    ) -> float:
        direct_gain = policy.distance_discount
        spillover_gain = max(n_outside_nodes - 1, 0) * (policy.distance_discount**2)
        raw_bridge = policy.benefit_to_cost_ratio * (
            direct_gain + (policy.outside_optimism * spillover_gain)
        ) - 1.0
        return float(
            np.clip(
                raw_bridge / policy._score_denom,
                -1.0,
                1.0,
            )
        )

    def test_constructor_stores_new_parameters(self) -> None:
        policy = self._build_policy(
            n_nodes=5,
            seed=11,
            exploration=1.0,
            outside_optimism=0.75,
        )

        self.assertEqual(policy.exploration, 1.0)
        self.assertEqual(policy.outside_optimism, 0.75)

    def test_constructor_rejects_interval_parameters_outside_unit_interval(self) -> None:
        invalid_parameters = (
            ("exploration", -0.1, r"exploration must be in \[0,1\]"),
            ("exploration", 1.1, r"exploration must be in \[0,1\]"),
            ("outside_optimism", -0.1, r"outside_optimism must be in \[0,1\]"),
            ("outside_optimism", 1.1, r"outside_optimism must be in \[0,1\]"),
        )

        for parameter_name, invalid_value, error_message in invalid_parameters:
            with self.subTest(parameter_name=parameter_name, value=invalid_value):
                with self.assertRaisesRegex(ValueError, error_message):
                    self._build_policy(**{parameter_name: invalid_value})

    def test_constructor_rejects_non_positive_benefit_ratio(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            r"benefit_to_cost_ratio must be > 0",
        ):
            self._build_policy(benefit_to_cost_ratio=0.0)

    def test_constructor_accepts_benefit_ratio_above_one(self) -> None:
        policy = self._build_policy(benefit_to_cost_ratio=1.5)

        self.assertAlmostEqual(policy.benefit_to_cost_ratio, 1.5)

    def test_constructor_rejects_non_positive_distance_discount(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            r"distance_discount must be in \(0,1\)",
        ):
            self._build_policy(distance_discount=0.0)

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

    def test_stable_log_mean_exp_matches_common_value_and_deterministic_limit(self) -> None:
        repeated_values = np.array([0.25, 0.25, 0.25], dtype=np.float32)
        mixed_values = np.array([-0.2, 0.6, 0.1], dtype=np.float32)

        self.assertAlmostEqual(stable_log_mean_exp(repeated_values, tau=0.7), 0.25)
        self.assertAlmostEqual(
            stable_log_mean_exp(mixed_values, tau=0.0),
            0.6,
        )

    def test_family_value_is_size_normalized_for_equal_action_scores(self) -> None:
        one_action = np.array([0.4], dtype=np.float32)
        many_actions = np.full(8, 0.4, dtype=np.float32)

        self.assertAlmostEqual(
            stable_log_mean_exp(one_action, tau=0.5),
            stable_log_mean_exp(many_actions, tau=0.5),
        )

    def test_deterministic_policy_passes_when_all_non_pass_values_are_negative(self) -> None:
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
            exploration=0.0,
            outside_optimism=0.0,
        )

        decision = policy.get_decision(observation, action_space)

        self.assertEqual(decision.actions, (Action(ActionType.PASS, source=0),))
        self.assertIsNotNone(decision.value)
        assert decision.value is not None
        self.assertAlmostEqual(float(decision.value["value"]), 0.0)
        self.assertEqual(decision.value["family"], ActionType.PASS.value)

    def test_positive_bridge_is_chosen_deterministically_on_large_empty_graph(self) -> None:
        adjacency = np.zeros((10, 10), dtype=np.int8)
        observation, action_space = self._build_observation_and_action_space(
            adjacency=adjacency,
            node_id=0,
            radius=3,
        )
        policy = self._build_policy(
            n_nodes=10,
            seed=11,
            benefit_to_cost_ratio=1.0,
            distance_discount=0.9,
            exploration=0.0,
            outside_optimism=1.0,
        )

        decision = policy.get_decision(observation, action_space)
        expected_bridge_score = self._expected_bridge_score(
            policy,
            len(observation.outside_node_idxs),
        )

        self.assertEqual(decision.actions[0].type, ActionType.BRIDGE)
        self.assertEqual(decision.actions[0], Action(ActionType.BRIDGE, source=0))
        self.assertIsNotNone(decision.value)
        assert decision.value is not None
        self.assertAlmostEqual(float(decision.value["value"]), expected_bridge_score)
        self.assertAlmostEqual(float(decision.value["family_value"]), expected_bridge_score)

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

        policy = self._build_policy(
            n_nodes=6,
            seed=23,
            benefit_to_cost_ratio=0.5,
            distance_discount=0.5,
            exploration=0.0,
            outside_optimism=0.0,
        )

        decision = policy.get_decision(observation, action_space)

        self.assertEqual(
            decision.actions,
            (Action(ActionType.DELETE, source=0, target=1),),
        )
        self.assertIsNotNone(decision.value)
        assert decision.value is not None
        self.assertAlmostEqual(float(decision.value["value"]), 3.0 / 14.0)

    def test_non_contiguous_bridge_action_returns_local_action(self) -> None:
        observation, action_space = self._build_observation_and_action_space(
            adjacency=np.zeros((10, 10), dtype=np.int8),
            node_id=4,
            radius=3,
        )
        self.assertEqual(observation.graph.vs["name"], [4])
        self.assertEqual(action_space[ActionType.PASS], [Action(ActionType.PASS, source=0)])
        self.assertEqual(action_space[ActionType.BRIDGE], [Action(ActionType.BRIDGE, source=0)])
        policy = self._build_policy(
            n_nodes=10,
            seed=29,
            benefit_to_cost_ratio=1.0,
            distance_discount=0.9,
            exploration=0.0,
            outside_optimism=1.0,
        )

        decision = policy.get_decision(observation, action_space)
        expected_bridge_score = self._expected_bridge_score(
            policy,
            len(observation.outside_node_idxs),
        )

        self.assertEqual(decision.actions[0].type, ActionType.BRIDGE)
        self.assertEqual(decision.actions[0].source, 0)
        self.assertIsNone(decision.actions[0].target)
        self.assertIsNotNone(decision.value)
        assert decision.value is not None
        self.assertAlmostEqual(float(decision.value["value"]), expected_bridge_score)

    def test_add_action_value_matches_one_step_delta(self) -> None:
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
        self.assertEqual(
            action_space[ActionType.ADD],
            [Action(ActionType.ADD, source=0, target=2)],
        )

        policy = self._build_policy(
            n_nodes=3,
            seed=31,
            benefit_to_cost_ratio=1.0,
            distance_discount=0.5,
            exploration=1.0,
            outside_optimism=0.0,
        )

        with patch(
            "network_creation.policies.best_response._sample_from_values",
            side_effect=[1, 0],
        ):
            decision = policy.get_decision(observation, action_space)

        self.assertEqual(
            decision.actions,
            (Action(ActionType.ADD, source=0, target=2),),
        )
        self.assertIsNotNone(decision.value)
        assert decision.value is not None
        self.assertAlmostEqual(float(decision.value["value"]), -0.25)


if __name__ == "__main__":
    unittest.main()
