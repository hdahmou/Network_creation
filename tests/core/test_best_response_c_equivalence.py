from __future__ import annotations

import math
import random
import unittest
from unittest.mock import patch

import igraph as ig
import numpy as np

from network_creation.core.graph import (
    build_observation_from_state,
    build_state_from_graph,
    get_action_space,
)
from network_creation.core.model import Action, ActionType
from network_creation.policies.best_response._kernel import (
    evaluate_best_response_terms,
)
from network_creation.policies.best_response import BestResponsePolicy
from tests.graph_factories import graph_from_adjacency_matrix


def _score_excluding_self(graph: ig.Graph, source: int, delta: float) -> float:
    distances = graph.distances(source=[source], mode="all")[0]
    total = 0.0
    for vertex, distance in enumerate(distances):
        if vertex == source or distance == math.inf:
            continue
        total += delta ** int(distance)
    return total


def _targets_for_source(graph: ig.Graph, source: int) -> tuple[list[int], list[int]]:
    add_targets = [
        vertex
        for vertex in range(graph.vcount())
        if vertex != source
        and graph.get_eid(source, vertex, directed=True, error=False) == -1
    ]
    delete_targets = [
        vertex for vertex in graph.neighbors(source, mode="out") if vertex != source
    ]
    return add_targets, delete_targets


def _reference_terms(
    graph: ig.Graph,
    source: int,
    delta: float,
    add_targets: list[int],
    delete_targets: list[int],
) -> tuple[float, list[float], list[float], list[float], list[int]]:
    base_score = _score_excluding_self(graph, source, delta)

    delete_scores: list[float] = []
    for delete_target in delete_targets:
        deleted_graph = graph.copy()
        edge_id = deleted_graph.get_eid(
            source,
            delete_target,
            directed=True,
            error=True,
        )
        deleted_graph.delete_edges([edge_id])
        delete_scores.append(_score_excluding_self(deleted_graph, source, delta))

    add_only_scores: list[float] = []
    best_swap_scores: list[float] = []
    best_swap_delete: list[int] = []
    for add_target in add_targets:
        add_graph = graph.copy()
        add_graph.add_edge(source, add_target)
        add_only_scores.append(_score_excluding_self(add_graph, source, delta))

        best_score = base_score
        best_delete = -1
        if delete_targets:
            best_score = -math.inf
            for delete_target in delete_targets:
                swap_graph = graph.copy()
                edge_id = swap_graph.get_eid(
                    source,
                    delete_target,
                    directed=True,
                    error=True,
                )
                swap_graph.delete_edges([edge_id])
                swap_graph.add_edge(source, add_target)
                candidate_score = _score_excluding_self(swap_graph, source, delta)
                if candidate_score > best_score:
                    best_score = candidate_score
                    best_delete = delete_target
            if best_delete == -1:
                best_score = base_score

        best_swap_scores.append(best_score)
        best_swap_delete.append(best_delete)

    return (
        base_score,
        delete_scores,
        add_only_scores,
        best_swap_scores,
        best_swap_delete,
    )


class BestResponseCKernelEquivalenceTest(unittest.TestCase):
    def assert_terms_match(
        self,
        graph: ig.Graph,
        source: int,
        delta: float,
        add_targets: list[int],
        delete_targets: list[int],
    ) -> None:
        actual = evaluate_best_response_terms(
            graph,
            source,
            delta,
            add_targets,
            delete_targets,
        )
        expected = _reference_terms(
            graph,
            source,
            delta,
            add_targets,
            delete_targets,
        )

        self.assertAlmostEqual(actual[0], expected[0], places=12)
        np.testing.assert_allclose(actual[1], expected[1], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(actual[2], expected[2], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(actual[3], expected[3], rtol=1e-12, atol=1e-12)
        if delete_targets:
            for delete_target in actual[4]:
                self.assertTrue(delete_target == -1 or delete_target in delete_targets)
        else:
            self.assertEqual(actual[4], expected[4])

    def test_c_terms_match_igraph_reference_on_random_graphs(self) -> None:
        delta = 0.91
        for n_nodes in range(5, 16):
            for seed in range(8):
                randomizer = random.Random(seed + 1000 * n_nodes)
                graph = ig.Graph(n=n_nodes, directed=True)
                edges: list[tuple[int, int]] = []
                for source in range(n_nodes):
                    for target in range(n_nodes):
                        if source == target:
                            continue
                        if randomizer.random() < 0.18:
                            edges.append((source, target))
                            if randomizer.random() < 0.03:
                                edges.append((source, target))
                graph.add_edges(edges)

                sources = randomizer.sample(range(n_nodes), min(3, n_nodes))
                for source in sources:
                    with self.subTest(n_nodes=n_nodes, seed=seed, source=source):
                        add_targets, delete_targets = _targets_for_source(
                            graph,
                            source,
                        )
                        self.assert_terms_match(
                            graph,
                            source,
                            delta,
                            add_targets,
                            delete_targets,
                        )

    def test_c_terms_cover_edge_cases(self) -> None:
        cases = [
            (4, [(1, 2)], 0),
            (4, [(0, 1), (0, 2), (0, 3)], 0),
            (5, [(0, 1), (2, 0), (1, 3)], 0),
        ]

        for n_nodes, edges, source in cases:
            with self.subTest(edges=edges):
                graph = ig.Graph(n=n_nodes, edges=edges, directed=True)
                add_targets, delete_targets = _targets_for_source(graph, source)
                self.assert_terms_match(
                    graph,
                    source,
                    0.84,
                    add_targets,
                    delete_targets,
                )

    def test_policy_values_match_c_terms_for_all_candidate_types(self) -> None:
        adjacency = np.zeros((6, 6), dtype=np.int8)
        adjacency[0, 1] = 1
        adjacency[1, 2] = 1
        adjacency[3, 0] = 1
        state = build_state_from_graph(graph_from_adjacency_matrix(adjacency))
        observation = build_observation_from_state(state, node_id=0, radius=1)
        action_space = get_action_space(observation)

        policy = BestResponsePolicy(
            n_nodes=6,
            seed=43,
            benefit_to_cost_ratio=0.75,
            distance_discount=0.84,
            exploration=0.5,
            outside_optimism=0.4,
        )

        delete_actions = action_space[ActionType.DELETE]
        add_actions = action_space[ActionType.ADD]
        bridge_actions = action_space[ActionType.BRIDGE]
        pass_action = action_space[ActionType.PASS][0]
        add_targets = [int(action.target) for action in add_actions]
        delete_targets = [int(action.target) for action in delete_actions]

        (
            base_score,
            delete_scores,
            add_only_scores,
            best_swap_scores,
            _best_swap_delete,
        ) = evaluate_best_response_terms(
            observation.graph,
            pass_action.source,
            policy.distance_discount,
            add_targets,
            delete_targets,
        )

        expected_actions: list[Action] = [*delete_actions, *add_actions]
        expected_values = [
            policy.benefit_to_cost_ratio * (score - base_score) + 1.0
            for score in delete_scores
        ]
        for add_only_score, swap_score in zip(
            add_only_scores,
            best_swap_scores,
            strict=True,
        ):
            add_only_delta = add_only_score - base_score
            swap_delta = swap_score - base_score
            add_only_value = policy.benefit_to_cost_ratio * add_only_delta - 1.0
            if delete_actions:
                swap_value = policy.benefit_to_cost_ratio * swap_delta
                expected_values.append(max(add_only_value, swap_value))
            else:
                expected_values.append(add_only_value)

        if bridge_actions:
            spillover_targets = max(len(observation.outside_node_idxs) - 1, 0)
            expected_bridge_gain = (
                policy.distance_discount
                + policy.outside_optimism
                * spillover_targets
                * policy.distance_discount**2
            )
            expected_actions.append(bridge_actions[0])
            expected_values.append(
                policy.benefit_to_cost_ratio * (expected_bridge_gain - base_score)
                - 1.0
            )
        expected_actions.append(pass_action)
        expected_values.append(0.0)

        self.assertGreater(len(delete_actions), 0)
        self.assertGreater(len(add_actions), 0)
        self.assertGreater(len(bridge_actions), 0)

        for chosen_index, expected_action in enumerate(expected_actions):
            with self.subTest(chosen_index=chosen_index, action=expected_action):
                with patch(
                    "network_creation.policies.best_response._sample_boltzmann",
                    return_value=chosen_index,
                ):
                    decision = policy.get_decision(observation, action_space)

                self.assertIsNotNone(decision.value)
                assert decision.value is not None
                self.assertAlmostEqual(
                    float(decision.value["value"]),
                    expected_values[chosen_index],
                    places=6,
                )

                if expected_action.type is ActionType.BRIDGE:
                    self.assertEqual(decision.actions[0].type, ActionType.BRIDGE)
                    self.assertEqual(decision.actions[0].source, expected_action.source)
                    self.assertIn(
                        int(decision.actions[0].target),
                        observation.outside_node_idxs,
                    )
                else:
                    self.assertEqual(decision.actions, (expected_action,))


if __name__ == "__main__":
    unittest.main()
