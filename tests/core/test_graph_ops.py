"""Tests for graph visibility, action spaces, and multi-action updates."""

from __future__ import annotations

import unittest

from bidict import bidict
import igraph as ig
import numpy as np

from network_creation.core.graph import (
    apply_actions,
    build_observation_from_state,
    build_state_from_graph,
    get_action_space,
)
from network_creation.core.model import Action, ActionSpace, ActionType
from tests.graph_factories import graph_from_adjacency_matrix


class GraphOpsTest(unittest.TestCase):
    @staticmethod
    def _larger_state():
        adjacency = np.array(
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
        return build_state_from_graph(graph_from_adjacency_matrix(adjacency))

    def test_build_observation_from_state_uses_radius(self) -> None:
        adjacency = np.array(
            [
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            dtype=np.int8,
        )
        state = build_state_from_graph(graph_from_adjacency_matrix(adjacency))

        radius_one = build_observation_from_state(state, node_id=0, radius=1)
        radius_two = build_observation_from_state(state, node_id=0, radius=2)

        self.assertEqual(radius_one.graph.vs["name"], [0, 1])
        np.testing.assert_array_equal(radius_one.outside_node_idxs, np.array([2, 3]))
        self.assertEqual(radius_two.graph.vs["name"], [0, 1, 2])
        np.testing.assert_array_equal(radius_two.outside_node_idxs, np.array([3]))

    def test_build_state_from_graph_copies_graph_and_assigns_vertex_names(self) -> None:
        graph = ig.Graph(n=2, edges=[(0, 1)], directed=True)

        state = build_state_from_graph(graph)

        self.assertIsNot(state.graph, graph)
        self.assertNotIn("name", graph.vs.attributes())
        self.assertEqual(state.graph.vs["name"], [0, 1])

    def test_build_state_from_graph_accepts_reciprocal_arcs(self) -> None:
        state = build_state_from_graph(
            ig.Graph(n=2, edges=[(0, 1), (1, 0)], directed=True)
        )

        self.assertEqual(sorted(state.graph.get_edgelist()), [(0, 1), (1, 0)])
        self.assertEqual({frozenset(edge) for edge in state.graph.get_edgelist()}, {frozenset((0, 1))})

    def test_observation_keeps_original_node_ids_in_mapping(self) -> None:
        adjacency = np.array(
            [
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            dtype=np.int8,
        )
        state = build_state_from_graph(graph_from_adjacency_matrix(adjacency))

        observation = build_observation_from_state(state, node_id=0, radius=2)

        self.assertEqual(observation.node_idx, 0)
        self.assertEqual(observation.graph.vs["name"], [0, 1, 2])
        np.testing.assert_array_equal(observation.outside_node_idxs, np.array([3]))
        self.assertEqual(observation.local_global, {0: 0, 1: 1, 2: 2, 3: 3})

    def test_action_space_separates_add_bridge_delete_and_pass(self) -> None:
        adjacency = np.array(
            [
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            dtype=np.int8,
        )
        state = build_state_from_graph(graph_from_adjacency_matrix(adjacency))
        observation = build_observation_from_state(state, node_id=0, radius=2)

        action_space = get_action_space(observation)

        self.assertIsInstance(action_space, ActionSpace)
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

    def test_action_space_changes_with_radius_on_larger_graph(self) -> None:
        state = self._larger_state()

        expected_by_radius = {
            1: {
                ActionType.ADD: [],
                ActionType.BRIDGE: [
                    Action(ActionType.BRIDGE, source=1, target=3),
                    Action(ActionType.BRIDGE, source=1, target=4),
                    Action(ActionType.BRIDGE, source=1, target=5),
                ],
                ActionType.DELETE: [],
                ActionType.PASS: [Action(ActionType.PASS, source=1)],
            },
            2: {
                ActionType.ADD: [
                    Action(ActionType.ADD, source=1, target=3),
                    Action(ActionType.ADD, source=1, target=4),
                ],
                ActionType.BRIDGE: [Action(ActionType.BRIDGE, source=1, target=5)],
                ActionType.DELETE: [],
                ActionType.PASS: [Action(ActionType.PASS, source=1)],
            },
            3: {
                ActionType.ADD: [
                    Action(ActionType.ADD, source=1, target=3),
                    Action(ActionType.ADD, source=1, target=4),
                    Action(ActionType.ADD, source=1, target=5),
                ],
                ActionType.BRIDGE: [],
                ActionType.DELETE: [],
                ActionType.PASS: [Action(ActionType.PASS, source=1)],
            },
        }

        for radius, expected in expected_by_radius.items():
            with self.subTest(radius=radius):
                observation = build_observation_from_state(state, node_id=1, radius=radius)
                action_space = get_action_space(observation)

                self.assertEqual(action_space[ActionType.ADD], expected[ActionType.ADD])
                self.assertEqual(
                    action_space[ActionType.BRIDGE],
                    expected[ActionType.BRIDGE],
                )
                self.assertEqual(
                    action_space[ActionType.DELETE],
                    expected[ActionType.DELETE],
                )
                self.assertEqual(action_space[ActionType.PASS], expected[ActionType.PASS])

    def test_action_space_uses_local_indices_when_subgraph_names_differ(self) -> None:
        state = self._larger_state()
        observation = build_observation_from_state(state, node_id=4, radius=2)

        self.assertEqual(observation.graph.vs["name"], [2, 3, 4])
        self.assertEqual(observation.node_idx, 2)
        np.testing.assert_array_equal(
            observation.outside_node_idxs,
            np.array([3, 4, 5]),
        )
        self.assertIsInstance(observation.local_global, bidict)
        self.assertEqual(
            observation.local_global,
            {0: 2, 1: 3, 2: 4, 3: 0, 4: 1, 5: 5},
        )
        self.assertEqual(observation.global_from_local(0), 2)
        self.assertEqual(observation.global_from_local(2), 4)
        self.assertEqual(observation.global_from_local(5), 5)
        self.assertEqual(observation.local_from_global(4), 2)
        self.assertEqual(observation.local_from_global(5), 5)

        action_space = get_action_space(observation)

        self.assertEqual(
            action_space[ActionType.ADD],
            [Action(ActionType.ADD, source=2, target=0)],
        )
        self.assertEqual(
            action_space[ActionType.BRIDGE],
            [
                Action(ActionType.BRIDGE, source=2, target=3),
                Action(ActionType.BRIDGE, source=2, target=4),
                Action(ActionType.BRIDGE, source=2, target=5),
            ],
        )
        self.assertEqual(action_space[ActionType.DELETE], [])
        self.assertEqual(
            action_space[ActionType.PASS],
            [Action(ActionType.PASS, source=2)],
        )

    def test_delete_actions_use_local_indices_when_subgraph_names_differ(self) -> None:
        state = self._larger_state()
        observation = build_observation_from_state(state, node_id=3, radius=1)

        self.assertEqual(observation.graph.vs["name"], [2, 3, 4])

        action_space = get_action_space(observation)

        self.assertEqual(
            action_space[ActionType.DELETE],
            [
                Action(ActionType.DELETE, source=1, target=0),
                Action(ActionType.DELETE, source=1, target=2),
            ],
        )

    def test_apply_actions_supports_simultaneous_pre_step_decisions(self) -> None:
        state = build_state_from_graph(
            graph_from_adjacency_matrix(np.zeros((2, 2), dtype=np.int8))
        )

        updated = apply_actions(
            state,
            actions=[
                Action(ActionType.BRIDGE, source=0, target=1),
                Action(ActionType.BRIDGE, source=1, target=0),
            ],
        )

        self.assertEqual(sorted(updated.graph.get_edgelist()), [(0, 1), (1, 0)])
        self.assertTrue(updated.graph.are_adjacent(0, 1))
        self.assertTrue(updated.graph.are_adjacent(1, 0))
        self.assertEqual(updated.graph.ecount(), 2)
        self.assertEqual(
            {frozenset(edge) for edge in updated.graph.get_edgelist()},
            {frozenset((0, 1))},
        )

    def test_apply_actions_applies_add_before_delete_for_same_edge(self) -> None:
        state = build_state_from_graph(
            graph_from_adjacency_matrix(np.zeros((2, 2), dtype=np.int8))
        )

        updated = apply_actions(
            state,
            actions=[
                Action(ActionType.ADD, source=0, target=1),
                Action(ActionType.DELETE, source=0, target=1),
            ],
        )

        self.assertEqual(updated.graph.get_edgelist(), [])
        self.assertEqual(updated.graph.ecount(), 0)


if __name__ == "__main__":
    unittest.main()
