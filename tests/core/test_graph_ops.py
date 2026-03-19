"""Tests for graph visibility, action spaces, and multi-action updates."""

from __future__ import annotations

import unittest

import numpy as np

from network_creation.core.graph import (
    apply_actions,
    build_observation_from_state,
    build_state_from_ownership_matrix,
    get_action_space,
    get_visible_node_ids,
)
from network_creation.core.model import Action, ActionSpace, ActionType


class GraphOpsTest(unittest.TestCase):
    @staticmethod
    def _larger_state():
        ownership = np.array(
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
        return build_state_from_ownership_matrix(ownership)

    def test_get_visible_node_ids_uses_radius(self) -> None:
        ownership = np.array(
            [
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            dtype=np.int8,
        )
        state = build_state_from_ownership_matrix(ownership)

        self.assertEqual(get_visible_node_ids(state.graph, node_id=0, radius=0), [0])
        self.assertEqual(get_visible_node_ids(state.graph, node_id=0, radius=1), [0, 1])
        self.assertEqual(get_visible_node_ids(state.graph, node_id=0, radius=2), [0, 1, 2])

    def test_observation_keeps_original_node_ids(self) -> None:
        ownership = np.array(
            [
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            dtype=np.int8,
        )
        state = build_state_from_ownership_matrix(ownership)

        observation = build_observation_from_state(state, node_id=0, radius=2)

        self.assertEqual(observation.node_id, 0)
        self.assertEqual(observation.graph.vs["name"], [0, 1, 2])
        np.testing.assert_array_equal(observation.outside_node_ids, np.array([3]))
        np.testing.assert_array_equal(observation.paid_node_ids, np.array([1]))

    def test_action_space_separates_add_bridge_delete_and_pass(self) -> None:
        ownership = np.array(
            [
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            dtype=np.int8,
        )
        state = build_state_from_ownership_matrix(ownership)
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
            0: {
                ActionType.ADD: [],
                ActionType.BRIDGE: [
                    Action(ActionType.BRIDGE, source=1, target=0),
                    Action(ActionType.BRIDGE, source=1, target=2),
                    Action(ActionType.BRIDGE, source=1, target=3),
                    Action(ActionType.BRIDGE, source=1, target=4),
                    Action(ActionType.BRIDGE, source=1, target=5),
                ],
                ActionType.DELETE: [],
                ActionType.PASS: [Action(ActionType.PASS, source=1)],
            },
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
                    Action(ActionType.ADD, source=1, target=5),
                ],
                ActionType.BRIDGE: [Action(ActionType.BRIDGE, source=1, target=4)],
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

    def test_action_space_uses_global_node_ids_when_subgraph_indices_differ(self) -> None:
        state = self._larger_state()
        observation = build_observation_from_state(state, node_id=4, radius=2)

        self.assertEqual(observation.graph.vs["name"], [2, 3, 4])

        action_space = get_action_space(observation)

        self.assertEqual(
            action_space[ActionType.ADD],
            [Action(ActionType.ADD, source=4, target=2)],
        )
        self.assertEqual(
            action_space[ActionType.BRIDGE],
            [
                Action(ActionType.BRIDGE, source=4, target=0),
                Action(ActionType.BRIDGE, source=4, target=1),
                Action(ActionType.BRIDGE, source=4, target=5),
            ],
        )
        self.assertEqual(action_space[ActionType.DELETE], [])
        self.assertEqual(
            action_space[ActionType.PASS],
            [Action(ActionType.PASS, source=4)],
        )

    def test_apply_actions_supports_simultaneous_pre_step_decisions(self) -> None:
        state = build_state_from_ownership_matrix(np.zeros((2, 2), dtype=np.int8))

        updated = apply_actions(
            state,
            actions=[
                Action(ActionType.BRIDGE, source=0, target=1),
                Action(ActionType.BRIDGE, source=1, target=0),
            ],
        )

        expected = np.array(
            [
                [0, 1],
                [1, 0],
            ],
            dtype=np.int8,
        )
        np.testing.assert_array_equal(updated.ownership_matrix, expected)
        self.assertTrue(updated.graph.are_adjacent(0, 1))
        self.assertEqual(updated.graph.ecount(), 1)

    def test_apply_actions_applies_delete_before_add(self) -> None:
        ownership = np.array(
            [
                [0, 1],
                [0, 0],
            ],
            dtype=np.int8,
        )
        state = build_state_from_ownership_matrix(ownership)

        updated = apply_actions(
            state,
            actions=[
                Action(ActionType.DELETE, source=0, target=1),
                Action(ActionType.ADD, source=1, target=0),
            ],
        )

        expected = np.array(
            [
                [0, 0],
                [1, 0],
            ],
            dtype=np.int8,
        )
        np.testing.assert_array_equal(updated.ownership_matrix, expected)
        self.assertTrue(updated.graph.are_adjacent(0, 1))


if __name__ == "__main__":
    unittest.main()
