"""Tests for core value objects."""

from __future__ import annotations

import unittest

from network_creation.core.local_global_map import LocalGlobalMap
from network_creation.core.model import Action, ActionType, ActorScheduler


class ActionTest(unittest.TestCase):
    def test_action_preserves_supplied_fields(self) -> None:
        action = Action(ActionType.PASS, source=0, target=1)

        self.assertEqual(action.type, ActionType.PASS)
        self.assertEqual(action.source, 0)
        self.assertEqual(action.target, 1)


class LocalGlobalMapTest(unittest.TestCase):
    def test_uses_dense_local_and_global_indices(self) -> None:
        local_global = LocalGlobalMap([2, 0, 1])

        self.assertEqual(local_global[0], 2)
        self.assertEqual(local_global[1], 0)
        self.assertEqual(local_global[2], 1)
        self.assertEqual(local_global.inverse[0], 1)
        self.assertEqual(local_global.inverse[1], 2)
        self.assertEqual(local_global.inverse[2], 0)

    def test_compares_equal_to_plain_mapping(self) -> None:
        local_global = LocalGlobalMap([2, 0, 1])

        self.assertEqual(local_global, {0: 2, 1: 0, 2: 1})

    def test_rejects_duplicate_global_node_ids(self) -> None:
        with self.assertRaisesRegex(ValueError, "unique"):
            LocalGlobalMap([0, 1, 1])

    def test_rejects_out_of_range_global_node_ids(self) -> None:
        with self.assertRaisesRegex(ValueError, "range"):
            LocalGlobalMap([0, 1, 3])


class ActorSchedulerTest(unittest.TestCase):
    def test_rejects_invalid_actor_count(self) -> None:
        cases = (
            (1, 0),
            (1, -1),
            (1, 2),
            (1, 1.0),
            (1, float("nan")),
            (1, True),
            (5, 2.5),
        )
        for n_agents, n_actors in cases:
            with self.subTest(n_agents=n_agents, n_actors=n_actors):
                with self.assertRaisesRegex(
                    ValueError, "1 <= n_actors <= n_agents"
                ):
                    ActorScheduler(seed=1, n_agents=n_agents, n_actors=n_actors)

    def test_rejects_non_positive_agent_count(self) -> None:
        with self.assertRaisesRegex(ValueError, "n_agents must be a positive integer"):
            ActorScheduler(seed=1, n_agents=0, n_actors=1)

    def test_select_agents_uses_actor_count_and_all_agent_fast_path(self) -> None:
        agents = ("a0", "a1", "a2", "a3", "a4")
        cases = (
            (ActorScheduler(seed=0, n_agents=5, n_actors=1), ("a4",)),
            (ActorScheduler(seed=0, n_agents=5, n_actors=2), ("a3", "a4")),
            (ActorScheduler(seed=0, n_agents=5, n_actors=5), agents),
        )

        for scheduler, expected_agents in cases:
            with self.subTest(scheduler=scheduler):
                selected_agents = scheduler.select_agents(agents)
                self.assertEqual(selected_agents, expected_agents)
                if expected_agents is agents:
                    self.assertIs(selected_agents, agents)

    def test_select_agents_samples_from_supplied_agents_tuple(self) -> None:
        scheduler = ActorScheduler(seed=0, n_agents=5, n_actors=1)
        agents = ("a0", "a1")

        self.assertIn(scheduler.select_agents(agents)[0], agents)


if __name__ == "__main__":
    unittest.main()
