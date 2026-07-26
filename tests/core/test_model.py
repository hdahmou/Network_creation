"""Tests for core value objects."""

from __future__ import annotations

import unittest

import numpy as np

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
                    ActorScheduler(
                        seed=1,
                        n_agents=n_agents,
                        n_actors=n_actors,
                        activation_weights=(1.0,) * n_agents,
                    )

    def test_rejects_non_positive_agent_count(self) -> None:
        with self.assertRaisesRegex(ValueError, "n_agents must be a positive integer"):
            ActorScheduler(
                seed=1,
                n_agents=0,
                n_actors=1,
                activation_weights=(),
            )

    def test_rejects_invalid_activation_weight_vectors(self) -> None:
        cases = (
            ((), "shape"),
            ((1.0, 2.0), "shape"),
            ((1.0, 2.0, 3.0, 4.0), "shape"),
            ([[1.0, 2.0, 3.0]], ".*"),
            ((1.0, float("nan"), 3.0), "finite and positive"),
            ((1.0, float("inf"), 3.0), "finite and positive"),
            ((1.0, 0.0, 3.0), "finite and positive"),
            ((1.0, -2.0, 3.0), "finite and positive"),
        )

        for activation_weights, message in cases:
            with self.subTest(activation_weights=activation_weights):
                with self.assertRaisesRegex(ValueError, message):
                    ActorScheduler(
                        seed=1,
                        n_agents=3,
                        n_actors=1,
                        activation_weights=activation_weights,
                    )

    def test_rejects_unequal_weights_for_multiple_actors(self) -> None:
        with self.assertRaisesRegex(ValueError, "unequal.*n_actors == 1"):
            ActorScheduler(
                seed=1,
                n_agents=3,
                n_actors=2,
                activation_weights=(1.0, 2.0, 3.0),
            )

    def test_weighted_selection_is_scale_invariant(self) -> None:
        scheduler = ActorScheduler(
            seed=1,
            n_agents=3,
            n_actors=1,
            activation_weights=(1.0, 2.0, 3.0),
        )
        scaled_scheduler = ActorScheduler(
            seed=1,
            n_agents=3,
            n_actors=1,
            activation_weights=(5.0e307, 1.0e308, 1.5e308),
        )

        agents = (0, 1, 2)
        self.assertEqual(
            tuple(scheduler.select_agents(agents) for _ in range(20)),
            tuple(scaled_scheduler.select_agents(agents) for _ in range(20)),
        )

    def test_weighted_selection_is_deterministic_for_seed(self) -> None:
        scheduler = ActorScheduler(
            seed=5,
            n_agents=3,
            n_actors=1,
            activation_weights=(1.0, 2.0, 7.0),
        )
        agents = ("a0", "a1", "a2")

        selected = tuple(scheduler.select_agents(agents)[0] for _ in range(10))

        self.assertEqual(
            selected,
            ("a2", "a2", "a2", "a1", "a0", "a2", "a2", "a0", "a0", "a2"),
        )

    def test_weighted_selection_tracks_expected_frequencies(self) -> None:
        scheduler = ActorScheduler(
            seed=17,
            n_agents=3,
            n_actors=1,
            activation_weights=(1.0, 2.0, 7.0),
        )
        agents = (0, 1, 2)
        counts = np.zeros(3, dtype=np.int_)

        for _ in range(20_000):
            counts[scheduler.select_agents(agents)[0]] += 1

        np.testing.assert_allclose(
            counts / counts.sum(),
            (0.1, 0.2, 0.7),
            atol=0.01,
        )

    def test_select_agents_uses_actor_count_and_all_agent_fast_path(self) -> None:
        agents = ("a0", "a1", "a2", "a3", "a4")
        uniform_weights = (1.0,) * 5
        cases = (
            (
                ActorScheduler(
                    seed=0,
                    n_agents=5,
                    n_actors=1,
                    activation_weights=uniform_weights,
                ),
                ("a4",),
            ),
            (
                ActorScheduler(
                    seed=0,
                    n_agents=5,
                    n_actors=2,
                    activation_weights=uniform_weights,
                ),
                ("a3", "a4"),
            ),
            (
                ActorScheduler(
                    seed=0,
                    n_agents=5,
                    n_actors=5,
                    activation_weights=uniform_weights,
                ),
                agents,
            ),
        )

        for scheduler, expected_agents in cases:
            with self.subTest(scheduler=scheduler):
                selected_agents = scheduler.select_agents(agents)
                self.assertEqual(selected_agents, expected_agents)
                if expected_agents is agents:
                    self.assertIs(selected_agents, agents)

    def test_select_agents_samples_from_supplied_agents_tuple(self) -> None:
        scheduler = ActorScheduler(
            seed=0,
            n_agents=5,
            n_actors=1,
            activation_weights=(1.0,) * 5,
        )
        agents = ("a0", "a1")

        self.assertIn(scheduler.select_agents(agents)[0], agents)

    def test_scalar_weight_preserves_uniform_selection(self) -> None:
        agents = ("a0", "a1", "a2", "a3", "a4")
        default_scheduler = ActorScheduler(
            seed=0,
            n_agents=5,
            n_actors=2,
            activation_weights=(1.0,) * 5,
        )
        equal_weight_scheduler = ActorScheduler(
            seed=0,
            n_agents=5,
            n_actors=2,
            activation_weights=9.0,
        )

        for _ in range(10):
            self.assertEqual(
                equal_weight_scheduler.select_agents(agents),
                default_scheduler.select_agents(agents),
            )


if __name__ == "__main__":
    unittest.main()
