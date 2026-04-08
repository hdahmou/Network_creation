"""Tests for core value objects."""

from __future__ import annotations

import unittest

from network_creation.core.model import Action, ActionType, ActorScheduler


class ActionTest(unittest.TestCase):
    def test_action_preserves_supplied_fields(self) -> None:
        action = Action(ActionType.PASS, source=0, target=1)

        self.assertEqual(action.type, ActionType.PASS)
        self.assertEqual(action.source, 0)
        self.assertEqual(action.target, 1)


class ActorSchedulerTest(unittest.TestCase):
    def test_rejects_invalid_fraction(self) -> None:
        cases = (
            (1, 0.0),
            (1, -0.1),
            (1, 1.1),
            (1, float("nan")),
            (1, True),
            (5, 0.1),
        )
        for n_agents, fraction in cases:
            with self.subTest(n_agents=n_agents, fraction=fraction):
                with self.assertRaisesRegex(
                    ValueError, "1 / n_agents <= fraction <= 1"
                ):
                    ActorScheduler(seed=1, n_agents=n_agents, fraction=fraction)

    def test_rejects_non_positive_agent_count(self) -> None:
        with self.assertRaisesRegex(ValueError, "n_agents must be positive"):
            ActorScheduler(seed=1, n_agents=0, fraction=1.0)

    def test_select_agents_resolves_fraction_and_all_agent_fast_path(self) -> None:
        agents = ("a0", "a1", "a2", "a3", "a4")
        cases = (
            (ActorScheduler(seed=0, n_agents=5, fraction=0.2), ("a4",)),
            (ActorScheduler(seed=0, n_agents=5, fraction=0.4), ("a3", "a4")),
            (ActorScheduler(seed=0, n_agents=5, fraction=1.0), agents),
        )

        for scheduler, expected_agents in cases:
            with self.subTest(scheduler=scheduler):
                selected_agents = scheduler.select_agents(agents)
                self.assertEqual(selected_agents, expected_agents)
                if expected_agents is agents:
                    self.assertIs(selected_agents, agents)

    def test_select_agents_samples_from_supplied_agents_tuple(self) -> None:
        scheduler = ActorScheduler(seed=0, n_agents=5, fraction=0.2)
        agents = ("a0", "a1")

        self.assertIn(scheduler.select_agents(agents)[0], agents)


if __name__ == "__main__":
    unittest.main()
