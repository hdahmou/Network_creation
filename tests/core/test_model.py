"""Tests for core value objects."""

from __future__ import annotations

import unittest

from network_creation.core.model import Action, ActionType


class ActionTest(unittest.TestCase):
    def test_pass_cannot_have_target(self) -> None:
        with self.assertRaises(ValueError):
            Action(ActionType.PASS, source=0, target=1)

    def test_non_pass_requires_target(self) -> None:
        with self.assertRaises(ValueError):
            Action(ActionType.ADD, source=0)

    def test_self_links_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            Action(ActionType.BRIDGE, source=2, target=2)


if __name__ == "__main__":
    unittest.main()

