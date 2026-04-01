"""Tests for core value objects."""

from __future__ import annotations

import unittest

from network_creation.core.model import Action, ActionType


class ActionTest(unittest.TestCase):
    def test_action_preserves_supplied_fields(self) -> None:
        action = Action(ActionType.PASS, source=0, target=1)

        self.assertEqual(action.type, ActionType.PASS)
        self.assertEqual(action.source, 0)
        self.assertEqual(action.target, 1)


if __name__ == "__main__":
    unittest.main()
