"""Tests for the task catalog lookup helper."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from tasks.catalog import load_task
from tests.tasks import support_tasks as _support_tasks


class TaskCatalogTest(unittest.TestCase):
    def test_load_task_returns_a_task_instance(self) -> None:
        with patch.dict("tasks.catalog.TASKS", _support_tasks.TEST_TASK_REGISTRY, clear=True):
            task = load_task("vector")

        self.assertIsInstance(task, _support_tasks.VectorTask)

    def test_load_task_raises_with_available_task_names(self) -> None:
        with patch.dict("tasks.catalog.TASKS", _support_tasks.TEST_TASK_REGISTRY, clear=True):
            with self.assertRaisesRegex(
                ValueError,
                "Unknown task 'missing'. Available tasks: generated, runtime_failure, structured, vector",
            ):
                load_task("missing")


if __name__ == "__main__":
    unittest.main()
