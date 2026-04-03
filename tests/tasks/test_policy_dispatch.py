"""Tests for policy result dispatch."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from tasks.policies import load_policy_results
from .policy_result_support import build_random_config, write_result_store


class PolicyDispatchTest(unittest.TestCase):
    def test_load_policy_results_rejects_unknown_task_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = write_result_store(
                Path(tmpdir),
                task_name="unsupported-policy",
                task_class="tests.tasks.support_tasks:VectorTask",
                config=build_random_config(),
                outputs_by_index=[
                    (
                        0,
                        {
                            "density.mean": np.asarray([0.0, 1.0, 2.0], dtype=np.float32)
                        },
                    )
                ],
                total_inputs=1,
            )

            with self.assertRaisesRegex(ValueError, "Unsupported policy task"):
                load_policy_results(store_path)
