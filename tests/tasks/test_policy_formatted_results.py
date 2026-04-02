"""Tests for the shared policy formatted-results contract."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from tasks.policies.formatted_results import PolicyFormattedResults


class PolicyFormattedResultsTest(unittest.TestCase):
    def test_dataclass_preserves_typed_sections(self) -> None:
        sample_labels = np.asarray([0, 1], dtype=np.int64)
        parameter_labels = ("a", "b")
        observable_labels = ("density.mean",)
        step_labels = np.asarray([0, 1, 2], dtype=np.int64)

        result = PolicyFormattedResults(
            inputs={
                "tensors": {
                    "parameters": np.asarray(
                        [[1.0, 2.0], [3.0, 4.0]],
                        dtype=np.float32,
                    )
                },
                "dims": {"parameters": ("sample", "parameter")},
                "labels": {
                    "sample": sample_labels,
                    "parameter": parameter_labels,
                },
            },
            outputs={
                "tensors": {
                    "final": np.asarray([[0.2], [0.4]], dtype=np.float32),
                    "timeseries": np.asarray(
                        [[[0.0, 0.1, 0.2]], [[0.2, 0.3, 0.4]]],
                        dtype=np.float32,
                    ),
                },
                "dims": {
                    "final": ("sample", "observable"),
                    "timeseries": ("sample", "observable", "step"),
                },
                "labels": {
                    "sample": sample_labels,
                    "observable": observable_labels,
                    "step": step_labels,
                },
            },
            metadata={
                "task_name": "myopic-policy",
                "task_class": "tasks.policies.myopic_policy:MyopicPolicyRun",
                "result_store_path": Path("/tmp/run.zarr"),
                "config_path": Path("/tmp/run.dispatch_config.json"),
                "average_completion_duration_ms": 12.5,
                "input_index": sample_labels,
                "output_schema": [],
            },
            failures={},
        )
        self.assertEqual(result.metadata["task_name"], "myopic-policy")
        self.assertEqual(result.metadata["average_completion_duration_ms"], 12.5)
        self.assertEqual(result.inputs["dims"]["parameters"], ("sample", "parameter"))
        self.assertEqual(result.outputs["dims"]["final"], ("sample", "observable"))
        np.testing.assert_allclose(
            result.outputs["tensors"]["final"],
            np.asarray([[0.2], [0.4]], dtype=np.float32),
        )
