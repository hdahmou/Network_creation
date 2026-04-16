"""Tests for the shared policy formatted-results contract."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from tasks.policies.results import PolicyFormattedResults


class PolicyFormattedResultsTest(unittest.TestCase):
    def test_dataclass_preserves_typed_sections(self) -> None:
        n_agents_labels = np.asarray([4, 8], dtype=np.int64)
        parameter_row_labels = np.asarray([0, 1], dtype=np.int64)
        parameter_labels = ("a", "b")
        observable_labels = ("density.mean",)
        step_labels = np.asarray([0, 1, 2], dtype=np.int64)

        result = PolicyFormattedResults(
            inputs={
                "tensors": {
                    "parameters": np.asarray(
                        [
                            [[1.0, 2.0], [3.0, 4.0]],
                            [[5.0, 6.0], [7.0, 8.0]],
                        ],
                        dtype=np.float32,
                    ),
                },
                "dims": {
                    "parameters": ("n_agents", "parameter_row", "parameter"),
                },
                "labels": {
                    "n_agents": n_agents_labels,
                    "parameter_row": parameter_row_labels,
                    "parameter": parameter_labels,
                },
            },
            outputs={
                "tensors": {
                    "final": np.asarray(
                        [[[0.2], [0.4]], [[0.6], [0.8]]],
                        dtype=np.float32,
                    ),
                    "timeseries": np.asarray(
                        [
                            [
                                [[0.0, 0.1, 0.2]],
                                [[0.2, 0.3, 0.4]],
                            ],
                            [
                                [[0.4, 0.5, 0.6]],
                                [[0.6, 0.7, 0.8]],
                            ],
                        ],
                        dtype=np.float32,
                    ),
                },
                "dims": {
                    "final": ("n_agents", "parameter_row", "observable"),
                    "timeseries": ("n_agents", "parameter_row", "observable", "step"),
                },
                "labels": {
                    "n_agents": n_agents_labels,
                    "parameter_row": parameter_row_labels,
                    "observable": observable_labels,
                    "step": step_labels,
                },
            },
            status={
                "tensors": {
                    "success": np.asarray(
                        [[True, True], [False, True]],
                        dtype=np.bool_,
                    ),
                    "error": np.asarray(
                        [["", ""], ["boom", ""]],
                        dtype="<U4",
                    ),
                    "duration_ms": np.asarray(
                        [[1.0, 2.0], [3.0, 4.0]],
                        dtype=np.float64,
                    ),
                },
                "dims": {
                    "success": ("n_agents", "parameter_row"),
                    "error": ("n_agents", "parameter_row"),
                    "duration_ms": ("n_agents", "parameter_row"),
                },
                "labels": {
                    "n_agents": n_agents_labels,
                    "parameter_row": parameter_row_labels,
                },
            },
            metadata={
                "task_name": "myopic-policy",
                "task_class": "tasks.policies.myopic:MyopicPolicyRun",
                "result_store_path": Path("/tmp/run.zarr"),
                "config_path": Path("/tmp/run.dispatch_config.json"),
                "average_completion_duration_ms": 12.5,
                "input_index": np.asarray([[0, 1], [2, 3]], dtype=np.int64),
                "output_schema": [],
                "simulation": {
                    "n_agents": 4,
                    "actor_scheduler": {"n_actors": 1},
                    "seed": 11,
                    "max_steps": 3,
                    "n_episodes": 2,
                    "stagnation_patience": None,
                    "initial_graph": {"kind": "empty"},
                },
            },
        )
        self.assertEqual(result.metadata["task_name"], "myopic-policy")
        self.assertEqual(result.metadata["average_completion_duration_ms"], 12.5)
        self.assertEqual(result.metadata["simulation"]["n_agents"], 4)
        self.assertEqual(
            result.inputs["dims"]["parameters"],
            ("n_agents", "parameter_row", "parameter"),
        )
        self.assertEqual(
            result.outputs["dims"]["final"],
            ("n_agents", "parameter_row", "observable"),
        )
        self.assertEqual(
            result.status["dims"]["success"],
            ("n_agents", "parameter_row"),
        )
        np.testing.assert_allclose(
            result.outputs["tensors"]["final"],
            np.asarray(
                [[[0.2], [0.4]], [[0.6], [0.8]]],
                dtype=np.float32,
            ),
        )
