"""Tests for random policy result formatting."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from tasks.policies.random_policy import RandomPolicyRun

from ..policy_result_support import build_random_config, write_result_store


class RandomPolicyResultFormatterTest(unittest.TestCase):
    def test_from_zarr_handles_empty_parameter_axis_with_timeseries_outputs(self) -> None:
        config = build_random_config()
        outputs_by_index = [
            (
                0,
                {
                    "density.mean": np.asarray([0.0, 1.0, 2.0], dtype=np.float32),
                    "social_welfare.mean": np.asarray([10.0, 11.0, 12.0], dtype=np.float32),
                },
            )
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = write_result_store(
                Path(tmpdir),
                task_name="random-policy",
                task_class="tasks.policies.random_policy:RandomPolicyRun",
                config=config,
                outputs_by_index=outputs_by_index,
                total_inputs=1,
            )

            formatted = RandomPolicyRun.from_zarr(store_path)

        self.assertEqual(formatted.inputs["tensors"]["parameters"].shape, (1, 0))
        self.assertEqual(formatted.metadata["average_completion_duration_ms"], 1.0)
        self.assertEqual(tuple(formatted.inputs["labels"]["parameter"]), ())
        self.assertEqual(formatted.outputs["tensors"]["timeseries"].shape, (1, 2, 3))
        self.assertEqual(formatted.outputs["tensors"]["final"].shape, (1, 2))
        np.testing.assert_allclose(
            formatted.outputs["tensors"]["final"],
            np.asarray([[2.0, 12.0]], dtype=np.float32),
        )

    def test_from_zarr_supports_scalar_outputs_without_timeseries(self) -> None:
        config = build_random_config()
        outputs_by_index = [
            (
                0,
                {
                    "density.final": np.asarray(0.25, dtype=np.float32),
                    "social_welfare.final": np.asarray(12.5, dtype=np.float32),
                },
            )
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = write_result_store(
                Path(tmpdir),
                task_name="random-policy",
                task_class="tasks.policies.random_policy:RandomPolicyRun",
                config=config,
                outputs_by_index=outputs_by_index,
                total_inputs=1,
            )

            formatted = RandomPolicyRun.from_zarr(store_path)

        self.assertNotIn("timeseries", formatted.outputs["tensors"])
        self.assertEqual(formatted.metadata["average_completion_duration_ms"], 1.0)
        self.assertEqual(formatted.outputs["tensors"]["final"].shape, (1, 2))
        np.testing.assert_allclose(
            formatted.outputs["tensors"]["final"],
            np.asarray([[0.25, 12.5]], dtype=np.float32),
        )
        self.assertEqual(
            tuple(formatted.outputs["labels"]["observable"]),
            ("density.final", "social_welfare.final"),
        )
