"""Tests for random policy result formatting."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from tasks.policies import load_policy_results

from ..policy_result_support import build_random_config, write_result_store


class RandomPolicyResultFormatterTest(unittest.TestCase):
    def test_from_zarr_reconstructs_timeseries_outputs(self) -> None:
        config = build_random_config()
        outputs_by_index = [
            (
                0,
                {
                    "density.mean": np.asarray([0.0, 1.0, 2.0], dtype=np.float32),
                    "centered_mean_welfare.mean": np.asarray([10.0, 11.0, 12.0], dtype=np.float32),
                },
            )
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = write_result_store(
                Path(tmpdir),
                task_name="random-policy",
                config=config,
                outputs_by_index=outputs_by_index,
                total_inputs=1,
            )

            ds = load_policy_results(store_path)

        self.assertNotIn("final", ds)
        self.assertNotIn("parameters", ds)
        self.assertNotIn("timeseries", ds)
        self.assertEqual(ds["theta"].shape, (1, 2))
        np.testing.assert_array_equal(
            ds.coords["input_index"].values,
            np.asarray([0], dtype=np.int64),
        )
        self.assertEqual(ds.attrs["average_completion_duration_ms"], 1.0)
        self.assertFalse(ds.attrs["has_failures"])
        self.assertEqual(
            tuple(ds.coords["theta_parameter"].values),
            ("n_agents", "visibility_distance"),
        )
        np.testing.assert_allclose(ds["prior_low"].values, [4.0, 4.0])
        np.testing.assert_allclose(ds["prior_high"].values, [4.0, 4.0])
        self.assertEqual(ds["x"].shape, (1, 3, 2))
        np.testing.assert_allclose(
            ds["x"].isel(step=-1).values,
            np.asarray([[2.0, 12.0]], dtype=np.float32),
        )

    def test_from_zarr_supports_scalar_outputs_without_timeseries(self) -> None:
        config = build_random_config()
        outputs_by_index = [
            (
                0,
                {
                    "density.final": np.asarray(0.25, dtype=np.float32),
                    "centered_mean_welfare.final": np.asarray(12.5, dtype=np.float32),
                },
            )
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = write_result_store(
                Path(tmpdir),
                task_name="random-policy",
                config=config,
                outputs_by_index=outputs_by_index,
                total_inputs=1,
            )

            ds = load_policy_results(store_path)

        self.assertNotIn("final", ds)
        self.assertEqual(ds["x"].shape, (1, 1, 2))
        self.assertEqual(ds.attrs["average_completion_duration_ms"], 1.0)
        self.assertEqual(
            tuple(ds.coords["theta_parameter"].values),
            ("n_agents", "visibility_distance"),
        )
        np.testing.assert_allclose(
            ds["x"].values,
            np.asarray([[[0.25, 12.5]]], dtype=np.float32),
        )
        self.assertEqual(
            tuple(ds.coords["observable"].values),
            ("density.final", "centered_mean_welfare.final"),
        )
