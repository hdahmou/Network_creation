"""Tests for random policy result formatting."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from tasks.policies import load_policy_results

from tests.tasks.policy_result_support import build_random_config, write_result_store


class RandomPolicyResultFormatterTest(unittest.TestCase):
    def test_from_zarr_reconstructs_timeseries_outputs(self) -> None:
        config = build_random_config()
        config["simulator"]["max_steps"] = 2
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
                config=config,
                outputs_by_index=outputs_by_index,
                total_inputs=1,
            )

            ds = load_policy_results(store_path)

        self.assertNotIn("final", ds)
        self.assertNotIn("parameters", ds)
        self.assertNotIn("timeseries", ds)
        self.assertEqual(ds["theta"].shape, (1, 1, 1, 4))
        self.assertEqual(ds["x"].dims, ("n_agents", "random_proportion", "parameter_row", "step", "observable"))
        np.testing.assert_array_equal(ds.coords["n_agents"].values, np.asarray([4]))
        np.testing.assert_allclose(ds.coords["random_proportion"].values, [1.0])
        np.testing.assert_array_equal(ds.coords["parameter_row"].values, np.asarray([0]))
        np.testing.assert_array_equal(ds.coords["step"].values, np.asarray([0, 1, 2]))
        np.testing.assert_array_equal(ds.coords["input_index"].values, np.asarray([[[0]]], dtype=np.int64))
        self.assertFalse(ds.attrs["has_failures"])
        self.assertEqual(
            tuple(ds.coords["theta_parameter"].values),
            (
                "n_agents",
                "benefit_to_cost_ratio",
                "distance_discount",
                "visibility_distance",
            ),
        )
        np.testing.assert_allclose(ds["prior_low"].values, [4.0, 0.5, 0.5, 4.0])
        np.testing.assert_allclose(ds["prior_high"].values, [4.0, 0.5, 0.5, 4.0])
        self.assertEqual(ds["x"].shape, (1, 1, 1, 3, 2))
        np.testing.assert_allclose(
            ds["x"].isel(step=-1).values,
            np.asarray([[[[2.0, 12.0]]]], dtype=np.float32),
        )

    def test_from_zarr_uses_observer_sampling_for_step_coordinates(self) -> None:
        config = build_random_config()
        config["simulator"]["max_steps"] = 10
        config["observers"]["defaults"]["sampling"]["every"] = 5
        outputs_by_index = [
            (
                0,
                {
                    "density.mean": np.asarray([0.0, 5.0, 10.0], dtype=np.float32),
                    "centered_mean_welfare.mean": np.asarray(
                        [20.0, 25.0, 30.0], dtype=np.float32
                    ),
                },
            )
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = write_result_store(
                Path(tmpdir),
                config=config,
                outputs_by_index=outputs_by_index,
                total_inputs=1,
            )

            ds = load_policy_results(store_path)

        np.testing.assert_array_equal(ds.coords["step"].values, np.asarray([0, 5, 10]))

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
                config=config,
                outputs_by_index=outputs_by_index,
                total_inputs=1,
            )

            ds = load_policy_results(store_path)

        self.assertNotIn("final", ds)
        self.assertEqual(ds["x"].shape, (1, 1, 1, 1, 2))
        self.assertEqual(
            tuple(ds.coords["theta_parameter"].values),
            (
                "n_agents",
                "benefit_to_cost_ratio",
                "distance_discount",
                "visibility_distance",
            ),
        )
        np.testing.assert_allclose(
            ds["x"].values,
            np.asarray([[[[[0.25, 12.5]]]]], dtype=np.float32),
        )
        self.assertEqual(
            tuple(ds.coords["observable"].values),
            ("density.final", "centered_mean_welfare.final"),
        )
