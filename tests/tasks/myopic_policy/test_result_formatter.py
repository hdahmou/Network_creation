"""Tests for myopic policy result formatting."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from tasks.policies import load_policy_results

from tests.tasks.policy_result_support import (
    build_myopic_config,
    build_policy_parameter_samples,
    write_result_store,
)


class MyopicPolicyResultFormatterTest(unittest.TestCase):
    def test_from_zarr_reconstructs_parameters_and_sorts_rows(self) -> None:
        config = build_myopic_config(log2_budget=2)
        parameter_samples = build_policy_parameter_samples(config)

        outputs_by_index = [
            (
                2,
                {
                    "density.mean": np.asarray([20.0, 21.0, 22.0, 23.0], dtype=np.float32),
                    "centered_mean_welfare.mean": np.asarray(
                        [200.0, 201.0, 202.0, 203.0], dtype=np.float32
                    ),
                },
            ),
            (
                0,
                {
                    "density.mean": np.asarray([0.0, 1.0, 2.0, 3.0], dtype=np.float32),
                    "centered_mean_welfare.mean": np.asarray(
                        [100.0, 101.0, 102.0, 103.0], dtype=np.float32
                    ),
                },
            ),
            (
                1,
                {
                    "density.mean": np.asarray([10.0, 11.0, 12.0, 13.0], dtype=np.float32),
                    "centered_mean_welfare.mean": np.asarray(
                        [110.0, 111.0, 112.0, 113.0], dtype=np.float32
                    ),
                },
            ),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = write_result_store(
                Path(tmpdir),
                config=config,
                outputs_by_index=outputs_by_index,
                total_inputs=4,
            )

            ds = load_policy_results(store_path)

        self.assertNotIn("final", ds)
        self.assertNotIn("parameters", ds)
        self.assertNotIn("timeseries", ds)
        self.assertFalse(ds.attrs["has_failures"])
        np.testing.assert_array_equal(
            ds.coords["input_index"].values,
            np.asarray([[[0, 1, 2, 3]]], dtype=np.int64),
        )
        np.testing.assert_allclose(
            ds["theta"].values,
            np.concatenate(
                [
                    np.full((1, 1, 4, 1), 4.0, dtype=np.float32),
                    parameter_samples.parameter_values[None, None, :, :],
                ],
                axis=-1,
            ),
        )
        np.testing.assert_array_equal(
            ds.coords["parameter_row"].values,
            np.asarray([0, 1, 2, 3], dtype=np.int64),
        )
        self.assertEqual(
            tuple(ds.coords["theta_parameter"].values),
            (
                "n_agents",
                "benefit_to_cost_ratio",
                "distance_discount",
                "visibility_distance",
                "bridge_probability",
                "agent.activation_weight",
            ),
        )
        np.testing.assert_allclose(
            ds["prior_low"].values,
            np.asarray([4.0, 0.5, 0.1, 4.0, 0.0, 1.0], dtype=np.float32),
        )
        np.testing.assert_allclose(
            ds["prior_high"].values,
            np.asarray([4.0, 0.5, 0.9, 4.0, 1.0, 1.0], dtype=np.float32),
        )
        self.assertEqual(
            tuple(ds.coords["observable"].values),
            ("density.mean", "centered_mean_welfare.mean"),
        )
        self.assertEqual(ds["x"].shape, (1, 1, 4, 4, 2))
        np.testing.assert_allclose(
            ds["x"].values[0, 0, 0, :, 0],
            np.asarray([0.0, 1.0, 2.0, 3.0], dtype=np.float32),
        )
        np.testing.assert_allclose(
            ds["x"].values[0, 0, 1, :, 1],
            np.asarray([110.0, 111.0, 112.0, 113.0], dtype=np.float32),
        )
        np.testing.assert_allclose(
            ds["x"].isel(step=-1).values,
            np.asarray(
                [[[
                    [3.0, 103.0],
                    [13.0, 113.0],
                    [23.0, 203.0],
                    [np.nan, np.nan],
                ]]],
                dtype=np.float32,
            ),
            equal_nan=True,
        )

    def test_from_zarr_reconstructs_n_agents_sweep_inputs(self) -> None:
        config = build_myopic_config(log2_budget=1)
        config["population"]["n_agents"] = [3, 5]
        parameter_samples = build_policy_parameter_samples(config)

        outputs_by_index = [
            (
                3,
                {
                    "density.mean": np.asarray([30.0, 31.0, 32.0, 33.0], dtype=np.float32),
                    "centered_mean_welfare.mean": np.asarray(
                        [300.0, 301.0, 302.0, 303.0], dtype=np.float32
                    ),
                },
            ),
            (
                0,
                {
                    "density.mean": np.asarray([0.0, 1.0, 2.0, 3.0], dtype=np.float32),
                    "centered_mean_welfare.mean": np.asarray(
                        [100.0, 101.0, 102.0, 103.0], dtype=np.float32
                    ),
                },
            ),
            (
                2,
                {
                    "density.mean": np.asarray([20.0, 21.0, 22.0, 23.0], dtype=np.float32),
                    "centered_mean_welfare.mean": np.asarray(
                        [200.0, 201.0, 202.0, 203.0], dtype=np.float32
                    ),
                },
            ),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = write_result_store(
                Path(tmpdir),
                config=config,
                outputs_by_index=outputs_by_index,
                total_inputs=4,
            )

            ds = load_policy_results(store_path)

        np.testing.assert_array_equal(
            ds.coords["input_index"].values,
            np.asarray([[[0, 1]], [[2, 3]]], dtype=np.int64),
        )
        np.testing.assert_allclose(
            ds["theta"].values,
            np.asarray(
                [
                    [[[3.0, *parameter_samples.parameter_values[0]],
                      [3.0, *parameter_samples.parameter_values[1]]]],
                    [[[5.0, *parameter_samples.parameter_values[0]],
                      [5.0, *parameter_samples.parameter_values[1]]]],
                ],
                dtype=np.float32,
            ),
        )
        np.testing.assert_array_equal(
            ds.coords["parameter_row"].values,
            np.asarray([0, 1], dtype=np.int64),
        )

    def test_from_zarr_uses_resolved_scaled_inverse_benefit_to_cost_ratio(self) -> None:
        config = build_myopic_config(log2_budget=0)
        config["population"]["n_agents"] = [5, 10]
        config["game"]["benefit_to_cost_ratio"] = {
            "kind": "scaled_inverse_agents",
            "numerator": 10.0,
            "factor": 2.0,
        }

        outputs_by_index = [
            (
                1,
                {
                    "density.mean": np.asarray([10.0, 11.0, 12.0, 13.0], dtype=np.float32),
                    "centered_mean_welfare.mean": np.asarray(
                        [110.0, 111.0, 112.0, 113.0], dtype=np.float32
                    ),
                },
            ),
            (
                0,
                {
                    "density.mean": np.asarray([0.0, 1.0, 2.0, 3.0], dtype=np.float32),
                    "centered_mean_welfare.mean": np.asarray(
                        [100.0, 101.0, 102.0, 103.0], dtype=np.float32
                    ),
                },
            ),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = write_result_store(
                Path(tmpdir),
                config=config,
                outputs_by_index=outputs_by_index,
                total_inputs=2,
            )

            ds = load_policy_results(store_path)

        np.testing.assert_allclose(
            ds["theta"].sel(theta_parameter="benefit_to_cost_ratio").values,
            np.asarray([[[1.0]], [[0.5]]], dtype=np.float32),
        )
        np.testing.assert_allclose(
            [
                ds["prior_low"]
                .sel(theta_parameter="benefit_to_cost_ratio")
                .item(),
                ds["prior_high"]
                .sel(theta_parameter="benefit_to_cost_ratio")
                .item(),
            ],
            np.asarray([0.5, 1.0], dtype=np.float32),
        )
