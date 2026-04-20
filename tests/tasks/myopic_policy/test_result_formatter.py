"""Tests for myopic policy result formatting."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from tasks.policies import load_policy_results
from tasks.policies.support import generate_parameter_table

from ..policy_result_support import build_myopic_config, write_result_store


class MyopicPolicyResultFormatterTest(unittest.TestCase):
    def test_from_zarr_reconstructs_parameters_and_sorts_rows(self) -> None:
        config = build_myopic_config(log2_budget=2)
        parameter_table = generate_parameter_table(config["input_spec"])

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
                task_name="myopic-policy",
                task_class="tasks.policies.myopic:MyopicPolicyRun",
                config=config,
                outputs_by_index=outputs_by_index,
                total_inputs=4,
            )

            formatted = load_policy_results(store_path)

        np.testing.assert_array_equal(
            formatted.metadata["input_index"],
            np.asarray([[0, 1, 2, 3]], dtype=np.int64),
        )
        self.assertEqual(formatted.metadata["average_completion_duration_ms"], 2.0)
        self.assertEqual(formatted.metadata["simulation"], config["simulation"])
        np.testing.assert_allclose(
            formatted.inputs["tensors"]["parameters"],
            parameter_table.parameter_values[None, :, :],
        )
        np.testing.assert_array_equal(
            formatted.inputs["labels"]["n_agents"],
            np.asarray([4], dtype=np.int64),
        )
        np.testing.assert_array_equal(
            formatted.inputs["labels"]["parameter_row"],
            np.asarray([0, 1, 2, 3], dtype=np.int64),
        )
        self.assertEqual(
            tuple(formatted.inputs["labels"]["parameter"]),
            (
                "benefit_to_cost_ratio",
                "distance_discount",
                "bridge_probability",
                "visibility_distance",
            ),
        )
        np.testing.assert_array_equal(
            formatted.status["tensors"]["success"],
            np.asarray([[True, True, True, False]], dtype=np.bool_),
        )
        self.assertEqual(
            tuple(formatted.outputs["labels"]["observable"]),
            ("density.mean", "centered_mean_welfare.mean"),
        )
        self.assertEqual(formatted.outputs["tensors"]["timeseries"].shape, (1, 4, 2, 4))
        self.assertEqual(formatted.outputs["tensors"]["final"].shape, (1, 4, 2))
        np.testing.assert_allclose(
            formatted.outputs["tensors"]["timeseries"][0, 0, 0],
            np.asarray([0.0, 1.0, 2.0, 3.0], dtype=np.float32),
        )
        np.testing.assert_allclose(
            formatted.outputs["tensors"]["timeseries"][0, 1, 1],
            np.asarray([110.0, 111.0, 112.0, 113.0], dtype=np.float32),
        )
        np.testing.assert_allclose(
            formatted.outputs["tensors"]["final"],
            np.asarray(
                [
                    [
                        [3.0, 103.0],
                        [13.0, 113.0],
                        [23.0, 203.0],
                        [0.0, 0.0],
                    ],
                ],
                dtype=np.float32,
            ),
        )

    def test_from_zarr_reconstructs_n_agents_sweep_inputs(self) -> None:
        config = build_myopic_config(log2_budget=1)
        config["simulation"]["n_agents"] = [3, 5]
        parameter_table = generate_parameter_table(config["input_spec"])

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
                task_name="myopic-policy",
                task_class="tasks.policies.myopic:MyopicPolicyRun",
                config=config,
                outputs_by_index=outputs_by_index,
                total_inputs=4,
            )

            formatted = load_policy_results(store_path)

        np.testing.assert_array_equal(
            formatted.metadata["input_index"],
            np.asarray([[0, 1], [2, 3]], dtype=np.int64),
        )
        np.testing.assert_allclose(
            formatted.inputs["tensors"]["parameters"],
            np.broadcast_to(
                parameter_table.parameter_values[None, :, :],
                (2, 2, 4),
            ),
        )
        np.testing.assert_array_equal(
            formatted.inputs["labels"]["n_agents"],
            np.asarray([3, 5], dtype=np.int64),
        )
        np.testing.assert_array_equal(
            formatted.status["tensors"]["success"],
            np.asarray([[True, False], [True, True]], dtype=np.bool_),
        )
        self.assertEqual(formatted.metadata["simulation"], config["simulation"])

    def test_from_zarr_uses_resolved_scaled_inverse_benefit_to_cost_ratio(self) -> None:
        config = build_myopic_config(log2_budget=0)
        config["simulation"]["n_agents"] = [5, 10]
        config["input_spec"]["parameters"]["benefit_to_cost_ratio"] = {
            "type": "scaled_inverse_n_agents",
            "payload": {
                "numerator": 10.0,
                "factor": 2.0,
            },
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
                task_name="myopic-policy",
                task_class="tasks.policies.myopic:MyopicPolicyRun",
                config=config,
                outputs_by_index=outputs_by_index,
                total_inputs=2,
            )

            formatted = load_policy_results(store_path)

        benefit_to_cost_ratio_index = tuple(formatted.inputs["labels"]["parameter"]).index(
            "benefit_to_cost_ratio"
        )
        np.testing.assert_allclose(
            formatted.inputs["tensors"]["parameters"][:, :, benefit_to_cost_ratio_index],
            np.asarray([[1.0], [0.5]], dtype=np.float32),
        )
