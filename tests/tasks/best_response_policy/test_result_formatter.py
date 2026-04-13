"""Tests for best-response policy result formatting."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from tasks.policies import load_policy_results
from tasks.policies.support import generate_parameter_table

from ..policy_result_support import build_best_response_config, write_result_store


class BestResponsePolicyResultFormatterTest(unittest.TestCase):
    def test_from_zarr_reconstructs_parameters_and_sorts_rows(self) -> None:
        config = build_best_response_config(log2_budget=2)
        parameter_table = generate_parameter_table(config["input_spec"])

        outputs_by_index = [
            (
                2,
                {
                    "density.mean": np.asarray([20.0, 21.0, 22.0, 23.0], dtype=np.float32),
                    "normalized_welfare.mean": np.asarray(
                        [200.0, 201.0, 202.0, 203.0], dtype=np.float32
                    ),
                },
            ),
            (
                0,
                {
                    "density.mean": np.asarray([0.0, 1.0, 2.0, 3.0], dtype=np.float32),
                    "normalized_welfare.mean": np.asarray(
                        [100.0, 101.0, 102.0, 103.0], dtype=np.float32
                    ),
                },
            ),
            (
                1,
                {
                    "density.mean": np.asarray([10.0, 11.0, 12.0, 13.0], dtype=np.float32),
                    "normalized_welfare.mean": np.asarray(
                        [110.0, 111.0, 112.0, 113.0], dtype=np.float32
                    ),
                },
            ),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = write_result_store(
                Path(tmpdir),
                task_name="best-response-policy",
                task_class="tasks.policies.best_response:BestResponsePolicyRun",
                config=config,
                outputs_by_index=outputs_by_index,
                total_inputs=4,
            )

            formatted = load_policy_results(store_path)

        np.testing.assert_array_equal(
            formatted.metadata["input_index"],
            np.asarray([0, 1, 2], dtype=np.int64),
        )
        np.testing.assert_allclose(
            formatted.inputs["tensors"]["parameters"],
            parameter_table.parameter_values[[0, 1, 2]],
        )
        self.assertEqual(
            tuple(formatted.inputs["labels"]["parameter"]),
            (
                "benefit_to_cost_ratio",
                "distance_discount",
                "bridge_value",
                "inertia_term",
                "tau",
                "visibility_fraction",
            ),
        )
        self.assertEqual(
            tuple(formatted.outputs["labels"]["observable"]),
            ("density.mean", "normalized_welfare.mean"),
        )
        self.assertEqual(formatted.outputs["tensors"]["timeseries"].shape, (3, 2, 4))
        self.assertEqual(formatted.outputs["tensors"]["final"].shape, (3, 2))
        np.testing.assert_allclose(
            formatted.outputs["tensors"]["final"],
            np.asarray(
                [
                    [3.0, 103.0],
                    [13.0, 113.0],
                    [23.0, 203.0],
                ],
                dtype=np.float32,
            ),
        )
