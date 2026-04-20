"""Tests for dispatch result loading."""

from __future__ import annotations

import importlib
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


class _FakeGroup(dict):
    def __init__(self, data: dict[str, object], *, attrs: dict[str, object] | None = None) -> None:
        super().__init__(data)
        self.attrs = attrs or {}


class TaskUtilsTest(unittest.TestCase):
    def test_load_result_zarr_returns_sorted_numpy_content(self) -> None:
        output_schema = [
            {"name": "density.mean", "shape": [3], "dtype": "float32"},
            {"name": "centered_mean_welfare.mean", "shape": [3], "dtype": "float32"},
        ]
        fake_root = _FakeGroup(
            {
                "results": _FakeGroup(
                    {
                        "input_index": np.asarray([2, 0, 1], dtype=np.int64),
                        "durations_ms": np.asarray([3.0, 1.0, 2.0], dtype=np.float64),
                        "outputs": _FakeGroup(
                            {
                                "density.mean": np.asarray(
                                    [[0.2, 0.3, 0.4], [0.0, 0.1, 0.2], [0.1, 0.2, 0.3]],
                                    dtype=np.float32,
                                ),
                                "centered_mean_welfare.mean": np.asarray(
                                    [[20.0, 21.0, 22.0], [0.0, 1.0, 2.0], [10.0, 11.0, 12.0]],
                                    dtype=np.float32,
                                ),
                            }
                        ),
                    }
                ),
                "failures": _FakeGroup(
                    {
                        "error": np.asarray(["boom"], dtype=object),
                    }
                ),
            },
            attrs={
                "task_name": "myopic-policy",
                "task_class": "tasks.policies.myopic:MyopicPolicyRun",
                "output_schema": output_schema,
                "total_inputs": 3,
            },
        )
        fake_zarr = types.SimpleNamespace(open_group=lambda path, mode="r": fake_root)

        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "run.zarr"
            store_path.mkdir()
            with mock.patch.dict(
                sys.modules,
                {
                    "zarr": fake_zarr,
                },
            ):
                sys.modules.pop("tasks.dispatch.result_store", None)
                load_result_zarr = importlib.import_module(
                    "tasks.dispatch.result_store"
                ).load_result_zarr
                result = load_result_zarr(store_path)

        self.assertEqual(result["path"], store_path.resolve())
        self.assertEqual(result["task_name"], "myopic-policy")
        self.assertEqual(
            result["task_class"],
            "tasks.policies.myopic:MyopicPolicyRun",
        )
        self.assertEqual(result["average_completion_duration_ms"], 2.0)
        self.assertEqual(result["output_schema"], output_schema)
        self.assertEqual(result["total_inputs"], 3)
        self.assertEqual(result["stored_results"], 3)
        self.assertEqual(result["stored_failures"], 1)
        np.testing.assert_array_equal(result["input_index"], np.asarray([0, 1, 2]))
        np.testing.assert_allclose(
            result["outputs"]["density.mean"],
            np.asarray(
                [[0.0, 0.1, 0.2], [0.1, 0.2, 0.3], [0.2, 0.3, 0.4]],
                dtype=np.float32,
            ),
        )
        np.testing.assert_allclose(
            result["outputs"]["centered_mean_welfare.mean"],
            np.asarray(
                [[0.0, 1.0, 2.0], [10.0, 11.0, 12.0], [20.0, 21.0, 22.0]],
                dtype=np.float32,
            ),
        )
        np.testing.assert_array_equal(
            result["failures"]["error"],
            np.asarray(["boom"], dtype=object),
        )
