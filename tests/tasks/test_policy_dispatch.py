"""Tests for policy result dispatch."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import zarr

from tasks.dispatch.dispatcher import dispatch_from_config
from tasks.policies import load_policy_results
from .policy_result_support import build_random_config, write_result_store


class _FakeFuture:
    def __init__(self, key: str, task_fn: object, task_input: object) -> None:
        self.key = key
        self._task_fn = task_fn
        self._task_input = task_input

    def result(self) -> object:
        return self._task_fn(self._task_input)

    def release(self) -> None:
        return None


class _FakeClient:
    def __init__(self, scheduler_address: str) -> None:
        self.scheduler_address = scheduler_address

    def __enter__(self) -> "_FakeClient":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        del exc_type, exc, tb

    def map(
        self,
        task_fn: object,
        task_inputs_list: list[object],
        *,
        key: str,
        pure: bool,
    ) -> list[_FakeFuture]:
        del pure
        return [
            _FakeFuture(f"{key}-{input_index}", task_fn, task_input)
            for input_index, task_input in enumerate(task_inputs_list)
        ]


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

    def test_dispatch_from_config_writes_dense_policy_results(self) -> None:
        config = build_random_config()
        config["simulation"]["n_agents"] = [2, 4]
        config["observers"] = [
            {
                "name": "density",
                "output": "mean",
                "kwargs": {
                    "observation_sampling": {"kind": "periodic", "every": 1},
                    "aggregations": ["mean"],
                },
            },
            {
                "name": "centered_mean_welfare",
                "output": "mean",
                "kwargs": {
                    "observation_sampling": {"kind": "periodic", "every": 1},
                    "aggregations": ["mean"],
                    "benefit_to_cost_ratio": 0.1,
                    "distance_discount": 0.5,
                },
            },
        ]
        config.update(
            {
                "scheduler_address": "inproc://tests",
                "write_batch_size": 2,
                "output_dir": "artifacts",
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")

            with patch("tasks.dispatch.dispatcher.Client", new=_FakeClient):
                with patch(
                    "tasks.dispatch.dispatcher.as_completed",
                    new=lambda futures: iter(futures),
                ):
                    summary = dispatch_from_config(str(config_path))

            root = zarr.open_group(summary["output_zarr"], mode="r")
            self.assertEqual(
                root["results"]["outputs"]["density.mean"].shape,
                (2, 4),
            )
            self.assertEqual(root["results"]["input_index"][:].tolist(), [0, 1])

            formatted = load_policy_results(summary["output_zarr"])

        self.assertEqual(formatted.outputs["tensors"]["timeseries"].shape, (2, 1, 2, 4))
        np.testing.assert_array_equal(
            formatted.status["tensors"]["success"],
            np.asarray([[True], [True]], dtype=np.bool_),
        )
