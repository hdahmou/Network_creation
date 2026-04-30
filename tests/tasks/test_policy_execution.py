"""Tests for policy result execution."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import zarr

from tasks.execution.runner import run_from_config
from tasks.policies import load_policy_results
from tests.tasks.config_support import execution_config
from tests.tasks.policy_result_support import build_random_config


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


class PolicyExecutionTest(unittest.TestCase):
    def test_run_from_config_writes_dense_policy_results(self) -> None:
        config = build_random_config()
        config["population"]["n_agents"] = [2, 4]
        config = execution_config(
            n_agents=[2, 4],
            n_actors=2,
            game=config["game"],
            visibility_distance=config["population"]["visibility_distance"],
            policies=config["population"]["policies"],
            observers=config["observers"],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")

            with patch("tasks.execution.runner.Client", new=_FakeClient):
                with patch(
                    "tasks.execution.runner.as_completed",
                    new=lambda futures: iter(futures),
                ):
                    summary = run_from_config(str(config_path))

            root = zarr.open_group(summary["output_zarr"], mode="r")
            self.assertEqual(
                root["results"]["outputs"]["density.mean"].shape,
                (2, 4),
            )
            self.assertEqual(root["results"]["input_index"][:].tolist(), [0, 1])
            self.assertEqual(
                root.attrs["parameter_names"],
                [
                    "random_proportion",
                    "random_count",
                    "benefit_to_cost_ratio",
                    "distance_discount",
                    "visibility_distance",
                ],
            )
            np.testing.assert_array_equal(
                root["metadata"]["input_coordinates"][:],
                np.asarray([[0, 0, 0], [1, 0, 0]], dtype=np.int64),
            )

            ds = load_policy_results(summary["output_zarr"])

        np.testing.assert_array_equal(
            ds.coords["input_index"].values,
            np.asarray([[[0]], [[1]]], dtype=np.int64),
        )
        self.assertEqual(ds["x"].shape, (2, 1, 1, 4, 2))
        np.testing.assert_array_equal(
            ds["theta"].sel(theta_parameter="n_agents").values,
            np.asarray([[[2]], [[4]]], dtype=np.float32),
        )
