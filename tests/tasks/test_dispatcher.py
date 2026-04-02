from __future__ import annotations

import contextlib
import io
import json
import logging
import tempfile
import unittest
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from unittest.mock import patch

import zarr

from tasks.dispatch.dispatcher import dispatch_from_config, run_tasks
from tests.tasks import support_tasks as _support_tasks


def run_tasks_inline(
    scheduler_address: str,
    task_name: str,
    task_fn: Any,
    task_inputs: list[Any],
    on_result: Any,
    on_failure: Any,
    *,
    logger: Any,
) -> dict[str, int]:
    del scheduler_address, logger
    run_tasks_inline.last_task_name = task_name

    success_count = 0
    failure_count = 0
    for input_index, task_input in enumerate(task_inputs):
        duration_ms = float(input_index + 1)
        try:
            task_output = task_fn(task_input)
        except Exception as exc:
            failure_count += 1
            on_failure(input_index, task_input, duration_ms, exc)
            continue

        try:
            on_result(input_index, task_input, task_output, duration_ms)
        except Exception as exc:
            failure_count += 1
            on_failure(input_index, task_input, duration_ms, exc)
            continue

        success_count += 1

    return {"success_count": success_count, "failure_count": failure_count}


run_tasks_inline.last_task_name = None


class _FakeFuture:
    def __init__(self, key: str, task_fn: Any, task_input: Any) -> None:
        self.key = key
        self._task_fn = task_fn
        self._task_input = task_input
        self.release_calls = 0

    def result(self) -> Any:
        return self._task_fn(self._task_input)

    def release(self) -> None:
        self.release_calls += 1


class _FakeClient:
    instances: list["_FakeClient"] = []

    def __init__(self, scheduler_address: str) -> None:
        self.scheduler_address = scheduler_address
        self.dashboard_link = "http://dashboard.test/status"
        self.futures: list[_FakeFuture] = []
        self.__class__.instances.append(self)

    def __enter__(self) -> "_FakeClient":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        del exc_type, exc, tb

    def map(
        self,
        task_fn: Any,
        task_inputs_list: list[Any],
        *,
        key: str,
        pure: bool,
    ) -> list[_FakeFuture]:
        del pure
        self.futures = [
            _FakeFuture(f"{key}-{input_index}", task_fn, task_input)
            for input_index, task_input in enumerate(task_inputs_list)
        ]
        return self.futures


class DispatcherTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.workdir = Path(self.temp_dir.name)

    def write_inputs(self, payload: Any) -> Path:
        inputs_path = self.workdir / "inputs.json"
        inputs_path.write_text(json.dumps(payload), encoding="utf-8")
        return inputs_path

    def write_config(
        self,
        task: str,
        inputs_path: Path | None = None,
        batch_size: int = 2,
        extra_config: Mapping[str, Any] | None = None,
    ) -> Path:
        config_path = self.workdir / "config.json"
        payload: dict[str, Any] = {
            "scheduler_address": "inproc://tests",
            "task": task,
            "write_batch_size": batch_size,
            "output_dir": "artifacts",
        }
        if inputs_path is not None:
            payload["inputs_json"] = inputs_path.name
        if extra_config is not None:
            payload.update(dict(extra_config))
        config_path.write_text(json.dumps(payload), encoding="utf-8")
        return config_path

    def open_store(self, path: str) -> zarr.Group:
        return zarr.open_group(path, mode="r")

    def test_run_tasks_releases_completed_futures_immediately(self) -> None:
        _FakeClient.instances.clear()
        release_snapshots: list[tuple[str, list[int]]] = []

        def task_fn(value: int) -> int:
            if value == 2:
                raise ValueError("boom")
            return value * 10

        def on_result(
            input_index: int,
            task_input: int,
            task_output: int,
            duration_ms: float,
        ) -> None:
            del task_input, task_output, duration_ms
            futures = _FakeClient.instances[-1].futures
            release_snapshots.append(("result", [future.release_calls for future in futures]))
            if input_index == 0:
                self.assertEqual([future.release_calls for future in futures], [0, 0, 0])
            if input_index == 2:
                self.assertEqual([future.release_calls for future in futures], [1, 1, 0])

        def on_failure(
            input_index: int,
            task_input: int,
            duration_ms: float,
            error: BaseException,
        ) -> None:
            del task_input, duration_ms, error
            futures = _FakeClient.instances[-1].futures
            release_snapshots.append(("failure", [future.release_calls for future in futures]))
            if input_index == 1:
                self.assertEqual([future.release_calls for future in futures], [1, 0, 0])

        with patch("tasks.dispatch.dispatcher.Client", new=_FakeClient):
            with patch(
                "tasks.dispatch.dispatcher.as_completed",
                new=lambda futures: iter(futures),
            ):
                stats = run_tasks(
                    scheduler_address="inproc://tests",
                    task_name="vector",
                    task_fn=task_fn,
                    task_inputs=[1, 2, 3],
                    on_result=on_result,
                    on_failure=on_failure,
                    logger=logging.getLogger("test"),
                )

        self.assertEqual(stats, {"success_count": 2, "failure_count": 1})
        self.assertEqual(
            release_snapshots,
            [
                ("result", [0, 0, 0]),
                ("failure", [1, 0, 0]),
                ("result", [1, 1, 0]),
            ],
        )
        self.assertEqual(
            [future.release_calls for future in _FakeClient.instances[-1].futures],
            [1, 1, 1],
        )

    def test_dispatcher_writes_success_rows_and_uses_task_name_for_output(self) -> None:
        inputs_path = self.write_inputs([3, 7, 11])
        config_path = self.write_config("vector", inputs_path)

        with patch("tasks.dispatch.dispatcher._run_started_slug", return_value="20260324T160000"):
            with patch("tasks.dispatch.dispatcher.run_tasks", new=run_tasks_inline):
                summary = dispatch_from_config(str(config_path))

        self.assertEqual(run_tasks_inline.last_task_name, "vector")
        self.assertEqual(summary["started_at"], "20260324T160000")
        self.assertTrue(
            summary["output_zarr"].endswith("artifacts/vector-20260324T160000.zarr")
        )
        self.assertTrue(
            summary["config_copy_path"].endswith(
                "artifacts/vector-20260324T160000.dispatch_config.json"
            )
        )
        self.assertTrue(Path(summary["output_zarr"]).exists())
        self.assertTrue(Path(summary["config_copy_path"]).exists())
        self.assertEqual(summary["success_count"], 3)
        self.assertEqual(summary["failure_count"], 0)
        self.assertEqual(summary["success_rows_written"], 3)
        self.assertEqual(summary["failure_rows_written"], 0)

        config_copy = json.loads(
            Path(summary["config_copy_path"]).read_text(encoding="utf-8")
        )
        self.assertEqual(config_copy["task"], "vector")
        self.assertEqual(config_copy["inputs_json"], str(inputs_path.resolve()))
        self.assertEqual(config_copy["output_dir"], str((self.workdir / "artifacts").resolve()))

        root = self.open_store(summary["output_zarr"])
        self.assertEqual(root.attrs["task_name"], "vector")
        self.assertEqual(root.attrs["task_class"], "tests.tasks.support_tasks:VectorTask")
        self.assertEqual(
            root.attrs["output_schema"],
            [{"name": "vector", "shape": [2], "dtype": "int64"}],
        )
        self.assertEqual(root.attrs["total_inputs"], 3)
        self.assertEqual(root["results"]["input_index"][:].tolist(), [0, 1, 2])
        self.assertEqual(root["results"]["input_label"][:].tolist(), ["3", "7", "11"])
        self.assertEqual(
            root["results"]["outputs"]["vector"][:].tolist(),
            [[3, 4], [7, 8], [11, 12]],
        )
        self.assertEqual(root["results"]["durations_ms"][:].tolist(), [1.0, 2.0, 3.0])
        self.assertEqual(set(root["failures"].keys()), {"error"})
        self.assertEqual(root["failures"]["error"][:].tolist(), [])

    def test_dispatcher_writes_results_to_configured_output_dir(self) -> None:
        inputs_path = self.write_inputs([4, 6])
        config_path = self.write_config(
            "vector",
            inputs_path,
            extra_config={"output_dir": "artifacts/custom"},
        )

        with patch("tasks.dispatch.dispatcher._run_started_slug", return_value="20260324T160050"):
            with patch("tasks.dispatch.dispatcher.run_tasks", new=run_tasks_inline):
                summary = dispatch_from_config(str(config_path))

        expected_output = (
            self.workdir / "artifacts" / "custom" / "vector-20260324T160050.zarr"
        )
        expected_config_copy = (
            self.workdir
            / "artifacts"
            / "custom"
            / "vector-20260324T160050.dispatch_config.json"
        )

        self.assertEqual(Path(summary["output_zarr"]), expected_output.resolve())
        self.assertEqual(Path(summary["config_copy_path"]), expected_config_copy.resolve())
        self.assertTrue(expected_output.exists())
        self.assertTrue(expected_config_copy.exists())

        config_copy = json.loads(expected_config_copy.read_text(encoding="utf-8"))
        self.assertEqual(config_copy["task"], "vector")
        self.assertEqual(config_copy["inputs_json"], str(inputs_path.resolve()))
        self.assertEqual(
            config_copy["output_dir"],
            str((self.workdir / "artifacts" / "custom").resolve()),
        )

        root = self.open_store(summary["output_zarr"])
        self.assertEqual(root["results"]["outputs"]["vector"][:].tolist(), [[4, 5], [6, 7]])

    def test_dispatcher_requires_output_dir(self) -> None:
        inputs_path = self.write_inputs([3])
        config_path = self.write_config("vector", inputs_path)
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        payload.pop("output_dir")
        config_path.write_text(json.dumps(payload), encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "output_dir"):
            dispatch_from_config(str(config_path))

    def test_dispatcher_requires_task(self) -> None:
        config_path = self.write_config("vector")
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        payload.pop("task")
        config_path.write_text(json.dumps(payload), encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "task"):
            dispatch_from_config(str(config_path))

    def test_dispatcher_rejects_unknown_task(self) -> None:
        config_path = self.write_config("unknown-task")

        with self.assertRaisesRegex(ValueError, "Unknown task"):
            dispatch_from_config(str(config_path))

    def test_runtime_failures_are_stored_in_failures_group(self) -> None:
        inputs_path = self.write_inputs([5, -1, 8])
        config_path = self.write_config("runtime_failure", inputs_path)

        with patch("tasks.dispatch.dispatcher._run_started_slug", return_value="20260324T160100"):
            with patch("tasks.dispatch.dispatcher.run_tasks", new=run_tasks_inline):
                summary = dispatch_from_config(str(config_path))

        self.assertEqual(summary["success_count"], 2)
        self.assertEqual(summary["failure_count"], 1)
        self.assertTrue(summary["output_zarr"].endswith("runtime_failure-20260324T160100.zarr"))
        root = self.open_store(summary["output_zarr"])
        self.assertEqual(root["results"]["input_index"][:].tolist(), [0, 2])
        self.assertEqual(set(root["failures"].keys()), {"error"})
        self.assertIn("negative values are not allowed: -1", str(root["failures"]["error"][0]))

    def test_dispatcher_writes_structured_outputs_by_field(self) -> None:
        inputs_path = self.write_inputs([2, 5])
        config_path = self.write_config("structured", inputs_path)

        with patch("tasks.dispatch.dispatcher._run_started_slug", return_value="20260324T160200"):
            with patch("tasks.dispatch.dispatcher.run_tasks", new=run_tasks_inline):
                summary = dispatch_from_config(str(config_path))

        root = self.open_store(summary["output_zarr"])
        self.assertEqual(
            root.attrs["output_schema"],
            [
                {"name": "vector", "shape": [2], "dtype": "int64"},
                {"name": "parity", "shape": [], "dtype": "int8"},
            ],
        )
        self.assertEqual(root["results"]["outputs"]["vector"][:].tolist(), [[2, 3], [5, 6]])
        self.assertEqual(root["results"]["outputs"]["parity"][:].tolist(), [0, 1])

    def test_dispatcher_allows_task_defined_inputs_without_inputs_json(self) -> None:
        config_path = self.write_config(
            "generated",
            extra_config={"input_spec": {"start": 4, "count": 3}},
        )

        with patch("tasks.dispatch.dispatcher._run_started_slug", return_value="20260324T160250"):
            with patch("tasks.dispatch.dispatcher.run_tasks", new=run_tasks_inline):
                summary = dispatch_from_config(str(config_path))

        self.assertEqual(summary["success_count"], 3)
        self.assertEqual(summary["failure_count"], 0)
        root = self.open_store(summary["output_zarr"])
        self.assertEqual(root.attrs["task_name"], "generated")
        self.assertEqual(root.attrs["total_inputs"], 3)
        self.assertEqual(root["results"]["input_label"][:].tolist(), ["4", "5", "6"])
        self.assertEqual(root["results"]["outputs"]["vector"][:].tolist(), [[4, 5], [5, 6], [6, 7]])

    def test_dispatcher_logs_use_input_label_and_omit_builtin_task_name_field(self) -> None:
        inputs_path = self.write_inputs([3])
        config_path = self.write_config("vector", inputs_path)
        log_stream = io.StringIO()

        with contextlib.redirect_stdout(log_stream):
            with patch("tasks.dispatch.dispatcher._run_started_slug", return_value="20260324T160257"):
                with patch("tasks.dispatch.dispatcher.run_tasks", new=run_tasks_inline):
                    dispatch_from_config(str(config_path), log_level="DEBUG")

        payloads = [
            json.loads(line)
            for line in log_stream.getvalue().splitlines()
            if line.strip()
        ]
        completed = next(payload for payload in payloads if payload["message"] == "Task completed")
        self.assertEqual(completed["input_label"], "3")
        self.assertNotIn("task_input", completed)
        for payload in payloads:
            self.assertNotIn("taskName", payload)
