from __future__ import annotations

import contextlib
import io
import json
import logging
import tempfile
import unittest
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import zarr

from tasks.execution.runner import run_from_config, execute_inputs
from tests.tasks import support_tasks as _support_tasks


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


class RunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.workdir = Path(self.temp_dir.name)
        self.loader_patch = patch(
            "tasks.execution.runner.prepare_policy_run",
            new=self.prepare_test_task_from_config,
        )
        self.loader_patch.start()
        self.addCleanup(self.loader_patch.stop)

    @staticmethod
    def prepare_test_task_from_config(config: Mapping[str, Any]) -> Any:
        policy_name = str(config["test_task"])
        try:
            task_class = _support_tasks.TEST_POLICY_REGISTRY[policy_name]
        except KeyError as exc:
            available = ", ".join(sorted(_support_tasks.TEST_POLICY_REGISTRY))
            raise ValueError(
                f"Unknown policy {policy_name!r}. Available policies: {available}"
            ) from exc
        task = task_class()
        return SimpleNamespace(
            prepared=task.prepare(config),
        )

    def write_inputs(self, payload: Any) -> Path:
        inputs_path = self.workdir / "inputs.json"
        inputs_path.write_text(json.dumps(payload), encoding="utf-8")
        return inputs_path

    def write_config(
        self,
        policy: str,
        inputs_path: Path | None = None,
        batch_size: int = 2,
        output_dir: str = "artifacts",
        extra_config: Mapping[str, Any] | None = None,
    ) -> Path:
        config_path = self.workdir / "config.json"
        payload: dict[str, Any] = {
            "execution": {
                "scheduler_address": "inproc://tests",
                "write_batch_size": batch_size,
                "output_dir": output_dir,
            },
            "test_task": policy,
        }
        if inputs_path is not None:
            payload["inputs_json"] = inputs_path.name
        if extra_config is not None:
            payload.update(dict(extra_config))
        config_path.write_text(json.dumps(payload), encoding="utf-8")
        return config_path

    def open_store(self, path: str) -> zarr.Group:
        return zarr.open_group(path, mode="r")

    def execution_with_fake_client(
        self,
        config_path: Path,
        *,
        log_level: str = "INFO",
    ) -> dict[str, Any]:
        with patch("tasks.execution.runner.Client", new=_FakeClient):
            with patch(
                "tasks.execution.runner.as_completed",
                new=lambda futures: iter(futures),
            ):
                return run_from_config(str(config_path), log_level=log_level)

    def test_execute_inputs_releases_completed_futures_immediately(self) -> None:
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

        with patch("tasks.execution.runner.Client", new=_FakeClient):
            with patch(
                "tasks.execution.runner.as_completed",
                new=lambda futures: iter(futures),
            ):
                stats = execute_inputs(
                    scheduler_address="inproc://tests",
                    run_key="vector",
                    run_fn=task_fn,
                    run_inputs=[1, 2, 3],
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

    def test_runner_writes_success_rows_and_uses_run_name_for_output(self) -> None:
        inputs_path = self.write_inputs([3, 7, 11])
        config_path = self.write_config("vector", inputs_path)

        summary = self.execution_with_fake_client(config_path)
        self.assertTrue(
            summary["output_zarr"].endswith(
                f"artifacts/run-{summary['started_at']}.zarr"
            )
        )
        self.assertTrue(
            summary["config_copy_path"].endswith(
                f"artifacts/run-{summary['started_at']}.run_config.json"
            )
        )
        self.assertTrue(Path(summary["output_zarr"]).exists())
        self.assertTrue(Path(summary["config_copy_path"]).exists())
        self.assertEqual(summary["success_count"], 3)
        self.assertEqual(summary["failure_count"], 0)
        self.assertEqual(summary["success_rows_written"], 3)
        self.assertFalse(summary["has_failures"])

        config_copy = json.loads(
            Path(summary["config_copy_path"]).read_text(encoding="utf-8")
        )
        self.assertEqual(config_copy["test_task"], "vector")
        self.assertEqual(config_copy["inputs_json"], str(inputs_path.resolve()))
        self.assertEqual(
            config_copy["execution"]["output_dir"],
            str((self.workdir / "artifacts").resolve()),
        )

        root = self.open_store(summary["output_zarr"])
        self.assertEqual(
            root.attrs["output_schema"],
            [{"name": "vector", "shape": [2], "dtype": "int64"}],
        )
        self.assertEqual(root.attrs["total_inputs"], 3)
        self.assertEqual(root.attrs["config_path"], summary["config_copy_path"])
        self.assertFalse(root.attrs["has_failures"])
        self.assertEqual(root["results"]["input_index"][:].tolist(), [0, 1, 2])
        self.assertEqual(
            root["results"]["outputs"]["vector"][:].tolist(),
            [[3, 4], [7, 8], [11, 12]],
        )
        self.assertEqual(len(root["results"]["durations_ms"][:].tolist()), 3)
        self.assertNotIn("failures", root)

    def test_runner_writes_results_to_configured_output_dir(self) -> None:
        inputs_path = self.write_inputs([4, 6])
        config_path = self.write_config(
            "vector",
            inputs_path,
            output_dir="artifacts/custom",
        )

        summary = self.execution_with_fake_client(config_path)

        expected_output = (
            self.workdir / "artifacts" / "custom" / f"run-{summary['started_at']}.zarr"
        )
        expected_config_copy = (
            self.workdir
            / "artifacts"
            / "custom"
            / f"run-{summary['started_at']}.run_config.json"
        )

        self.assertEqual(Path(summary["output_zarr"]), expected_output.resolve())
        self.assertEqual(Path(summary["config_copy_path"]), expected_config_copy.resolve())
        self.assertTrue(expected_output.exists())
        self.assertTrue(expected_config_copy.exists())

        config_copy = json.loads(expected_config_copy.read_text(encoding="utf-8"))
        self.assertEqual(config_copy["test_task"], "vector")
        self.assertEqual(config_copy["inputs_json"], str(inputs_path.resolve()))
        self.assertEqual(
            config_copy["execution"]["output_dir"],
            str((self.workdir / "artifacts" / "custom").resolve()),
        )

        root = self.open_store(summary["output_zarr"])
        self.assertEqual(root["results"]["outputs"]["vector"][:].tolist(), [[4, 5], [6, 7]])

    def test_runner_rejects_unknown_policy(self) -> None:
        config_path = self.write_config("unknown-task")

        with self.assertRaisesRegex(ValueError, "Unknown policy"):
            run_from_config(str(config_path))

    def test_runner_rejects_empty_prepared_input_set(self) -> None:
        config_path = self.write_config(
            "generated",
            extra_config={"input_spec": {"start": 4, "count": 0}},
        )

        with self.assertRaisesRegex(ValueError, "No run inputs found"):
            run_from_config(str(config_path))

    def test_runtime_failures_are_recorded_as_store_attrs(self) -> None:
        inputs_path = self.write_inputs([5, -1, 8])
        config_path = self.write_config("runtime_failure", inputs_path)

        summary = self.execution_with_fake_client(config_path)

        self.assertEqual(summary["success_count"], 2)
        self.assertEqual(summary["failure_count"], 1)
        self.assertTrue(
            summary["output_zarr"].endswith(
                f"run-{summary['started_at']}.zarr"
            )
        )
        root = self.open_store(summary["output_zarr"])
        self.assertEqual(root["results"]["input_index"][:].tolist(), [0, 2])
        self.assertTrue(root.attrs["has_failures"])
        self.assertTrue(summary["has_failures"])
        self.assertNotIn("failures", root)

    def test_runner_writes_structured_outputs_by_field(self) -> None:
        inputs_path = self.write_inputs([2, 5])
        config_path = self.write_config("structured", inputs_path)

        summary = self.execution_with_fake_client(config_path)

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

    def test_runner_allows_task_defined_inputs_without_inputs_json(self) -> None:
        config_path = self.write_config(
            "generated",
            extra_config={"input_spec": {"start": 4, "count": 3}},
        )

        summary = self.execution_with_fake_client(config_path)

        self.assertEqual(summary["success_count"], 3)
        self.assertEqual(summary["failure_count"], 0)
        root = self.open_store(summary["output_zarr"])
        self.assertEqual(root.attrs["total_inputs"], 3)
        self.assertEqual(
            root["results"]["outputs"]["vector"][:].tolist(),
            [[4, 5], [5, 6], [6, 7]],
        )

    def test_runner_logs_human_readable_lifecycle_messages(self) -> None:
        inputs_path = self.write_inputs([3])
        config_path = self.write_config("vector", inputs_path)
        log_stream = io.StringIO()

        with contextlib.redirect_stdout(log_stream):
            self.execution_with_fake_client(
                config_path,
                log_level="DEBUG",
            )

        output = log_stream.getvalue()
        self.assertIn("INFO Preparing run", output)
        self.assertIn("INFO Initialized output", output)
        self.assertIn("INFO Connected to scheduler", output)
        self.assertIn("INFO Run finished", output)
