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
from unittest.mock import MagicMock, patch

import zarr

from tasks.execution.runner import run_from_config, execute_inputs
from tests.tasks import support_tasks as _support_tasks


class _FakeFuture:
    _unset = object()

    def __init__(self, key: str, compute: Any) -> None:
        self.key = key
        self._compute = compute
        self._value: Any = self._unset
        self._error: BaseException | None = None
        self.release_calls = 0

    def result(self) -> Any:
        if self._value is self._unset and self._error is None:
            try:
                self._value = self._compute()
            except BaseException as exc:
                self._error = exc
        if self._error is not None:
            raise self._error
        return self._value

    def release(self) -> None:
        self.release_calls += 1


class _FakeClient:
    instances: list["_FakeClient"] = []
    thread_counts: list[int] = [1]

    def __init__(self, scheduler_address: str) -> None:
        self.scheduler_address = scheduler_address
        self.dashboard_link = "http://dashboard.test/status"
        self.futures: list[_FakeFuture] = []
        self.runtime_future: _FakeFuture | None = None
        self.map_batch_sizes: list[int] = []
        self._nthreads_call_count = 0
        self.__class__.instances.append(self)

    def __enter__(self) -> "_FakeClient":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        del exc_type, exc, tb

    @staticmethod
    def _resolve(value: Any) -> Any:
        return value.result() if isinstance(value, _FakeFuture) else value

    def submit(
        self,
        task_fn: Any,
        *args: Any,
        key: str,
        pure: bool,
    ) -> _FakeFuture:
        del pure
        future = _FakeFuture(
            key,
            lambda: task_fn(*(self._resolve(arg) for arg in args)),
        )
        self.runtime_future = future
        return future

    def map(
        self,
        task_fn: Any,
        task_inputs_list: list[Any],
        *,
        key: list[str],
        pure: bool,
        **kwargs: Any,
    ) -> list[_FakeFuture]:
        del pure
        self.map_batch_sizes.append(len(task_inputs_list))
        futures = [
            _FakeFuture(
                task_key,
                lambda task_input=task_input: task_fn(
                    task_input,
                    **{
                        name: self._resolve(value)
                        for name, value in kwargs.items()
                    },
                ),
            )
            for task_key, task_input in zip(key, task_inputs_list, strict=True)
        ]
        self.futures.extend(futures)
        return futures

    def nthreads(self) -> dict[str, int]:
        index = min(
            self._nthreads_call_count,
            len(self.thread_counts) - 1,
        )
        self._nthreads_call_count += 1
        return {
            f"worker-{worker_index}": 1
            for worker_index in range(self.thread_counts[index])
        }


class _FakeAsCompleted:
    def __init__(self) -> None:
        self._futures: list[_FakeFuture] = []

    def update(self, futures: list[_FakeFuture]) -> None:
        self._futures.extend(futures)

    def next_batch(self) -> list[_FakeFuture]:
        if not self._futures:
            raise StopIteration
        futures = self._futures
        self._futures = []
        return futures


class RunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        _FakeClient.instances.clear()
        _FakeClient.thread_counts = [1]
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
        output_dir: str = "artifacts/simulation_runs",
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
                new=_FakeAsCompleted,
            ):
                return run_from_config(str(config_path), log_level=log_level)

    def test_execute_inputs_releases_completed_futures_immediately(self) -> None:
        release_snapshots: list[tuple[str, list[int]]] = []

        def task_fn(value: int) -> int:
            if value == 2:
                raise ValueError("boom")
            return value * 10

        def on_result(
            input_index: int,
            task_input: int,
            task_output: int,
        ) -> None:
            del task_input, task_output
            futures = _FakeClient.instances[-1].futures
            release_snapshots.append(("result", [future.release_calls for future in futures]))
            if input_index == 0:
                self.assertEqual([future.release_calls for future in futures], [0, 0])
            if input_index == 2:
                self.assertEqual([future.release_calls for future in futures], [1, 1, 0])

        def on_failure(
            input_index: int,
            task_input: int,
            error: BaseException,
        ) -> None:
            del task_input, error
            futures = _FakeClient.instances[-1].futures
            release_snapshots.append(("failure", [future.release_calls for future in futures]))
            if input_index == 1:
                self.assertEqual([future.release_calls for future in futures], [1, 0])

        with patch("tasks.execution.runner.Client", new=_FakeClient):
            with patch(
                "tasks.execution.runner.as_completed",
                new=_FakeAsCompleted,
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
                ("result", [0, 0]),
                ("failure", [1, 0]),
                ("result", [1, 1, 0]),
            ],
        )
        client = _FakeClient.instances[-1]
        self.assertEqual(client.map_batch_sizes, [2, 1])
        self.assertIsNotNone(client.runtime_future)
        assert client.runtime_future is not None
        self.assertEqual(client.runtime_future.release_calls, 1)
        self.assertEqual(
            [future.release_calls for future in client.futures],
            [1, 1, 1],
        )

    def test_execute_inputs_expands_window_when_workers_join(self) -> None:
        _FakeClient.thread_counts = [1, 2]
        results: list[int] = []

        with patch("tasks.execution.runner.Client", new=_FakeClient):
            with patch(
                "tasks.execution.runner.as_completed",
                new=_FakeAsCompleted,
            ):
                stats = execute_inputs(
                    scheduler_address="inproc://tests",
                    run_key="dynamic",
                    run_fn=lambda value: {"value": value},
                    run_inputs=list(range(8)),
                    on_result=lambda _index, value, _output: results.append(value),
                    on_failure=lambda _index, _value, error: self.fail(str(error)),
                    logger=logging.getLogger("test"),
                )

        self.assertEqual(stats, {"success_count": 8, "failure_count": 0})
        self.assertEqual(results, list(range(8)))
        self.assertEqual(_FakeClient.instances[-1].map_batch_sizes, [2, 4, 2])

    def test_execute_inputs_reports_exact_terminal_progress(self) -> None:
        progress = MagicMock()
        progress_context = MagicMock()
        progress_context.__enter__.return_value = progress

        with patch("tasks.execution.runner.Client", new=_FakeClient):
            with patch(
                "tasks.execution.runner.as_completed",
                new=_FakeAsCompleted,
            ):
                with patch(
                    "tasks.execution.runner.tqdm",
                    return_value=progress_context,
                ) as progress_factory:
                    execute_inputs(
                        scheduler_address="inproc://tests",
                        run_key="progress",
                        run_fn=lambda value: {"value": value},
                        run_inputs=[1, 2, 3],
                        on_result=lambda _index, _value, _output: None,
                        on_failure=lambda _index, _value, error: self.fail(str(error)),
                        logger=logging.getLogger("test"),
                    )

        self.assertEqual(progress_factory.call_args.kwargs["total"], 3)
        self.assertEqual(
            sum(call.args[0] for call in progress.update.call_args_list),
            3,
        )
        progress.set_postfix.assert_called_with(
            successes=3,
            failures=0,
            workers=1,
            refresh=False,
        )

    def test_runner_writes_success_rows_and_uses_run_name_for_output(self) -> None:
        inputs_path = self.write_inputs([3, 7, 11])
        config_path = self.write_config("vector", inputs_path)

        summary = self.execution_with_fake_client(config_path)
        self.assertTrue(
            summary["output_zarr"].endswith(
                f"artifacts/simulation_runs/run-{summary['started_at']}.zarr"
            )
        )
        self.assertTrue(
            summary["config_copy_path"].endswith(
                "artifacts/simulation_runs/"
                f"run-{summary['started_at']}.zarr/run_config.json"
            )
        )
        self.assertTrue(Path(summary["output_zarr"]).exists())
        self.assertTrue(Path(summary["config_copy_path"]).exists())
        self.assertTrue((Path(summary["output_zarr"]) / "done").is_file())
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
            str((self.workdir / "artifacts" / "simulation_runs").resolve()),
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
        self.assertNotIn("durations_ms", root["results"])
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
            / f"run-{summary['started_at']}.zarr"
            / "run_config.json"
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
        self.assertTrue((Path(summary["output_zarr"]) / "done").is_file())
        self.assertNotIn("failures", root)

    def test_runner_does_not_write_done_marker_when_execution_stops(self) -> None:
        inputs_path = self.write_inputs([5, 8])
        config_path = self.write_config("vector", inputs_path)

        with patch(
            "tasks.execution.runner.execute_inputs",
            side_effect=RuntimeError("scheduler stopped"),
        ):
            with self.assertRaisesRegex(RuntimeError, "scheduler stopped"):
                run_from_config(str(config_path))

        output_dirs = list(
            (self.workdir / "artifacts" / "simulation_runs").glob("run-*.zarr")
        )
        self.assertEqual(len(output_dirs), 1)
        self.assertFalse((output_dirs[0] / "done").exists())

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
