"""Zarr-backed task result writer and reader."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import zarr

from .runtime import OutputSchema, TaskOutput


INPUT_LABEL_DTYPE = "<U1024"
ERROR_DTYPE = "<U4096"


def task_class_path(task: object) -> str:
    return f"{task.__class__.__module__}:{task.__class__.__qualname__}"


def load_result_zarr(path: str | Path) -> dict[str, Any]:
    """Load one task result Zarr store into NumPy arrays."""
    store_path = Path(path).expanduser()
    if not store_path.exists():
        raise FileNotFoundError(f"Result store not found: {store_path}")

    root = zarr.open_group(store_path, mode="r")
    results_group = root["results"]
    outputs_group = results_group["outputs"]
    failures_group = root["failures"]

    input_index = np.asarray(results_group["input_index"])
    order = np.argsort(input_index)
    output_schema = list(root.attrs["output_schema"])
    outputs = {
        field["name"]: np.asarray(outputs_group[field["name"]])[order]
        for field in output_schema
    }
    failures = {
        name: np.asarray(failures_group[name]) for name in failures_group.keys()
    }
    failure_count = 0
    if failures:
        failure_count = int(next(iter(failures.values())).shape[0])

    return {
        "path": store_path.resolve(),
        "schema_version": root.attrs.get("schema_version"),
        "task_name": root.attrs.get("task_name"),
        "task_class": root.attrs.get("task_class"),
        "output_schema": output_schema,
        "total_inputs": root.attrs.get("total_inputs"),
        "stored_results": int(input_index.shape[0]),
        "stored_failures": failure_count,
        "input_index": input_index[order],
        "outputs": outputs,
        "failures": failures,
    }


class ResultStore:
    def __init__(
        self,
        output_path: str,
        *,
        task_name: str,
        task_class: str,
        output_schema: OutputSchema,
        batch_size: int,
        total_inputs: int,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if total_inputs < 0:
            raise ValueError("total_inputs must be >= 0")

        self.batch_size = batch_size
        self.success_count = 0
        self.failure_count = 0
        self._output_fields = output_schema.fields
        self._pending_success_indices: list[int] = []
        self._pending_success_labels: list[str] = []
        self._pending_success_outputs: dict[str, list[np.ndarray[Any, Any]]] = {
            field.name: [] for field in self._output_fields
        }
        self._pending_success_durations_ms: list[float] = []
        self._pending_failure_indices: list[int] = []
        self._pending_failure_labels: list[str] = []
        self._pending_failure_durations_ms: list[float] = []
        self._pending_failure_errors: list[str] = []

        root = zarr.open_group(output_path, mode="w", zarr_format=2)
        root.attrs.update(
            {
                "schema_version": 2,
                "task_name": task_name,
                "task_class": task_class,
                "output_schema": output_schema.metadata(),
                "total_inputs": int(total_inputs),
            }
        )

        results = root.create_group("results", overwrite=True)
        outputs = results.create_group("outputs", overwrite=True)
        failures = root.create_group("failures", overwrite=True)

        self.results_input_index = results.create_array(
            "input_index",
            shape=(0,),
            chunks=(batch_size,),
            dtype="i8",
            overwrite=True,
        )
        self.results_input_label = results.create_array(
            "input_label",
            shape=(0,),
            chunks=(batch_size,),
            dtype=INPUT_LABEL_DTYPE,
            overwrite=True,
        )
        self.results_durations_ms = results.create_array(
            "durations_ms",
            shape=(0,),
            chunks=(batch_size,),
            dtype="f8",
            overwrite=True,
        )
        self.results_outputs: dict[str, Any] = {}
        for field in self._output_fields:
            output_shape = (0, *field.shape)
            output_chunks = (batch_size, *field.shape) if field.shape else (batch_size,)
            self.results_outputs[field.name] = outputs.create_array(
                field.name,
                shape=output_shape,
                chunks=output_chunks,
                dtype=field.resolved_dtype,
                overwrite=True,
            )

        self.failures_input_index = failures.create_array(
            "input_index",
            shape=(0,),
            chunks=(batch_size,),
            dtype="i8",
            overwrite=True,
        )
        self.failures_input_label = failures.create_array(
            "input_label",
            shape=(0,),
            chunks=(batch_size,),
            dtype=INPUT_LABEL_DTYPE,
            overwrite=True,
        )
        self.failures_durations_ms = failures.create_array(
            "durations_ms",
            shape=(0,),
            chunks=(batch_size,),
            dtype="f8",
            overwrite=True,
        )
        self.failures_error = failures.create_array(
            "error",
            shape=(0,),
            chunks=(batch_size,),
            dtype=ERROR_DTYPE,
            overwrite=True,
        )

    def add_success(
        self,
        input_index: int,
        input_label: str,
        output: TaskOutput,
        duration_ms: float,
    ) -> tuple[int, int]:
        normalized_output: dict[str, np.ndarray[Any, Any]] = {}
        for field in self._output_fields:
            array = np.asarray(output[field.name], dtype=field.resolved_dtype)
            if tuple(array.shape) != field.shape:
                raise ValueError(
                    f"field {field.name!r} exported shape {tuple(array.shape)} "
                    f"but expected {field.shape}"
                )
            normalized_output[field.name] = array

        self._pending_success_indices.append(int(input_index))
        self._pending_success_labels.append(str(input_label))
        for field_name, array in normalized_output.items():
            self._pending_success_outputs[field_name].append(array)
        self._pending_success_durations_ms.append(float(duration_ms))
        self.success_count += 1

        if len(self._pending_success_indices) >= self.batch_size:
            return self._flush_successes(), 0
        return 0, 0

    def add_failure(
        self,
        input_index: int,
        input_label: str,
        duration_ms: float,
        error: str,
    ) -> tuple[int, int]:
        self._pending_failure_indices.append(int(input_index))
        self._pending_failure_labels.append(str(input_label))
        self._pending_failure_durations_ms.append(float(duration_ms))
        self._pending_failure_errors.append(str(error))
        self.failure_count += 1

        if len(self._pending_failure_indices) >= self.batch_size:
            return 0, self._flush_failures()
        return 0, 0

    def flush(self) -> tuple[int, int]:
        return self._flush_successes(), self._flush_failures()

    def _flush_successes(self) -> int:
        if not self._pending_success_indices:
            return 0

        batch_len = len(self._pending_success_indices)
        self.results_input_index.append(
            np.asarray(self._pending_success_indices, dtype=self.results_input_index.dtype),
            axis=0,
        )
        self.results_input_label.append(
            np.asarray(self._pending_success_labels, dtype=self.results_input_label.dtype),
            axis=0,
        )
        for field in self._output_fields:
            stacked = np.stack(self._pending_success_outputs[field.name], axis=0).astype(
                self.results_outputs[field.name].dtype,
                copy=False,
            )
            self.results_outputs[field.name].append(stacked, axis=0)
            self._pending_success_outputs[field.name].clear()
        self.results_durations_ms.append(
            np.asarray(self._pending_success_durations_ms, dtype=self.results_durations_ms.dtype),
            axis=0,
        )

        self._pending_success_indices.clear()
        self._pending_success_labels.clear()
        self._pending_success_durations_ms.clear()
        return batch_len

    def _flush_failures(self) -> int:
        if not self._pending_failure_indices:
            return 0

        batch_len = len(self._pending_failure_indices)
        self.failures_input_index.append(
            np.asarray(self._pending_failure_indices, dtype=self.failures_input_index.dtype),
            axis=0,
        )
        self.failures_input_label.append(
            np.asarray(self._pending_failure_labels, dtype=self.failures_input_label.dtype),
            axis=0,
        )
        self.failures_durations_ms.append(
            np.asarray(self._pending_failure_durations_ms, dtype=self.failures_durations_ms.dtype),
            axis=0,
        )
        self.failures_error.append(
            np.asarray(self._pending_failure_errors, dtype=self.failures_error.dtype),
            axis=0,
        )

        self._pending_failure_indices.clear()
        self._pending_failure_labels.clear()
        self._pending_failure_durations_ms.clear()
        self._pending_failure_errors.clear()
        return batch_len
