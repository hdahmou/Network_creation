"""Zarr-backed run result writer and reader."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np
import zarr
import xarray as xr

from .schema import OutputSpec, RunOutput


def load_zarr_results(path: str | Path) -> xr.Dataset:
    """Load one run result Zarr store into an in-memory xarray dataset."""
    store_path = Path(path).expanduser()
    if not store_path.exists():
        raise FileNotFoundError(f"Result store not found: {store_path}")

    root = cast(Any, zarr.open_group(store_path, mode="r"))
    results_group = root["results"]
    outputs_group = results_group["outputs"]

    input_index = np.asarray(results_group["input_index"])
    durations_ms = np.asarray(results_group["durations_ms"], dtype=np.float64)
    order = np.argsort(input_index)
    output_schema = list(root.attrs["output_schema"])
    root_attrs = dict(root.attrs)
    metadata_group = root["metadata"]
    metadata = {
        name: np.asarray(metadata_group[name]) for name in metadata_group.keys()
    }

    coords: dict[str, Any] = {
        "result": np.arange(input_index.shape[0], dtype=np.int64),
        "input_index": ("result", np.asarray(input_index[order], dtype=np.int64)),
    }
    data_vars: dict[str, Any] = {
        "duration_ms": (
            ("result",),
            np.asarray(durations_ms[order], dtype=np.float64),
        ),
    }
    for field in output_schema:
        name = str(field["name"])
        values = np.asarray(outputs_group[name])[order]
        dims = (
            "result",
            *tuple(f"{name}_dim_{index}" for index in range(values.ndim - 1)),
        )
        data_vars[name] = (dims, values)

    return xr.Dataset(
        data_vars=data_vars,
        coords=coords,
        attrs={
            **root_attrs,
            "path": store_path.resolve(),
            "output_schema": output_schema,
            "stored_results": int(input_index.shape[0]),
            "has_failures": bool(root_attrs["has_failures"]),
            "average_completion_duration_ms": (
                float(np.mean(durations_ms)) if durations_ms.size else None
            ),
            "metadata": metadata,
        },
    )


class ZarrResultStore:
    def __init__(
        self,
        output_path: str,
        *,
        output_schema: OutputSpec,
        batch_size: int,
        total_inputs: int,
        metadata_attrs: Mapping[str, Any] | None = None,
        metadata_arrays: Mapping[str, Any] | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if total_inputs < 0:
            raise ValueError("total_inputs must be >= 0")

        self.batch_size = batch_size
        self.success_count = 0
        self.has_failures = False
        self._output_fields = output_schema.fields
        self._pending_success_indices: list[int] = []
        self._pending_success_outputs: dict[str, list[np.ndarray[Any, Any]]] = {
            field.name: [] for field in self._output_fields
        }
        self._pending_success_durations_ms: list[float] = []

        root = cast(Any, zarr.open_group(output_path, mode="w", zarr_format=2))
        self._root = root
        attrs = {
            "output_schema": output_schema.metadata(),
            "total_inputs": int(total_inputs),
            "has_failures": False,
        }
        if metadata_attrs is not None:
            attrs.update(dict(metadata_attrs))
        root.attrs.update(attrs)

        results = root.create_group("results", overwrite=True)
        outputs = results.create_group("outputs", overwrite=True)
        metadata = root.create_group("metadata", overwrite=True)
        for name, value in (metadata_arrays or {}).items():
            metadata.create_array(
                str(name),
                data=np.asarray(value),
                overwrite=True,
            )

        self.results_input_index = results.create_array(
            "input_index",
            shape=(0,),
            chunks=(batch_size,),
            dtype="i8",
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

    def add_success(
        self,
        input_index: int,
        output: RunOutput,
        duration_ms: float,
    ) -> None:
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
        for field_name, array in normalized_output.items():
            self._pending_success_outputs[field_name].append(array)
        self._pending_success_durations_ms.append(float(duration_ms))
        self.success_count += 1

        if len(self._pending_success_indices) >= self.batch_size:
            self._flush_successes()

    def add_failure(self) -> None:
        self.has_failures = True
        self._root.attrs["has_failures"] = True

    def flush(self) -> None:
        self._flush_successes()

    def _flush_successes(self) -> None:
        if not self._pending_success_indices:
            return

        self.results_input_index.append(
            np.asarray(
                self._pending_success_indices,
                dtype=self.results_input_index.dtype,
            ),
            axis=0,
        )
        for field in self._output_fields:
            stacked = np.stack(
                self._pending_success_outputs[field.name],
                axis=0,
            ).astype(self.results_outputs[field.name].dtype, copy=False)
            self.results_outputs[field.name].append(stacked, axis=0)
            self._pending_success_outputs[field.name].clear()
        self.results_durations_ms.append(
            np.asarray(
                self._pending_success_durations_ms,
                dtype=self.results_durations_ms.dtype,
            ),
            axis=0,
        )

        self._pending_success_indices.clear()
        self._pending_success_durations_ms.clear()
