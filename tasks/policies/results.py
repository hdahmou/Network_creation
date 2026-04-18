"""Policy result contracts and loading."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict, cast

import numpy as np
from numpy.typing import NDArray

from ..dispatch.result_store import load_result_zarr
from .support import generate_parameter_table, load_simulation_sweep


type TensorDims = tuple[str, ...]
type PolicyTensor = NDArray[np.number[Any]] | NDArray[np.bool_]
type StatusTensor = (
    NDArray[np.bool_]
    | NDArray[np.float64]
    | NDArray[np.int64]
    | NDArray[np.str_]
)
type DimLabels = Sequence[str] | NDArray[np.integer[Any]] | NDArray[np.floating[Any]]


class PolicyTensorGroup(TypedDict):
    tensors: dict[str, PolicyTensor]
    dims: dict[str, TensorDims]
    labels: dict[str, DimLabels]


class PolicyStatusGroup(TypedDict):
    tensors: dict[str, StatusTensor]
    dims: dict[str, TensorDims]
    labels: dict[str, DimLabels]


class PolicyResultMetadata(TypedDict):
    task_name: str
    task_class: str
    result_store_path: Path
    config_path: Path
    average_completion_duration_ms: float | None
    input_index: NDArray[np.int64]
    output_schema: list[dict[str, object]]
    simulation: dict[str, object]


@dataclass(frozen=True)
class PolicyFormattedResults:
    inputs: PolicyTensorGroup
    outputs: PolicyTensorGroup
    status: PolicyStatusGroup
    metadata: PolicyResultMetadata


def load_policy_results(zarr_path: str | Path) -> PolicyFormattedResults:
    result_data = load_result_zarr(zarr_path)
    result_store_path = Path(result_data["path"])
    config_path = result_store_path.with_suffix(".dispatch_config.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(
            f"Dispatch config at {config_path} must contain a JSON object"
        )

    from tasks.catalog import TASKS

    task_name = str(result_data["task_name"])
    task_class = TASKS.get(task_name)
    if task_class is None:
        raise ValueError(f"Unsupported policy task: {task_name}")

    simulation = config.get("simulation")
    if not isinstance(simulation, dict):
        raise ValueError(
            f"Dispatch config at {config_path} must contain a 'simulation' object"
        )

    input_spec = cast(dict[str, object], config["input_spec"])
    parameter_specs = cast(dict[str, dict[str, object]], input_spec["parameters"])
    parameter_table = generate_parameter_table(input_spec)
    simulation_sweep = load_simulation_sweep(
        cast(dict[str, object], simulation),
        parameter_table,
        parameter_specs=parameter_specs,
    )
    n_agents_count = len(simulation_sweep.n_agents_values)
    n_parameter_rows = simulation_sweep.n_parameter_rows
    input_index = np.arange(
        simulation_sweep.total_inputs,
        dtype=np.int64,
    ).reshape(n_agents_count, n_parameter_rows)
    success_index = np.asarray(result_data["input_index"], dtype=np.int64)
    success_durations = np.asarray(result_data["durations_ms"], dtype=np.float64)
    output_schema = list(result_data["output_schema"])
    outputs: dict[str, NDArray[np.generic]] = {}
    for field in output_schema:
        name = str(field["name"])
        flat_values = np.asarray(result_data["outputs"][name])
        dense_values = np.zeros(
            (n_agents_count, n_parameter_rows, *flat_values.shape[1:]),
            dtype=flat_values.dtype,
        )
        for row_index, flat_input_index in enumerate(success_index):
            n_agents_index, parameter_row_index = simulation_sweep.coordinates(
                int(flat_input_index)
            )
            dense_values[n_agents_index, parameter_row_index] = flat_values[row_index]
        outputs[name] = dense_values
    output_group = task_class.build_output_group(
        outputs=cast(dict[str, NDArray[np.generic]], outputs),
    )
    output_group["labels"] = {
        "n_agents": np.asarray(simulation_sweep.n_agents_values, dtype=np.int64),
        "parameter_row": np.arange(n_parameter_rows, dtype=np.int64),
        **output_group["labels"],
    }

    parameter_values = np.asarray(
        simulation_sweep.configuration_values[:, :, 1:],
        dtype=np.float32,
    )
    success_mask = np.zeros((n_agents_count, n_parameter_rows), dtype=np.bool_)
    duration_ms = np.full((n_agents_count, n_parameter_rows), np.nan, dtype=np.float64)
    for row_index, flat_input_index in enumerate(success_index):
        n_agents_index, parameter_row_index = simulation_sweep.coordinates(
            int(flat_input_index)
        )
        success_mask[n_agents_index, parameter_row_index] = True
        duration_ms[n_agents_index, parameter_row_index] = success_durations[row_index]
    finite_durations = duration_ms[np.isfinite(duration_ms)]
    error = np.full((n_agents_count, n_parameter_rows), "", dtype="<U1")

    return PolicyFormattedResults(
        inputs={
            "tensors": {
                "parameters": parameter_values,
            },
            "dims": {
                "parameters": ("n_agents", "parameter_row", "parameter"),
            },
            "labels": {
                "n_agents": np.asarray(simulation_sweep.n_agents_values, dtype=np.int64),
                "parameter_row": np.arange(n_parameter_rows, dtype=np.int64),
                "parameter": parameter_table.parameter_names,
            },
        },
        outputs=output_group,
        status={
            "tensors": {
                "success": success_mask,
                "error": error,
                "duration_ms": duration_ms,
            },
            "dims": {
                "success": ("n_agents", "parameter_row"),
                "error": ("n_agents", "parameter_row"),
                "duration_ms": ("n_agents", "parameter_row"),
            },
            "labels": {
                "n_agents": np.asarray(simulation_sweep.n_agents_values, dtype=np.int64),
                "parameter_row": np.arange(n_parameter_rows, dtype=np.int64),
            },
        },
        metadata={
            "task_name": task_name,
            "task_class": str(result_data["task_class"]),
            "result_store_path": result_store_path,
            "config_path": config_path,
            "average_completion_duration_ms": (
                float(np.mean(finite_durations)) if finite_durations.size else None
            ),
            "input_index": input_index,
            "output_schema": output_schema,
            "simulation": cast(dict[str, object], simulation),
        },
    )
