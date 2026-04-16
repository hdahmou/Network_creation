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
from .support import generate_parameter_table


type TensorDims = tuple[str, ...]
type PolicyTensor = NDArray[np.number[Any]] | NDArray[np.bool_]
type DimLabels = Sequence[str] | NDArray[np.integer[Any]] | NDArray[np.floating[Any]]
type FailureArray = NDArray[np.int64] | NDArray[np.float64] | NDArray[np.str_]


class PolicyTensorGroup(TypedDict):
    tensors: dict[str, PolicyTensor]
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
    metadata: PolicyResultMetadata
    failures: dict[str, FailureArray]


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

    parameter_table = generate_parameter_table(cast(dict[str, object], config["input_spec"]))
    input_index = np.asarray(result_data["input_index"], dtype=np.int64)
    parameter_values = np.asarray(
        parameter_table.parameter_values[input_index],
        dtype=np.float32,
    )
    output_group = task_class.build_output_group(
        outputs=cast(dict[str, NDArray[np.generic]], result_data["outputs"]),
        input_index=input_index,
    )

    return PolicyFormattedResults(
        inputs={
            "tensors": {"parameters": parameter_values},
            "dims": {"parameters": ("sample", "parameter")},
            "labels": {
                "sample": input_index,
                "parameter": parameter_table.parameter_names,
            },
        },
        outputs=output_group,
        metadata={
            "task_name": task_name,
            "task_class": str(result_data["task_class"]),
            "result_store_path": result_store_path,
            "config_path": config_path,
            "average_completion_duration_ms": cast(
                float | None, result_data["average_completion_duration_ms"]
            ),
            "input_index": input_index,
            "output_schema": list(result_data["output_schema"]),
            "simulation": cast(dict[str, object], simulation),
        },
        failures=cast(
            dict[str, NDArray[np.int64] | NDArray[np.float64] | NDArray[np.str_]],
            result_data["failures"],
        ),
    )
