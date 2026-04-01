"""Typed notebook-facing result contract and shared policy result loading."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict, cast

import numpy as np
from numpy.typing import NDArray

from ..dispatch.result_store import load_result_zarr
from .utils import ParameterTable, generate_parameter_table


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
    input_index: NDArray[np.int64]
    output_schema: list[dict[str, object]]

@dataclass(frozen=True)
class PolicyFormattedResults:
    inputs: PolicyTensorGroup
    outputs: PolicyTensorGroup
    metadata: PolicyResultMetadata
    failures: dict[str, FailureArray]


def load_policy_result_data(
    zarr_path: str | Path,
) -> tuple[
    dict[str, Any],
    Path,
    NDArray[np.int64],
    ParameterTable,
    NDArray[np.float32],
]:
    result_data = load_result_zarr(zarr_path)
    config_path = Path(result_data["path"]).with_suffix(".dispatch_config.json")
    if not config_path.exists():
        raise FileNotFoundError(f"Dispatch config copy not found: {config_path}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"Dispatch config must be a JSON object: {config_path}")

    parameter_table = generate_parameter_table(cast(dict[str, object], config["input_spec"]))
    input_index = np.asarray(result_data["input_index"], dtype=np.int64)
    parameter_values = np.asarray(parameter_table.parameter_values[input_index], dtype=np.float32)

    return result_data, config_path, input_index, parameter_table, parameter_values
