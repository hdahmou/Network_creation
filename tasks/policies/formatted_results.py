"""Typed formatted policy result contract."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
from numpy.typing import NDArray

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

@dataclass(frozen=True)
class PolicyFormattedResults:
    inputs: PolicyTensorGroup
    outputs: PolicyTensorGroup
    metadata: PolicyResultMetadata
    failures: dict[str, FailureArray]
