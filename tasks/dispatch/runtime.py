"""Shared task runtime types for repo-owned dispatch."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

import numpy as np


TaskInputT = TypeVar("TaskInputT")
TaskOutput = Mapping[str, Any]


@dataclass(frozen=True)
class OutputField:
    name: str
    shape: tuple[int, ...]
    dtype: np.dtype[Any] | str | type[np.generic]

    @property
    def resolved_dtype(self) -> np.dtype[Any]:
        return np.dtype(self.dtype)

    def metadata(self) -> dict[str, object]:
        return {
            "name": self.name,
            "shape": [int(dim) for dim in self.shape],
            "dtype": str(self.resolved_dtype),
        }


@dataclass(frozen=True)
class OutputSchema:
    fields: tuple[OutputField, ...]

    def __post_init__(self) -> None:
        field_names = [field.name for field in self.fields]
        if len(set(field_names)) != len(field_names):
            raise ValueError("output schema field names must be unique")

    def metadata(self) -> list[dict[str, object]]:
        return [field.metadata() for field in self.fields]


@dataclass(frozen=True)
class PreparedTask(Generic[TaskInputT]):
    inputs: Sequence[TaskInputT]
    output_schema: OutputSchema
    run: Callable[[TaskInputT], TaskOutput]
    result_attrs: Mapping[str, Any] = field(default_factory=dict)
    result_arrays: Mapping[str, Any] = field(default_factory=dict)
