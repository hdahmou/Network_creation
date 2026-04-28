"""Shared schema types for executable runs."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

import numpy as np


RunInputT = TypeVar("RunInputT")
RunOutput = Mapping[str, Any]


@dataclass(frozen=True)
class OutputArraySpec:
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
class OutputSpec:
    fields: tuple[OutputArraySpec, ...]

    def __post_init__(self) -> None:
        field_names = [field.name for field in self.fields]
        if len(set(field_names)) != len(field_names):
            raise ValueError("output schema field names must be unique")

    def metadata(self) -> list[dict[str, object]]:
        return [field.metadata() for field in self.fields]


@dataclass(frozen=True)
class PreparedRun(Generic[RunInputT]):
    inputs: Sequence[RunInputT]
    output_schema: OutputSpec
    run: Callable[[RunInputT], RunOutput]
    result_attrs: Mapping[str, Any] = field(default_factory=dict)
    result_arrays: Mapping[str, Any] = field(default_factory=dict)
