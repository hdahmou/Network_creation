"""Shared task runtime types for repo-owned dispatch."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, Generic, Protocol, TypeVar

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

    @property
    def field_map(self) -> dict[str, OutputField]:
        return {field.name: field for field in self.fields}

    def metadata(self) -> list[dict[str, object]]:
        return [field.metadata() for field in self.fields]


@dataclass(frozen=True)
class PreparedTask(Generic[TaskInputT]):
    inputs: Sequence[TaskInputT]
    output_schema: OutputSchema
    run: Callable[[TaskInputT], TaskOutput]
    describe_input_fn: Callable[[TaskInputT], str] | None = None
    describe_output_fn: Callable[[TaskOutput], str] | None = None

    def describe_input(self, task_input: TaskInputT) -> str:
        if self.describe_input_fn is None:
            return str(task_input)
        return str(self.describe_input_fn(task_input))

    def describe_output(self, task_output: TaskOutput) -> str:
        if self.describe_output_fn is not None:
            return str(self.describe_output_fn(task_output))

        parts: list[str] = []
        for field_name, value in task_output.items():
            array = np.asarray(value)
            shape = tuple(int(dim) for dim in array.shape)
            parts.append(f"{field_name}:shape={shape} dtype={array.dtype}")
        return "; ".join(parts)


class Task(Protocol[TaskInputT]):
    name: ClassVar[str]

    def prepare(self, config: Mapping[str, Any]) -> PreparedTask[TaskInputT]:
        """Load inputs from the task config and return a prepared task ready for dispatch."""
