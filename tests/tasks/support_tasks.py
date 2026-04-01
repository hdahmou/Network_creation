from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from tasks.dispatch.registry import register_task
from tasks.dispatch.runtime import OutputField, OutputSchema, PreparedTask


def _load_int_inputs(inputs_json: str) -> list[int]:
    payload = json.loads(Path(inputs_json).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected a list of integers, got {type(payload)}")
    return [int(item) for item in payload]


@register_task
class VectorTask:
    name: ClassVar[str] = "vector"

    def prepare(self, config: Mapping[str, Any]) -> PreparedTask[int]:
        inputs_json = str(config["inputs_json"])
        inputs = _load_int_inputs(inputs_json)
        return PreparedTask(
            inputs=inputs,
            output_schema=OutputSchema(
                fields=(OutputField(name="vector", shape=(2,), dtype=np.int64),)
            ),
            run=self.run,
        )

    def run(self, task_input: int) -> dict[str, np.ndarray[Any, Any]]:
        return {"vector": np.asarray([task_input, task_input + 1], dtype=np.int64)}


@register_task
class RuntimeFailureTask(VectorTask):
    name: ClassVar[str] = "runtime_failure"

    def run(self, task_input: int) -> dict[str, np.ndarray[Any, Any]]:
        if task_input < 0:
            raise RuntimeError(f"negative values are not allowed: {task_input}")
        return super().run(task_input)


@register_task
class StructuredTask:
    name: ClassVar[str] = "structured"

    def prepare(self, config: Mapping[str, Any]) -> PreparedTask[int]:
        inputs_json = str(config["inputs_json"])
        inputs = _load_int_inputs(inputs_json)
        return PreparedTask(
            inputs=inputs,
            output_schema=OutputSchema(
                fields=(
                    OutputField(name="vector", shape=(2,), dtype=np.int64),
                    OutputField(name="parity", shape=(), dtype=np.int8),
                )
            ),
            run=self.run,
        )

    def run(self, task_input: int) -> dict[str, Any]:
        return {
            "vector": np.asarray([task_input, task_input + 1], dtype=np.int64),
            "parity": np.asarray(task_input % 2, dtype=np.int8),
        }


@register_task
class GeneratedTask:
    name: ClassVar[str] = "generated"

    def prepare(self, config: Mapping[str, Any]) -> PreparedTask[int]:
        raw_spec = config["input_spec"]
        if not isinstance(raw_spec, Mapping):
            raise ValueError(f"Expected input_spec to be a mapping, got {type(raw_spec)!r}")
        start = int(raw_spec.get("start", 0))
        count = int(raw_spec["count"])
        return PreparedTask(
            inputs=[start + offset for offset in range(count)],
            output_schema=OutputSchema(
                fields=(OutputField(name="vector", shape=(2,), dtype=np.int64),)
            ),
            run=self.run,
        )

    def run(self, task_input: int) -> dict[str, np.ndarray[Any, Any]]:
        return {"vector": np.asarray([task_input, task_input + 1], dtype=np.int64)}
