from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from tasks.execution.schema import OutputArraySpec, OutputSpec, PreparedRun


def _load_int_inputs(inputs_json: str) -> list[int]:
    payload = json.loads(Path(inputs_json).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected a list of integers, got {type(payload)}")
    return [int(item) for item in payload]


class VectorTask:
    name: ClassVar[str] = "vector"

    def prepare(self, config: Mapping[str, Any]) -> PreparedRun[int]:
        inputs_json = str(config["inputs_json"])
        inputs = _load_int_inputs(inputs_json)
        return PreparedRun(
            inputs=inputs,
            output_schema=OutputSpec(
                fields=(OutputArraySpec(name="vector", shape=(2,), dtype=np.int64),)
            ),
            run=self.run,
        )

    def run(self, task_input: int) -> dict[str, np.ndarray[Any, Any]]:
        return {"vector": np.asarray([task_input, task_input + 1], dtype=np.int64)}


class RuntimeFailureTask(VectorTask):
    name: ClassVar[str] = "runtime_failure"

    def run(self, task_input: int) -> dict[str, np.ndarray[Any, Any]]:
        if task_input < 0:
            raise RuntimeError(f"negative values are not allowed: {task_input}")
        return super().run(task_input)


class StructuredTask:
    name: ClassVar[str] = "structured"

    def prepare(self, config: Mapping[str, Any]) -> PreparedRun[int]:
        inputs_json = str(config["inputs_json"])
        inputs = _load_int_inputs(inputs_json)
        return PreparedRun(
            inputs=inputs,
            output_schema=OutputSpec(
                fields=(
                    OutputArraySpec(name="vector", shape=(2,), dtype=np.int64),
                    OutputArraySpec(name="parity", shape=(), dtype=np.int8),
                )
            ),
            run=self.run,
        )

    def run(self, task_input: int) -> dict[str, Any]:
        return {
            "vector": np.asarray([task_input, task_input + 1], dtype=np.int64),
            "parity": np.asarray(task_input % 2, dtype=np.int8),
        }


class GeneratedTask:
    name: ClassVar[str] = "generated"

    def prepare(self, config: Mapping[str, Any]) -> PreparedRun[int]:
        raw_spec = config["input_spec"]
        if not isinstance(raw_spec, Mapping):
            raise ValueError(f"Expected input_spec to be a mapping, got {type(raw_spec)!r}")
        start = int(raw_spec.get("start", 0))
        count = int(raw_spec["count"])
        return PreparedRun(
            inputs=[start + offset for offset in range(count)],
            output_schema=OutputSpec(
                fields=(OutputArraySpec(name="vector", shape=(2,), dtype=np.int64),)
            ),
            run=self.run,
        )

    def run(self, task_input: int) -> dict[str, np.ndarray[Any, Any]]:
        return {"vector": np.asarray([task_input, task_input + 1], dtype=np.int64)}


TEST_POLICY_REGISTRY = {
    VectorTask.name: VectorTask,
    RuntimeFailureTask.name: RuntimeFailureTask,
    StructuredTask.name: StructuredTask,
    GeneratedTask.name: GeneratedTask,
}
