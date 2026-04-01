"""Helpers for policy result formatter tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from tasks.dispatch.result_store import ResultStore
from tasks.dispatch.runtime import OutputField, OutputSchema


def build_myopic_config(*, log2_budget: int = 2) -> dict[str, object]:
    return {
        "task": "myopic-policy",
        "input_spec": {
            "log2_budget": log2_budget,
            "sampler_seed": 123,
            "parameters": {
                "link_cost": 20.0,
                "decay": {"min": 0.1, "max": 0.9},
                "per_link_benefit": 10.0,
                "bridge_probability": {"min": 0.0, "max": 1.0},
            },
        },
    }


def build_random_config() -> dict[str, object]:
    return {
        "task": "random-policy",
        "input_spec": {
            "log2_budget": 0,
            "sampler_seed": 123,
            "parameters": {},
        },
    }


def write_result_store(
    base_dir: Path,
    *,
    task_name: str,
    task_class: str,
    config: dict[str, object],
    outputs_by_index: list[tuple[int, dict[str, np.ndarray[Any, Any]]]],
    total_inputs: int,
) -> Path:
    if not outputs_by_index:
        raise ValueError("outputs_by_index must not be empty")

    first_output = outputs_by_index[0][1]
    output_schema = OutputSchema(
        fields=tuple(
            OutputField(
                name=name,
                shape=tuple(np.asarray(values).shape),
                dtype=np.asarray(values).dtype,
            )
            for name, values in first_output.items()
        )
    )

    store_path = base_dir / f"{task_name}.zarr"
    writer = ResultStore(
        str(store_path),
        task_name=task_name,
        task_class=task_class,
        output_schema=output_schema,
        batch_size=2,
        total_inputs=total_inputs,
    )
    for input_index, output in outputs_by_index:
        writer.add_success(
            input_index=input_index,
            input_label=f"input-{input_index}",
            output=output,
            duration_ms=float(input_index + 1),
        )
    writer.flush()

    config_path = store_path.with_suffix(".dispatch_config.json")
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return store_path
