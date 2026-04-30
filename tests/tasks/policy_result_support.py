"""Helpers for policy result formatter tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from tasks.execution.zarr_store import ZarrResultStore
from tasks.execution.schema import OutputArraySpec, OutputSpec
from tasks.policies.sweep import (
    collect_parameter_specs,
    sample_parameters,
    parse_policy_entries,
    build_sweep,
    infer_parameter_bounds,
)
from .config_support import continuous, fixed, policy, task_config


def build_myopic_config(*, log2_budget: int = 2) -> dict[str, object]:
    return task_config(
        n_agents=4,
        n_actors=2,
        log2_budget=log2_budget,
        game={
            "benefit_to_cost_ratio": fixed(0.5),
            "distance_discount": continuous(0.1, 0.9),
        },
        visibility_distance=fixed(4.0),
        policies=[
            policy(
                "myopic",
                "myopic",
                {"bridge_probability": continuous(0.0, 1.0)},
            )
        ],
    )


def build_best_response_config(*, log2_budget: int = 2) -> dict[str, object]:
    return task_config(
        n_agents=4,
        n_actors=2,
        log2_budget=log2_budget,
        game={
            "benefit_to_cost_ratio": fixed(0.5),
            "distance_discount": continuous(0.1, 0.9),
        },
        visibility_distance=fixed(4.0),
        policies=[
            policy(
                "best_response",
                "best-response",
                {
                    "outside_optimism": continuous(0.0, 1.0),
                    "exploration": continuous(1e-3, 1.0),
                },
            )
        ],
    )


def build_random_config() -> dict[str, object]:
    return task_config(
        n_agents=4,
        n_actors=2,
        visibility_distance=fixed(4.0),
        policies=[policy("random", "random")],
    )


def build_policy_parameter_samples(config: dict[str, object]):
    input_spec = collect_parameter_specs(config)
    return sample_parameters(input_spec)


def write_result_store(
    base_dir: Path,
    *,
    config: dict[str, object],
    outputs_by_index: list[tuple[int, dict[str, np.ndarray[Any, Any]]]],
    total_inputs: int,
) -> Path:
    if not outputs_by_index:
        raise ValueError("outputs_by_index must not be empty")

    first_output = outputs_by_index[0][1]
    output_schema = OutputSpec(
        fields=tuple(
            OutputArraySpec(
                name=name,
                shape=tuple(np.asarray(values).shape),
                dtype=np.asarray(values).dtype,
            )
            for name, values in first_output.items()
        )
    )
    metadata_attrs, metadata_arrays = _policy_metadata(config)

    store_path = base_dir / "results.zarr"
    writer = ZarrResultStore(
        str(store_path),
        output_schema=output_schema,
        batch_size=2,
        total_inputs=total_inputs,
        metadata_attrs={
            **metadata_attrs,
            "config_path": str(store_path.with_suffix(".run_config.json")),
        },
        metadata_arrays=metadata_arrays,
    )
    for input_index, output in outputs_by_index:
        writer.add_success(
            input_index=input_index,
            output=output,
        )
    writer.flush()

    config_path = store_path.with_suffix(".run_config.json")
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return store_path


def _policy_metadata(
    config: dict[str, object],
) -> tuple[dict[str, object], dict[str, np.ndarray[Any, Any]]]:
    input_spec = collect_parameter_specs(config)
    parameter_specs = input_spec["parameters"]
    parameter_samples = sample_parameters(input_spec)
    policy_entries = parse_policy_entries(config)
    sweep = build_sweep(
        config,
        parameter_samples,
        parameter_specs=parameter_specs,
        policy_ids=tuple(policy_id for policy_id, _ in policy_entries),
    )
    parameter_names = sweep.parameter_names_for_results
    parameters = sweep.parameter_values_for_results()
    parameter_low, parameter_high = infer_parameter_bounds(
        parameters,
        parameter_names=parameter_names,
        parameter_specs=parameter_specs,
    )
    input_coordinates = np.asarray(
        [
            sweep.coordinates(input_index)
            for input_index in range(sweep.total_inputs)
        ],
        dtype=np.int64,
    )
    return (
        {
            "parameter_names": list(parameter_names),
            "policy_config": {
                key: config[key]
                for key in ("sampling", "simulator", "game", "population", "observers")
                if key in config
            },
        },
        {
            "n_agents": np.asarray(sweep.n_agents_values, dtype=np.int64),
            "parameters": parameters,
            "parameter_low": parameter_low,
            "parameter_high": parameter_high,
            "input_coordinates": input_coordinates,
        },
    )
