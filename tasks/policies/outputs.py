"""Policy result tensor assembly."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from tasks.execution.schema import OutputArraySpec, OutputSpec

from .parameter_space import ParameterSpec
from .sweep import SweepPlan, infer_parameter_bounds


SAMPLED_AGENT_PARAMETERS_FIELD = "sampled_agent_parameters"
SAMPLED_AGENT_PARAMETER_NAMES_ATTR = "sampled_agent_parameter_names"


@dataclass(frozen=True)
class PolicyOutput:
    timeseries: NDArray[np.float32]
    observable_names: tuple[str, ...]
    step_labels: NDArray[np.int64]


def stack_result_tensors(
    outputs: Mapping[str, NDArray[np.generic]],
) -> tuple[NDArray[np.float32], tuple[str, ...]]:
    series_names: list[str] = []
    series_arrays: list[NDArray[np.float32]] = []
    expected_steps: int | None = None

    for name, values in outputs.items():
        array = np.asarray(values)
        if array.ndim != 2:
            continue
        if expected_steps is None:
            expected_steps = array.shape[1]
        elif array.shape[1] != expected_steps:
            raise ValueError("time-series outputs must share the same step axis length")
        series_names.append(name)
        series_arrays.append(np.asarray(array, dtype=np.float32))

    if series_arrays:
        return (
            np.stack(series_arrays, axis=2).astype(np.float32, copy=False),
            tuple(series_names),
        )

    scalar_names: list[str] = []
    scalar_arrays: list[NDArray[np.float32]] = []
    for name, values in outputs.items():
        array = np.asarray(values)
        if array.ndim == 1:
            scalar_names.append(name)
            scalar_arrays.append(np.asarray(array, dtype=np.float32))

    if scalar_arrays:
        scalar_values = np.stack(scalar_arrays, axis=1).astype(
            np.float32,
            copy=False,
        )
        return scalar_values[:, None, :], tuple(scalar_names)

    raise ValueError("results must contain scalar or time-series outputs")


def build_policy_outputs(
    outputs: dict[str, NDArray[np.generic]],
    *,
    step_labels: NDArray[np.int64] | None = None,
) -> PolicyOutput:
    timeseries, observable_names = stack_result_tensors(outputs)
    labels = (
        np.arange(timeseries.shape[1], dtype=np.int64)
        if step_labels is None
        else np.asarray(step_labels, dtype=np.int64)
    )
    if labels.shape != (timeseries.shape[1],):
        raise ValueError("step labels must match the output step axis")
    return PolicyOutput(
        timeseries=timeseries,
        observable_names=observable_names,
        step_labels=labels,
    )


def build_policy_output_spec(
    observer_fields: Sequence[OutputArraySpec],
    *,
    sampled_agent_parameter_names: Sequence[str],
    max_n_agents: int,
) -> OutputSpec:
    output_fields = tuple(observer_fields)
    if sampled_agent_parameter_names:
        output_fields = (
            *output_fields,
            OutputArraySpec(
                name=SAMPLED_AGENT_PARAMETERS_FIELD,
                shape=(max_n_agents, len(sampled_agent_parameter_names)),
                dtype=np.float32,
            ),
        )
    return OutputSpec(fields=output_fields)


def build_result_metadata(
    *,
    config: Mapping[str, Any],
    sweep: SweepPlan,
    parameter_specs: Mapping[str, ParameterSpec],
    sampled_agent_parameter_names: Sequence[str],
) -> tuple[dict[str, Any], dict[str, np.ndarray[Any, Any]]]:
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

    attrs: dict[str, Any] = {
        "parameter_names": list(parameter_names),
        "sweep_layout": (
            "sampled_composition"
            if sweep.policy_composition.kind == "balanced"
            else "cartesian"
        ),
        "policy_config": {
            key: config[key]
            for key in ("sampling", "simulator", "game", "population", "observers")
            if key in config
        },
    }
    if sampled_agent_parameter_names:
        attrs[SAMPLED_AGENT_PARAMETER_NAMES_ATTR] = list(
            sampled_agent_parameter_names
        )

    return (
        attrs,
        {
            "n_agents": np.asarray(
                sweep.n_agents_values,
                dtype=np.int64,
            ),
            "raw_step_horizons": np.asarray(
                sweep.raw_step_horizons,
                dtype=np.int64,
            ),
            "parameters": parameters,
            "parameter_low": parameter_low,
            "parameter_high": parameter_high,
            "input_coordinates": input_coordinates,
        },
    )
