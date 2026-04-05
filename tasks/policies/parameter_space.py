"""Parameter-space expansion helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast

import numpy as np
from numpy.typing import NDArray


type ParameterSpec = Mapping[str, object]
type ParameterConfig = Mapping[str, ParameterSpec]


def build_parameter_values(
    parameter_type: str,
    payload: object,
    *,
    sample_budget: int,
    sampler_seed: int,
) -> NDArray[np.float32]:
    if parameter_type == "fixed":
        return np.full(
            sample_budget,
            float(cast(int | float, payload)),
            dtype=np.float32,
        )

    if parameter_type == "discrete":
        values = cast(Sequence[int | float], payload)
        rng = np.random.default_rng(sampler_seed)
        sampled = rng.choice(
            np.asarray(values, dtype=np.float32),
            size=sample_budget,
            replace=True,
        )
        return np.asarray(sampled, dtype=np.float32)

    if parameter_type == "sampled_discrete_range":
        range_spec = cast(Mapping[str, object], payload)
        values = np.arange(
            int(range_spec["min"]),
            int(range_spec["max"]) + 1,
            int(range_spec.get("step", 1)),
            dtype=np.float32,
        )
        rng = np.random.default_rng(sampler_seed)
        sampled = rng.choice(values, size=sample_budget, replace=True)
        return np.asarray(sampled, dtype=np.float32)

    raise ValueError(f"Unsupported parameter type {parameter_type!r}")


def build_continuous_parameter_values(
    continuous_parameters: Sequence[tuple[str, object]],
    *,
    sample_budget: int,
    sampler_seed: int,
) -> tuple[tuple[str, ...], NDArray[np.float32]]:
    if not continuous_parameters:
        return (), np.zeros((sample_budget, 0), dtype=np.float32)

    from scipy.stats import qmc

    sampler = qmc.Sobol(
        d=len(continuous_parameters),
        scramble=True,
        optimization="random-cd",
        rng=sampler_seed,
    )
    unit_samples = np.asarray(
        sampler.random(n=sample_budget),
        dtype=np.float32,
    )
    continuous_parameter_values = np.empty_like(unit_samples)

    for column_index, (_, payload) in enumerate(continuous_parameters):
        continuous_range = cast(Mapping[str, object], payload)
        minimum = float(continuous_range["min"])
        maximum = float(continuous_range["max"])
        continuous_parameter_values[:, column_index] = minimum + (
            unit_samples[:, column_index] * (maximum - minimum)
        )

    return (
        tuple(name for name, _ in continuous_parameters),
        continuous_parameter_values,
    )


def generate_parameter_space(
    parameters: ParameterConfig,
    *,
    sample_budget: int,
    sampler_seed: int,
) -> tuple[tuple[str, ...], NDArray[np.float32]]:
    parameter_names = tuple(parameters)
    continuous_parameters: list[tuple[str, object]] = []
    non_continuous_parameter_names: list[str] = []
    non_continuous_parameter_columns: list[NDArray[np.float32]] = []
    effective_sample_budget = (
        sample_budget
        if any(str(spec["type"]) != "fixed" for spec in parameters.values())
        else 1
    )

    for parameter_index, (name, spec) in enumerate(parameters.items()):
        parameter_type = str(spec["type"])
        payload = spec["payload"]
        if parameter_type == "continuous":
            continuous_parameters.append((name, payload))
            continue

        column_values = build_parameter_values(
            parameter_type,
            payload,
            sample_budget=effective_sample_budget,
            sampler_seed=sampler_seed + parameter_index,
        )
        non_continuous_parameter_names.append(name)
        non_continuous_parameter_columns.append(column_values)

    continuous_parameter_names, continuous_parameter_values = (
        build_continuous_parameter_values(
            continuous_parameters,
            sample_budget=effective_sample_budget,
            sampler_seed=sampler_seed,
        )
    )

    output = np.empty((effective_sample_budget, len(parameter_names)), dtype=np.float32)
    continuous_parameter_column_index = {
        name: index for index, name in enumerate(continuous_parameter_names)
    }
    non_continuous_parameter_column_index = {
        name: index for index, name in enumerate(non_continuous_parameter_names)
    }

    for column_index, name in enumerate(parameter_names):
        if name in continuous_parameter_column_index:
            output[:, column_index] = continuous_parameter_values[
                :,
                continuous_parameter_column_index[name],
            ]
        else:
            output[:, column_index] = non_continuous_parameter_columns[
                non_continuous_parameter_column_index[name]
            ]

    return parameter_names, output
