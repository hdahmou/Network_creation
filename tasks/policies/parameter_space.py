"""Parameter-space expansion helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Real
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray


type ParameterSpec = Mapping[str, object]
type ParameterConfig = Mapping[str, ParameterSpec]


def _read_finite_real(spec: ParameterSpec, key: str, *, parameter_name: str) -> float:
    value = spec.get(key)
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{parameter_name}.{key} must be a finite number")
    numeric_value = float(value)
    if not np.isfinite(numeric_value):
        raise ValueError(f"{parameter_name}.{key} must be a finite number")
    return numeric_value


def validate_continuous_parameter_spec(
    name: str,
    spec: ParameterSpec,
) -> tuple[float, float, float]:
    minimum = _read_finite_real(spec, "min", parameter_name=name)
    maximum = _read_finite_real(spec, "max", parameter_name=name)
    std = _read_finite_real(spec, "std", parameter_name=name)
    if minimum >= maximum:
        raise ValueError(f"{name}.min must be less than {name}.max")
    if std < 0.0:
        raise ValueError(f"{name}.std must be non-negative")
    return minimum, maximum, std


def build_parameter_values(
    spec: ParameterSpec,
    *,
    sample_budget: int,
    sampler_seed: int,
) -> NDArray[np.float32]:
    kind = str(spec["kind"])

    if kind == "fixed":
        return np.full(
            sample_budget,
            float(cast(int | float, spec["value"])),
            dtype=np.float32,
        )

    if kind == "scaled_inverse_agents":
        return np.full(sample_budget, np.nan, dtype=np.float32)

    if kind == "sampled_choice":
        values = cast(Sequence[int | float], spec["values"])
        rng = np.random.default_rng(sampler_seed)
        sampled = rng.choice(
            np.asarray(values, dtype=np.float32),
            size=sample_budget,
            replace=True,
        )
        return np.asarray(sampled, dtype=np.float32)

    if kind == "sampled_discrete_range":
        range_values = np.arange(
            int(cast(Any, spec["min"])),
            int(cast(Any, spec["max"])) + 1,
            int(cast(Any, spec.get("step", 1))),
            dtype=np.float32,
        )
        rng = np.random.default_rng(sampler_seed)
        sampled = rng.choice(range_values, size=sample_budget, replace=True)
        return np.asarray(sampled, dtype=np.float32)

    raise ValueError(f"Unsupported parameter kind {kind!r}")


def build_continuous_parameter_values(
    continuous_parameters: Sequence[tuple[str, ParameterSpec]],
    *,
    sample_budget: int,
    sampler_seed: int,
) -> tuple[tuple[str, ...], NDArray[np.float32]]:
    if not continuous_parameters:
        return (), np.zeros((sample_budget, 0), dtype=np.float32)

    from scipy.stats import qmc  # type: ignore[import-untyped]

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

    for column_index, (name, spec) in enumerate(continuous_parameters):
        minimum, maximum, _ = validate_continuous_parameter_spec(name, spec)
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
    continuous_parameters: list[tuple[str, ParameterSpec]] = []
    non_continuous_parameter_names: list[str] = []
    non_continuous_parameter_columns: list[NDArray[np.float32]] = []
    effective_sample_budget = (
        sample_budget
        if any(
            str(spec["kind"]) not in {"fixed", "scaled_inverse_agents"}
            for spec in parameters.values()
        )
        else 1
    )

    for parameter_index, (name, spec) in enumerate(parameters.items()):
        kind = str(spec["kind"])
        if kind == "continuous":
            continuous_parameters.append((name, spec))
            continue

        column_values = build_parameter_values(
            spec,
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
