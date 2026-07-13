"""Resolve model-inversion theta and conditioning schemas."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from network_creation.policy_result_metadata import (
    configured_n_actors as _configured_n_actors,
    configured_output_step_labels as _configured_output_step_labels,
    policy_composition_axis_name as _policy_composition_axis_name,
    theta_parameter_columns as _theta_parameter_columns,
)
from network_creation.time_normalization import common_normalized_time_grid

from .source import InversionSource


@dataclass(frozen=True)
class InversionSchema:
    observed_outputs: tuple[str, ...]
    continuous_input_names: tuple[str, ...]
    discrete_input_names: tuple[str, ...]
    continuous_theta_names: tuple[str, ...]
    discrete_theta_names: tuple[str, ...]
    prior_low: NDArray[np.float32]
    prior_high: NDArray[np.float32]
    x_components: tuple[str, ...]
    x_event_shape: tuple[int, ...]
    discrete_input_values: dict[str, tuple[int, ...]]
    discrete_theta_values: dict[str, tuple[int, ...]]
    constant_theta: dict[str, float]
    x_representation: Literal["final", "timeseries"]
    normalize_time: bool
    time_axis: str | None
    raw_step_labels: NDArray[np.float64] | None
    step_labels: NDArray[np.float64] | None
    final_step_label: int | float | None
    n_actors: int | None

    @property
    def theta_order(self) -> tuple[str, ...]:
        return self.continuous_theta_names + self.discrete_theta_names

    @property
    def sbi_method(self) -> Literal["NPE", "MNPE"]:
        return "MNPE" if self.discrete_theta_names else "NPE"

    @property
    def category_counts(self) -> tuple[int, ...]:
        return tuple(
            len(self.discrete_theta_values[name])
            for name in self.discrete_theta_names
        )


def resolve_inversion_schema(source: InversionSource) -> InversionSchema:
    config = source.config
    data = config.data
    output_names = tuple(source.observed_output_arrays)
    observed_outputs = _resolve_names(
        data.observed_outputs,
        output_names,
        kind="observed output",
    )
    available_theta_names, parameter_index = _available_theta_names(source)

    discrete_input_names = _resolve_names(
        data.discrete_input_names,
        available_theta_names,
        kind="discrete input",
    )
    continuous_input_names = _resolve_names(
        data.continuous_input_names,
        available_theta_names,
        kind="continuous input",
    )
    discrete_theta_names = _resolve_names(
        data.discrete_theta_names,
        available_theta_names,
        kind="discrete theta",
    )
    if data.continuous_theta_names is None:
        blocked = set(discrete_theta_names) | set(discrete_input_names) | set(
            continuous_input_names
        )
        continuous_theta_names = tuple(
            name for name in available_theta_names if name not in blocked
        )
    else:
        continuous_theta_names = _resolve_names(
            data.continuous_theta_names,
            available_theta_names,
            kind="continuous theta",
        )

    theta_overlap = set(continuous_theta_names) & set(discrete_theta_names)
    if theta_overlap:
        raise ValueError(
            "theta parameters cannot be both continuous and discrete: "
            f"{sorted(theta_overlap)}"
        )
    input_overlap = set(continuous_input_names) & set(discrete_input_names)
    if input_overlap:
        raise ValueError(
            "input parameters cannot be both continuous and discrete: "
            f"{sorted(input_overlap)}"
        )
    theta_names = continuous_theta_names + discrete_theta_names
    input_names = continuous_input_names + discrete_input_names
    overlap = set(theta_names) & set(input_names)
    if overlap:
        raise ValueError(
            "input parameters cannot also be inferred theta parameters: "
            f"{sorted(overlap)}"
        )

    theta_values = _parameter_matrix(source, theta_names, parameter_index)
    prior_low, prior_high = _prior_bounds(source, theta_names, parameter_index)
    (
        continuous_theta_names,
        discrete_theta_names,
        prior_low,
        prior_high,
        constant_theta,
        theta_values,
    ) = _drop_constant_theta(
        continuous_theta_names=continuous_theta_names,
        discrete_theta_names=discrete_theta_names,
        values=theta_values,
        prior_low=prior_low,
        prior_high=prior_high,
    )
    if not np.all(prior_low < prior_high):
        raise ValueError("prior_low must be strictly less than prior_high")

    input_values = _parameter_matrix(source, input_names, parameter_index)
    discrete_input_values = _categorical_values(
        input_values[:, len(continuous_input_names) :],
        discrete_input_names,
        kind="input",
    )
    discrete_theta_values = _categorical_values(
        theta_values[:, len(continuous_theta_names) :],
        discrete_theta_names,
        kind="theta",
    )
    x_components = _x_components(
        observed_outputs=observed_outputs,
        continuous_input_names=continuous_input_names,
        discrete_input_names=discrete_input_names,
        discrete_input_values=discrete_input_values,
    )
    time_metadata = _time_metadata(source, observed_outputs)
    x_event_shape = _x_event_shape(
        source,
        observed_outputs=observed_outputs,
        x_components=x_components,
        step_labels=time_metadata["step_labels"],
    )

    return InversionSchema(
        observed_outputs=observed_outputs,
        continuous_input_names=continuous_input_names,
        discrete_input_names=discrete_input_names,
        continuous_theta_names=continuous_theta_names,
        discrete_theta_names=discrete_theta_names,
        prior_low=np.asarray(prior_low, dtype=np.float32),
        prior_high=np.asarray(prior_high, dtype=np.float32),
        x_components=x_components,
        x_event_shape=x_event_shape,
        discrete_input_values=discrete_input_values,
        discrete_theta_values=discrete_theta_values,
        constant_theta=constant_theta,
        x_representation=source.config.data.x_representation,
        normalize_time=bool(source.config.data.normalize_time),
        time_axis=time_metadata["time_axis"],
        raw_step_labels=time_metadata["raw_step_labels"],
        step_labels=time_metadata["step_labels"],
        final_step_label=time_metadata["final_step_label"],
        n_actors=time_metadata["n_actors"],
    )


def parameter_values_for_indices(
    source: InversionSource,
    names: Sequence[str],
    *,
    source_positions: Sequence[int] | NDArray[np.integer[Any]] | None = None,
) -> NDArray[np.float32]:
    _, parameter_index = _available_theta_names(source)
    if source_positions is None:
        return _parameter_matrix(source, names, parameter_index)
    positions = np.asarray(source_positions, dtype=np.int64)
    input_index = np.asarray(source.input_index[positions], dtype=np.int64)
    return _parameter_matrix_for_input_index(source, names, parameter_index, input_index)


def _available_theta_names(
    source: InversionSource,
) -> tuple[tuple[str, ...], dict[str, int]]:
    raw_parameter_names = [str(name) for name in source.attrs["parameter_names"]]
    composition_axis_name = _policy_composition_axis_name(
        source.attrs.get("policy_config", {}),
        raw_parameter_names,
    )
    keep_names, keep_indices = _theta_parameter_columns(
        raw_parameter_names,
        policy_config=source.attrs.get("policy_config", {}),
        composition_axis_name=composition_axis_name,
    )
    names = ("n_agents", *tuple(keep_names))
    parameter_index = {
        name: index for name, index in zip(keep_names, keep_indices, strict=True)
    }
    return names, parameter_index


def _parameter_matrix(
    source: InversionSource,
    names: Sequence[str],
    parameter_index: Mapping[str, int],
) -> NDArray[np.float32]:
    return _parameter_matrix_for_input_index(
        source,
        names,
        parameter_index,
        np.asarray(source.input_index, dtype=np.int64),
    )


def _parameter_matrix_for_input_index(
    source: InversionSource,
    names: Sequence[str],
    parameter_index: Mapping[str, int],
    input_index: NDArray[np.int64],
) -> NDArray[np.float32]:
    names = tuple(str(name) for name in names)
    if not names:
        return np.empty((input_index.shape[0], 0), dtype=np.float32)

    input_coordinates = np.asarray(
        source.metadata_arrays["input_coordinates"],
        dtype=np.int64,
    )
    coordinates = input_coordinates[input_index]
    n_agents_values = np.asarray(source.metadata_arrays["n_agents"], dtype=np.float32)
    parameters = np.asarray(source.metadata_arrays["parameters"], dtype=np.float32)
    result = np.empty((input_index.shape[0], len(names)), dtype=np.float32)
    for output_column, name in enumerate(names):
        if name == "n_agents":
            result[:, output_column] = n_agents_values[coordinates[:, 0]]
            continue
        parameter_column = parameter_index[name]
        result[:, output_column] = parameters[
            coordinates[:, 0],
            coordinates[:, 1],
            coordinates[:, 2],
            parameter_column,
        ]
    return result


def _prior_bounds(
    source: InversionSource,
    names: Sequence[str],
    parameter_index: Mapping[str, int],
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    n_agents = np.asarray(source.metadata_arrays["n_agents"], dtype=np.float32)
    parameter_low = np.asarray(source.metadata_arrays["parameter_low"], dtype=np.float32)
    parameter_high = np.asarray(
        source.metadata_arrays["parameter_high"],
        dtype=np.float32,
    )
    low: list[float] = []
    high: list[float] = []
    for name in names:
        if name == "n_agents":
            low.append(float(np.min(n_agents)))
            high.append(float(np.max(n_agents)))
            continue
        index = parameter_index[str(name)]
        low.append(float(parameter_low[index]))
        high.append(float(parameter_high[index]))
    return np.asarray(low, dtype=np.float32), np.asarray(high, dtype=np.float32)


def _drop_constant_theta(
    *,
    continuous_theta_names: tuple[str, ...],
    discrete_theta_names: tuple[str, ...],
    values: NDArray[np.float32],
    prior_low: NDArray[np.float32],
    prior_high: NDArray[np.float32],
) -> tuple[
    tuple[str, ...],
    tuple[str, ...],
    NDArray[np.float32],
    NDArray[np.float32],
    dict[str, float],
    NDArray[np.float32],
]:
    theta_names = continuous_theta_names + discrete_theta_names
    if not theta_names:
        raise ValueError("at least one theta parameter is required")
    observed_low = np.nanmin(values, axis=0)
    observed_high = np.nanmax(values, axis=0)
    is_constant = np.isclose(observed_low, observed_high, rtol=0.0, atol=1e-12)
    constant_theta = {
        name: float(observed_low[index])
        for index, name in enumerate(theta_names)
        if bool(is_constant[index])
    }
    keep_indices = np.nonzero(~is_constant)[0]
    if keep_indices.size == 0:
        raise ValueError("all requested theta parameters are constant")

    continuous_count = len(continuous_theta_names)
    kept_continuous: list[str] = []
    kept_discrete: list[str] = []
    for index in keep_indices:
        name = theta_names[int(index)]
        if int(index) < continuous_count:
            kept_continuous.append(name)
        else:
            kept_discrete.append(name)
    return (
        tuple(kept_continuous),
        tuple(kept_discrete),
        np.asarray(prior_low[keep_indices], dtype=np.float32),
        np.asarray(prior_high[keep_indices], dtype=np.float32),
        constant_theta,
        np.asarray(values[:, keep_indices], dtype=np.float32),
    )


def _categorical_values(
    values: NDArray[np.float32],
    names: Sequence[str],
    *,
    kind: str,
) -> dict[str, tuple[int, ...]]:
    categories: dict[str, tuple[int, ...]] = {}
    for column, name in enumerate(names):
        raw = np.asarray(values[:, column], dtype=np.float32)
        if not np.isfinite(raw).all():
            raise ValueError(f"discrete {kind} parameter {name!r} must be finite")
        rounded = np.rint(raw)
        if not np.allclose(raw, rounded, rtol=0.0, atol=1e-6):
            raise ValueError(
                f"discrete {kind} parameter {name!r} contains non-integer values"
            )
        categories[str(name)] = tuple(
            int(value) for value in np.unique(rounded.astype(np.int64)).tolist()
        )
    return categories


def _x_components(
    *,
    observed_outputs: Sequence[str],
    continuous_input_names: Sequence[str],
    discrete_input_names: Sequence[str],
    discrete_input_values: Mapping[str, Sequence[int]],
) -> tuple[str, ...]:
    components = [str(name) for name in observed_outputs]
    components.extend(str(name) for name in continuous_input_names)
    for name in discrete_input_names:
        components.extend(
            f"{name}={int(value)}" for value in discrete_input_values[str(name)]
        )
    return tuple(components)


def _x_event_shape(
    source: InversionSource,
    *,
    observed_outputs: Sequence[str],
    x_components: Sequence[str],
    step_labels: NDArray[np.float64] | None,
) -> tuple[int, ...]:
    representation = source.config.data.x_representation
    if representation == "final":
        return (len(x_components),)
    if representation != "timeseries":
        raise ValueError(f"unsupported x_representation: {representation}")
    if step_labels is None:
        raise ValueError("timeseries representation requires step labels")
    return (int(step_labels.shape[0]), len(x_components))


def _time_metadata(
    source: InversionSource,
    observed_outputs: Sequence[str],
) -> dict[str, Any]:
    output_shapes = [
        tuple(int(value) for value in source.observed_output_arrays[name].shape[1:])
        for name in observed_outputs
    ]
    if any(len(shape) > 1 for shape in output_shapes):
        raise ValueError("model inversion supports scalar or one-dimensional outputs")
    step_count = max((shape[0] if shape else 1) for shape in output_shapes)
    raw_step_labels = _raw_step_labels(source, observed_outputs, step_count)
    n_actors = None
    step_labels = raw_step_labels
    time_axis = "raw_simulator_steps"
    final_step_label: int | float | None = _numeric_scalar(raw_step_labels[-1])

    if source.config.data.normalize_time:
        n_actors = _configured_n_actors(source.attrs.get("policy_config", {}))
        grid = common_normalized_time_grid(
            raw_step_labels,
            n_agents_values=np.asarray(source.metadata_arrays["n_agents"]),
            n_actors=n_actors,
        )
        step_labels = grid.step_labels
        time_axis = "activation_opportunities_per_agent"
        final_step_label = float(grid.step_labels[-1])

    if source.config.data.x_representation == "final":
        stored_step_labels = None
    else:
        stored_step_labels = np.asarray(step_labels, dtype=np.float64)

    return {
        "time_axis": time_axis,
        "raw_step_labels": np.asarray(raw_step_labels, dtype=np.float64),
        "step_labels": stored_step_labels,
        "final_step_label": final_step_label,
        "n_actors": n_actors,
    }


def _raw_step_labels(
    source: InversionSource,
    observed_outputs: Sequence[str],
    step_count: int,
) -> NDArray[np.float64]:
    policy_config = source.attrs.get("policy_config", {})
    labels_by_output = _configured_output_step_labels(policy_config)
    labels: NDArray[np.int64] | None = None
    scalar_labels: NDArray[np.int64] | None = None
    for name in observed_outputs:
        shape = tuple(int(value) for value in source.observed_output_arrays[name].shape[1:])
        output_step_count = shape[0] if shape else 1
        output_labels = labels_by_output.get(str(name))
        if output_labels is None:
            output_labels = np.arange(output_step_count, dtype=np.int64)
        if output_labels.shape != (output_step_count,):
            raise ValueError(
                f"configured step labels for {name!r} do not match output shape"
            )
        if output_step_count == 1:
            if scalar_labels is None:
                scalar_labels = output_labels
            continue
        if labels is None:
            labels = output_labels
        elif not np.array_equal(labels, output_labels):
            raise ValueError("non-scalar observed outputs must share step labels")
    if labels is None:
        labels = (
            scalar_labels
            if scalar_labels is not None
            else np.arange(step_count, dtype=np.int64)
        )
    return np.asarray(labels, dtype=np.float64)


def _resolve_names(
    requested: Sequence[str],
    available: Sequence[str],
    *,
    kind: str,
) -> tuple[str, ...]:
    available_tuple = tuple(str(name) for name in available)
    resolved: list[str] = []
    for raw_name in requested:
        name = str(raw_name).strip()
        if name in available_tuple:
            resolved.append(name)
            continue
        matches = [
            candidate for candidate in available_tuple if candidate.startswith(f"{name}.")
        ]
        if len(matches) == 1:
            resolved.append(matches[0])
            continue
        if len(matches) > 1:
            raise ValueError(f"ambiguous {kind} {name!r}; choose one of {matches}")
        raise ValueError(
            f"unknown {kind} {name!r}; available values are {list(available_tuple)}"
        )
    return tuple(resolved)


def _numeric_scalar(value: Any) -> int | float:
    scalar = np.asarray(value).item()
    if isinstance(scalar, (np.integer, int)):
        return int(scalar)
    return float(scalar)
