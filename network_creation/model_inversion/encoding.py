"""Canonical observation encoding for training and posterior use."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from network_creation.time_normalization import normalize_time_tensor

from .artifact_metadata import ArtifactDatasetMetadata
from .schema import InversionSchema


@dataclass(frozen=True)
class ObservationEncoder:
    observed_outputs: tuple[str, ...]
    continuous_input_names: tuple[str, ...]
    discrete_input_names: tuple[str, ...]
    discrete_input_values: dict[str, tuple[int, ...]]
    x_components: tuple[str, ...]
    x_event_shape: tuple[int, ...]
    x_representation: str
    normalize_time: bool
    raw_step_labels: NDArray[np.float64] | None
    step_labels: NDArray[np.float64] | None
    final_step_label: int | float | None
    n_actors: int | None

    @classmethod
    def from_schema(cls, schema: InversionSchema) -> "ObservationEncoder":
        return cls(
            observed_outputs=tuple(schema.observed_outputs),
            continuous_input_names=tuple(schema.continuous_input_names),
            discrete_input_names=tuple(schema.discrete_input_names),
            discrete_input_values={
                str(name): tuple(int(value) for value in values)
                for name, values in schema.discrete_input_values.items()
            },
            x_components=tuple(schema.x_components),
            x_event_shape=tuple(schema.x_event_shape),
            x_representation=schema.x_representation,
            normalize_time=schema.normalize_time,
            raw_step_labels=_optional_float_array(schema.raw_step_labels),
            step_labels=_optional_float_array(schema.step_labels),
            final_step_label=schema.final_step_label,
            n_actors=schema.n_actors,
        )

    @classmethod
    def from_metadata(cls, dataset: ArtifactDatasetMetadata) -> "ObservationEncoder":
        return cls(
            observed_outputs=dataset.observed_outputs,
            continuous_input_names=dataset.continuous_input_names,
            discrete_input_names=dataset.discrete_input_names,
            discrete_input_values=dataset.discrete_input_values_by_name,
            x_components=dataset.x_components,
            x_event_shape=dataset.x_event_shape,
            x_representation=dataset.x_representation,
            normalize_time=dataset.normalize_time,
            raw_step_labels=_optional_float_array(dataset.raw_step_labels),
            step_labels=_optional_float_array(dataset.step_labels),
            final_step_label=dataset.final_step_label,
            n_actors=dataset.n_actors,
        )

    def encode(
        self,
        observed_outputs: Mapping[str, Any],
        input_values: Mapping[str, Any] | None = None,
        *,
        n_agents: int | float | None = None,
        n_actors: int | None = None,
    ) -> NDArray[np.float32]:
        resolved_inputs = {} if input_values is None else dict(input_values)
        if n_agents is not None and "n_agents" not in resolved_inputs:
            resolved_inputs["n_agents"] = n_agents

        observed = _stack_observed_outputs(observed_outputs, self.observed_outputs)
        if self.normalize_time:
            observed = self._normalize(
                observed,
                n_agents=_required_integer(
                    n_agents if n_agents is not None else resolved_inputs.get("n_agents"),
                    name="n_agents",
                ),
                n_actors=_required_integer(
                    n_actors if n_actors is not None else self.n_actors,
                    name="n_actors",
                ),
            )

        values = observed[-1] if self.x_representation == "final" else observed
        encoded_inputs = self._encode_inputs(resolved_inputs)
        if encoded_inputs.size:
            if values.ndim == 1:
                values = np.concatenate((values, encoded_inputs), axis=0)
            else:
                repeated = np.repeat(encoded_inputs[None, :], values.shape[0], axis=0)
                values = np.concatenate((values, repeated), axis=1)

        result = np.asarray(values, dtype=np.float32)
        if tuple(result.shape) != self.x_event_shape:
            raise ValueError(
                f"prepared x shape {tuple(result.shape)} does not match "
                f"x_event_shape {self.x_event_shape}"
            )
        return result

    def _normalize(
        self,
        values: NDArray[np.float32],
        *,
        n_agents: int,
        n_actors: int,
    ) -> NDArray[np.float32]:
        if self.raw_step_labels is None:
            raise ValueError("normalized x requires raw_step_labels")
        if self.x_representation == "final":
            if self.final_step_label is None:
                raise ValueError("normalized final x requires final_step_label")
            target_times = np.asarray([self.final_step_label], dtype=np.float64)
        else:
            if self.step_labels is None:
                raise ValueError("normalized timeseries x requires step_labels")
            target_times = self.step_labels
        return normalize_time_tensor(
            values,
            raw_step_labels=self.raw_step_labels,
            n_agents=n_agents,
            n_actors=n_actors,
            step_axis=0,
            target_times=target_times,
        ).values

    def _encode_inputs(
        self,
        input_values: Mapping[str, Any],
    ) -> NDArray[np.float32]:
        encoded: list[float] = []
        for name in self.continuous_input_names:
            if name not in input_values:
                raise ValueError(f"missing input value: {name}")
            value = float(input_values[name])
            if not np.isfinite(value):
                raise ValueError(f"input value {name!r} must be finite")
            encoded.append(value)
        for name in self.discrete_input_names:
            if name not in input_values:
                raise ValueError(f"missing input value: {name}")
            value = _required_integer(input_values[name], name=name)
            categories = self.discrete_input_values[name]
            if value not in categories:
                raise ValueError(
                    f"unknown categorical input value {value!r} for {name!r}; "
                    f"known values are {list(categories)}"
                )
            encoded.extend(1.0 if value == category else 0.0 for category in categories)
        return np.asarray(encoded, dtype=np.float32)


def _stack_observed_outputs(
    outputs: Mapping[str, Any],
    observed_order: Sequence[str],
) -> NDArray[np.float32]:
    arrays = [_output_series(outputs[name], name=name) for name in observed_order]
    step_count = max(int(array.shape[0]) for array in arrays)
    expanded: list[NDArray[np.float32]] = []
    for name, array in zip(observed_order, arrays, strict=True):
        if array.shape[0] == step_count:
            expanded.append(array)
        elif array.shape[0] == 1:
            expanded.append(np.repeat(array, step_count))
        else:
            raise ValueError(
                f"observed output {name!r} has {array.shape[0]} steps; "
                f"expected 1 or {step_count}"
            )
    return np.stack(expanded, axis=1).astype(np.float32)


def _output_series(values: Any, *, name: str) -> NDArray[np.float32]:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim == 0:
        return array.reshape(1)
    if array.ndim == 1:
        return array
    raise ValueError(f"observed output {name!r} must be scalar or one-dimensional")


def _required_integer(value: Any, *, name: str) -> int:
    if value is None:
        raise ValueError(f"prepared x requires {name}")
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 0:
        raise ValueError(f"{name!r} must be scalar")
    scalar = float(array.item())
    if not np.isfinite(scalar):
        raise ValueError(f"{name!r} must be finite")
    rounded = round(scalar)
    if not np.isclose(scalar, rounded, rtol=0.0, atol=1e-6):
        raise ValueError(f"{name!r} must be integer-valued")
    return int(rounded)


def _optional_float_array(values: Any) -> NDArray[np.float64] | None:
    if values is None:
        return None
    return np.asarray(values, dtype=np.float64)
