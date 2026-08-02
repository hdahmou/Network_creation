"""Canonical observation encoding for training and posterior use."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from network_creation.core.observability import ObservationSampling
from network_creation.policy_result_metadata import observation_step_labels
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
    n_agents_values: NDArray[np.int64]
    raw_step_horizons: NDArray[np.int64]
    n_actors: int
    output_samplings: tuple[ObservationSampling, ...]
    step_labels: NDArray[np.float64]

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
            n_agents_values=np.asarray(schema.n_agents, dtype=np.int64),
            raw_step_horizons=np.asarray(
                schema.raw_step_horizons,
                dtype=np.int64,
            ),
            n_actors=schema.n_actors,
            output_samplings=tuple(schema.output_samplings),
            step_labels=np.asarray(schema.step_labels, dtype=np.float64),
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
            n_agents_values=np.asarray(dataset.n_agents, dtype=np.int64),
            raw_step_horizons=np.asarray(
                dataset.raw_step_horizons,
                dtype=np.int64,
            ),
            n_actors=dataset.n_actors,
            output_samplings=tuple(
                ObservationSampling(
                    kind=sampling.kind,
                    every=sampling.every,
                )
                for sampling in dataset.output_samplings
            ),
            step_labels=np.asarray(dataset.step_labels, dtype=np.float64),
        )

    def encode(
        self,
        observed_outputs: Mapping[str, Any],
        input_values: Mapping[str, Any] | None = None,
        *,
        n_agents: int | float | None = None,
    ) -> NDArray[np.float32]:
        resolved_inputs = {} if input_values is None else dict(input_values)
        resolved_n_agents = _required_integer(
            n_agents if n_agents is not None else resolved_inputs.get("n_agents"),
            name="n_agents",
        )
        if "n_agents" in resolved_inputs and _required_integer(
            resolved_inputs["n_agents"],
            name="n_agents",
        ) != resolved_n_agents:
            raise ValueError("n_agents does not match the supplied input value")
        resolved_inputs.setdefault("n_agents", resolved_n_agents)

        observed = self._normalize_outputs(
            observed_outputs,
            n_agents=resolved_n_agents,
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

    def _normalize_outputs(
        self,
        outputs: Mapping[str, Any],
        *,
        n_agents: int,
    ) -> NDArray[np.float32]:
        matches = np.flatnonzero(self.n_agents_values == n_agents)
        if matches.shape[0] != 1:
            raise ValueError(f"unknown n_agents value: {n_agents}")
        population_index = int(matches[0])
        raw_step_horizon = int(self.raw_step_horizons[population_index])
        storage_raw_step_horizon = int(np.max(self.raw_step_horizons))

        normalized_outputs: list[NDArray[np.float32]] = []
        for name, sampling in zip(
            self.observed_outputs,
            self.output_samplings,
            strict=True,
        ):
            values = _output_series(outputs[name], name=name)
            raw_labels = observation_step_labels(raw_step_horizon, sampling)
            valid_count = int(raw_labels.shape[0])
            storage_count = int(
                observation_step_labels(
                    storage_raw_step_horizon,
                    sampling,
                ).shape[0]
            )
            sample_count = int(values.shape[0])
            if sample_count not in {valid_count, storage_count}:
                expected = (
                    f"exactly {valid_count} sample"
                    f"{'s' if valid_count != 1 else ''}"
                    if valid_count == storage_count
                    else (
                        f"either {valid_count} native samples or "
                        f"{storage_count} fixed-storage samples"
                    )
                )
                raise ValueError(
                    f"observed output {name!r} must contain {expected} for "
                    f"n_agents={n_agents}; got {sample_count}"
                )
            valid_values = np.asarray(values[:valid_count], dtype=np.float32)
            if not np.isfinite(valid_values).all():
                raise ValueError(
                    f"observed output {name!r} has non-finite values inside its "
                    "structurally valid prefix"
                )
            normalized_outputs.append(
                normalize_time_tensor(
                    valid_values,
                    raw_step_labels=raw_labels,
                    n_agents=n_agents,
                    n_actors=self.n_actors,
                    step_axis=0,
                    target_times=self.step_labels,
                ).values
            )
        return np.stack(normalized_outputs, axis=1).astype(np.float32, copy=False)

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
