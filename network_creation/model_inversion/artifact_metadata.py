"""Typed contract for persisted model-inversion artifact metadata."""

from __future__ import annotations

import math
from datetime import datetime
from typing import Annotated, Literal

from pydantic import Field, model_validator

from .config import (
    InversionModelConfig,
    InversionTrainingConfig,
    ObservationRepresentation,
)
from .contracts import Name, NonNegativeInt, PositiveInt, StrictModel

Numeric = int | float
CategoryValues = Annotated[tuple[int, ...], Field(min_length=1)]
StepLabels = Annotated[tuple[Numeric, ...], Field(min_length=1)]
PopulationSizes = Annotated[tuple[PositiveInt, ...], Field(min_length=1)]
RawStepHorizons = Annotated[tuple[NonNegativeInt, ...], Field(min_length=1)]


class ArtifactOutputSampling(StrictModel):
    kind: Literal["periodic", "terminal"]
    every: PositiveInt | None = None

    @model_validator(mode="after")
    def validate_sampling(self) -> "ArtifactOutputSampling":
        if self.kind == "periodic" and self.every is None:
            raise ValueError("periodic output sampling requires every")
        if self.kind == "terminal" and self.every is not None:
            raise ValueError("terminal output sampling does not accept every")
        return self


class ArtifactDatasetMetadata(StrictModel):
    result_store_path: Name
    training_config_path: Name
    simulation_config_path: Name
    observed_outputs: tuple[Name, ...] = Field(min_length=1)
    x_components: tuple[Name, ...] = Field(min_length=1)
    x_event_shape: tuple[PositiveInt, ...]
    x_representation: ObservationRepresentation
    time_axis: Literal["activation_opportunities_per_agent"]
    n_agents: PopulationSizes
    raw_step_horizons: RawStepHorizons
    n_actors: PositiveInt
    output_samplings: tuple[ArtifactOutputSampling, ...] = Field(min_length=1)
    step_labels: StepLabels
    continuous_theta_names: tuple[Name, ...]
    discrete_theta_names: tuple[Name, ...]
    continuous_input_names: tuple[Name, ...]
    discrete_input_names: tuple[Name, ...]
    discrete_input_values: tuple[CategoryValues, ...]
    discrete_theta_values: tuple[CategoryValues, ...]
    prior_low: tuple[float, ...]
    prior_high: tuple[float, ...]
    constant_theta: dict[Name, float]
    train_count: NonNegativeInt
    diagnostic_count: NonNegativeInt
    n_samples: PositiveInt

    @model_validator(mode="after")
    def validate_positional_metadata(self) -> "ArtifactDatasetMetadata":
        for label, names in (
            ("observed_outputs", self.observed_outputs),
            ("theta names", self.theta_order),
            (
                "input names",
                self.continuous_input_names + self.discrete_input_names,
            ),
        ):
            if len(names) != len(set(names)):
                raise ValueError(f"{label} must not contain duplicates")
        if len(self.discrete_input_names) != len(self.discrete_input_values):
            raise ValueError(
                "discrete_input_values must align with discrete_input_names"
            )
        if len(self.discrete_theta_names) != len(self.discrete_theta_values):
            raise ValueError(
                "discrete_theta_values must align with discrete_theta_names"
            )
        theta_count = len(self.theta_order)
        if theta_count == 0:
            raise ValueError("artifact metadata must contain at least one theta name")
        if len(self.prior_low) != theta_count or len(self.prior_high) != theta_count:
            raise ValueError("prior bounds must align with theta names")
        expected_rank = 1 if self.x_representation == "final" else 2
        if len(self.x_event_shape) != expected_rank:
            raise ValueError(
                f"{self.x_representation} x requires a rank-{expected_rank} event shape"
            )
        expected_components = [
            *self.observed_outputs,
            *self.continuous_input_names,
        ]
        for name, values in zip(
            self.discrete_input_names,
            self.discrete_input_values,
            strict=True,
        ):
            expected_components.extend(f"{name}={value}" for value in values)
        if self.x_components != tuple(expected_components):
            raise ValueError("x_components do not match the stored observation schema")
        if self.x_event_shape[-1] != len(expected_components):
            raise ValueError("x_event_shape does not match the observation feature count")
        if len(self.n_agents) != len(self.raw_step_horizons):
            raise ValueError("raw_step_horizons must align with n_agents")
        if len(self.n_agents) != len(set(self.n_agents)):
            raise ValueError("n_agents must contain unique population sizes")
        if len(self.output_samplings) != len(self.observed_outputs):
            raise ValueError("output_samplings must align with observed_outputs")
        if any(
            sampling != self.output_samplings[0]
            for sampling in self.output_samplings[1:]
        ):
            raise ValueError("observed outputs must have compatible sampling")
        if self.x_representation == "final" and len(self.step_labels) != 1:
            raise ValueError("final step_labels must contain one target value")
        if self.x_representation == "timeseries" and len(
            self.step_labels
        ) != self.x_event_shape[0]:
            raise ValueError("timeseries step_labels must align with the x step dimension")
        if not all(math.isfinite(float(value)) for value in self.step_labels):
            raise ValueError("step_labels must contain only finite values")
        if any(
            float(right) <= float(left)
            for left, right in zip(self.step_labels, self.step_labels[1:])
        ):
            raise ValueError("step_labels must be strictly increasing")
        for name, values in (
            *zip(self.discrete_input_names, self.discrete_input_values, strict=True),
            *zip(self.discrete_theta_names, self.discrete_theta_values, strict=True),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"categorical values for {name!r} must be unique")
        return self

    @property
    def theta_order(self) -> tuple[str, ...]:
        return self.continuous_theta_names + self.discrete_theta_names

    @property
    def discrete_input_values_by_name(self) -> dict[str, tuple[int, ...]]:
        return dict(
            zip(self.discrete_input_names, self.discrete_input_values, strict=True)
        )

    @property
    def discrete_theta_values_by_name(self) -> dict[str, tuple[int, ...]]:
        return dict(
            zip(self.discrete_theta_names, self.discrete_theta_values, strict=True)
        )


class ArtifactPriorSamplesMetadata(StrictModel):
    source: Literal["training_theta_rows"]
    count: NonNegativeInt
    selection_seed: int


class ArtifactSbiMetadata(StrictModel):
    method: Literal["NPE", "MNPE"]
    amortized: Literal[True]
    mixed_parameters: bool
    posterior_file: Name
    training_metrics_file: Name
    diagnostic_data_file: Name
    prior_samples_file: Name
    tensorboard_dir: Name
    prior_samples: ArtifactPriorSamplesMetadata


class ArtifactMetadata(StrictModel):
    created_at: datetime
    dataset: ArtifactDatasetMetadata
    sbi: ArtifactSbiMetadata
    training: InversionTrainingConfig
    model: InversionModelConfig

    @model_validator(mode="after")
    def validate_sbi_method(self) -> "ArtifactMetadata":
        mixed_parameters = bool(self.dataset.discrete_theta_names)
        expected_method = "MNPE" if mixed_parameters else "NPE"
        if self.sbi.method != expected_method:
            raise ValueError(
                f"sbi.method must be {expected_method!r} for the stored theta schema"
            )
        if self.sbi.mixed_parameters != mixed_parameters:
            raise ValueError(
                "sbi.mixed_parameters must match the stored discrete theta schema"
            )
        return self
