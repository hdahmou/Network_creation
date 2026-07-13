"""Typed contract for persisted model-inversion artifact metadata."""

from __future__ import annotations

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


class ArtifactDatasetMetadata(StrictModel):
    result_store_path: Name
    training_config_path: Name
    simulation_config_path: Name
    observed_outputs: tuple[Name, ...] = Field(min_length=1)
    x_components: tuple[Name, ...] = Field(min_length=1)
    x_event_shape: tuple[PositiveInt, ...]
    x_representation: ObservationRepresentation
    normalize_time: bool
    time_axis: Name | None
    raw_step_labels: StepLabels | None
    step_labels: StepLabels | None
    final_step_label: Numeric | None
    n_actors: PositiveInt | None
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
        if self.x_representation == "final" and self.step_labels is not None:
            raise ValueError("final x metadata must not contain step_labels")
        if self.x_representation == "timeseries" and (
            self.step_labels is None
            or len(self.step_labels) != self.x_event_shape[0]
        ):
            raise ValueError("timeseries step_labels must align with the x step dimension")
        if self.normalize_time and (
            self.raw_step_labels is None
            or self.n_actors is None
            or (
                self.x_representation == "final"
                and self.final_step_label is None
            )
        ):
            raise ValueError(
                "normalized x metadata requires raw_step_labels, n_actors, and target labels"
            )
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
