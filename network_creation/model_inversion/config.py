"""Strict configuration contract for zarr-backed inversion training."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import Field, Strict, ValidationInfo, field_validator, model_validator

from .contracts import (
    Name,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    StrictModel,
    UnitFraction,
)
from .networks import DensityEstimatorConfig, EmbeddingConfig

ObservationRepresentation = Literal["final", "timeseries"]
ConfigPath = Annotated[Path, Strict(strict=False)]


class InversionDataConfig(StrictModel):
    zarr_path: ConfigPath
    observed_outputs: tuple[Name, ...] = Field(min_length=1)
    continuous_theta_names: tuple[Name, ...] | None = None
    discrete_theta_names: tuple[Name, ...] = ()
    continuous_input_names: tuple[Name, ...] = ()
    discrete_input_names: tuple[Name, ...] = ()
    x_representation: ObservationRepresentation = "final"

    @field_validator("zarr_path", mode="before")
    @classmethod
    def resolve_zarr_path(cls, value: Any, info: ValidationInfo) -> Any:
        return _resolve_config_path(value, info)

    @field_validator(
        "observed_outputs",
        "continuous_theta_names",
        "discrete_theta_names",
        "continuous_input_names",
        "discrete_input_names",
    )
    @classmethod
    def require_unique_names(
        cls,
        names: tuple[str, ...] | None,
    ) -> tuple[str, ...] | None:
        if names is not None and len(names) != len(set(names)):
            raise ValueError("parameter names must not contain duplicates")
        return names


class InversionModelConfig(StrictModel):
    density_estimator: DensityEstimatorConfig = Field(
        default_factory=DensityEstimatorConfig
    )
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)


class InversionTrainingConfig(StrictModel):
    seed: int = 0
    device: Name = "cpu"
    batch_size: PositiveInt = 200
    learning_rate: NonNegativeFloat = 5e-4
    validation_fraction: UnitFraction = Field(
        default=0.1,
        description=(
            "Fraction of non-diagnostic parameter samples assigned to validation"
        ),
    )
    stop_after_epochs: PositiveInt = 20
    max_num_epochs: PositiveInt = 500
    clip_max_norm: PositiveFloat | None = 5.0
    show_progress_bars: bool = True
    num_workers: NonNegativeInt = 0
    pin_memory: bool | None = None
    persistent_workers: bool = False
    materialize_dataset: bool = False
    calibration_sample_count: PositiveInt = 200

    @model_validator(mode="after")
    def validate_worker_settings(self) -> "InversionTrainingConfig":
        if self.persistent_workers and self.num_workers == 0:
            raise ValueError("persistent_workers requires num_workers > 0")
        return self


class InversionOutputConfig(StrictModel):
    output_root: ConfigPath
    artifact_name: Name | None = None
    save_training_data: bool = False
    prior_sample_count: PositiveInt = 4096
    training_data_sample_count: PositiveInt = 4096

    @field_validator("output_root", mode="before")
    @classmethod
    def resolve_output_root(cls, value: Any, info: ValidationInfo) -> Any:
        return _resolve_config_path(value, info)

    @field_validator("artifact_name")
    @classmethod
    def require_directory_name(cls, value: str | None) -> str | None:
        if value is not None and Path(value).name != value:
            raise ValueError("artifact_name must be a directory name")
        return value


class InversionConfig(StrictModel):
    data: InversionDataConfig
    model: InversionModelConfig = Field(default_factory=InversionModelConfig)
    training: InversionTrainingConfig = Field(default_factory=InversionTrainingConfig)
    output: InversionOutputConfig
    diagnostic_holdout_count_per_group: PositiveInt = Field(
        description=(
            "Number of parameter samples held out per categorical stratum; "
            "n_agents is not part of the stratum"
        )
    )

    @model_validator(mode="after")
    def validate_embedding_representation(self) -> "InversionConfig":
        expected_embedding = (
            "rnn" if self.data.x_representation == "timeseries" else "identity"
        )
        if self.model.embedding.kind != expected_embedding:
            raise ValueError(
                f"{self.data.x_representation} x requires a "
                f"{expected_embedding!r} embedding"
            )
        return self


def load_inversion_config(config_path: str | Path) -> InversionConfig:
    path = Path(config_path).expanduser().resolve()
    return InversionConfig.model_validate_json(
        path.read_text(encoding="utf-8"),
        context={"config_dir": path.parent},
    )


def _resolve_config_path(value: Any, info: ValidationInfo) -> str:
    if not isinstance(value, (str, Path)):
        raise TypeError("config paths must be strings")
    path = Path(value).expanduser()
    if path.is_absolute():
        return str(path)
    context = info.context if isinstance(info.context, dict) else {}
    config_dir = context.get("config_dir")
    if not isinstance(config_dir, Path):
        raise ValueError("config path resolution requires config_dir context")
    return str((config_dir / path).resolve())
