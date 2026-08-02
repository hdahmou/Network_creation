"""Artifact writing for trained model-inversion posteriors."""

from __future__ import annotations

import pickle
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from network_creation.io import write_json

from .artifact_metadata import (
    ArtifactDatasetMetadata,
    ArtifactMetadata,
    ArtifactOutputSampling,
    ArtifactPriorSamplesMetadata,
    ArtifactSbiMetadata,
)
from .config import InversionConfig
from .dataset import InversionRowIndex, ZarrInversionDataset
from .schema import InversionSchema
from .source import InversionSource
from .splits import InversionSplit, PriorSamples
from .training import EstimatorTrainingResult

SBI_LOG_DIRNAME = "sbi_logs"
POSTERIOR_FILENAME = "posterior.pkl"
METADATA_FILENAME = "metadata.json"
TRAINING_METRICS_FILENAME = "training_metrics.json"
DIAGNOSTIC_DATA_FILENAME = "diagnostic_data.npz"
TRAINING_DATA_FILENAME = "training_data.npz"
PRIOR_SAMPLES_FILENAME = "prior_samples.npz"


@dataclass(frozen=True)
class TrainingRunSummary:
    artifact_dir: str
    posterior_file: str
    metadata_file: str
    tensorboard_dir: str
    training_metrics_file: str
    diagnostic_data_file: str
    prior_samples_file: str
    prior_sample_count: int
    best_validation_loss: float
    final_validation_loss: float | None
    best_validation_log_prob: float
    final_validation_log_prob: float | None
    sbi_method: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def prepare_artifact_dir(config: InversionConfig, schema: InversionSchema) -> Path:
    artifact_name = config.output.artifact_name
    if artifact_name is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        artifact_name = f"{schema.sbi_method.lower()}-{stamp}"
    artifact_dir = config.output.output_root / artifact_name
    artifact_dir.mkdir(parents=True, exist_ok=False)
    return artifact_dir


def write_inversion_artifact(
    *,
    posterior: Any,
    source: InversionSource,
    schema: InversionSchema,
    row_index: InversionRowIndex,
    split: InversionSplit,
    prior_samples: PriorSamples,
    training: EstimatorTrainingResult,
    config: InversionConfig,
    dataset: ZarrInversionDataset,
    artifact_dir: Path | None = None,
) -> TrainingRunSummary:
    artifact_dir = (
        prepare_artifact_dir(config, schema)
        if artifact_dir is None
        else artifact_dir
    )
    if not artifact_dir.is_dir():
        raise FileNotFoundError(f"artifact directory does not exist: {artifact_dir}")
    for filename in (
        POSTERIOR_FILENAME,
        METADATA_FILENAME,
        TRAINING_METRICS_FILENAME,
        DIAGNOSTIC_DATA_FILENAME,
        TRAINING_DATA_FILENAME,
        PRIOR_SAMPLES_FILENAME,
    ):
        if (artifact_dir / filename).exists():
            raise FileExistsError(f"artifact file already exists: {artifact_dir / filename}")
    tensorboard_dir = artifact_dir / SBI_LOG_DIRNAME
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    posterior_path = artifact_dir / POSTERIOR_FILENAME
    metadata_path = artifact_dir / METADATA_FILENAME
    metrics_path = artifact_dir / TRAINING_METRICS_FILENAME
    diagnostic_path = artifact_dir / DIAGNOSTIC_DATA_FILENAME
    prior_samples_path = artifact_dir / PRIOR_SAMPLES_FILENAME

    metadata = _metadata(
        source=source,
        schema=schema,
        row_index=row_index,
        split=split,
        prior_samples=prior_samples,
        config=config,
    )
    move_to_cpu = getattr(posterior, "to", None)
    if callable(move_to_cpu):
        moved = move_to_cpu("cpu")
        if moved is not None:
            posterior = moved
    write_json(metrics_path, training.to_metrics())
    with posterior_path.open("wb") as handle:
        pickle.dump(posterior, handle)
    np.savez_compressed(
        prior_samples_path,
        theta=np.asarray(prior_samples.theta, dtype=np.float32),
        source_sample_ids=np.asarray(prior_samples.source_indices, dtype=np.int64),
        continuous_theta_names=np.asarray(schema.continuous_theta_names),
        discrete_theta_names=np.asarray(schema.discrete_theta_names),
    )
    _write_npz(diagnostic_path, dataset=dataset, indices=split.diagnostic_indices)
    if config.output.save_training_data:
        training_indices = _bounded_indices(
            split.train_indices,
            max_count=config.output.training_data_sample_count,
            seed=config.training.seed,
        )
        _write_npz(
            artifact_dir / TRAINING_DATA_FILENAME,
            dataset=dataset,
            indices=training_indices,
        )
    write_json(metadata_path, metadata.model_dump(mode="json"))

    return TrainingRunSummary(
        artifact_dir=str(artifact_dir),
        posterior_file=str(posterior_path),
        metadata_file=str(metadata_path),
        tensorboard_dir=str(tensorboard_dir),
        training_metrics_file=str(metrics_path),
        diagnostic_data_file=str(diagnostic_path),
        prior_samples_file=str(prior_samples_path),
        prior_sample_count=int(prior_samples.theta.shape[0]),
        best_validation_loss=float(training.best_validation_loss),
        final_validation_loss=training.final_validation_loss,
        best_validation_log_prob=float(training.best_validation_log_prob),
        final_validation_log_prob=training.final_validation_log_prob,
        sbi_method=schema.sbi_method,
    )


def _metadata(
    *,
    source: InversionSource,
    schema: InversionSchema,
    row_index: InversionRowIndex,
    split: InversionSplit,
    prior_samples: PriorSamples,
    config: InversionConfig,
) -> ArtifactMetadata:
    return ArtifactMetadata(
        created_at=datetime.now(timezone.utc),
        dataset=ArtifactDatasetMetadata(
            result_store_path=str(source.zarr_path),
            training_config_path=str(source.config_path),
            simulation_config_path=str(source.attrs["config_path"]),
            observed_outputs=tuple(schema.observed_outputs),
            x_components=tuple(schema.x_components),
            x_event_shape=tuple(schema.x_event_shape),
            x_representation=schema.x_representation,
            time_axis=schema.time_axis,
            n_agents=tuple(int(value) for value in schema.n_agents),
            raw_step_horizons=tuple(
                int(value) for value in schema.raw_step_horizons
            ),
            n_actors=schema.n_actors,
            output_samplings=tuple(
                ArtifactOutputSampling(
                    kind=sampling.kind,
                    every=sampling.every,
                )
                for sampling in schema.output_samplings
            ),
            step_labels=tuple(float(value) for value in schema.step_labels),
            continuous_theta_names=tuple(schema.continuous_theta_names),
            discrete_theta_names=tuple(schema.discrete_theta_names),
            continuous_input_names=tuple(schema.continuous_input_names),
            discrete_input_names=tuple(schema.discrete_input_names),
            discrete_input_values=tuple(
                tuple(schema.discrete_input_values[name])
                for name in schema.discrete_input_names
            ),
            discrete_theta_values=tuple(
                tuple(schema.discrete_theta_values[name])
                for name in schema.discrete_theta_names
            ),
            prior_low=tuple(float(value) for value in schema.prior_low),
            prior_high=tuple(float(value) for value in schema.prior_high),
            constant_theta={
                str(name): float(value) for name, value in schema.constant_theta.items()
            },
            train_count=int(split.train_indices.shape[0]),
            diagnostic_count=int(split.diagnostic_indices.shape[0]),
            n_samples=int(row_index.row_count),
        ),
        sbi=ArtifactSbiMetadata(
            method=schema.sbi_method,
            amortized=True,
            mixed_parameters=bool(schema.discrete_theta_names),
            posterior_file=POSTERIOR_FILENAME,
            training_metrics_file=TRAINING_METRICS_FILENAME,
            diagnostic_data_file=DIAGNOSTIC_DATA_FILENAME,
            prior_samples_file=PRIOR_SAMPLES_FILENAME,
            tensorboard_dir=SBI_LOG_DIRNAME,
            prior_samples=ArtifactPriorSamplesMetadata(
                source="training_theta_rows",
                count=int(prior_samples.theta.shape[0]),
                selection_seed=int(prior_samples.seed),
            ),
        ),
        training=config.training,
        model=config.model,
    )


def _write_npz(
    path: Path,
    *,
    dataset: ZarrInversionDataset,
    indices: NDArray[Any],
) -> None:
    theta_rows: list[NDArray[Any]] = []
    x_rows: list[NDArray[Any]] = []
    for index in indices:
        theta, x = dataset.item_arrays(int(index))
        theta_rows.append(theta)
        x_rows.append(x)
    data: dict[str, Any] = {
        "theta": np.stack(theta_rows, axis=0).astype(np.float32),
        "x": np.stack(x_rows, axis=0).astype(np.float32),
        "source_sample_ids": dataset.source_sample_ids_for_indices(indices),
        "continuous_theta_names": np.asarray(dataset.schema.continuous_theta_names),
        "discrete_theta_names": np.asarray(dataset.schema.discrete_theta_names),
        "x_components": np.asarray(dataset.schema.x_components),
    }
    np.savez_compressed(path, **data)


def _bounded_indices(
    indices: NDArray[Any],
    *,
    max_count: int,
    seed: int,
) -> NDArray[np.int64]:
    values = np.asarray(indices, dtype=np.int64)
    if values.shape[0] <= max_count:
        return values
    rng = np.random.default_rng(seed)
    return np.sort(
        rng.choice(values, size=max_count, replace=False).astype(np.int64)
    )
