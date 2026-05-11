#!/usr/bin/env python3
"""Train and save an amortized SBI NPE/MNPE posterior from a JSON config.

Example:
    uv run python scripts/train_npe_from_zarr.py \
      configs/model_inversion.baseline.train_npe_config.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray


LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SBI_LOG_DIRNAME = "sbi_logs"
DIAGNOSTIC_DATA_FILENAME = "diagnostic_data.npz"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from network_creation.io import read_json, write_json  # noqa: E402
from network_creation.model_inversion.data import (  # noqa: E402
    ObservationRepresentation,
    filter_inversion_dataset,
    inversion_dataset_metadata,
    move_discrete_theta_to_end,
)
from network_creation.model_inversion.networks import (  # noqa: E402
    ContinuousDensityEstimatorName,
    DensityEstimatorConfig,
    DensityEstimatorName,
    EmbeddingConfig,
    EmbeddingKind,
    RNNKind,
)
from network_creation.model_inversion.posterior import (  # noqa: E402
    attach_prior_samples_to_posterior,
    select_prior_samples,
)
from network_creation.model_inversion.training import (  # noqa: E402
    NPETrainingConfig,
    train_amortized_npe,
)


def load_policy_results(zarr_path: str | Path):
    from tasks.policies import load_policy_results as _load_policy_results

    return _load_policy_results(zarr_path)


@dataclass(frozen=True)
class DiagnosticsConfig:
    reserve_count: int

    def to_metadata(self) -> dict[str, int]:
        return {"reserve_count": int(self.reserve_count)}


@dataclass(frozen=True)
class TrainNPEConfig:
    zarr_path: Path
    observers: tuple[str, ...]
    theta_parameters: tuple[str, ...] | None
    drop_constant_theta: bool
    x_representation: ObservationRepresentation
    output_root: Path
    run_name: str | None
    save_training_data: bool
    density: DensityEstimatorConfig
    embedding: EmbeddingConfig
    training: NPETrainingConfig
    diagnostics: DiagnosticsConfig
    discrete_theta_parameters: tuple[str, ...] | None = None
    prior_sample_count: int = 4096


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train one amortized sbi NPE/MNPE posterior from a JSON config."
    )
    parser.add_argument(
        "config_path",
        type=Path,
        help="Path to the model-inversion training config JSON file.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Python logging level for progress messages. Defaults to INFO.",
    )
    return parser.parse_args(argv)


def load_train_config(config_path: str | Path) -> TrainNPEConfig:
    path = Path(config_path).expanduser().resolve()
    data = read_json(path)

    config_dir = path.parent
    dataset = data["dataset"]
    artifact = data["artifact"]
    density = data["density_estimator"]
    embedding = data["embedding"]
    training = data["training"]
    diagnostics = data.get("diagnostics")
    if not isinstance(diagnostics, dict) or "reserve_count" not in diagnostics:
        raise ValueError("diagnostics.reserve_count is required")

    observers = tuple(str(observer) for observer in dataset["observers"])
    theta_parameters = _optional_str_tuple(dataset.get("theta_parameters"))
    discrete_theta_parameters = _optional_str_tuple(
        dataset.get("discrete_theta_parameters")
    )
    x_representation = cast(
        ObservationRepresentation,
        dataset.get("x_representation", "final"),
    )

    return TrainNPEConfig(
        zarr_path=_resolve_config_path(config_dir, dataset["zarr_path"]),
        observers=observers,
        theta_parameters=theta_parameters,
        discrete_theta_parameters=discrete_theta_parameters,
        drop_constant_theta=dataset.get("drop_constant_theta", True),
        x_representation=x_representation,
        output_root=_resolve_config_path(
            config_dir,
            artifact.get("output_root", "artifacts/model_inversion"),
        ),
        run_name=_optional_str(artifact.get("run_name")),
        save_training_data=artifact.get("save_training_data", False),
        prior_sample_count=int(artifact.get("prior_sample_count", 4096)),
        density=DensityEstimatorConfig(
            model=cast(DensityEstimatorName, density.get("model", "nsf")),
            flow_model=cast(
                ContinuousDensityEstimatorName,
                density.get("flow_model", "nsf"),
            ),
            hidden_features=density.get("hidden_features", 50),
            num_transforms=density.get("num_transforms", 5),
            num_bins=density.get("num_bins", 10),
            discrete_hidden_features=_optional_int(
                density.get("discrete_hidden_features")
            ),
            discrete_hidden_layers=density.get("discrete_hidden_layers", 2),
            continuous_hidden_features=_optional_int(
                density.get("continuous_hidden_features")
            ),
            z_score_theta=density.get("z_score_theta", "independent"),
            z_score_x=density.get("z_score_x", "independent"),
        ),
        embedding=EmbeddingConfig(
            kind=cast(EmbeddingKind, embedding.get("kind", "identity")),
            output_dim=embedding.get("output_dim", 32),
            rnn_hidden_size=embedding.get("rnn_hidden_size", 64),
            rnn_num_layers=embedding.get("rnn_num_layers", 1),
            rnn_type=cast(RNNKind, embedding.get("rnn_type", "gru")),
            rnn_bidirectional=embedding.get("rnn_bidirectional", False),
        ),
        training=NPETrainingConfig(
            seed=training.get("seed", 0),
            device=training.get("device", "cpu"),
            training_batch_size=training.get("training_batch_size", 200),
            learning_rate=training.get("learning_rate", 5e-4),
            validation_fraction=training.get("validation_fraction", 0.1),
            stop_after_epochs=training.get("stop_after_epochs", 20),
            max_num_epochs=training.get("max_num_epochs", 500),
            clip_max_norm=_optional_float(training.get("clip_max_norm", 5.0)),
            show_progress_bars=training.get("show_progress_bars", True),
        ),
        diagnostics=DiagnosticsConfig(
            reserve_count=_positive_int(
                diagnostics["reserve_count"],
                field_name="diagnostics.reserve_count",
            ),
        ),
    )


def train_from_config(config: TrainNPEConfig) -> dict[str, Any]:
    np.random.seed(config.training.seed)

    LOGGER.info(
        "[stage 2/10] Loading zarr dataset from %s",
        config.zarr_path,
    )
    raw_results = load_policy_results(config.zarr_path)
    LOGGER.info(
        "[stage 2/10] Loaded zarr dataset with dimensions %s",
        _format_mapping(raw_results.sizes),
    )

    LOGGER.info(
        "[stage 3/10] Filtering inversion dataset: observers=%s "
        "x_representation=%s theta_parameters=%s discrete_theta_parameters=%s "
        "drop_constant_theta=%s",
        list(config.observers),
        config.x_representation,
        None if config.theta_parameters is None else list(config.theta_parameters),
        (
            "auto"
            if config.discrete_theta_parameters is None
            else list(config.discrete_theta_parameters)
        ),
        config.drop_constant_theta,
    )
    dataset = filter_inversion_dataset(
        raw_results,
        observers=config.observers,
        theta_parameters=config.theta_parameters,
        drop_constant_theta=config.drop_constant_theta,
        x_representation=config.x_representation,
    )
    dataset = move_discrete_theta_to_end(
        dataset,
        discrete_theta_parameters=config.discrete_theta_parameters,
    )
    theta = np.asarray(dataset["theta"].values, dtype=np.float32)
    x = np.asarray(dataset["x"].values, dtype=np.float32)
    source_sample = np.asarray(dataset.coords["sample"].values, dtype=np.int64)
    dataset_metadata = inversion_dataset_metadata(dataset)
    train_indices, diagnostic_indices = split_diagnostic_holdout(
        n_simulations=theta.shape[0],
        reserve_count=config.diagnostics.reserve_count,
        seed=config.training.seed,
    )
    train_theta = theta[train_indices]
    train_x = x[train_indices]
    diagnostic_theta = theta[diagnostic_indices]
    diagnostic_x = x[diagnostic_indices]
    diagnostic_source_sample = source_sample[diagnostic_indices]
    category_counts = tuple(
        int(value)
        for value in dataset_metadata.get("num_categories_per_discrete_theta", [])
    )
    sbi_method = "MNPE" if category_counts else "NPE"
    prior_samples, prior_sample_indices = select_prior_samples(
        train_theta,
        max_samples=config.prior_sample_count,
        seed=config.training.seed,
    )
    LOGGER.info(
        "[stage 3/10] Prepared inversion tensors: theta_shape=%s x_shape=%s "
        "n_simulations=%s train_count=%s diagnostic_count=%s x_event_shape=%s "
        "theta_order=%s discrete_theta=%s stored_prior_samples=%s",
        theta.shape,
        x.shape,
        dataset_metadata["n_simulations"],
        int(train_theta.shape[0]),
        int(diagnostic_theta.shape[0]),
        tuple(dataset_metadata["x_event_shape"]),
        list(dataset_metadata["theta_names"]),
        list(dataset_metadata.get("discrete_theta_names", [])),
        int(prior_samples.shape[0]),
    )

    run_name = config.run_name
    if run_name is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_name = f"{sbi_method.lower()}-{stamp}"
    artifact_dir = config.output_root / run_name
    LOGGER.info(
        "[stage 4/10] Creating artifact directory %s",
        artifact_dir,
    )
    artifact_dir.mkdir(parents=True, exist_ok=False)
    tensorboard_log_dir = artifact_dir / SBI_LOG_DIRNAME
    LOGGER.info(
        "[stage 5/10] Starting %s training: device=%s batch_size=%s "
        "max_num_epochs=%s stop_after_epochs=%s density=%s flow_model=%s "
        "embedding=%s tensorboard_log_dir=%s",
        sbi_method,
        config.training.device,
        config.training.training_batch_size,
        config.training.max_num_epochs,
        config.training.stop_after_epochs,
        config.density.model,
        config.density.flow_model,
        config.embedding.kind,
        tensorboard_log_dir,
    )
    posterior = train_amortized_npe(
        theta=train_theta,
        x=train_x,
        prior_low=dataset_metadata["prior_low"],
        prior_high=dataset_metadata["prior_high"],
        x_event_shape=tuple(dataset_metadata["x_event_shape"]),
        tensorboard_log_dir=tensorboard_log_dir,
        training=config.training,
        density=config.density,
        embedding=config.embedding,
        num_categories_per_discrete_theta=category_counts,
    )
    posterior = attach_prior_samples_to_posterior(
        posterior,
        prior_samples=prior_samples,
        prior_sample_indices=prior_sample_indices,
        theta_names=list(dataset_metadata["theta_names"]),
        seed=config.training.seed,
    )
    LOGGER.info(
        "[stage 6/10] Finished %s training and attached %s prior samples",
        sbi_method,
        int(prior_samples.shape[0]),
    )

    metadata = build_metadata(
        dataset_metadata,
        theta_dim=theta.shape[1],
        sbi_method=sbi_method,
        prior_sample_count=int(prior_samples.shape[0]),
        train_count=int(train_theta.shape[0]),
        diagnostic_count=int(diagnostic_theta.shape[0]),
        diagnostics=config.diagnostics,
        training=config.training,
        density=config.density,
        embedding=config.embedding,
    )

    metadata_path = artifact_dir / "metadata.json"
    posterior_path = artifact_dir / "posterior.pkl"
    LOGGER.info("[stage 7/10] Writing metadata to %s", metadata_path)
    write_json(metadata_path, metadata)
    LOGGER.info("[stage 8/10] Writing posterior to %s", posterior_path)
    with posterior_path.open("wb") as handle:
        pickle.dump(posterior, handle)
    diagnostic_data_path = artifact_dir / DIAGNOSTIC_DATA_FILENAME
    LOGGER.info("[stage 9/10] Writing diagnostic holdout to %s", diagnostic_data_path)
    np.savez_compressed(
        diagnostic_data_path,
        theta=diagnostic_theta,
        x=diagnostic_x,
        source_sample=diagnostic_source_sample,
        theta_names=np.asarray(dataset_metadata["theta_names"]),
        x_names=np.asarray(dataset_metadata["x_names"]),
        continuous_theta_names=np.asarray(
            dataset_metadata.get("continuous_theta_names", [])
        ),
        discrete_theta_names=np.asarray(dataset_metadata.get("discrete_theta_names", [])),
    )
    if config.save_training_data:
        training_data_path = artifact_dir / "training_data.npz"
        LOGGER.info("[stage 9/10] Writing training data to %s", training_data_path)
        np.savez_compressed(
            training_data_path,
            theta=train_theta,
            x=train_x,
            kept_mask=np.asarray(dataset["kept_mask"].values, dtype=np.bool_),
            source_sample=source_sample[train_indices],
            theta_names=np.asarray(dataset_metadata["theta_names"]),
            x_names=np.asarray(dataset_metadata["x_names"]),
            continuous_theta_names=np.asarray(
                dataset_metadata.get("continuous_theta_names", [])
            ),
            discrete_theta_names=np.asarray(
                dataset_metadata.get("discrete_theta_names", [])
            ),
        )
    else:
        LOGGER.info("[stage 9/10] Skipping training data write")
    LOGGER.info("[stage 10/10] Training run complete: artifact_dir=%s", artifact_dir)

    return {
        "artifact_dir": str(artifact_dir),
        "posterior_file": str(posterior_path),
        "metadata_file": str(metadata_path),
        "tensorboard_log_dir": str(tensorboard_log_dir),
        "sbi_method": sbi_method,
        "n_simulations": dataset_metadata["n_simulations"],
        "train_count": int(train_theta.shape[0]),
        "diagnostic_count": int(diagnostic_theta.shape[0]),
        "diagnostic_data_file": DIAGNOSTIC_DATA_FILENAME,
        "prior_sample_count": int(prior_samples.shape[0]),
        "theta_names": list(dataset_metadata["theta_names"]),
        "continuous_theta_names": list(
            dataset_metadata.get("continuous_theta_names", [])
        ),
        "discrete_theta_names": list(dataset_metadata.get("discrete_theta_names", [])),
        "x_names": list(dataset_metadata["x_names"]),
        "x_representation": dataset_metadata["x_representation"],
    }


def build_metadata(
    dataset_metadata: dict[str, Any],
    *,
    theta_dim: int,
    sbi_method: str,
    prior_sample_count: int,
    train_count: int,
    diagnostic_count: int,
    diagnostics: DiagnosticsConfig,
    training: NPETrainingConfig,
    density: DensityEstimatorConfig,
    embedding: EmbeddingConfig,
) -> dict[str, Any]:
    artifact_dataset_metadata = _artifact_dataset_metadata(dataset_metadata)
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sbi": {
            "method": sbi_method,
            "amortized": True,
            "mixed_parameters": sbi_method == "MNPE",
            "posterior_file": "posterior.pkl",
            "tensorboard_log_dir": SBI_LOG_DIRNAME,
            "prior_samples": {
                "stored_in_posterior": prior_sample_count > 0,
                "source": "training_theta_rows",
                "count": int(prior_sample_count),
            },
        },
        "dataset": {
            **artifact_dataset_metadata,
            "theta_dim": theta_dim,
            "train_count": int(train_count),
            "diagnostic_count": int(diagnostic_count),
        },
        "diagnostics": {
            **diagnostics.to_metadata(),
            "file": DIAGNOSTIC_DATA_FILENAME,
            "train_count": int(train_count),
            "diagnostic_count": int(diagnostic_count),
        },
        "density_estimator": density.to_metadata(),
        "embedding": embedding.to_metadata(),
        "training": training.to_metadata(),
        "notebook_usage": {
            "load": "from network_creation.model_inversion import load_posterior_artifact",
            "observation_order": list(dataset_metadata["x_names"]),
            "theta_order": list(dataset_metadata["theta_names"]),
            "continuous_theta_order": list(
                dataset_metadata.get("continuous_theta_names", [])
            ),
            "discrete_theta_order": list(dataset_metadata.get("discrete_theta_names", [])),
        },
    }


def _artifact_dataset_metadata(dataset_metadata: dict[str, Any]) -> dict[str, Any]:
    metadata = dict(dataset_metadata)
    metadata["result_store_path"] = _project_relative_path(
        metadata["result_store_path"]
    )
    metadata["config_path"] = _project_relative_path(metadata["config_path"])
    metadata["path_base"] = "project_root"
    return metadata


def _project_relative_path(raw_path: Any) -> str:
    path = Path(str(raw_path)).expanduser()
    absolute_path = path if path.is_absolute() else PROJECT_ROOT / path
    relative_path = os.path.relpath(absolute_path.resolve(), PROJECT_ROOT)
    return relative_path.replace(os.sep, "/")


def split_diagnostic_holdout(
    *,
    n_simulations: int,
    reserve_count: int,
    seed: int,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    if reserve_count <= 0:
        raise ValueError("diagnostics.reserve_count must be > 0")
    if reserve_count >= n_simulations:
        raise ValueError("diagnostics.reserve_count must be < n_simulations")

    rng = np.random.default_rng(seed)
    diagnostic_indices = np.sort(
        rng.choice(n_simulations, size=reserve_count, replace=False).astype(np.int64)
    )
    diagnostic_mask = np.zeros(n_simulations, dtype=np.bool_)
    diagnostic_mask[diagnostic_indices] = True
    train_indices = np.nonzero(~diagnostic_mask)[0].astype(np.int64)
    return train_indices, diagnostic_indices


def _optional_str_tuple(value: Any) -> tuple[str, ...] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in value)


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _positive_int(value: Any, *, field_name: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{field_name} must be > 0")
    return parsed


def _resolve_config_path(config_dir: Path, raw_path: Any) -> Path:
    path = Path(str(raw_path)).expanduser()
    if path.is_absolute():
        return path
    return (config_dir / path).resolve()


def _format_mapping(mapping: Any) -> dict[str, int]:
    return {str(key): int(value) for key, value in dict(mapping).items()}


def _configure_logging(log_level: str) -> None:
    level_name = str(log_level).upper()
    level = getattr(logging, level_name, None)
    if not isinstance(level, int):
        raise ValueError(f"unknown log level: {log_level!r}")
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    _configure_logging(args.log_level)
    config_path = args.config_path.expanduser().resolve()
    LOGGER.info("[stage 1/10] Loading config from %s", config_path)
    config = load_train_config(config_path)
    LOGGER.info(
        "[stage 1/10] Loaded config: zarr_path=%s observers=%s "
        "x_representation=%s output_root=%s",
        config.zarr_path,
        list(config.observers),
        config.x_representation,
        config.output_root,
    )
    print(json.dumps(train_from_config(config), indent=2))


if __name__ == "__main__":
    main()
