#!/usr/bin/env python3
"""Train and save an amortized SBI NPE posterior from a JSON config.

Example:
    uv run python scripts/train_npe_from_zarr.py \
      configs/model_inversion.baseline.train_npe_config.json
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import numpy as np


LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SBI_LOG_DIRNAME = "sbi_logs"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from network_creation.io import read_json, write_json  # noqa: E402
from network_creation.model_inversion.data import (  # noqa: E402
    ObservationRepresentation,
    filter_inversion_dataset,
    inversion_dataset_metadata,
)
from network_creation.model_inversion.networks import (  # noqa: E402
    DensityEstimatorConfig,
    DensityEstimatorName,
    EmbeddingConfig,
    EmbeddingKind,
    RNNKind,
)
from network_creation.model_inversion.training import (  # noqa: E402
    NPETrainingConfig,
    train_amortized_npe,
)
from tasks.policies import load_policy_results  # noqa: E402


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


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train one amortized sbi NPE posterior from a JSON config."
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

    observers = tuple(str(observer) for observer in dataset["observers"])
    theta_parameters = _optional_str_tuple(dataset.get("theta_parameters"))
    x_representation = cast(
        ObservationRepresentation,
        dataset.get("x_representation", "final"),
    )

    return TrainNPEConfig(
        zarr_path=_resolve_config_path(config_dir, dataset["zarr_path"]),
        observers=observers,
        theta_parameters=theta_parameters,
        drop_constant_theta=dataset.get("drop_constant_theta", True),
        x_representation=x_representation,
        output_root=_resolve_config_path(
            config_dir,
            artifact.get("output_root", "artifacts/model_inversion"),
        ),
        run_name=_optional_str(artifact.get("run_name")),
        save_training_data=artifact.get("save_training_data", False),
        density=DensityEstimatorConfig(
            model=cast(DensityEstimatorName, density.get("model", "nsf")),
            hidden_features=density.get("hidden_features", 50),
            num_transforms=density.get("num_transforms", 5),
            num_bins=density.get("num_bins", 10),
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
        "x_representation=%s theta_parameters=%s drop_constant_theta=%s",
        list(config.observers),
        config.x_representation,
        None if config.theta_parameters is None else list(config.theta_parameters),
        config.drop_constant_theta,
    )
    dataset = filter_inversion_dataset(
        raw_results,
        observers=config.observers,
        theta_parameters=config.theta_parameters,
        drop_constant_theta=config.drop_constant_theta,
        x_representation=config.x_representation,
    )
    theta = np.asarray(dataset["theta"].values, dtype=np.float32)
    x = np.asarray(dataset["x"].values, dtype=np.float32)
    dataset_metadata = inversion_dataset_metadata(dataset)
    LOGGER.info(
        "[stage 3/10] Prepared training tensors: theta_shape=%s x_shape=%s "
        "n_simulations=%s x_event_shape=%s",
        theta.shape,
        x.shape,
        dataset_metadata["n_simulations"],
        tuple(dataset_metadata["x_event_shape"]),
    )

    run_name = config.run_name
    if run_name is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_name = f"npe-{stamp}"
    artifact_dir = config.output_root / run_name
    LOGGER.info(
        "[stage 4/10] Creating artifact directory %s",
        artifact_dir,
    )
    artifact_dir.mkdir(parents=True, exist_ok=False)
    tensorboard_log_dir = artifact_dir / SBI_LOG_DIRNAME
    LOGGER.info(
        "[stage 5/10] Starting NPE training: device=%s batch_size=%s "
        "max_num_epochs=%s stop_after_epochs=%s density=%s embedding=%s "
        "tensorboard_log_dir=%s",
        config.training.device,
        config.training.training_batch_size,
        config.training.max_num_epochs,
        config.training.stop_after_epochs,
        config.density.model,
        config.embedding.kind,
        tensorboard_log_dir,
    )
    posterior = train_amortized_npe(
        theta=theta,
        x=x,
        prior_low=dataset_metadata["prior_low"],
        prior_high=dataset_metadata["prior_high"],
        x_event_shape=tuple(dataset_metadata["x_event_shape"]),
        tensorboard_log_dir=tensorboard_log_dir,
        training=config.training,
        density=config.density,
        embedding=config.embedding,
    )
    LOGGER.info("[stage 6/10] Finished NPE training")

    metadata = build_metadata(
        dataset_metadata,
        theta_dim=theta.shape[1],
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
    if config.save_training_data:
        training_data_path = artifact_dir / "training_data.npz"
        LOGGER.info("[stage 9/10] Writing training data to %s", training_data_path)
        np.savez_compressed(
            training_data_path,
            theta=theta,
            x=x,
            kept_mask=np.asarray(dataset["kept_mask"].values, dtype=np.bool_),
            theta_names=np.asarray(dataset_metadata["theta_names"]),
            x_names=np.asarray(dataset_metadata["x_names"]),
        )
    else:
        LOGGER.info("[stage 9/10] Skipping training data write")
    LOGGER.info("[stage 10/10] Training run complete: artifact_dir=%s", artifact_dir)

    return {
        "artifact_dir": str(artifact_dir),
        "posterior_file": str(posterior_path),
        "metadata_file": str(metadata_path),
        "tensorboard_log_dir": str(tensorboard_log_dir),
        "n_simulations": dataset_metadata["n_simulations"],
        "theta_names": list(dataset_metadata["theta_names"]),
        "x_names": list(dataset_metadata["x_names"]),
        "x_representation": dataset_metadata["x_representation"],
    }


def build_metadata(
    dataset_metadata: dict[str, Any],
    *,
    theta_dim: int,
    training: NPETrainingConfig,
    density: DensityEstimatorConfig,
    embedding: EmbeddingConfig,
) -> dict[str, Any]:
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sbi": {
            "method": "NPE",
            "amortized": True,
            "posterior_file": "posterior.pkl",
            "tensorboard_log_dir": SBI_LOG_DIRNAME,
        },
        "dataset": {
            **dataset_metadata,
            "theta_dim": theta_dim,
        },
        "density_estimator": density.to_metadata(),
        "embedding": embedding.to_metadata(),
        "training": training.to_metadata(),
        "notebook_usage": {
            "load": "from network_creation.model_inversion import load_posterior_artifact",
            "observation_order": list(dataset_metadata["x_names"]),
            "theta_order": list(dataset_metadata["theta_names"]),
        },
    }


def _optional_str_tuple(value: Any) -> tuple[str, ...] | None:
    if value is None:
        return None
    return tuple(str(item) for item in value)


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


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
