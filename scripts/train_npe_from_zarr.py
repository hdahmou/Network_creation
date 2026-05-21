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
from typing import Any

import numpy as np
from numpy.typing import NDArray


LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SBI_LOG_DIRNAME = "sbi_logs"
DIAGNOSTIC_DATA_FILENAME = "diagnostic_data.npz"
TRAINING_METRICS_FILENAME = "training_metrics.json"
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
    DensityEstimatorConfig,
    EmbeddingConfig,
)
from network_creation.model_inversion.posterior import (  # noqa: E402
    attach_prior_samples_to_posterior,
    select_prior_samples,
)
from network_creation.model_inversion.training import (  # noqa: E402
    NPETrainingConfig,
    train_amortized_npe,
)


def load_policy_results(zarr_path: str | Path, *, normalize_time: bool):
    from tasks.policies import load_policy_results as _load_policy_results

    return _load_policy_results(zarr_path, normalize_time=normalize_time)


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
    normalize_time: bool = True
    input_parameter_names: tuple[str, ...] = ()


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
    diagnostics = data["diagnostics"]

    observers = tuple(str(observer) for observer in dataset["observers"])
    theta_parameters = _optional_name_tuple(dataset.get("theta_parameters"))
    input_parameter_names = tuple(
        str(item) for item in dataset.get("input_parameter_names", ())
    )
    discrete_theta_parameters = _optional_name_tuple(
        dataset.get("discrete_theta_parameters")
    )
    x_representation = dataset.get("x_representation", "final")
    normalize_time = bool(dataset.get("normalize_time", True))
    embedding_kind = embedding.get("kind", "identity")

    return TrainNPEConfig(
        zarr_path=_resolve_config_path(config_dir, dataset["zarr_path"]),
        observers=observers,
        theta_parameters=theta_parameters,
        input_parameter_names=input_parameter_names,
        discrete_theta_parameters=discrete_theta_parameters,
        drop_constant_theta=dataset.get("drop_constant_theta", True),
        x_representation=x_representation,
        normalize_time=normalize_time,
        output_root=_resolve_config_path(
            config_dir,
            artifact.get("output_root", "artifacts/model_inversion"),
        ),
        run_name=(
            None if artifact.get("run_name") is None else str(artifact.get("run_name"))
        ),
        save_training_data=artifact.get("save_training_data", False),
        prior_sample_count=int(artifact.get("prior_sample_count", 4096)),
        density=DensityEstimatorConfig(
            model=density.get("model", "nsf"),
            flow_model=density.get("flow_model", "nsf"),
            hidden_features=density.get("hidden_features", 50),
            num_transforms=density.get("num_transforms", 5),
            num_bins=density.get("num_bins", 10),
            discrete_hidden_features=(
                None
                if density.get("discrete_hidden_features") is None
                else int(density["discrete_hidden_features"])
            ),
            discrete_hidden_layers=density.get("discrete_hidden_layers", 2),
            continuous_hidden_features=(
                None
                if density.get("continuous_hidden_features") is None
                else int(density["continuous_hidden_features"])
            ),
            z_score_theta=density.get("z_score_theta", "independent"),
            z_score_x=density.get("z_score_x", "independent"),
            sbi_kwargs=dict(density.get("sbi_kwargs", {})),
        ),
        embedding=EmbeddingConfig(
            kind=embedding_kind,
            output_dim=embedding.get("output_dim", 32),
            rnn_hidden_size=embedding.get("rnn_hidden_size", 64),
            rnn_num_layers=embedding.get("rnn_num_layers", 1),
            rnn_type=embedding.get("rnn_type", "gru"),
            rnn_bidirectional=embedding.get("rnn_bidirectional", False),
        ),
        training=NPETrainingConfig(
            seed=training.get("seed", 0),
            device=training.get("device", "cpu"),
            data_device=training.get("data_device", "cpu"),
            training_batch_size=training.get("training_batch_size", 200),
            learning_rate=training.get("learning_rate", 5e-4),
            validation_fraction=training.get("validation_fraction", 0.1),
            stop_after_epochs=training.get("stop_after_epochs", 20),
            max_num_epochs=training.get("max_num_epochs", 500),
            clip_max_norm=(
                None
                if training.get("clip_max_norm", 5.0) is None
                else float(training.get("clip_max_norm", 5.0))
            ),
            show_progress_bars=training.get("show_progress_bars", True),
            dataloader_kwargs=dict(training.get("dataloader_kwargs", {})),
        ),
        diagnostics=DiagnosticsConfig(
            reserve_count=int(diagnostics["reserve_count"]),
        ),
    )


def train_from_config(config: TrainNPEConfig) -> dict[str, Any]:
    np.random.seed(config.training.seed)
    run = _prepare_run(config)
    _train_run(run)
    _write_run_artifacts(run)
    LOGGER.info("[complete] Training run complete: artifact_dir=%s", run["artifact_dir"])
    return _run_result(run)


def _prepare_run(config: TrainNPEConfig) -> dict[str, Any]:
    LOGGER.info(
        "[data] Loading zarr dataset from %s normalize_time=%s",
        config.zarr_path,
        config.normalize_time,
    )
    raw_results = load_policy_results(
        config.zarr_path,
        normalize_time=config.normalize_time,
    )
    raw_results = raw_results.assign_attrs(
        {
            **raw_results.attrs,
            "normalize_time": bool(config.normalize_time),
            "time_axis": raw_results.attrs.get(
                "time_axis",
                (
                    "activation_opportunities_per_agent"
                    if config.normalize_time
                    else "raw_simulator_steps"
                ),
            ),
        }
    )
    LOGGER.info(
        "[data] Loaded zarr dataset with dimensions %s",
        _format_mapping(raw_results.sizes),
    )

    LOGGER.info(
        "[data] Filtering inversion dataset: observers=%s "
        "x_representation=%s normalize_time=%s theta_parameters=%s "
        "input_parameter_names=%s discrete_theta_parameters=%s drop_constant_theta=%s",
        list(config.observers),
        config.x_representation,
        config.normalize_time,
        None if config.theta_parameters is None else list(config.theta_parameters),
        list(config.input_parameter_names),
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
        input_parameter_names=config.input_parameter_names,
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
    kept_mask = np.asarray(dataset["kept_mask"].values, dtype=np.bool_)
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
        "[data] Prepared inversion tensors: theta_shape=%s x_shape=%s "
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
    LOGGER.info("[artifacts] Creating artifact directory %s", artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=False)
    return {
        "config": config,
        "dataset_metadata": dataset_metadata,
        "theta": theta,
        "x": x,
        "train_theta": train_theta,
        "train_x": train_x,
        "diagnostic_theta": diagnostic_theta,
        "diagnostic_x": diagnostic_x,
        "diagnostic_source_sample": diagnostic_source_sample,
        "source_sample": source_sample,
        "kept_mask": kept_mask,
        "train_indices": train_indices,
        "diagnostic_indices": diagnostic_indices,
        "category_counts": category_counts,
        "sbi_method": sbi_method,
        "prior_samples": prior_samples,
        "prior_sample_indices": prior_sample_indices,
        "artifact_dir": artifact_dir,
        "posterior_path": artifact_dir / "posterior.pkl",
        "metadata_path": artifact_dir / "metadata.json",
        "tensorboard_log_dir": artifact_dir / SBI_LOG_DIRNAME,
        "training_metrics_path": artifact_dir / TRAINING_METRICS_FILENAME,
        "diagnostic_data_path": artifact_dir / DIAGNOSTIC_DATA_FILENAME,
    }


def _train_run(run: dict[str, Any]) -> None:
    config = run["config"]
    LOGGER.info(
        "[training] Starting %s training: device=%s data_device=%s batch_size=%s "
        "max_num_epochs=%s stop_after_epochs=%s density=%s flow_model=%s "
        "embedding=%s tensorboard_log_dir=%s",
        run["sbi_method"],
        config.training.device,
        config.training.data_device,
        config.training.training_batch_size,
        config.training.max_num_epochs,
        config.training.stop_after_epochs,
        config.density.model,
        config.density.flow_model,
        config.embedding.kind,
        run["tensorboard_log_dir"],
    )
    training_result = train_amortized_npe(
        theta=run["train_theta"],
        x=run["train_x"],
        prior_low=run["dataset_metadata"]["prior_low"],
        prior_high=run["dataset_metadata"]["prior_high"],
        x_event_shape=tuple(run["dataset_metadata"]["x_event_shape"]),
        tensorboard_log_dir=run["tensorboard_log_dir"],
        training=config.training,
        density=config.density,
        embedding=config.embedding,
        num_categories_per_discrete_theta=run["category_counts"],
    )
    run["training_summary"] = training_result.training_summary
    run["posterior"] = attach_prior_samples_to_posterior(
        training_result.posterior,
        prior_samples=run["prior_samples"],
        prior_sample_indices=run["prior_sample_indices"],
        theta_names=list(run["dataset_metadata"]["theta_names"]),
        seed=config.training.seed,
    )
    LOGGER.info(
        "[training] Finished %s training and attached %s prior samples",
        run["sbi_method"],
        int(run["prior_samples"].shape[0]),
    )


def _write_run_artifacts(run: dict[str, Any]) -> None:
    config = run["config"]
    metadata = _build_metadata(run)

    metadata_path = run["metadata_path"]
    posterior_path = run["posterior_path"]
    training_metrics_path = run["training_metrics_path"]
    LOGGER.info("[artifacts] Writing metadata to %s", metadata_path)
    write_json(metadata_path, metadata)
    LOGGER.info("[artifacts] Writing training metrics to %s", training_metrics_path)
    write_json(training_metrics_path, run["training_summary"])
    LOGGER.info("[artifacts] Writing posterior to %s", posterior_path)
    with posterior_path.open("wb") as handle:
        pickle.dump(run["posterior"], handle)
    diagnostic_data_path = run["diagnostic_data_path"]
    LOGGER.info("[artifacts] Writing diagnostic holdout to %s", diagnostic_data_path)
    np.savez_compressed(
        diagnostic_data_path,
        theta=run["diagnostic_theta"],
        x=run["diagnostic_x"],
        source_sample=run["diagnostic_source_sample"],
        theta_names=np.asarray(run["dataset_metadata"]["theta_names"]),
        x_names=np.asarray(run["dataset_metadata"]["x_names"]),
        continuous_theta_names=np.asarray(
            run["dataset_metadata"].get("continuous_theta_names", [])
        ),
        discrete_theta_names=np.asarray(
            run["dataset_metadata"].get("discrete_theta_names", [])
        ),
    )
    if config.save_training_data:
        training_data_path = run["artifact_dir"] / "training_data.npz"
        run["training_data_path"] = training_data_path
        LOGGER.info("[artifacts] Writing training data to %s", training_data_path)
        np.savez_compressed(
            training_data_path,
            theta=run["train_theta"],
            x=run["train_x"],
            kept_mask=run["kept_mask"],
            source_sample=run["source_sample"][run["train_indices"]],
            theta_names=np.asarray(run["dataset_metadata"]["theta_names"]),
            x_names=np.asarray(run["dataset_metadata"]["x_names"]),
            continuous_theta_names=np.asarray(
                run["dataset_metadata"].get("continuous_theta_names", [])
            ),
            discrete_theta_names=np.asarray(
                run["dataset_metadata"].get("discrete_theta_names", [])
            ),
        )
    else:
        LOGGER.info("[artifacts] Skipping training data write")


def _run_result(run: dict[str, Any]) -> dict[str, Any]:
    config = run["config"]
    training_summary = run["training_summary"]
    return {
        "artifact_dir": str(run["artifact_dir"]),
        "posterior_file": str(run["posterior_path"]),
        "metadata_file": str(run["metadata_path"]),
        "tensorboard_log_dir": str(run["tensorboard_log_dir"]),
        "training_metrics_file": TRAINING_METRICS_FILENAME,
        "best_validation_loss": training_summary.get("best_validation_loss"),
        "final_validation_loss": training_summary.get("final_validation_loss"),
        "best_validation_log_prob": training_summary.get("best_validation_log_prob"),
        "final_validation_log_prob": training_summary.get("final_validation_log_prob"),
        "sbi_method": run["sbi_method"],
        "n_simulations": run["dataset_metadata"]["n_simulations"],
        "train_count": int(run["train_theta"].shape[0]),
        "diagnostic_count": int(run["diagnostic_theta"].shape[0]),
        "diagnostic_data_file": DIAGNOSTIC_DATA_FILENAME,
        "prior_sample_count": int(run["prior_samples"].shape[0]),
        "theta_names": list(run["dataset_metadata"]["theta_names"]),
        "input_parameter_names": list(config.input_parameter_names),
        "continuous_theta_names": list(
            run["dataset_metadata"].get("continuous_theta_names", [])
        ),
        "discrete_theta_names": list(
            run["dataset_metadata"].get("discrete_theta_names", [])
        ),
        "x_names": list(run["dataset_metadata"]["x_names"]),
        "x_representation": run["dataset_metadata"]["x_representation"],
        "normalize_time": bool(config.normalize_time),
    }


def _build_metadata(run: dict[str, Any]) -> dict[str, Any]:
    config = run["config"]
    dataset_metadata = run["dataset_metadata"]
    artifact_dataset_metadata = _artifact_dataset_metadata(dataset_metadata)
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sbi": {
            "method": run["sbi_method"],
            "amortized": True,
            "mixed_parameters": run["sbi_method"] == "MNPE",
            "posterior_file": "posterior.pkl",
            "training_metrics_file": TRAINING_METRICS_FILENAME,
            "tensorboard_log_dir": SBI_LOG_DIRNAME,
            "prior_samples": {
                "stored_in_posterior": int(run["prior_samples"].shape[0]) > 0,
                "source": "training_theta_rows",
                "count": int(run["prior_samples"].shape[0]),
            },
        },
        "dataset": {
            **artifact_dataset_metadata,
            "theta_dim": int(run["theta"].shape[1]),
            "train_count": int(run["train_theta"].shape[0]),
            "diagnostic_count": int(run["diagnostic_theta"].shape[0]),
        },
        "diagnostics": {
            **config.diagnostics.to_metadata(),
            "file": DIAGNOSTIC_DATA_FILENAME,
            "train_count": int(run["train_theta"].shape[0]),
            "diagnostic_count": int(run["diagnostic_theta"].shape[0]),
        },
        "density_estimator": config.density.to_metadata(),
        "embedding": config.embedding.to_metadata(),
        "training": config.training.to_metadata(),
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


def _optional_name_tuple(value: Any) -> tuple[str, ...] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in value)


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
    LOGGER.info("[config] Loading config from %s", config_path)
    config = load_train_config(config_path)
    LOGGER.info(
        "[config] Loaded config: zarr_path=%s observers=%s "
        "x_representation=%s output_root=%s",
        config.zarr_path,
        list(config.observers),
        config.x_representation,
        config.output_root,
    )
    print(json.dumps(train_from_config(config), indent=2))


if __name__ == "__main__":
    main()
