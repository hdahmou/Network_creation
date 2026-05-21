"""SBI NPE/MNPE training orchestration."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

from .networks import DensityEstimatorConfig, EmbeddingConfig, build_density_estimator


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class NPETrainingConfig:
    seed: int = 0
    device: str = "cpu"
    data_device: str = "cpu"
    training_batch_size: int = 200
    learning_rate: float = 5e-4
    validation_fraction: float = 0.1
    stop_after_epochs: int = 20
    max_num_epochs: int = 500
    clip_max_norm: float | None = 5.0
    show_progress_bars: bool = True
    dataloader_kwargs: dict[str, Any] = field(default_factory=dict)

    def to_metadata(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class NPETrainingResult:
    posterior: Any
    training_summary: dict[str, Any]


def train_amortized_npe(
    *,
    theta: Any,
    x: Any,
    prior_low: Any,
    prior_high: Any,
    x_event_shape: tuple[int, ...],
    tensorboard_log_dir: Path,
    training: NPETrainingConfig,
    density: DensityEstimatorConfig,
    embedding: EmbeddingConfig,
    num_categories_per_discrete_theta: Sequence[int] | None = None,
) -> NPETrainingResult:
    """Train a single-round amortized NPE/MNPE posterior from precomputed tensors.

    Passing ``num_categories_per_discrete_theta`` switches the SBI inference class
    to MNPE and expects the corresponding discrete theta columns to be the last
    columns in ``theta``.
    """

    from sbi.inference import MNPE, NPE
    from sbi.utils import BoxUniform
    from sbi.utils.tracking import TensorBoardTracker
    from torch.utils.tensorboard.writer import SummaryWriter

    torch.manual_seed(training.seed)
    device = torch.device(training.device)
    data_device = torch.device(training.data_device)
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)
    category_counts = tuple(
        int(value) for value in (num_categories_per_discrete_theta or ())
    )
    method_name = "MNPE" if category_counts else "NPE"

    LOGGER.info(
        "Preparing tensors for amortized %s: training_device=%s data_device=%s seed=%s",
        method_name,
        device,
        data_device,
        training.seed,
    )
    theta_tensor = torch.as_tensor(theta, dtype=torch.float32, device=data_device)
    x_tensor = torch.as_tensor(x, dtype=torch.float32, device=data_device)
    LOGGER.info(
        "Prepared tensors: theta_shape=%s x_shape=%s x_event_shape=%s",
        tuple(theta_tensor.shape),
        tuple(x_tensor.shape),
        x_event_shape,
    )
    LOGGER.info("Creating uniform support prior for theta_dim=%s", theta_tensor.shape[-1])
    prior = BoxUniform(
        low=torch.as_tensor(prior_low, dtype=torch.float32, device=device),
        high=torch.as_tensor(prior_high, dtype=torch.float32, device=device),
    )
    LOGGER.info(
        "Building density estimator: model=%s embedding=%s discrete_categories=%s",
        density.model,
        embedding.kind,
        category_counts,
    )
    density_estimator_builder = build_density_estimator(
        density,
        embedding,
        x_event_shape=x_event_shape,
        num_categories_per_discrete_theta=category_counts,
    )
    LOGGER.info(
        "Creating SBI %s inference object: tensorboard_log_dir=%s",
        method_name,
        tensorboard_log_dir,
    )
    summary_writer = SummaryWriter(str(tensorboard_log_dir))
    try:
        inference_class = MNPE if category_counts else NPE
        inference = inference_class(
            prior=prior,
            density_estimator=density_estimator_builder,
            device=str(device),
            tracker=TensorBoardTracker(summary_writer),
            show_progress_bars=training.show_progress_bars,
        )
        LOGGER.info(
            "Appending %s simulations to SBI inference on data_device=%s",
            theta_tensor.shape[0],
            data_device,
        )
        inference = inference.append_simulations(
            theta_tensor,
            x_tensor,
            data_device=str(data_device),
        )
        LOGGER.info(
            "Starting SBI train call: method=%s batch_size=%s learning_rate=%s "
            "validation_fraction=%s max_num_epochs=%s stop_after_epochs=%s "
            "dataloader_kwargs=%s",
            method_name,
            training.training_batch_size,
            training.learning_rate,
            training.validation_fraction,
            training.max_num_epochs,
            training.stop_after_epochs,
            training.dataloader_kwargs,
        )
        density_estimator = inference.train(
            training_batch_size=training.training_batch_size,
            learning_rate=training.learning_rate,
            validation_fraction=training.validation_fraction,
            stop_after_epochs=training.stop_after_epochs,
            max_num_epochs=training.max_num_epochs,
            clip_max_norm=training.clip_max_norm,
            show_train_summary=True,
            dataloader_kwargs=training.dataloader_kwargs,
        )
        LOGGER.info("SBI train call finished; building posterior")
        posterior = inference.build_posterior(density_estimator)
        training_summary = _training_summary_from_inference(inference)
        LOGGER.info("Built amortized %s posterior", method_name)
        return NPETrainingResult(
            posterior=posterior,
            training_summary=training_summary,
        )
    finally:
        summary_writer.close()


def _training_summary_from_inference(inference: Any) -> dict[str, Any]:
    return _training_summary_from_sbi_summary(getattr(inference, "_summary", {}))


def _training_summary_from_sbi_summary(summary: Any) -> dict[str, Any]:
    if not isinstance(summary, dict):
        summary = {}

    training_loss = _json_float_list(summary.get("training_loss", []))
    validation_loss = _json_float_list(summary.get("validation_loss", []))
    best_validation_loss_values = _json_float_list(
        summary.get("best_validation_loss", [])
    )
    epochs_trained_values = _json_int_list(summary.get("epochs_trained", []))

    result: dict[str, Any] = {
        "epochs_trained": (
            epochs_trained_values[-1] if epochs_trained_values else None
        ),
        "training_loss": training_loss,
        "validation_loss": validation_loss,
        "best_validation_loss": (
            best_validation_loss_values[-1]
            if best_validation_loss_values
            else None
        ),
        "final_training_loss": training_loss[-1] if training_loss else None,
        "final_validation_loss": validation_loss[-1] if validation_loss else None,
    }
    if "epoch_durations_sec" in summary:
        result["epoch_durations_sec"] = _json_float_list(
            summary.get("epoch_durations_sec", [])
        )

    best_validation_loss = result["best_validation_loss"]
    final_validation_loss = result["final_validation_loss"]
    result["best_validation_log_prob"] = (
        -best_validation_loss if best_validation_loss is not None else None
    )
    result["final_validation_log_prob"] = (
        -final_validation_loss if final_validation_loss is not None else None
    )
    return result


def _json_float_list(values: Any) -> list[float]:
    if values is None:
        return []
    try:
        iterator = iter(values)
    except TypeError:
        return [float(values)]
    return [float(value) for value in iterator]


def _json_int_list(values: Any) -> list[int]:
    if values is None:
        return []
    try:
        iterator = iter(values)
    except TypeError:
        return [int(values)]
    return [int(value) for value in iterator]
