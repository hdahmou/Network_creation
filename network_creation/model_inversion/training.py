"""SBI NPE/MNPE training orchestration."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from .networks import DensityEstimatorConfig, EmbeddingConfig, build_density_estimator


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class NPETrainingConfig:
    seed: int = 0
    device: str = "cpu"
    training_batch_size: int = 200
    learning_rate: float = 5e-4
    validation_fraction: float = 0.1
    stop_after_epochs: int = 20
    max_num_epochs: int = 500
    clip_max_norm: float | None = 5.0
    show_progress_bars: bool = True

    def to_metadata(self) -> dict[str, object]:
        return asdict(self)


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
) -> Any:
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
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)
    category_counts = tuple(int(value) for value in (num_categories_per_discrete_theta or ()))
    method_name = "MNPE" if category_counts else "NPE"

    LOGGER.info(
        "Preparing tensors for amortized %s: device=%s seed=%s",
        method_name,
        device,
        training.seed,
    )
    theta_tensor = torch.as_tensor(theta, dtype=torch.float32, device=device)
    x_tensor = torch.as_tensor(x, dtype=torch.float32, device=device)
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
        LOGGER.info("Appending %s simulations to SBI inference", theta_tensor.shape[0])
        inference = inference.append_simulations(theta_tensor, x_tensor)
        LOGGER.info(
            "Starting SBI train call: method=%s batch_size=%s learning_rate=%s "
            "validation_fraction=%s max_num_epochs=%s stop_after_epochs=%s",
            method_name,
            training.training_batch_size,
            training.learning_rate,
            training.validation_fraction,
            training.max_num_epochs,
            training.stop_after_epochs,
        )
        density_estimator = inference.train(
            training_batch_size=training.training_batch_size,
            learning_rate=training.learning_rate,
            validation_fraction=training.validation_fraction,
            stop_after_epochs=training.stop_after_epochs,
            max_num_epochs=training.max_num_epochs,
            clip_max_norm=training.clip_max_norm,
            show_train_summary=True,
        )
        LOGGER.info("SBI train call finished; building posterior")
        posterior = inference.build_posterior(density_estimator)
        LOGGER.info("Built amortized %s posterior", method_name)
        return posterior
    finally:
        summary_writer.close()
