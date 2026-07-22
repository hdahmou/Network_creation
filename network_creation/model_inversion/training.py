"""SBI NPE/MNPE training with zarr-backed lazy datasets."""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from math import isfinite, prod
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import InversionConfig
from .dataset import ZarrInversionDataset
from .networks import build_density_estimator as make_density_estimator_builder
from .schema import InversionSchema
from .splits import InversionSplit


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainingLoaders:
    train: Any
    validation: Any


@dataclass(frozen=True)
class EstimatorTrainingResult:
    estimator: Any
    epochs_trained: int
    train_loss: list[float]
    validation_loss: list[float]
    best_validation_loss: float
    final_train_loss: float | None
    final_validation_loss: float | None
    best_validation_log_prob: float
    final_validation_log_prob: float | None

    def to_metrics(self) -> dict[str, Any]:
        return {
            "epochs_trained": int(self.epochs_trained),
            "train_loss": [float(value) for value in self.train_loss],
            "validation_loss": [float(value) for value in self.validation_loss],
            "best_validation_loss": float(self.best_validation_loss),
            "final_train_loss": (
                None if self.final_train_loss is None else float(self.final_train_loss)
            ),
            "final_validation_loss": (
                None
                if self.final_validation_loss is None
                else float(self.final_validation_loss)
            ),
            "best_validation_log_prob": float(self.best_validation_log_prob),
            "final_validation_log_prob": (
                None
                if self.final_validation_log_prob is None
                else float(self.final_validation_log_prob)
            ),
        }


def build_posterior_estimator(
    dataset: ZarrInversionDataset,
    split: InversionSplit,
    schema: InversionSchema,
    config: InversionConfig,
) -> Any:
    theta, x = _materialize_rows(dataset, split.calibration_indices)
    _validate_training_boundary(theta, x, schema)
    device = torch.device(config.training.device)
    LOGGER.info(
        "Building %s estimator with calibration_shape theta=%s x=%s device=%s",
        schema.sbi_method,
        tuple(theta.shape),
        tuple(x.shape),
        device,
    )
    with torch.random.fork_rng(devices=_rng_fork_devices(device)):
        torch.manual_seed(int(config.training.seed))
        density_estimator_builder = make_density_estimator_builder(
            config.model.density_estimator,
            config.model.embedding,
            x_event_shape=schema.x_event_shape,
            num_categories_per_discrete_theta=schema.category_counts,
        )
        with torch.no_grad():
            estimator = density_estimator_builder(theta.to("cpu"), x.to("cpu"))
    return estimator.to(device)


def make_training_loaders(
    dataset: ZarrInversionDataset,
    split: InversionSplit,
    config: InversionConfig,
) -> TrainingLoaders:
    from torch.utils import data

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(config.training.seed))
    training = config.training
    if training.materialize_dataset:
        loader_dataset = _materialize_dataset(dataset, config)
        train_subset = data.Subset(loader_dataset, split.train_indices.tolist())
        validation_subset = data.Subset(
            loader_dataset,
            split.validation_indices.tolist(),
        )
        train_batches = data.BatchSampler(
            data.RandomSampler(train_subset, generator=generator),
            batch_size=int(training.batch_size),
            drop_last=False,
        )
        validation_batches = data.BatchSampler(
            data.SequentialSampler(validation_subset),
            batch_size=int(training.batch_size),
            drop_last=False,
        )
        return TrainingLoaders(
            train=data.DataLoader(
                train_subset,
                batch_size=None,
                sampler=train_batches,
                generator=generator,
                num_workers=0,
                pin_memory=False,
            ),
            validation=data.DataLoader(
                validation_subset,
                batch_size=None,
                sampler=validation_batches,
                num_workers=0,
                pin_memory=False,
            ),
        )

    loader_kwargs: dict[str, Any] = {
        "num_workers": int(training.num_workers),
        "pin_memory": _resolve_pin_memory(config),
    }
    if training.num_workers > 0:
        loader_kwargs["persistent_workers"] = bool(training.persistent_workers)
    train_subset = data.Subset(dataset, split.train_indices.tolist())
    validation_subset = data.Subset(dataset, split.validation_indices.tolist())
    return TrainingLoaders(
        train=data.DataLoader(
            train_subset,
            batch_size=min(int(training.batch_size), len(train_subset)),
            shuffle=True,
            generator=generator,
            **loader_kwargs,
        ),
        validation=data.DataLoader(
            validation_subset,
            batch_size=min(int(training.batch_size), len(validation_subset)),
            shuffle=False,
            **loader_kwargs,
        ),
    )


def _materialize_dataset(
    dataset: ZarrInversionDataset,
    config: InversionConfig,
) -> Any:
    from torch.utils import data

    row_count = len(dataset)
    device = torch.device(config.training.device)
    theta_shape = (row_count, len(dataset.schema.theta_order))
    x_shape = (row_count, *dataset.schema.x_event_shape)
    estimated_bytes = 4 * (prod(theta_shape) + prod(x_shape))
    LOGGER.info(
        "Materializing inversion dataset on %s: rows=%s theta_shape=%s "
        "x_shape=%s estimated_size=%.2f GiB",
        device,
        row_count,
        theta_shape,
        x_shape,
        estimated_bytes / (1024**3),
    )
    observed_output_arrays: dict[str, Any] = {}
    for name in dataset.schema.observed_outputs:
        LOGGER.info("Reading observed output %s into CPU memory", name)
        observed_output_arrays[name] = np.asarray(
            dataset.source.observed_output_arrays[name],
            dtype=np.float32,
        )
    memory_source = replace(
        dataset.source,
        observed_output_arrays=observed_output_arrays,
    )
    memory_dataset = ZarrInversionDataset(
        source=memory_source,
        row_index=dataset.row_index,
        schema=dataset.schema,
    )
    theta = torch.empty(theta_shape, dtype=torch.float32)
    x = torch.empty(x_shape, dtype=torch.float32)
    rows: Any = range(row_count)
    if config.training.show_progress_bars:
        from tqdm.auto import tqdm  # type: ignore[import-untyped]

        rows = tqdm(
            rows,
            desc="Materializing dataset",
            unit="sample",
            mininterval=1.0,
            miniters=100,
        )
    try:
        for index in rows:
            theta_row, x_row = memory_dataset[int(index)]
            theta[int(index)].copy_(theta_row)
            x[int(index)].copy_(x_row)
    finally:
        close = getattr(rows, "close", None)
        if close is not None:
            close()
    del memory_dataset, memory_source, observed_output_arrays
    theta = theta.to(device)
    x = x.to(device)
    LOGGER.info("Finished materializing inversion dataset")
    return data.TensorDataset(theta, x)


def fit_posterior_estimator(
    estimator: Any,
    loaders: TrainingLoaders,
    config: InversionConfig,
    *,
    tensorboard_dir: Path | None = None,
) -> EstimatorTrainingResult:
    from torch.optim import Adam
    from torch.utils.tensorboard.writer import SummaryWriter

    training = config.training
    device = torch.device(training.device)
    optimizer = Adam(list(estimator.parameters()), lr=float(training.learning_rate))
    non_blocking = _resolve_pin_memory(config) and device.type == "cuda"
    train_loss_values: list[float] = []
    validation_loss_values: list[float] = []
    best_validation_loss = float("inf")
    best_state_dict = _cpu_state_dict(estimator)
    epochs_since_improvement = 0
    epochs_trained = 0
    progress_bar = _epoch_progress_bar(config=config)
    summary_writer = None
    if tensorboard_dir is not None:
        tensorboard_dir.mkdir(parents=True, exist_ok=True)
        summary_writer = SummaryWriter(str(tensorboard_dir))

    try:
        with torch.random.fork_rng(devices=_rng_fork_devices(device)):
            torch.manual_seed(int(training.seed))
            for epoch in progress_bar:
                estimator.train()
                train_loss = _run_epoch(
                    density_estimator=estimator,
                    loader=loaders.train,
                    device=device,
                    optimizer=optimizer,
                    clip_max_norm=training.clip_max_norm,
                    loader_name="training",
                    non_blocking=non_blocking,
                )
                estimator.eval()
                validation_loss = _run_epoch(
                    density_estimator=estimator,
                    loader=loaders.validation,
                    device=device,
                    optimizer=None,
                    clip_max_norm=None,
                    loader_name="validation",
                    non_blocking=non_blocking,
                )
                epochs_trained = int(epoch) + 1
                train_loss_values.append(float(train_loss))
                validation_loss_values.append(float(validation_loss))
                if summary_writer is not None:
                    summary_writer.add_scalar("train_loss", train_loss, epoch)
                    summary_writer.add_scalar("validation_loss", validation_loss, epoch)

                if validation_loss < best_validation_loss:
                    best_validation_loss = float(validation_loss)
                    best_state_dict = _cpu_state_dict(estimator)
                    epochs_since_improvement = 0
                else:
                    epochs_since_improvement += 1
                _update_progress_bar(
                    progress_bar,
                    train_loss=train_loss,
                    validation_loss=validation_loss,
                    best_validation_loss=best_validation_loss,
                )
                if epochs_since_improvement >= int(training.stop_after_epochs):
                    break
    finally:
        close = getattr(progress_bar, "close", None)
        if close is not None:
            close()
        if summary_writer is not None:
            summary_writer.close()

    estimator.load_state_dict(best_state_dict)
    estimator.zero_grad(set_to_none=True)
    final_train_loss = train_loss_values[-1] if train_loss_values else None
    final_validation_loss = validation_loss_values[-1] if validation_loss_values else None
    return EstimatorTrainingResult(
        estimator=estimator,
        epochs_trained=epochs_trained,
        train_loss=train_loss_values,
        validation_loss=validation_loss_values,
        best_validation_loss=float(best_validation_loss),
        final_train_loss=final_train_loss,
        final_validation_loss=final_validation_loss,
        best_validation_log_prob=-float(best_validation_loss),
        final_validation_log_prob=(
            None if final_validation_loss is None else -float(final_validation_loss)
        ),
    )


def build_direct_posterior(
    training: EstimatorTrainingResult,
    schema: InversionSchema,
    config: InversionConfig,
) -> Any:
    from sbi.inference.posteriors import DirectPosterior  # type: ignore[import-untyped]
    from sbi.utils import BoxUniform  # type: ignore[import-untyped]

    device = torch.device(config.training.device)
    prior_low = torch.as_tensor(schema.prior_low, dtype=torch.float32, device=device)
    prior_high = torch.as_tensor(schema.prior_high, dtype=torch.float32, device=device)
    prior = BoxUniform(low=prior_low, high=prior_high)
    return DirectPosterior(training.estimator, prior=prior, device=str(device))


def _materialize_rows(
    dataset: ZarrInversionDataset,
    indices: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    theta_rows: list[torch.Tensor] = []
    x_rows: list[torch.Tensor] = []
    for index in indices:
        theta, x = dataset[int(index)]
        theta_rows.append(theta)
        x_rows.append(x)
    if not theta_rows:
        raise ValueError("cannot materialize an empty row set")
    return torch.stack(theta_rows, dim=0), torch.stack(x_rows, dim=0)


def _validate_training_boundary(
    theta: torch.Tensor,
    x: torch.Tensor,
    schema: InversionSchema,
) -> None:
    if theta.ndim != 2:
        raise ValueError("theta must have shape (batch, parameter)")
    if x.ndim < 2:
        raise ValueError("x must have shape (batch, ...)")
    if int(theta.shape[0]) != int(x.shape[0]):
        raise ValueError("theta and x must have the same batch size")
    if int(theta.shape[1]) != len(schema.theta_order):
        raise ValueError("theta dimension must match theta_order")
    if tuple(int(value) for value in x.shape[1:]) != schema.x_event_shape:
        raise ValueError("x_event_shape must match prepared x rows")
    if not bool(torch.isfinite(theta).all().item()):
        raise ValueError("theta contains non-finite values")
    if not bool(torch.isfinite(x).all().item()):
        raise ValueError("x contains non-finite values")


def _run_epoch(
    *,
    density_estimator: Any,
    loader: Any,
    device: torch.device,
    optimizer: Any | None,
    clip_max_norm: float | None,
    loader_name: str,
    non_blocking: bool = False,
) -> float:
    from torch.nn.utils.clip_grad import clip_grad_norm_

    is_training = optimizer is not None
    loss_sum = 0.0
    sample_count = 0
    context = torch.enable_grad() if is_training else torch.no_grad()
    with context:
        for theta_batch, x_batch in loader:
            theta_batch = theta_batch.to(device, non_blocking=non_blocking)
            x_batch = x_batch.to(device, non_blocking=non_blocking)
            batch_size = int(theta_batch.shape[0])
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            losses = density_estimator.loss(theta_batch, condition=x_batch)
            loss = torch.mean(losses)
            loss_value = float(loss.detach().item())
            if not isfinite(loss_value):
                raise ValueError(f"{loader_name} loss contains non-finite values")
            loss_sum += loss_value * batch_size
            sample_count += batch_size
            if optimizer is not None:
                loss.backward()
                if clip_max_norm is not None:
                    clip_grad_norm_(
                        density_estimator.parameters(),
                        max_norm=float(clip_max_norm),
                    )
                optimizer.step()
    if sample_count <= 0:
        raise ValueError(f"{loader_name} loader processed zero samples")
    return loss_sum / sample_count


def _resolve_pin_memory(config: InversionConfig) -> bool:
    if config.training.pin_memory is not None:
        return bool(config.training.pin_memory)
    return torch.device(config.training.device).type == "cuda"


def _epoch_progress_bar(*, config: InversionConfig) -> Any:
    if not config.training.show_progress_bars:
        return range(config.training.max_num_epochs)
    from tqdm.auto import tqdm  # type: ignore[import-untyped]

    return tqdm(
        range(config.training.max_num_epochs),
        desc="Posterior training",
        unit="epoch",
    )


def _update_progress_bar(
    progress_bar: Any,
    *,
    train_loss: float,
    validation_loss: float,
    best_validation_loss: float,
) -> None:
    set_postfix = getattr(progress_bar, "set_postfix", None)
    if set_postfix is None:
        return
    set_postfix(
        {
            "train_loss": f"{train_loss:.4g}",
            "val_loss": f"{validation_loss:.4g}",
            "best_val": f"{best_validation_loss:.4g}",
        },
        refresh=False,
    )


def _cpu_state_dict(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().to("cpu").clone()
        for key, value in module.state_dict().items()
    }


def _rng_fork_devices(device: torch.device) -> list[int]:
    if device.type != "cuda":
        return []
    index = 0 if device.index is None else int(device.index)
    return [index]
