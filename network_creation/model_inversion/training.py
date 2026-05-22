"""SBI NPE/MNPE training orchestration."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from .networks import (
    DensityEstimatorConfig,
    EmbeddingConfig,
    build_density_estimator as make_density_estimator_builder,
)


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
    num_workers: int = 0
    pin_memory: bool | None = None
    persistent_workers: bool = False

    def to_metadata(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class NPETrainingResult:
    posterior: Any
    training_summary: dict[str, Any]


@dataclass(frozen=True)
class _TrainingTensors:
    theta: torch.Tensor
    x: torch.Tensor
    prior: Any
    category_counts: tuple[int, ...]
    train_count: int
    validation_count: int
    method_name: str


@dataclass(frozen=True)
class _TrainValidationSplit:
    train_indices: torch.Tensor
    validation_indices: torch.Tensor


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

    Passing ``num_categories_per_discrete_theta`` selects MNPE. The corresponding
    discrete theta columns must be the last columns in ``theta``.
    """

    from sbi.inference.posteriors import DirectPosterior
    from torch.utils.tensorboard.writer import SummaryWriter

    device = torch.device(training.device)
    data_device = torch.device(training.data_device)
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)

    tensors = _prepare_training_tensors(
        theta=theta,
        x=x,
        prior_low=prior_low,
        prior_high=prior_high,
        x_event_shape=x_event_shape,
        training=training,
        device=device,
        data_device=data_device,
        num_categories_per_discrete_theta=num_categories_per_discrete_theta,
    )
    LOGGER.info(
        "Prepared tensors for amortized %s: theta_shape=%s x_shape=%s "
        "x_event_shape=%s training_device=%s data_device=%s seed=%s",
        tensors.method_name,
        tuple(tensors.theta.shape),
        tuple(tensors.x.shape),
        x_event_shape,
        device,
        data_device,
        training.seed,
    )

    density_estimator_builder = make_density_estimator_builder(
        density,
        embedding,
        x_event_shape=x_event_shape,
        num_categories_per_discrete_theta=tensors.category_counts,
    )
    summary_writer = SummaryWriter(str(tensorboard_log_dir))
    try:
        density_estimator, summary = _train_density_estimator(
            density_estimator_builder=density_estimator_builder,
            tensors=tensors,
            training=training,
            summary_writer=summary_writer,
        )
        posterior = DirectPosterior(
            density_estimator,
            prior=tensors.prior,
            device=str(device),
        )
        return NPETrainingResult(
            posterior=posterior,
            training_summary=summary,
        )
    finally:
        summary_writer.close()


def _prepare_training_tensors(
    *,
    theta: Any,
    x: Any,
    prior_low: Any,
    prior_high: Any,
    x_event_shape: tuple[int, ...],
    training: NPETrainingConfig,
    device: torch.device,
    data_device: torch.device,
    num_categories_per_discrete_theta: Sequence[int] | None,
) -> _TrainingTensors:
    from sbi.utils import BoxUniform

    _validate_training_settings(training, data_device=data_device)

    theta_tensor = torch.as_tensor(theta, dtype=torch.float32, device=data_device)
    x_tensor = torch.as_tensor(x, dtype=torch.float32, device=data_device)
    category_counts = tuple(
        int(value) for value in (num_categories_per_discrete_theta or ())
    )

    if theta_tensor.ndim != 2:
        raise ValueError("theta must have shape (simulation, parameter)")
    if x_tensor.ndim < 2:
        raise ValueError("x must have shape (simulation, ...)")
    if int(theta_tensor.shape[0]) != int(x_tensor.shape[0]):
        raise ValueError("theta and x must have the same number of simulations")
    if tuple(int(value) for value in x_tensor.shape[1:]) != tuple(x_event_shape):
        raise ValueError("x_event_shape must match x.shape[1:]")
    if int(theta_tensor.shape[0]) <= 0:
        raise ValueError("cannot train on an empty dataset")

    prior_low_tensor = torch.as_tensor(prior_low, dtype=torch.float32, device=device)
    prior_high_tensor = torch.as_tensor(prior_high, dtype=torch.float32, device=device)
    expected_prior_shape = (int(theta_tensor.shape[1]),)
    if tuple(prior_low_tensor.shape) != expected_prior_shape:
        raise ValueError("prior_low must have one value per theta column")
    if tuple(prior_high_tensor.shape) != expected_prior_shape:
        raise ValueError("prior_high must have one value per theta column")
    if not bool(torch.all(prior_low_tensor < prior_high_tensor).item()):
        raise ValueError("prior_low must be strictly less than prior_high")

    train_count = int((1.0 - training.validation_fraction) * int(theta_tensor.shape[0]))
    validation_count = int(theta_tensor.shape[0]) - train_count
    if train_count <= 0 or validation_count <= 0:
        raise ValueError(
            "validation_fraction must leave at least one training and validation sample"
        )

    theta_dim = int(theta_tensor.shape[1])
    if len(category_counts) > theta_dim:
        raise ValueError("too many discrete theta category counts")
    if any(count <= 0 for count in category_counts):
        raise ValueError("discrete theta category counts must be positive")

    if category_counts:
        theta_cpu = theta_tensor.detach().to("cpu")
        discrete_start = theta_dim - len(category_counts)
        for offset, category_count in enumerate(category_counts):
            column = theta_cpu[:, discrete_start + offset]
            if not bool(torch.isfinite(column).all().item()):
                raise ValueError("discrete theta columns must contain only finite values")
            rounded_column = torch.round(column)
            if not bool(torch.equal(column, rounded_column)):
                raise ValueError("discrete theta columns must contain integer-coded values")
            observed_count = int(torch.unique(rounded_column.to(torch.long)).numel())
            if observed_count != category_count:
                raise ValueError(
                    "configured category count must match observed discrete theta values"
                )

    return _TrainingTensors(
        theta=theta_tensor,
        x=x_tensor,
        prior=BoxUniform(low=prior_low_tensor, high=prior_high_tensor),
        category_counts=category_counts,
        train_count=train_count,
        validation_count=validation_count,
        method_name="MNPE" if category_counts else "NPE",
    )


def _validate_training_settings(
    training: NPETrainingConfig,
    *,
    data_device: torch.device,
) -> None:
    if training.training_batch_size <= 0:
        raise ValueError("training_batch_size must be positive")
    if training.max_num_epochs <= 0:
        raise ValueError("max_num_epochs must be positive")
    if training.stop_after_epochs <= 0:
        raise ValueError("stop_after_epochs must be positive")
    if not 0.0 < training.validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    if training.num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    if training.persistent_workers and training.num_workers <= 0:
        raise ValueError("persistent_workers requires num_workers > 0")
    if training.num_workers > 0 and data_device.type != "cpu":
        raise ValueError("num_workers > 0 requires CPU-resident training tensors")
    if training.pin_memory is True and data_device.type != "cpu":
        raise ValueError("pin_memory=True requires CPU-resident training tensors")


def _train_density_estimator(
    *,
    density_estimator_builder: Any,
    tensors: _TrainingTensors,
    training: NPETrainingConfig,
    summary_writer: Any,
) -> tuple[Any, dict[str, Any]]:
    from torch.optim import Adam

    device = torch.device(training.device)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(training.seed))
    non_blocking = (
        _resolve_pin_memory(
            training,
            data_device=tensors.theta.device,
            training_device=device,
        )
        and tensors.theta.device.type == "cpu"
        and device.type == "cuda"
    )
    split = _make_train_validation_split(
        tensors=tensors,
        generator=generator,
    )

    with torch.random.fork_rng(devices=_rng_fork_devices(device)):
        torch.manual_seed(int(training.seed))
        density_estimator = _build_density_estimator(
            density_estimator_builder=density_estimator_builder,
            tensors=tensors,
            split=split,
            training=training,
            generator=generator,
        )
        loaders = _make_train_validation_loaders(
            tensors=tensors,
            split=split,
            training=training,
            generator=generator,
        )
        optimizer = Adam(
            list(density_estimator.parameters()),
            lr=training.learning_rate,
        )
        return _fit_density_estimator(
            density_estimator=density_estimator,
            loaders=loaders,
            optimizer=optimizer,
            device=device,
            non_blocking=non_blocking,
            training=training,
            method_name=tensors.method_name,
            summary_writer=summary_writer,
        )


def _build_density_estimator(
    *,
    density_estimator_builder: Any,
    tensors: _TrainingTensors,
    split: _TrainValidationSplit,
    training: NPETrainingConfig,
    generator: torch.Generator,
) -> Any:
    device = torch.device(training.device)
    train_theta = _index_rows(tensors.theta, split.train_indices)
    train_x = _index_rows(tensors.x, split.train_indices)
    required_indices = _indices_covering_discrete_values(
        train_theta,
        num_discrete_columns=len(tensors.category_counts),
    )
    init_count = min(
        max(int(training.training_batch_size), int(required_indices.numel())),
        int(train_theta.shape[0]),
    )
    init_indices = _initialization_indices(
        size=int(train_theta.shape[0]),
        count=init_count,
        required_indices=required_indices,
        generator=generator,
        target_device=train_theta.device,
    )

    LOGGER.info(
        "Building %s density estimator with init_count=%s n_simulations=%s "
        "batch_size=%s discrete_categories=%s",
        tensors.method_name,
        init_count,
        int(train_theta.shape[0]),
        training.training_batch_size,
        tensors.category_counts,
    )
    with torch.no_grad():
        density_estimator = density_estimator_builder(
            train_theta[init_indices].to("cpu"),
            train_x[init_indices].to("cpu"),
        )
    return density_estimator.to(device)


def _make_train_validation_split(
    *,
    tensors: _TrainingTensors,
    generator: torch.Generator,
) -> _TrainValidationSplit:
    sample_count = int(tensors.theta.shape[0])
    permutation = torch.randperm(
        sample_count,
        generator=generator,
        device=torch.device("cpu"),
    )
    validation_indices = torch.sort(permutation[:tensors.validation_count]).values
    train_indices = torch.sort(permutation[tensors.validation_count:]).values
    split = _TrainValidationSplit(
        train_indices=train_indices,
        validation_indices=validation_indices,
    )
    _validate_split_discrete_category_coverage(
        tensors=tensors,
        indices=split.train_indices,
        split_name="training split",
    )
    _validate_split_discrete_category_coverage(
        tensors=tensors,
        indices=split.validation_indices,
        split_name="validation split",
    )
    return split


def _make_train_validation_loaders(
    *,
    tensors: _TrainingTensors,
    split: _TrainValidationSplit,
    training: NPETrainingConfig,
    generator: torch.Generator,
) -> tuple[Any, Any]:
    from torch.utils import data

    dataset = data.TensorDataset(tensors.theta, tensors.x)
    train_dataset = data.Subset(dataset, split.train_indices.tolist())
    validation_dataset = data.Subset(dataset, split.validation_indices.tolist())
    loader_kwargs: dict[str, Any] = {
        "num_workers": int(training.num_workers),
        "pin_memory": _resolve_pin_memory(
            training,
            data_device=tensors.theta.device,
            training_device=torch.device(training.device),
        ),
    }
    if training.num_workers > 0:
        loader_kwargs["persistent_workers"] = bool(training.persistent_workers)

    return (
        data.DataLoader(
            train_dataset,
            batch_size=min(
                training.training_batch_size,
                int(split.train_indices.numel()),
            ),
            shuffle=True,
            generator=generator,
            **loader_kwargs,
        ),
        data.DataLoader(
            validation_dataset,
            batch_size=min(
                training.training_batch_size,
                int(split.validation_indices.numel()),
            ),
            shuffle=False,
            **loader_kwargs,
        ),
    )


def _fit_density_estimator(
    *,
    density_estimator: Any,
    loaders: tuple[Any, Any],
    optimizer: Any,
    device: torch.device,
    non_blocking: bool,
    training: NPETrainingConfig,
    method_name: str,
    summary_writer: Any,
) -> tuple[Any, dict[str, Any]]:
    train_loader, validation_loader = loaders
    training_loss_values: list[float] = []
    validation_loss_values: list[float] = []
    best_validation_loss = float("inf")
    best_state_dict = _cpu_state_dict(density_estimator)
    epochs_since_improvement = 0
    epochs_trained = 0
    progress_bar = _epoch_progress_bar(
        training=training,
        method_name=method_name,
    )

    try:
        for epoch in progress_bar:
            density_estimator.train()
            training_loss = _run_epoch(
                density_estimator=density_estimator,
                loader=train_loader,
                device=device,
                optimizer=optimizer,
                clip_max_norm=training.clip_max_norm,
                loader_name="training",
                non_blocking=non_blocking,
            )

            density_estimator.eval()
            validation_loss = _run_epoch(
                density_estimator=density_estimator,
                loader=validation_loader,
                device=device,
                optimizer=None,
                clip_max_norm=None,
                loader_name="validation",
                non_blocking=non_blocking,
            )

            epochs_trained = epoch + 1
            training_loss_values.append(float(training_loss))
            validation_loss_values.append(float(validation_loss))
            summary_writer.add_scalar("training_loss", training_loss, epoch)
            summary_writer.add_scalar("validation_loss", validation_loss, epoch)

            if validation_loss < best_validation_loss:
                best_validation_loss = float(validation_loss)
                best_state_dict = _cpu_state_dict(density_estimator)
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1

            _update_progress_bar(
                progress_bar,
                training_loss=training_loss,
                validation_loss=validation_loss,
                best_validation_loss=best_validation_loss,
            )
            if epochs_since_improvement >= training.stop_after_epochs:
                break
    finally:
        close = getattr(progress_bar, "close", None)
        if close is not None:
            close()

    density_estimator.load_state_dict(best_state_dict)
    density_estimator.zero_grad(set_to_none=True)
    summary_writer.add_scalar(
        "best_validation_loss",
        best_validation_loss,
        epochs_trained,
    )
    LOGGER.info(
        "Finished %s manual SBI training after %s epochs: best_validation_loss=%s",
        method_name,
        epochs_trained,
        best_validation_loss,
    )
    final_training_loss = training_loss_values[-1] if training_loss_values else None
    final_validation_loss = (
        validation_loss_values[-1] if validation_loss_values else None
    )
    summary: dict[str, Any] = {
        "epochs_trained": int(epochs_trained),
        "training_loss": training_loss_values,
        "validation_loss": validation_loss_values,
        "best_validation_loss": float(best_validation_loss),
        "final_training_loss": final_training_loss,
        "final_validation_loss": final_validation_loss,
        "best_validation_log_prob": -float(best_validation_loss),
        "final_validation_log_prob": (
            -final_validation_loss if final_validation_loss is not None else None
        ),
    }
    return density_estimator, summary


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
            if is_training:
                optimizer.zero_grad(set_to_none=True)

            losses = density_estimator.loss(theta_batch, condition=x_batch)
            if not bool(torch.isfinite(losses).all().item()):
                raise ValueError(f"{loader_name} loss contains non-finite values")
            loss = torch.mean(losses)
            if not bool(torch.isfinite(loss).item()):
                raise ValueError(f"{loader_name} loss contains non-finite values")
            loss_sum += float(loss.detach().item()) * batch_size
            sample_count += batch_size

            if is_training:
                loss.backward()
                if clip_max_norm is not None:
                    clip_grad_norm_(
                        density_estimator.parameters(),
                        max_norm=clip_max_norm,
                    )
                optimizer.step()

    if sample_count <= 0:
        raise ValueError(f"{loader_name} loader processed zero samples")
    return loss_sum / sample_count


def _epoch_progress_bar(
    *,
    training: NPETrainingConfig,
    method_name: str,
) -> Any:
    if not training.show_progress_bars:
        return range(training.max_num_epochs)

    from tqdm.auto import tqdm

    return tqdm(
        range(training.max_num_epochs),
        desc=f"{method_name} training",
        unit="epoch",
    )


def _update_progress_bar(
    progress_bar: Any,
    *,
    training_loss: float,
    validation_loss: float,
    best_validation_loss: float,
) -> None:
    set_postfix = getattr(progress_bar, "set_postfix", None)
    if set_postfix is None:
        return
    set_postfix(
        {
            "train_loss": f"{training_loss:.4g}",
            "val_loss": f"{validation_loss:.4g}",
            "best_val": f"{best_validation_loss:.4g}",
        },
        refresh=False,
    )


def _validate_split_discrete_category_coverage(
    *,
    tensors: _TrainingTensors,
    indices: torch.Tensor,
    split_name: str,
) -> None:
    if not tensors.category_counts:
        return

    discrete_columns = _last_discrete_columns(
        tensors.theta,
        num_discrete_columns=len(tensors.category_counts),
    )
    if discrete_columns is None:
        return
    split_values = discrete_columns[indices.detach().to("cpu", dtype=torch.long)]
    for column, category_count in enumerate(tensors.category_counts):
        full_categories = torch.unique(discrete_columns[:, column].to(torch.long))
        observed_categories = torch.unique(split_values[:, column].to(torch.long))
        if (
            int(observed_categories.numel()) != int(category_count)
            or not bool(torch.equal(full_categories, observed_categories))
        ):
            raise ValueError(
                f"{split_name} must contain all discrete theta categories; "
                "change seed, add data, or adjust validation_fraction"
            )


def _index_rows(tensor: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    local_indices = indices.detach().to(dtype=torch.long)
    if tensor.device.type != "cpu":
        local_indices = local_indices.to(tensor.device)
    return torch.index_select(tensor, 0, local_indices)


def _last_discrete_columns(
    theta_tensor: torch.Tensor,
    *,
    num_discrete_columns: int,
) -> torch.Tensor | None:
    if num_discrete_columns <= 0:
        return None
    theta_cpu = theta_tensor.detach().to("cpu")
    discrete_start = int(theta_cpu.shape[1]) - int(num_discrete_columns)
    return theta_cpu[:, discrete_start:]


def _indices_covering_discrete_values(
    theta_tensor: torch.Tensor,
    *,
    num_discrete_columns: int,
) -> torch.Tensor:
    if num_discrete_columns <= 0:
        return torch.empty((0,), dtype=torch.long)

    theta_cpu = theta_tensor.detach().to("cpu")
    discrete_start = int(theta_cpu.shape[1]) - int(num_discrete_columns)
    required: set[int] = set()
    for column in range(discrete_start, int(theta_cpu.shape[1])):
        for value in torch.unique(theta_cpu[:, column]):
            matches = torch.nonzero(theta_cpu[:, column] == value, as_tuple=False)
            required.add(int(matches[0, 0]))
    return torch.as_tensor(sorted(required), dtype=torch.long)


def _initialization_indices(
    *,
    size: int,
    count: int,
    required_indices: torch.Tensor,
    generator: torch.Generator,
    target_device: torch.device,
) -> torch.Tensor:
    selected = [int(index) for index in required_indices.detach().cpu().tolist()]
    selected_set = set(selected)
    for index in torch.randperm(size, generator=generator, device=torch.device("cpu")):
        item = int(index)
        if item in selected_set:
            continue
        selected.append(item)
        selected_set.add(item)
        if len(selected) >= count:
            break
    indices = torch.as_tensor(selected[:count], dtype=torch.long)
    if target_device.type != "cpu":
        indices = indices.to(target_device)
    return indices


def _cpu_state_dict(density_estimator: Any) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in density_estimator.state_dict().items()
    }


def _resolve_pin_memory(
    training: NPETrainingConfig,
    *,
    data_device: torch.device,
    training_device: torch.device,
) -> bool:
    if training.pin_memory is not None:
        return bool(training.pin_memory)
    return data_device.type == "cpu" and training_device.type == "cuda"


def _rng_fork_devices(device: torch.device) -> list[int]:
    if device.type != "cuda" or not torch.cuda.is_available():
        return []
    if device.index is None:
        return [int(torch.cuda.current_device())]
    return [int(device.index)]
