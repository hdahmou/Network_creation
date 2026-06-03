"""Notebook-friendly helpers for using a saved posterior artifact."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from network_creation.io import read_json

PRIOR_SAMPLES_ATTR = "prior_samples"
PRIOR_SAMPLE_INDICES_ATTR = "prior_sample_indices"
PRIOR_SAMPLE_METADATA_ATTR = "prior_sample_metadata"


@dataclass(frozen=True)
class PosteriorWithPriorSamples:
    """Fallback wrapper for posterior classes that disallow new attributes."""

    posterior: Any
    prior_samples: NDArray[np.float32]
    prior_sample_indices: NDArray[np.int64] | None
    prior_sample_metadata: dict[str, Any]

    def sample(self, *args: Any, **kwargs: Any) -> Any:
        return self.posterior.sample(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.posterior, name)


@dataclass(frozen=True)
class PosteriorArtifact:
    artifact_dir: Path
    posterior: Any
    metadata: dict[str, Any]

    @property
    def observation_order(self) -> list[str]:
        return list(self.metadata["dataset"]["x_names"])

    @property
    def theta_order(self) -> list[str]:
        return list(self.metadata["dataset"]["theta_names"])

    @property
    def continuous_theta_order(self) -> list[str]:
        return list(self.metadata["dataset"]["continuous_theta_names"])

    @property
    def discrete_theta_order(self) -> list[str]:
        return list(self.metadata["dataset"]["discrete_theta_names"])

    @property
    def input_parameters(self) -> list[str]:
        return list(self.metadata["dataset"]["input_parameters"])

    @property
    def continuous_input_parameters(self) -> list[str]:
        return list(self.metadata["dataset"]["continuous_input_parameters"])

    @property
    def discrete_input_parameters(self) -> list[str]:
        return list(self.metadata["dataset"]["discrete_input_parameters"])

    @property
    def tensorboard_log_dir(self) -> Path:
        path = Path(self.metadata["sbi"]["tensorboard_log_dir"])
        if path.is_absolute():
            return path
        return self.artifact_dir / path

    @property
    def prior_samples(self) -> NDArray[np.float32]:
        """Prior-reference samples stored on the posterior during training.

        These are a deterministic subset of the training theta rows.  They should
        be used for notebook overlays instead of reconstructing/sampling a prior
        after training, which is especially error-prone for MNPE categorical
        parameters.
        """

        raw_samples = _posterior_value(self.posterior, PRIOR_SAMPLES_ATTR)
        if raw_samples is None:
            raise AttributeError(
                "This posterior artifact does not contain stored prior_samples. "
                "Retrain it with the current train_npe_from_zarr.py script."
            )
        samples = np.asarray(raw_samples, dtype=np.float32)
        if samples.ndim == 1:
            samples = samples.reshape(1, -1)
        return decode_discrete_theta_samples(samples, self.metadata)

    @property
    def prior_sample_indices(self) -> NDArray[np.int64] | None:
        raw_indices = _posterior_value(self.posterior, PRIOR_SAMPLE_INDICES_ATTR)
        if raw_indices is None:
            return None
        return np.asarray(raw_indices, dtype=np.int64)

    def category_values(self, name: str) -> NDArray[np.int64]:
        return discrete_category_values(self.metadata, name)

    def input_category_values(self, name: str) -> NDArray[np.int64]:
        return discrete_input_category_values(self.metadata, name)

    def is_discrete_theta(self, name: str) -> bool:
        return name in set(self.discrete_theta_order)

    def input_parameter_values_from_x(
        self,
        x: Any,
    ) -> dict[str, NDArray[np.float32]]:
        return decode_input_parameter_values(x, self.metadata)

    def to_device(self, device: str | torch.device) -> torch.device:
        torch_device = torch.device(device)
        _move_posterior_to_device(self.posterior, torch_device)
        return torch_device

    def sample(
        self,
        x: NDArray[np.floating[Any]] | list[float],
        *,
        num_samples: int,
        seed: int | None = None,
        device: str = "cpu",
        **sample_kwargs: Any,
    ) -> tuple[NDArray[np.float32], list[str]]:
        """Draw posterior samples for one observation and decode categorical columns."""

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        torch_device = self.to_device(device)

        dataset_meta = self.metadata["dataset"]
        x_event_shape = tuple(dataset_meta["x_event_shape"])
        x_array = np.asarray(x, dtype=np.float32).reshape(x_event_shape)
        x_tensor = torch.as_tensor(
            x_array[None, ...],
            dtype=torch.float32,
            device=torch_device,
        )

        try:
            samples = self.posterior.sample((num_samples,), x=x_tensor, **sample_kwargs)
        except RuntimeError as error:
            if not _should_retry_with_batched_sampling(
                posterior=self.posterior,
                metadata=self.metadata,
                x_event_shape=x_event_shape,
                error=error,
            ):
                raise
            repeat_factors = (num_samples,) + (1,) * len(x_event_shape)
            repeated_x = x_tensor.repeat(repeat_factors)
            samples = self.posterior.sample_batched(
                (1,),
                x=repeated_x,
                **sample_kwargs,
            ).squeeze(0)
        samples_np = np.asarray(samples.detach().cpu().numpy(), dtype=np.float32)
        if samples_np.ndim == 1:
            samples_np = samples_np.reshape(1, -1)
        return (
            decode_discrete_theta_samples(samples_np, self.metadata),
            self.theta_order,
        )


def load_posterior_artifact(artifact_dir: str | Path) -> PosteriorArtifact:
    artifact_dir = Path(artifact_dir)
    metadata = read_json(artifact_dir / "metadata.json")
    posterior_file = metadata["sbi"]["posterior_file"]
    with (artifact_dir / posterior_file).open("rb") as handle:
        posterior = pickle.load(handle)
    return PosteriorArtifact(
        artifact_dir=artifact_dir,
        posterior=posterior,
        metadata=metadata,
    )


def select_prior_samples(
    theta: Any,
    *,
    max_samples: int,
    seed: int,
) -> tuple[NDArray[np.float32], NDArray[np.int64]]:
    """Keep a deterministic subset of training theta rows as prior samples."""

    theta_array = np.asarray(theta, dtype=np.float32)
    if theta_array.ndim != 2:
        raise ValueError("theta must be a 2D array")
    if max_samples <= 0 or theta_array.shape[0] == 0:
        return (
            np.empty((0, theta_array.shape[1]), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
        )

    n_keep = min(int(max_samples), int(theta_array.shape[0]))
    if n_keep == theta_array.shape[0]:
        indices = np.arange(theta_array.shape[0], dtype=np.int64)
    else:
        rng = np.random.default_rng(seed)
        indices = np.sort(
            rng.choice(theta_array.shape[0], size=n_keep, replace=False).astype(
                np.int64
            )
        )
    return np.asarray(theta_array[indices], dtype=np.float32), indices


def attach_prior_samples_to_posterior(
    posterior: Any,
    *,
    prior_samples: Any,
    prior_sample_indices: Any | None,
    theta_names: list[str] | tuple[str, ...],
    seed: int,
) -> Any:
    """Attach prior-reference samples to the object that will be pickled."""

    samples = np.asarray(prior_samples, dtype=np.float32)
    indices = (
        None
        if prior_sample_indices is None
        else np.asarray(prior_sample_indices, dtype=np.int64)
    )
    sample_metadata = {
        "source": "training_theta_rows",
        "count": int(samples.shape[0]),
        "theta_order": [str(name) for name in theta_names],
        "selection_seed": int(seed),
    }

    if isinstance(posterior, dict):
        posterior[PRIOR_SAMPLES_ATTR] = samples
        posterior[PRIOR_SAMPLE_INDICES_ATTR] = indices
        posterior[PRIOR_SAMPLE_METADATA_ATTR] = sample_metadata
        return posterior

    try:
        setattr(posterior, PRIOR_SAMPLES_ATTR, samples)
        setattr(posterior, PRIOR_SAMPLE_INDICES_ATTR, indices)
        setattr(posterior, PRIOR_SAMPLE_METADATA_ATTR, sample_metadata)
        return posterior
    except (AttributeError, TypeError):
        return PosteriorWithPriorSamples(
            posterior=posterior,
            prior_samples=samples,
            prior_sample_indices=indices,
            prior_sample_metadata=sample_metadata,
        )


def decode_discrete_theta_samples(
    samples: Any,
    metadata: dict[str, Any],
) -> NDArray[np.float32]:
    """Return samples with MNPE categorical columns decoded to stored values.

    SBI categorical heads may return either stored category values or 0-based
    category indices depending on the internal estimator.  This helper accepts
    both and maps the output back to the theta values recorded in metadata.
    """

    decoded = np.asarray(samples, dtype=np.float32).copy()
    if decoded.ndim == 1:
        decoded = decoded.reshape(1, -1)

    dataset_meta = metadata.get("dataset", metadata)
    theta_names = [str(name) for name in dataset_meta["theta_names"]]

    for name in dataset_meta["discrete_theta_names"]:
        name = str(name)
        if name not in theta_names:
            raise ValueError(f"unknown discrete theta parameter {name!r}")
        index = theta_names.index(name)
        categories = discrete_category_values(metadata, name)
        rounded = np.rint(decoded[:, index]).astype(np.int64)

        if categories.size == 0:
            raise ValueError(f"discrete theta parameter {name!r} has no categories")

        unique_values = set(np.unique(rounded).tolist())
        category_set = set(categories.tolist())
        if unique_values.issubset(category_set):
            values = rounded
        elif (
            unique_values
            and min(unique_values) >= 0
            and max(unique_values) < len(categories)
        ):
            values = categories[rounded]
        else:
            nearest = np.argmin(
                np.abs(rounded[:, None] - categories[None, :]),
                axis=1,
            )
            values = categories[nearest]
        decoded[:, index] = values.astype(np.float32)

    return decoded


def discrete_category_values(
    metadata: dict[str, Any],
    name: str,
) -> NDArray[np.int64]:
    dataset_meta = metadata.get("dataset", metadata)
    discrete_names = [
        str(value) for value in dataset_meta["discrete_theta_names"]
    ]
    for discrete_name, values in zip(
        discrete_names,
        dataset_meta["discrete_theta_values"],
        strict=True,
    ):
        if discrete_name == name:
            return np.asarray([int(value) for value in values], dtype=np.int64)
    return np.empty((0,), dtype=np.int64)


def discrete_input_category_values(
    metadata: dict[str, Any],
    name: str,
) -> NDArray[np.int64]:
    dataset_meta = metadata.get("dataset", metadata)
    discrete_names = [
        str(value) for value in dataset_meta["discrete_input_parameters"]
    ]
    for discrete_name, values in zip(
        discrete_names,
        dataset_meta["discrete_input_values"],
        strict=True,
    ):
        if discrete_name == name:
            return np.asarray([int(value) for value in values], dtype=np.int64)
    return np.empty((0,), dtype=np.int64)


def decode_input_parameter_values(
    x: Any,
    metadata: dict[str, Any],
) -> dict[str, NDArray[np.float32]]:
    dataset_meta = metadata.get("dataset", metadata)
    input_parameters = [str(value) for value in dataset_meta["input_parameters"]]
    if not input_parameters:
        return {}

    encoded = _encoded_input_columns(x, dataset_meta)
    discrete_names = [
        str(value) for value in dataset_meta["discrete_input_parameters"]
    ]
    discrete_values_by_name = {
        name: np.asarray([int(item) for item in values], dtype=np.float32)
        for name, values in zip(
            discrete_names,
            dataset_meta["discrete_input_values"],
            strict=True,
        )
    }

    result: dict[str, NDArray[np.float32]] = {}
    column = 0
    for name in input_parameters:
        if name in discrete_values_by_name:
            categories = discrete_values_by_name[name]
            width = int(categories.shape[0])
            block = encoded[..., column : column + width]
            result[name] = _decode_one_hot_input_block(
                block,
                categories=categories,
                name=name,
            )
            column += width
            continue

        result[name] = np.asarray(encoded[..., column], dtype=np.float32)
        column += 1
    return result


def _decode_one_hot_input_block(
    block: NDArray[np.float32],
    *,
    categories: NDArray[np.float32],
    name: str,
) -> NDArray[np.float32]:
    if categories.size == 0:
        raise ValueError(f"discrete input parameter {name!r} has no categories")
    if block.shape[-1] != categories.shape[0]:
        raise ValueError(
            f"encoded input block for {name!r} has width {block.shape[-1]}, "
            f"expected {categories.shape[0]}"
        )
    if not np.isfinite(block).all():
        raise ValueError(f"encoded input block for {name!r} contains non-finite values")

    is_zero = np.isclose(block, 0.0, rtol=0.0, atol=1e-6)
    is_one = np.isclose(block, 1.0, rtol=0.0, atol=1e-6)
    if not np.all(is_zero | is_one):
        raise ValueError(f"encoded input block for {name!r} is not one-hot")
    if not np.all(is_one.sum(axis=-1) == 1):
        raise ValueError(f"encoded input block for {name!r} is not one-hot")

    return categories[np.argmax(is_one, axis=-1)].astype(np.float32)


def _encoded_input_columns(
    x: Any,
    dataset_meta: dict[str, Any],
) -> NDArray[np.float32]:
    array = np.asarray(x, dtype=np.float32)
    observer_count = len(dataset_meta["observer_names"])
    x_event_shape = tuple(int(value) for value in dataset_meta["x_event_shape"])
    if len(x_event_shape) == 1:
        return array[..., observer_count:]
    if len(x_event_shape) == 2:
        if array.ndim < 2:
            raise ValueError("timeseries x must include step and feature axes")
        return array[..., 0, observer_count:]
    raise ValueError(f"unsupported x_event_shape for input decoding: {x_event_shape}")


def category_probabilities(values: Any, categories: Any) -> NDArray[np.float64]:
    categories = np.asarray(categories, dtype=np.int64)
    if categories.size == 0:
        return np.empty((0,), dtype=np.float64)
    rounded = np.rint(np.asarray(values)).astype(np.int64)
    counts = np.asarray(
        [np.sum(rounded == category) for category in categories],
        dtype=np.float64,
    )
    total = float(counts.sum())
    if total:
        counts /= total
    return counts


def _posterior_value(posterior: Any, name: str) -> Any:
    if isinstance(posterior, dict):
        return posterior.get(name)
    return getattr(posterior, name, None)


def _move_posterior_to_device(posterior: Any, device: torch.device) -> None:
    move = getattr(posterior, "to", None)
    if callable(move):
        move(device)


def _should_retry_with_batched_sampling(
    *,
    posterior: Any,
    metadata: dict[str, Any],
    x_event_shape: tuple[int, ...],
    error: RuntimeError,
) -> bool:
    dataset_meta = metadata.get("dataset", metadata)
    has_discrete_theta = bool(dataset_meta["discrete_theta_names"])
    return (
        has_discrete_theta
        and len(x_event_shape) > 1
        and hasattr(posterior, "sample_batched")
        and "doesn't match the broadcast shape" in str(error)
    )
