"""User-facing helpers for saved model-inversion posterior artifacts."""

from __future__ import annotations

import pickle
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from .artifact_metadata import ArtifactDatasetMetadata, ArtifactMetadata
from .encoding import ObservationEncoder


@dataclass(frozen=True)
class PosteriorSchema:
    observation: ObservationEncoder
    continuous_theta_names: tuple[str, ...]
    discrete_theta_names: tuple[str, ...]
    discrete_theta_values: dict[str, tuple[int, ...]]

    @classmethod
    def from_metadata(cls, dataset: ArtifactDatasetMetadata) -> "PosteriorSchema":
        return cls(
            observation=ObservationEncoder.from_metadata(dataset),
            continuous_theta_names=dataset.continuous_theta_names,
            discrete_theta_names=dataset.discrete_theta_names,
            discrete_theta_values=dataset.discrete_theta_values_by_name,
        )

    @property
    def theta_order(self) -> tuple[str, ...]:
        return self.continuous_theta_names + self.discrete_theta_names


@dataclass(frozen=True)
class PosteriorArtifact:
    artifact_dir: Path
    posterior: Any
    metadata: ArtifactMetadata
    prior_samples: NDArray[np.float32]
    prior_sample_indices: NDArray[np.int64]
    schema: PosteriorSchema = field(init=False, repr=False)

    def __post_init__(self) -> None:
        schema = PosteriorSchema.from_metadata(self.metadata.dataset)
        samples = np.asarray(self.prior_samples, dtype=np.float32)
        indices = np.asarray(self.prior_sample_indices, dtype=np.int64)
        if samples.ndim != 2 or samples.shape[1] != len(schema.theta_order):
            raise ValueError("prior samples must have shape (sample, theta)")
        if indices.shape != (samples.shape[0],):
            raise ValueError("prior sample indices must match prior samples")
        if self.metadata.sbi.prior_samples.count != samples.shape[0]:
            raise ValueError("prior sample count does not match artifact metadata")
        object.__setattr__(self, "schema", schema)
        object.__setattr__(self, "prior_samples", samples)
        object.__setattr__(self, "prior_sample_indices", indices)

    @property
    def observation_encoder(self) -> ObservationEncoder:
        return self.schema.observation

    @property
    def theta_order(self) -> list[str]:
        return list(self.schema.theta_order)

    @property
    def continuous_theta_names(self) -> list[str]:
        return list(self.schema.continuous_theta_names)

    @property
    def discrete_theta_names(self) -> list[str]:
        return list(self.schema.discrete_theta_names)

    @property
    def observed_output_order(self) -> list[str]:
        return list(self.schema.observation.observed_outputs)

    @property
    def x_components(self) -> list[str]:
        return list(self.schema.observation.x_components)

    @property
    def x_event_shape(self) -> tuple[int, ...]:
        return self.schema.observation.x_event_shape

    @property
    def continuous_input_names(self) -> list[str]:
        return list(self.schema.observation.continuous_input_names)

    @property
    def discrete_input_names(self) -> list[str]:
        return list(self.schema.observation.discrete_input_names)

    @property
    def tensorboard_dir(self) -> Path:
        return self.artifact_dir / self.metadata.sbi.tensorboard_dir

    def category_values(self, theta_name: str) -> NDArray[np.int64]:
        try:
            values = self.schema.discrete_theta_values[str(theta_name)]
        except KeyError as error:
            raise ValueError(f"unknown discrete theta parameter: {theta_name}") from error
        return np.asarray(values, dtype=np.int64)

    def input_category_values(self, name: str) -> NDArray[np.int64]:
        try:
            values = self.schema.observation.discrete_input_values[str(name)]
        except KeyError as error:
            raise ValueError(f"unknown discrete input parameter: {name}") from error
        return np.asarray(values, dtype=np.int64)

    def is_discrete_theta(self, name: str) -> bool:
        return str(name) in self.schema.discrete_theta_names

    def input_values_from_x(self, x: Any) -> dict[str, NDArray[np.float32]]:
        return decode_input_parameter_values(x, self.metadata.dataset)

    def to_device(self, device: str | torch.device) -> torch.device:
        torch_device = torch.device(device)
        _move_posterior_to_device(self.posterior, torch_device)
        return torch_device

    def sample(
        self,
        x: Any,
        *,
        num_samples: int,
        seed: int | None = None,
        device: str = "cpu",
        **sample_kwargs: Any,
    ) -> tuple[NDArray[np.float32], list[str]]:
        samples = sample_posterior_observation(
            self.posterior,
            x,
            num_samples=num_samples,
            x_event_shape=self.x_event_shape,
            theta_names=self.theta_order,
            discrete_theta_names=self.discrete_theta_names,
            discrete_theta_values=tuple(
                self.schema.discrete_theta_values[name]
                for name in self.schema.discrete_theta_names
            ),
            seed=seed,
            device=device,
            **sample_kwargs,
        )
        return samples, self.theta_order

    def sample_cases(
        self,
        x: Any,
        *,
        num_samples: int,
        seed: int | None = None,
        device: str = "cpu",
        max_samples_per_call: int | None = None,
        **sample_kwargs: Any,
    ) -> NDArray[np.float32]:
        return sample_posterior_cases(
            self,
            x,
            num_samples=num_samples,
            seed=seed,
            device=device,
            max_samples_per_call=max_samples_per_call,
            **sample_kwargs,
        )

    def build_x(
        self,
        observed_outputs: Mapping[str, Any],
        input_values: Mapping[str, Any] | None = None,
        *,
        n_agents: int | None = None,
        n_actors: int | None = None,
    ) -> NDArray[np.float32]:
        return self.observation_encoder.encode(
            observed_outputs,
            input_values,
            n_agents=n_agents,
            n_actors=n_actors,
        )


def load_posterior_artifact(artifact_dir: str | Path) -> PosteriorArtifact:
    artifact_dir = Path(artifact_dir).expanduser().resolve()
    metadata = ArtifactMetadata.model_validate_json(
        (artifact_dir / "metadata.json").read_text(encoding="utf-8")
    )
    with (artifact_dir / metadata.sbi.posterior_file).open("rb") as handle:
        posterior = pickle.load(handle)
    with np.load(artifact_dir / metadata.sbi.prior_samples_file) as data:
        prior_samples = np.asarray(data["theta"], dtype=np.float32)
        prior_sample_indices = np.asarray(data["source_sample_ids"], dtype=np.int64)
        continuous_names = tuple(
            str(name) for name in data["continuous_theta_names"].tolist()
        )
        discrete_names = tuple(
            str(name) for name in data["discrete_theta_names"].tolist()
        )
    if continuous_names != metadata.dataset.continuous_theta_names:
        raise ValueError("prior sample continuous theta names do not match metadata")
    if discrete_names != metadata.dataset.discrete_theta_names:
        raise ValueError("prior sample discrete theta names do not match metadata")
    return PosteriorArtifact(
        artifact_dir=artifact_dir,
        posterior=posterior,
        metadata=metadata,
        prior_samples=prior_samples,
        prior_sample_indices=prior_sample_indices,
    )


def sample_posterior_observation(
    posterior: Any,
    observation: Any,
    *,
    num_samples: int,
    x_event_shape: Sequence[int],
    theta_names: Sequence[str],
    discrete_theta_names: Sequence[str] = (),
    discrete_theta_values: Sequence[Sequence[int]] = (),
    seed: int | None = None,
    device: str = "cpu",
    **sample_kwargs: Any,
) -> NDArray[np.float32]:
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    torch_device = torch.device(device)
    _move_posterior_to_device(posterior, torch_device)
    event_shape = tuple(int(value) for value in x_event_shape)
    x_array = np.asarray(observation, dtype=np.float32)
    if tuple(x_array.shape) != event_shape:
        raise ValueError("observation must match x_event_shape")
    x_tensor = torch.as_tensor(
        x_array[None, ...],
        dtype=torch.float32,
        device=torch_device,
    )
    values_by_name = dict(
        zip(
            (str(name) for name in discrete_theta_names),
            discrete_theta_values,
            strict=True,
        )
    )

    with torch.random.fork_rng(devices=_rng_fork_devices(torch_device)):
        if seed is not None:
            torch.manual_seed(seed)
        try:
            samples = posterior.sample((num_samples,), x=x_tensor, **sample_kwargs)
        except RuntimeError as error:
            if not _should_retry_with_batched_sampling(
                posterior=posterior,
                has_discrete_theta=bool(values_by_name),
                x_event_shape=event_shape,
                error=error,
            ):
                raise
            repeat_factors = (num_samples,) + (1,) * len(event_shape)
            repeated_x = x_tensor.repeat(repeat_factors)
            samples = posterior.sample_batched(
                (1,),
                x=repeated_x,
                **sample_kwargs,
            ).squeeze(0)

    if hasattr(samples, "detach"):
        samples_np = np.asarray(samples.detach().cpu().numpy(), dtype=np.float32)
    else:
        samples_np = np.asarray(samples, dtype=np.float32)
    if samples_np.ndim == 1:
        samples_np = samples_np.reshape(1, -1)
    if samples_np.ndim != 2:
        raise ValueError("posterior samples must have shape (sample, theta)")
    if samples_np.shape != (num_samples, len(theta_names)):
        raise ValueError(
            "posterior sample shape does not match num_samples and theta_names"
        )
    return validate_discrete_theta_samples(
        samples_np,
        theta_names=theta_names,
        discrete_theta_values=values_by_name,
    )


def sample_posterior_cases(
    artifact: PosteriorArtifact,
    x: Any,
    *,
    num_samples: int,
    seed: int | None = None,
    device: str = "cpu",
    max_samples_per_call: int | None = None,
    **sample_kwargs: Any,
) -> NDArray[np.float32]:
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if max_samples_per_call is not None and max_samples_per_call <= 0:
        raise ValueError("max_samples_per_call must be positive")

    observations = _case_observations(x, x_event_shape=artifact.x_event_shape)
    chunk_size = int(max_samples_per_call or num_samples)
    results: list[NDArray[np.float32]] = []
    call_index = 0
    for observation in observations:
        chunks: list[NDArray[np.float32]] = []
        remaining = int(num_samples)
        while remaining > 0:
            this_count = min(chunk_size, remaining)
            call_seed = None if seed is None else int(seed) + call_index
            samples = sample_posterior_observation(
                artifact.posterior,
                observation,
                num_samples=this_count,
                x_event_shape=artifact.x_event_shape,
                theta_names=artifact.theta_order,
                discrete_theta_names=artifact.discrete_theta_names,
                discrete_theta_values=tuple(
                    artifact.schema.discrete_theta_values[name]
                    for name in artifact.schema.discrete_theta_names
                ),
                seed=call_seed,
                device=device,
                **sample_kwargs,
            )
            chunks.append(np.asarray(samples, dtype=np.float32))
            remaining -= int(samples.shape[0])
            call_index += 1
        results.append(np.concatenate(chunks, axis=0)[:num_samples])
    return np.stack(results, axis=0).astype(np.float32)


def validate_discrete_theta_samples(
    samples: Any,
    *,
    theta_names: Sequence[str],
    discrete_theta_values: Mapping[str, Sequence[int]],
) -> NDArray[np.float32]:
    decoded = np.asarray(samples, dtype=np.float32).copy()
    if decoded.ndim == 1:
        decoded = decoded.reshape(1, -1)
    if decoded.ndim != 2 or decoded.shape[1] != len(theta_names):
        raise ValueError("theta samples must have shape (sample, theta)")
    names = [str(name) for name in theta_names]
    for name, raw_categories in discrete_theta_values.items():
        name = str(name)
        try:
            index = names.index(name)
        except ValueError as error:
            raise ValueError(
                f"discrete theta parameter {name!r} is not in theta_names"
            ) from error
        categories = np.asarray(raw_categories, dtype=np.int64)
        rounded = np.rint(decoded[:, index]).astype(np.int64)
        if not np.isfinite(decoded[:, index]).all() or not np.allclose(
            decoded[:, index],
            rounded,
            rtol=0.0,
            atol=1e-6,
        ):
            raise ValueError(f"discrete theta samples for {name!r} must be integers")
        if not np.isin(rounded, categories).all():
            raise ValueError(
                f"discrete theta samples for {name!r} contain unknown categories"
            )
        decoded[:, index] = rounded.astype(np.float32)
    return decoded


def decode_input_parameter_values(
    x: Any,
    dataset: ArtifactDatasetMetadata,
) -> dict[str, NDArray[np.float32]]:
    continuous_names = dataset.continuous_input_names
    discrete_names = dataset.discrete_input_names
    if not continuous_names and not discrete_names:
        return {}
    encoded = _encoded_input_columns(x, dataset)
    result: dict[str, NDArray[np.float32]] = {}
    column = 0
    for name in continuous_names:
        result[name] = np.asarray(encoded[..., column], dtype=np.float32)
        column += 1
    values_by_name = dataset.discrete_input_values_by_name
    for name in discrete_names:
        categories = np.asarray(values_by_name[name], dtype=np.float32)
        width = int(categories.shape[0])
        result[name] = _decode_one_hot_input_block(
            encoded[..., column : column + width],
            categories=categories,
            name=name,
        )
        column += width
    return result


def category_probabilities(values: Any, categories: Any) -> NDArray[np.float64]:
    categories = np.asarray(categories, dtype=np.int64)
    if categories.size == 0:
        return np.empty((0,), dtype=np.float64)
    raw_values = np.asarray(values)
    rounded = np.rint(raw_values).astype(np.int64)
    if not np.isfinite(raw_values).all() or not np.allclose(
        raw_values,
        rounded,
        rtol=0.0,
        atol=1e-6,
    ):
        raise ValueError("categorical samples must be finite integers")
    if not np.isin(rounded, categories).all():
        raise ValueError("categorical samples contain unknown categories")
    counts = np.asarray(
        [np.sum(rounded == category) for category in categories],
        dtype=np.float64,
    )
    total = float(counts.sum())
    if total:
        counts /= total
    return counts


def _case_observations(x: Any, *, x_event_shape: Sequence[int]) -> NDArray[np.float32]:
    event_shape = tuple(int(value) for value in x_event_shape)
    array = np.asarray(x, dtype=np.float32)
    if tuple(array.shape) == event_shape:
        return array.reshape((1,) + event_shape)
    if (
        array.ndim == len(event_shape) + 1
        and tuple(array.shape[1:]) == event_shape
    ):
        return array
    raise ValueError("x must have shape matching one observation or a leading case axis")


def _encoded_input_columns(
    x: Any,
    dataset: ArtifactDatasetMetadata,
) -> NDArray[np.float32]:
    array = np.asarray(x, dtype=np.float32)
    observed_count = len(dataset.observed_outputs)
    x_event_shape = dataset.x_event_shape
    if len(x_event_shape) == 1:
        return array[..., observed_count:]
    if len(x_event_shape) == 2:
        if array.ndim < 2:
            raise ValueError("timeseries x must include step and feature axes")
        return array[..., 0, observed_count:]
    raise ValueError(f"unsupported x_event_shape for input decoding: {x_event_shape}")


def _decode_one_hot_input_block(
    block: NDArray[np.float32],
    *,
    categories: NDArray[np.float32],
    name: str,
) -> NDArray[np.float32]:
    if categories.size == 0:
        raise ValueError(f"discrete input parameter {name!r} has no categories")
    is_zero = np.isclose(block, 0.0, rtol=0.0, atol=1e-6)
    is_one = np.isclose(block, 1.0, rtol=0.0, atol=1e-6)
    if not np.all(is_zero | is_one) or not np.all(is_one.sum(axis=-1) == 1):
        raise ValueError(f"encoded input block for {name!r} is not one-hot")
    return categories[np.argmax(is_one, axis=-1)].astype(np.float32)


def _move_posterior_to_device(posterior: Any, device: torch.device) -> None:
    move = getattr(posterior, "to", None)
    if callable(move):
        move(device)


def _should_retry_with_batched_sampling(
    *,
    posterior: Any,
    has_discrete_theta: bool,
    x_event_shape: tuple[int, ...],
    error: RuntimeError,
) -> bool:
    return (
        has_discrete_theta
        and len(x_event_shape) > 1
        and hasattr(posterior, "sample_batched")
        and "doesn't match the broadcast shape" in str(error)
    )


def _rng_fork_devices(device: torch.device) -> list[int]:
    if device.type != "cuda":
        return []
    return [0 if device.index is None else int(device.index)]
