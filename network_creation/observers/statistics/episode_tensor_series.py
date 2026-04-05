"""Statistic observer skeleton for sampled episode tensor series."""

from __future__ import annotations

from abc import abstractmethod

import numpy as np
from numpy.typing import NDArray

from network_creation.core.observability import (
    EpisodeEndEvent,
    EpisodeStartEvent,
    ObservationSampling,
    StepEvent,
)

from .episode import EpisodeStatisticObserver


class EpisodeSeriesTensorStatisticObserver(EpisodeStatisticObserver):
    """Observer skeleton for episode statistics that produce one tensor per step."""

    label: str
    x_label: str
    y_label: str
    y_min: float | None
    y_max: float | None
    dimension_names: tuple[str, ...]

    def _reset_episode_state(self) -> None:
        self.n_observations = 0
        self._sample_shape = ()
        self._episode_steps = np.empty(0, dtype=np.int_)
        self._episode_values = np.empty(0, dtype=np.float32)

    def _start_episode(self, event: EpisodeStartEvent) -> None:
        initial_value = self.compute_initial_value(event)
        if initial_value.ndim == 0:
            raise ValueError("tensor-valued observers require non-scalar sampled values")

        n_slots = self.observation_sampling.fixed_sample_count(event.max_steps)
        self.n_observations = 1
        self._sample_shape = initial_value.shape
        self._validate_dimension_names()
        self._episode_steps = np.empty(n_slots, dtype=np.int_)
        self._episode_values = np.empty((n_slots, *self.sample_shape), dtype=np.float32)
        self._episode_steps[0] = 0
        self._episode_values[0] = initial_value

    def _record_step(self, event: StepEvent) -> None:
        step_value = self.compute_step_value(event)
        if step_value.shape != self.sample_shape:
            raise ValueError(
                "all sampled tensor values must share the same shape as the initial value"
            )

        if self.observation_sampling.kind == "terminal":
            self._episode_steps[0] = event.step + 1
            self._episode_values[0] = step_value
            return

        observation_idx = self.n_observations
        self._episode_steps[observation_idx] = event.step + 1
        self._episode_values[observation_idx] = step_value
        self.n_observations += 1

    def _get_episode_values(self, event: EpisodeEndEvent) -> NDArray[np.float32]:
        del event
        return self.values

    def _get_result_steps(self) -> NDArray[np.int_]:
        return self.steps

    def _get_result_sample_shape(self) -> tuple[int, ...]:
        return self.sample_shape

    def _get_result_dimension_names(self) -> tuple[str, ...]:
        return self.dimension_names

    def _validate_dimension_names(self) -> None:
        if not hasattr(self, "dimension_names") or self.dimension_names is None:
            raise ValueError("tensor-valued observers must define dimension_names")

        self.dimension_names = tuple(self.dimension_names)
        if len(self.dimension_names) != self.n_dimensions:
            raise ValueError(
                "dimension_names must match the flattened sampled value size"
            )

    @abstractmethod
    def compute_initial_value(self, event: EpisodeStartEvent) -> NDArray[np.float32]:
        """Compute the initial tensor value at step `0`."""

    @abstractmethod
    def compute_step_value(self, event: StepEvent) -> NDArray[np.float32]:
        """Compute one sampled tensor value after a completed step."""

    @property
    def steps(self) -> NDArray[np.int_]:
        return self._episode_steps[: self.n_observations]

    @property
    def sample_shape(self) -> tuple[int, ...]:
        return self._sample_shape

    @property
    def n_dimensions(self) -> int:
        return int(np.prod(self.sample_shape, dtype=np.int_))

    @property
    def values(self) -> NDArray[np.float32]:
        return self._episode_values[: self.n_observations]

    @classmethod
    def fixed_output_shape(
        cls,
        *,
        max_steps: int,
        observation_sampling: ObservationSampling,
    ) -> tuple[int, ...]:
        if not getattr(cls, "dimension_names", ()):
            raise ValueError(f"observer {cls.__name__!r} must define dimension_names")
        return (
            observation_sampling.fixed_sample_count(max_steps),
            len(tuple(cls.dimension_names)),
        )
