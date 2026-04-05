"""Statistic observer skeleton for sampled episode series."""

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


class EpisodeSeriesStatisticObserver(EpisodeStatisticObserver):
    """Observer skeleton for episode statistics that produce one scalar series."""

    label: str
    x_label: str
    y_label: str
    y_min: float | None
    y_max: float | None

    def _reset_episode_state(self) -> None:
        self.n_observations = 0
        self._episode_steps = np.empty(0, dtype=np.int_)
        self._episode_values = np.empty(0, dtype=np.float32)

    def _start_episode(self, event: EpisodeStartEvent) -> None:
        n_slots = self.observation_sampling.fixed_sample_count(event.max_steps)
        self.n_observations = 1
        self._episode_steps = np.empty(n_slots, dtype=np.int_)
        self._episode_values = np.empty(n_slots, dtype=np.float32)
        self._episode_steps[0] = 0
        self._episode_values[0] = self.compute_initial_value(event)

    def _record_step(self, event: StepEvent) -> None:
        if self.observation_sampling.kind == "terminal":
            self._episode_steps[0] = event.step + 1
            self._episode_values[0] = self.compute_step_value(event)
            return
        observation_idx = self.n_observations
        self._episode_steps[observation_idx] = event.step + 1
        self._episode_values[observation_idx] = self.compute_step_value(event)
        self.n_observations += 1

    def _get_episode_values(self, event: EpisodeEndEvent) -> NDArray[np.float32]:
        del event
        return self.values

    def _get_result_steps(self) -> NDArray[np.int_]:
        return self.steps

    @abstractmethod
    def compute_initial_value(self, event: EpisodeStartEvent) -> float:
        """Compute the initial value at step `0`."""

    @abstractmethod
    def compute_step_value(self, event: StepEvent) -> float:
        """Compute one sampled value after a completed step."""

    @property
    def steps(self) -> NDArray[np.int_]:
        return self._episode_steps[: self.n_observations]

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
        del cls
        return (observation_sampling.fixed_sample_count(max_steps),)
