"""Core lifecycle for cumulative statistic observers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from numpy.typing import NDArray

from network_creation.core.observability import (
    EpisodeEndEvent,
    EpisodeStartEvent,
    SimulationObserver,
    StatisticResult,
    StepEvent,
)
from network_creation.stats.aggregation.base import (
    AggregationOutputs,
    StatisticAccumulator,
)
from network_creation.stats.aggregation.registry import build_accumulators


class EpisodeStatisticObserver(SimulationObserver, ABC):
    """Observer base that turns one episode event stream into one episode result."""

    name: str = "statistic"
    observation_frequency: int
    aggregations: tuple[str, ...]
    n_episodes: int
    _accumulators: tuple[StatisticAccumulator, ...]

    def __init__(
        self,
        observation_frequency: int,
        aggregations: tuple[str, ...],
    ) -> None:
        if observation_frequency < 1:
            raise ValueError("observation_frequency must be positive")
        self.observation_frequency = observation_frequency
        self.aggregations = tuple(aggregations)
        self.reset()

    def reset(self) -> None:
        self.n_episodes = 0
        self._accumulators = build_accumulators(self.aggregations)
        self._reset_episode_state()

    def on_episode_start(self, event: EpisodeStartEvent) -> None:
        self._reset_episode_state()
        self._start_episode(event)

    def on_step_end(self, event: StepEvent) -> None:
        self._record_step(event)

    def on_episode_end(self, event: EpisodeEndEvent) -> None:
        values = np.asarray(self._get_episode_values(event), dtype=np.float32)
        for accumulator in self._accumulators:
            accumulator.update(values)
        self.n_episodes += 1

    def get(self) -> StatisticResult:
        outputs = self._export_outputs()
        return StatisticResult(
            steps=self._get_result_steps(),
            n_episodes=self.n_episodes,
            outputs=outputs,
            sample_shape=self._get_result_sample_shape(),
            dimension_names=self._get_result_dimension_names(),
        )

    def _export_outputs(self) -> AggregationOutputs:
        outputs: AggregationOutputs = {}
        for accumulator in self._accumulators:
            outputs.update(accumulator.export())
        return outputs

    def _reset_episode_state(self) -> None:
        """Clear state used only for the current episode."""
        return None

    def _start_episode(self, event: EpisodeStartEvent) -> None:
        del event

    def _record_step(self, event: StepEvent) -> None:
        """Record one sampled step inside the current episode."""
        del event

    def _get_result_steps(self) -> NDArray[np.int_] | None:
        """Return the x-axis values for exported results, if any."""
        return None

    def _get_result_sample_shape(self) -> tuple[int, ...]:
        """Return the per-sample tensor shape for exported results."""
        return ()

    def _get_result_dimension_names(self) -> tuple[str, ...] | None:
        """Return flattened scalar labels for exported results, if any."""
        return None

    @classmethod
    def fixed_output_shape(
        cls,
        *,
        max_steps: int,
        observation_frequency: int,
    ) -> tuple[int, ...]:
        raise TypeError(f"observer {cls.__name__!r} does not define a fixed output shape")

    @classmethod
    def fixed_output_dtype(cls) -> np.dtype[Any]:
        del cls
        return np.dtype(np.float32)

    @abstractmethod
    def _get_episode_values(self, event: EpisodeEndEvent) -> NDArray[np.float32]:
        """Return the finalized value array for the current episode."""
