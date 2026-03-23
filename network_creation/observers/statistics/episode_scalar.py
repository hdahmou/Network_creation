"""Statistic observer skeleton for final-state-only metrics."""

from __future__ import annotations

from abc import abstractmethod

import numpy as np
from numpy.typing import NDArray

from network_creation.core.observability import EpisodeEndEvent

from .episode import EpisodeStatisticObserver


class EpisodeScalarStatisticObserver(EpisodeStatisticObserver):
    """Observer skeleton for episode statistics that produce one scalar."""

    def __init__(
        self,
        *,
        aggregations: tuple[str, ...],
    ) -> None:
        super().__init__(observation_frequency=1, aggregations=aggregations)

    def _get_episode_values(self, event: EpisodeEndEvent) -> NDArray[np.float32]:
        return np.array([self.compute_terminal_value(event)], dtype=np.float32)

    @abstractmethod
    def compute_terminal_value(self, event: EpisodeEndEvent) -> float:
        """Compute the final episode value."""
