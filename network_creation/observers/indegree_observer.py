"""Observer that tracks average indegree across simulator steps."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from network_creation.core.observability import (
    EpisodeStartEvent,
    StepEvent,
)
from network_creation.observers.statistics.episode_series import (
    EpisodeSeriesStatisticObserver,
)


def _compute_average_indegree(ownership_matrix: NDArray[np.int8]) -> float:
    n_nodes = ownership_matrix.shape[0]
    if n_nodes < 2:
        return 0.0

    indegrees = ownership_matrix.sum(axis=0, dtype=np.int_)
    return indegrees.mean()


class IndegreeObserver(EpisodeSeriesStatisticObserver):
    """Collect average indegree values per sampled simulator step."""

    label = "Average In-Degree"
    x_label = "step"
    y_label = "average in-degree"
    y_min = 0.0
    y_max = None

    @property
    def average_indegrees(self) -> NDArray[np.float32]:
        return self.values

    def compute_initial_value(self, event: EpisodeStartEvent) -> float:
        return _compute_average_indegree(event.initial_state.ownership_matrix)

    def compute_step_value(self, event: StepEvent) -> float:
        return _compute_average_indegree(event.post_state.ownership_matrix)
