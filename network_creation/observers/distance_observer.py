"""Observer that tracks average distance across simulator steps."""

from __future__ import annotations

import igraph as ig
import numpy as np
from numpy.typing import NDArray

from network_creation.core.observability import (
    EpisodeStartEvent,
    StepEvent,
)
from network_creation.observers.statistics.episode_series import (
    EpisodeSeriesStatisticObserver,
)


def _compute_average_distance(graph: ig.Graph) -> float:
    giant_component = graph.connected_components(mode="weak").giant()
    if giant_component.vcount() < 2:
        return 0.0

    return giant_component.average_path_length(directed=False)


class DistanceObserver(EpisodeSeriesStatisticObserver):
    """Collect average-distance values per sampled step."""

    label = "Average Distance"
    x_label = "step"
    y_label = "average distance"
    y_min = 0.0
    y_max = None

    @property
    def average_distances(self) -> NDArray[np.float32]:
        return self.values

    def compute_initial_value(self, event: EpisodeStartEvent) -> float:
        return _compute_average_distance(event.initial_state.graph)

    def compute_step_value(self, event: StepEvent) -> float:
        return _compute_average_distance(event.post_state.graph)
