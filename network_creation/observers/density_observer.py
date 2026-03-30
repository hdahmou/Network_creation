"""Observer that tracks graph density across simulator steps."""

from __future__ import annotations

import igraph as ig
import numpy as np

from network_creation.core.observability import EpisodeStartEvent, StepEvent
from network_creation.observers.statistics.episode_series import (
    EpisodeSeriesStatisticObserver,
)


def compute_density(graph: ig.Graph) -> float:
    return float(graph.as_undirected(mode="collapse").density(loops=False))


class DensityObserver(EpisodeSeriesStatisticObserver):
    """Collect density values per episode and aggregate them across episodes."""

    label = "Density"
    x_label = "step"
    y_label = "density"
    y_min = 0.0
    y_max = None

    def compute_initial_value(self, event: EpisodeStartEvent) -> float:
        return compute_density(event.initial_state.graph)

    def compute_step_value(self, event: StepEvent) -> float:
        return compute_density(event.post_state.graph)
