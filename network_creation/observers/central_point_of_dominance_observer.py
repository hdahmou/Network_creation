"""Observer that tracks central point of dominance across simulator steps."""

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


def _compute_central_point_of_dominance(graph: ig.Graph) -> float:
    n_nodes = graph.vcount()

    betweenness = np.asarray(
        graph.betweenness(directed=False),
        dtype=np.float64,
    )
    betweenness *= 2.0 / ((n_nodes - 1) * (n_nodes - 2))

    return float((betweenness.max() - betweenness).sum() / (n_nodes - 1))


class CentralPointOfDominanceObserver(EpisodeSeriesStatisticObserver):
    """Collect central-point-of-dominance values per sampled step."""

    # Hernández, Javier Martın, and Piet Van Mieghem. "Classification of graph metrics." Delft University of Technology: Mekelweg, The Netherlands 1 (2011).

    label = "Central Point Of Dominance"
    x_label = "step"
    y_label = "central point of dominance"
    y_min = 0.0
    y_max = 1.0

    @property
    def central_points_of_dominance(self) -> NDArray[np.float32]:
        return self.values

    def compute_initial_value(self, event: EpisodeStartEvent) -> float:
        return _compute_central_point_of_dominance(event.initial_state.graph)

    def compute_step_value(self, event: StepEvent) -> float:
        return _compute_central_point_of_dominance(event.post_state.graph)
