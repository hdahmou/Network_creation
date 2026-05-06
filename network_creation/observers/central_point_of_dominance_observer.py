"""Observer that tracks central point of dominance across simulator steps."""

from __future__ import annotations

from collections.abc import Mapping

import igraph as ig
import numpy as np

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

    def compute_initial_value(self, event: EpisodeStartEvent) -> float:
        return _compute_central_point_of_dominance(event.initial_state.graph)

    def compute_step_value(self, event: StepEvent) -> float:
        return _compute_central_point_of_dominance(event.post_state.graph)

    @classmethod
    def reference_values(
        cls,
        context: Mapping[str, object],
        references: Mapping[str, float | None] | None = None,
    ) -> dict[str, float | None]:
        del cls, context
        result = dict(references or {})
        result.update({"empty": 0.0, "star": 1.0})
        return result
