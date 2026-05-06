"""Observer that tracks global harmonic closeness centralization."""

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


def _compute_global_closeness_centrality(graph: ig.Graph) -> float:
    n_nodes = graph.vcount()
    if n_nodes < 3:
        return 0.0

    harmonic_scores = np.asarray(
        graph.harmonic_centrality(mode="all", normalized=True),
        dtype=np.float64,
    )
    max_score = float(harmonic_scores.max(initial=0.0))
    numerator = float((max_score - harmonic_scores).sum())
    denominator = (n_nodes - 2) * (n_nodes - 1) / (2 * n_nodes - 2)
    return 0.0 if denominator == 0.0 else numerator / denominator


class GlobalClosenessCentralityObserver(EpisodeSeriesStatisticObserver):
    """Collect harmonic-closeness centralization values per sampled step."""

    label = "Global Closeness Centrality"
    x_label = "step"
    y_label = "global closeness centrality"
    y_min = 0.0
    y_max = 1.0

    def compute_initial_value(self, event: EpisodeStartEvent) -> float:
        return _compute_global_closeness_centrality(event.initial_state.graph)

    def compute_step_value(self, event: StepEvent) -> float:
        return _compute_global_closeness_centrality(event.post_state.graph)

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
