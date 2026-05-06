"""Observer that tracks average indegree across simulator steps."""

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


def _compute_average_indegree(graph: ig.Graph) -> float:
    indegrees = np.asarray(graph.degree(mode="in"), dtype=np.int_)
    return indegrees.mean()


class IndegreeObserver(EpisodeSeriesStatisticObserver):
    """Collect average indegree values per sampled simulator step."""

    label = "Average In-Degree"
    x_label = "step"
    y_label = "average in-degree"
    y_min = 0.0
    y_max = None

    def compute_initial_value(self, event: EpisodeStartEvent) -> float:
        return _compute_average_indegree(event.initial_state.graph)

    def compute_step_value(self, event: StepEvent) -> float:
        return _compute_average_indegree(event.post_state.graph)

    @classmethod
    def reference_values(
        cls,
        context: Mapping[str, object],
        references: Mapping[str, float | None] | None = None,
    ) -> dict[str, float | None]:
        del cls
        result = dict(references or {})
        result["empty"] = 0.0
        try:
            n_agents = int(context["n_agents"])
        except (KeyError, TypeError, ValueError):
            return result
        result["star"] = float((n_agents - 1) / n_agents)
        return result
