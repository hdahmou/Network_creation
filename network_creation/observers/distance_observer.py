"""Observer that tracks average distance across simulator steps."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import igraph as ig
import numpy as np

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

    def compute_initial_value(self, event: EpisodeStartEvent) -> float:
        return _compute_average_distance(event.initial_state.graph)

    def compute_step_value(self, event: StepEvent) -> float:
        return _compute_average_distance(event.post_state.graph)

    @classmethod
    def reference_values(
        cls,
        context: Mapping[str, object],
        references: Mapping[str, float | None] | None = None,
    ) -> dict[str, float | None]:
        del cls
        result = dict(references or {})
        try:
            n_agents = int(cast(Any, context["n_agents"]))
        except (KeyError, TypeError, ValueError):
            return result
        result.update({"empty": 0.0, "star": float(2.0 * (n_agents - 1) / n_agents)})
        return result
