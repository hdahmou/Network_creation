"""Observer that tracks graph density across simulator steps."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

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
        result.update({"empty": 0.0, "star": float(2.0 / n_agents)})
        return result
