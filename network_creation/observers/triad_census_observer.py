"""Observer that tracks selected triad census counts across simulator steps."""

from __future__ import annotations

from collections.abc import Mapping
from math import comb

import igraph as ig
import numpy as np
from numpy.typing import NDArray

from network_creation.core.observability import EpisodeStartEvent, StepEvent
from network_creation.observers.statistics.episode_tensor_series import (
    EpisodeSeriesTensorStatisticObserver,
)

TRIAD_TYPE_NAMES = (
    "003",
    "012",
    "021D",
    "021U",
    "021C",
    "030T",
    "030C",
)


def _compute_triad_proportions(
    graph: ig.Graph,
) -> NDArray[np.float32]:
    total_possible_triads = comb(graph.vcount(), 3)
    if total_possible_triads == 0:
        return np.zeros(len(TRIAD_TYPE_NAMES), dtype=np.float32)

    triad_census = graph.triad_census()
    return np.array(
        [
            triad_census[triad_type_name] / total_possible_triads
            for triad_type_name in TRIAD_TYPE_NAMES
        ],
        dtype=np.float32,
    )


class TriadCensusObserver(EpisodeSeriesTensorStatisticObserver):
    """Collect selected triad census proportions per sampled step."""

    label = "Triad Census"
    x_label = "step"
    y_label = "triad proportion"
    y_min = 0.0
    y_max = None
    dimension_names = TRIAD_TYPE_NAMES

    @property
    def triad_type_names(self) -> tuple[str, ...]:
        return TRIAD_TYPE_NAMES

    def compute_initial_value(self, event: EpisodeStartEvent) -> NDArray[np.float32]:
        return _compute_triad_proportions(event.initial_state.graph)

    def compute_step_value(self, event: StepEvent) -> NDArray[np.float32]:
        return _compute_triad_proportions(event.post_state.graph)

    @classmethod
    def reference_values(
        cls,
        context: Mapping[str, object],
        references: Mapping[str, float | None] | None = None,
    ) -> dict[str, float | None]:
        del cls
        result = dict(references or {})
        try:
            n_agents = int(context["n_agents"])
        except (KeyError, TypeError, ValueError):
            return result

        dimension_name = context.get("dimension_name")
        if dimension_name == "003":
            result.update({"empty": 1.0, "star": float((n_agents - 3) / n_agents)})
        elif dimension_name == "021U":
            result.update({"empty": 0.0, "star": float(3.0 / n_agents)})
        elif dimension_name in TRIAD_TYPE_NAMES:
            result.update({"empty": 0.0, "star": 0.0})
        return result
