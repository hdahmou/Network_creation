"""Observer for the sponsorship share paid by the current graph hubs."""

from __future__ import annotations

from collections.abc import Mapping

import igraph as ig
import numpy as np

from network_creation.core.observability import EpisodeStartEvent, StepEvent
from network_creation.observers.statistics.episode_series import (
    EpisodeSeriesStatisticObserver,
)


def compute_hub_sponsorship_share(graph: ig.Graph) -> float:
    """Return the share of hub-incident sponsorship arcs paid by current hubs."""

    if graph.vcount() == 0:
        return 0.5

    degrees = np.asarray(
        [
            len(set(graph.neighbors(node_id, mode="all")))
            for node_id in range(graph.vcount())
        ],
        dtype=np.int64,
    )
    hub_ids = np.flatnonzero(degrees == degrees.max()).tolist()
    hub_outdegree = np.asarray(graph.degree(hub_ids, mode="out"), dtype=np.float64)
    hub_indegree = np.asarray(graph.degree(hub_ids, mode="in"), dtype=np.float64)
    incident_sponsorships = float((hub_outdegree + hub_indegree).sum())
    if incident_sponsorships == 0.0:
        return 0.5
    return float(hub_outdegree.sum() / incident_sponsorships)


class HubSponsorshipObserver(EpisodeSeriesStatisticObserver):
    """Track the share of hub-incident sponsorship paid by current hubs."""

    label = "Hub Sponsorship"
    x_label = "step"
    y_label = "hub sponsorship share"
    y_min = 0.0
    y_max = 1.0

    def compute_initial_value(self, event: EpisodeStartEvent) -> float:
        return compute_hub_sponsorship_share(event.initial_state.graph)

    def compute_step_value(self, event: StepEvent) -> float:
        return compute_hub_sponsorship_share(event.post_state.graph)

    @classmethod
    def reference_values(
        cls,
        context: Mapping[str, object],
        references: Mapping[str, float | None] | None = None,
    ) -> dict[str, float | None]:
        del cls, context
        result = dict(references or {})
        result.update(
            {
                "Periphery_sponsored_star": 0.0,
                "Empty/Balanced": 0.5,
                "Center_sponsored_star": 1.0,
            }
        )
        return result


__all__ = ["HubSponsorshipObserver", "compute_hub_sponsorship_share"]
