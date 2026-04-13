"""Observer that tracks normalized welfare across simulator steps."""

from __future__ import annotations

r"""
.. math::
    \begin{aligned}
        \text{Network Structure:} & \quad g = (V, E) \text{ is directed; } \bar{g} \text{ is the undirected skeleton.} \\
        \\
        \text{Individual Payoff:} & \quad \hat{\Pi}_i = \frac{1}{2n-1} \left[ \frac{V}{k} \sum_{j \in N} \delta^{d(i, j, \bar{g})} - \eta_i(g) + (n-1) \right] \in [0, 1] \\
        \\
        \text{Network Value (Sum):} & \quad \mathcal{V}(g) = \sum_{i \in N} \hat{\Pi}_i = \frac{1}{2n-1} \left[ \frac{V}{k} \mathcal{B}(\bar{g}) - |E(g)| + n(n-1) \right] \in [0, n] \\
        \\
        \text{Bounded Aggregate:} & \quad \bar{\Pi}_{agg} = \frac{\mathcal{V}(g)}{n} = \frac{1}{n(2n-1)} \left[ \frac{V}{k} \mathcal{B}(\bar{g}) - |E(g)| + n(n-1) \right] \in [0, 1] \\
        \\
        \text{Where:} & \quad \mathcal{B}(\bar{g}) = \text{Total Benefit} = \sum_{i \in N} \sum_{j \in N} \delta^{d(i,j, \bar{g})} \\
        & \quad |E(g)| = \text{Total Directed Edges} = \sum_{i \in N} \eta_i(g)
   \end{aligned}
"""


import igraph as ig
import numpy as np

from network_creation.core.observability import (
    EpisodeStartEvent,
    ObservationSampling,
    StepEvent,
)
from network_creation.observers.statistics.episode_series import (
    EpisodeSeriesStatisticObserver,
)


def _compute_normalized_welfare(
    graph: ig.Graph,
    benefit_to_cost_ratio: float,
    distance_discount: float,
) -> float:
    n = graph.vcount()
    total_benefit = float(
        np.power(distance_discount, graph.distances(mode="all")).sum()
    )
    return (
        benefit_to_cost_ratio * total_benefit
        - float(graph.ecount())
        + float(n * (n - 1))
    ) / float(n * (2 * n - 1))


class NormalizedWelfareObserver(EpisodeSeriesStatisticObserver):

    label = "Normalized Welfare"
    x_label = "step"
    y_label = "normalized welfare (no units)"
    y_min = None
    y_max = None

    def __init__(
        self,
        observation_sampling: ObservationSampling,
        benefit_to_cost_ratio: float,
        distance_discount: float,
        aggregations: tuple[str, ...],
    ) -> None:
        if not 0.0 <= benefit_to_cost_ratio <= 1.0:
            raise ValueError(
                "benefit_to_cost_ratio must satisfy 0 <= " "benefit_to_cost_ratio <= 1"
            )
        if not 0.0 <= distance_discount < 1.0:
            raise ValueError(
                "distance_discount must satisfy 0 <= distance_discount < 1"
            )

        self.benefit_to_cost_ratio = benefit_to_cost_ratio
        self.distance_discount = distance_discount
        super().__init__(
            observation_sampling=observation_sampling,
            aggregations=aggregations,
        )

    def compute_initial_value(self, event: EpisodeStartEvent) -> float:
        return _compute_normalized_welfare(
            event.initial_state.graph,
            benefit_to_cost_ratio=self.benefit_to_cost_ratio,
            distance_discount=self.distance_discount,
        )

    def compute_step_value(self, event: StepEvent) -> float:
        return _compute_normalized_welfare(
            event.post_state.graph,
            benefit_to_cost_ratio=self.benefit_to_cost_ratio,
            distance_discount=self.distance_discount,
        )
