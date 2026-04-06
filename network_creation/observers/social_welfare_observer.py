"""Observer that tracks normalized welfare across simulator steps."""

from __future__ import annotations

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
    """Return mean normalized node utility."""
    n = graph.vcount()
    discounted_distance_sum = float(
        np.power(distance_discount, graph.distances(mode="all")).sum()
    )
    return (
        benefit_to_cost_ratio * (discounted_distance_sum - float(n))
        - float(graph.ecount())
    ) / float(n * (n - 1))


class NormalizedWelfareObserver(EpisodeSeriesStatisticObserver):
    r"""Collect normalized welfare per sampled step.

    The statistic is

    .. math::

       \bar U
       =
       \frac{1}{n}\sum_{i=1}^n
       \left(
       \eta \frac{1}{n-1}\sum_{j \neq i}\delta^{d(i, j)}
       -
       \frac{m_i}{n-1}
       \right)

    where ``benefit_to_cost_ratio = \eta = v/k`` and
    ``distance_discount = \delta``.

    Disconnected pairs contribute zero, self-terms are excluded, and for
    :math:`n \ge 2` the returned value lies in :math:`[-1, 1]`.
    """

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
                "benefit_to_cost_ratio must satisfy 0 <= "
                "benefit_to_cost_ratio <= 1"
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
