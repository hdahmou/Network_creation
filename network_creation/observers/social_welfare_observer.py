"""Observer that tracks social welfare across simulator steps."""

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


def _compute_social_welfare(
    graph: ig.Graph,
    ownership_matrix: NDArray[np.int8],
    link_cost: float,
    decay: float,
    per_link_benefit: float,
) -> float:
    n_paid_links = float(ownership_matrix.sum(axis=None))
    benefit = per_link_benefit * float(np.power(decay, graph.distances()).sum())
    cost = link_cost * n_paid_links
    return benefit - cost


class SocialWelfareObserver(EpisodeSeriesStatisticObserver):
    r"""Collect social welfare values per sampled step.

    .. math::

       W = b \sum_{i=1}^{n} \sum_{j=1}^{n} \delta^{d(i, j)}
       - c \sum_{i=1}^{n} \sum_{j=1}^{n} A_{ij}

    If ``max_welfare`` and ``baseline_welfare`` are provided, the observer
    returns the min-max normalized value

    .. math::

       \widetilde{W}
       = \frac{W - W_{\mathrm{baseline}}}
       {W_{\max} - W_{\mathrm{baseline}}}

    where :math:`b` is ``per_link_benefit``, :math:`c` is ``link_cost``,
    :math:`\delta` is ``decay``, :math:`d(i, j)` is the undirected geodesic
    distance between nodes :math:`i` and :math:`j`, and :math:`A` is the
    ownership matrix. Self-terms are included because :math:`d(i, i) = 0`, and
    disconnected pairs contribute zero because :math:`\delta^{\infty} = 0` for
    :math:`0 \leq \delta < 1`. Each directed paid link in :math:`A` contributes
    one full unit of link cost.
    """

    label = "Social Welfare"
    x_label = "step"
    y_label = "social welfare"
    y_min = None
    y_max = None

    def __init__(
        self,
        observation_frequency: int,
        link_cost: float,
        decay: float,
        per_link_benefit: float,
        aggregations: tuple[str, ...],
        max_welfare: float | None = None,
        baseline_welfare: float | None = None,
    ) -> None:
        if not 0.0 <= decay < 1.0:
            raise ValueError("decay must satisfy 0 <= decay < 1")
        if (max_welfare is None) != (baseline_welfare is None):
            raise ValueError(
                "max_welfare and baseline_welfare must be provided together"
            )
        if (
            max_welfare is not None
            and baseline_welfare is not None
            and max_welfare <= baseline_welfare
        ):
            raise ValueError("max_welfare must be greater than baseline_welfare")

        self.link_cost = link_cost
        self.decay = decay
        self.per_link_benefit = per_link_benefit
        self.max_welfare = max_welfare
        self.baseline_welfare = baseline_welfare
        super().__init__(
            observation_frequency=observation_frequency,
            aggregations=aggregations,
        )

    @property
    def social_welfares(self) -> NDArray[np.float32]:
        return self.values

    def _normalize_welfare(self, welfare: float) -> float:
        if self.max_welfare is None or self.baseline_welfare is None:
            return welfare
        return (welfare - self.baseline_welfare) / (
            self.max_welfare - self.baseline_welfare
        )

    def compute_initial_value(self, event: EpisodeStartEvent) -> float:
        return self._normalize_welfare(
            _compute_social_welfare(
                event.initial_state.graph,
                event.initial_state.ownership_matrix,
                link_cost=self.link_cost,
                decay=self.decay,
                per_link_benefit=self.per_link_benefit,
            )
        )

    def compute_step_value(self, event: StepEvent) -> float:
        return self._normalize_welfare(
            _compute_social_welfare(
                event.post_state.graph,
                event.post_state.ownership_matrix,
                link_cost=self.link_cost,
                decay=self.decay,
                per_link_benefit=self.per_link_benefit,
            )
        )
