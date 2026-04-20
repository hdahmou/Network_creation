"""Observers for centered social welfare in the linking game."""

from __future__ import annotations

r"""
The standard welfare object in the Bala-Goyal style linking game is the sum of
agents' payoffs. For reporting across network sizes, the project uses the mean
centered payoff per agent

.. math::
    u_i(g)
    = \lambda \sum_{j \ne i} \delta^{d(i,j;\bar g)} - \eta_i(g),

and therefore

.. math::
    \bar W(g)
    = \frac{1}{n} \sum_{i=1}^n u_i(g)
    = \frac{1}{n}
      \left[
        \lambda \sum_{i \ne j} \delta^{d(i,j;\bar g)} - |E(g)|
      \right].

This convention removes the network-invariant self term, so the empty network
has welfare exactly zero.

For an :math:`n`-node star,

.. math::
    \bar W(\text{star})
    = \frac{1}{n}
      \left[
        \lambda \left(2(n-1)\delta + (n-1)(n-2)\delta^2\right) - (n-1)
      \right],

while for the empty network,

.. math::
    \bar W(\emptyset) = 0.
"""

from network_creation.core.observability import (
    EpisodeStartEvent,
    ObservationSampling,
    StepEvent,
)
from network_creation.observers.statistics.episode_series import (
    EpisodeSeriesStatisticObserver,
)
from network_creation.stats.er_payoff import expected_aggregate_payoff
from network_creation.stats.payoff import centered_mean_welfare, centered_mean_welfare_star


def _compute_centered_mean_welfare(
    graph,
    benefit_to_cost_ratio: float,
    distance_discount: float,
) -> float:
    return centered_mean_welfare(
        graph,
        benefit_to_cost_ratio=benefit_to_cost_ratio,
        distance_discount=distance_discount,
    )


class CenteredMeanWelfareObserver(EpisodeSeriesStatisticObserver):
    """Track mean centered welfare per agent over simulator steps."""

    label = "Centered Mean Welfare"
    x_label = "step"
    y_label = "mean centered welfare (payoff units)"
    y_min = None
    y_max = None

    def __init__(
        self,
        observation_sampling: ObservationSampling,
        benefit_to_cost_ratio: float,
        distance_discount: float,
        aggregations: tuple[str, ...],
    ) -> None:
        if benefit_to_cost_ratio <= 0.0:
            raise ValueError(
                "benefit_to_cost_ratio must satisfy benefit_to_cost_ratio > 0"
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

    @staticmethod
    def reference_values(
        n: int,
        distance_discount: float,
        benefit_to_cost_ratio: float,
    ) -> dict[str, float]:
        if n < 1:
            raise ValueError("n must satisfy n >= 1")
        if benefit_to_cost_ratio <= 0.0:
            raise ValueError(
                "benefit_to_cost_ratio must satisfy benefit_to_cost_ratio > 0"
            )
        if not 0.0 <= distance_discount < 1.0:
            raise ValueError(
                "distance_discount must satisfy 0 <= distance_discount < 1"
            )

        star = centered_mean_welfare_star(
            n,
            benefit_to_cost_ratio=benefit_to_cost_ratio,
            distance_discount=distance_discount,
        )
        empty = 0.0
        random_reference = expected_aggregate_payoff(
            n,
            benefit_to_cost_ratio,
            p=None,
            delta=distance_discount,
        )
        random = (
            float(random_reference[0])
            if isinstance(random_reference, tuple)
            else float(random_reference)
        )
        return {"star": star, "empty": empty, "random": random}

    def compute_initial_value(self, event: EpisodeStartEvent) -> float:
        return _compute_centered_mean_welfare(
            event.initial_state.graph,
            benefit_to_cost_ratio=self.benefit_to_cost_ratio,
            distance_discount=self.distance_discount,
        )

    def compute_step_value(self, event: StepEvent) -> float:
        return _compute_centered_mean_welfare(
            event.post_state.graph,
            benefit_to_cost_ratio=self.benefit_to_cost_ratio,
            distance_discount=self.distance_discount,
        )