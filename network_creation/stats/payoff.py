"""Payoff helpers for the Bala-Goyal style linking game."""

from __future__ import annotations

import igraph as ig
import numpy as np


def centered_total_benefit(
    graph: ig.Graph,
    distance_discount: float,
) -> float:
    r"""Return total discounted access excluding self terms.

    This equals

    .. math:: \sum_i \sum_{j \ne i} \delta^{d(i,j;\bar g)}.
    """

    distances = np.asarray(graph.distances(mode="all"), dtype=np.float32)
    discounted = np.empty_like(distances, dtype=np.float32)
    np.power(distance_discount, distances, out=discounted)
    return float(discounted.sum(dtype=np.float64) - float(graph.vcount()))


def centered_total_welfare(
    graph: ig.Graph,
    benefit_to_cost_ratio: float,
    distance_discount: float,
) -> float:
    """Return the standard aggregate welfare: sum of centered payoffs."""

    return (
        benefit_to_cost_ratio
        * centered_total_benefit(graph, distance_discount)
        - float(graph.ecount())
    )


def centered_mean_welfare(
    graph: ig.Graph,
    benefit_to_cost_ratio: float,
    distance_discount: float,
) -> float:
    """Return mean centered welfare per agent."""

    n = graph.vcount()
    if n < 1:
        raise ValueError("graph must contain at least one node")
    return centered_total_welfare(
        graph,
        benefit_to_cost_ratio=benefit_to_cost_ratio,
        distance_discount=distance_discount,
    ) / float(n)


def centered_mean_welfare_star(
    n: int,
    benefit_to_cost_ratio: float,
    distance_discount: float,
) -> float:
    """Closed-form mean centered welfare of an n-node star."""

    if n < 1:
        raise ValueError("n must satisfy n >= 1")
    if n == 1:
        return 0.0

    external_benefit = (
        2.0 * float(n - 1) * distance_discount
        + float((n - 1) * (n - 2)) * distance_discount**2
    )
    total_welfare = benefit_to_cost_ratio * external_benefit - float(n - 1)
    return total_welfare / float(n)
