from __future__ import annotations

from functools import cache, lru_cache
from typing import Literal

import igraph as ig
import numpy as np
from numpy.typing import NDArray
from scipy.special import gammaln  # type: ignore[import-untyped]

type WelfareReference = Literal["star", "empty", "random", "max"]


def centered_total_welfare(
    graph: ig.Graph,
    benefit_to_cost_ratio: float,
    distance_discount: float,
) -> float:
    hist = graph.path_length_hist(directed=False)
    total_benefit = 0.0
    for left, _, count in hist.bins():
        total_benefit += 2.0 * count * (distance_discount ** int(left))
    total_cost = float(graph.ecount())
    return benefit_to_cost_ratio * total_benefit - total_cost


def agent_payoff(
    graph: ig.Graph,
    benefit_to_cost_ratio: float,
    distance_discount: float,
    n_agent_slots: int | None = None,
) -> NDArray[np.float32]:
    """Return normalized payoff per agent, optionally padded to a fixed slot count."""

    n_agents = graph.vcount()
    if n_agent_slots is None:
        n_agent_slots = n_agents
    if n_agent_slots < n_agents:
        raise ValueError("n_agent_slots must be greater than or equal to graph size")

    values = np.full(n_agent_slots, np.nan, dtype=np.float32)
    normalization = benefit_to_cost_ratio * (n_agents - 1) * distance_discount
    if n_agents == 0 or normalization == 0.0:
        return values

    distances = np.asarray(graph.distances(mode="all"), dtype=np.float64)
    finite_non_self = np.isfinite(distances)
    np.fill_diagonal(finite_non_self, False)

    benefit_terms = np.zeros_like(distances, dtype=np.float64)
    np.power(
        distance_discount,
        distances,
        out=benefit_terms,
        where=finite_non_self,
    )
    benefits = benefit_terms.sum(axis=1)
    costs = np.asarray(graph.degree(mode="out"), dtype=np.float64)
    values[:n_agents] = (
        (benefit_to_cost_ratio * benefits - costs) / normalization
    ).astype(np.float32)
    return values


def centered_mean_welfare(
    graph: ig.Graph,
    benefit_to_cost_ratio: float,
    distance_discount: float,
) -> float:
    n = graph.vcount()
    return (
        centered_total_welfare(
            graph,
            benefit_to_cost_ratio=benefit_to_cost_ratio,
            distance_discount=distance_discount,
        )
        / n
    )


@cache
def reference_centered_mean_welfare(
    reference: WelfareReference,
    n_agents: int,
    benefit_to_cost_ratio: float,
    distance_discount: float,
) -> float:
    if reference == "star":
        return star_centered_mean_welfare(
            n_agents=n_agents,
            benefit_to_cost_ratio=benefit_to_cost_ratio,
            distance_discount=distance_discount,
        )

    if reference == "empty":
        return 0.0

    if reference == "random":
        return max_er_centered_mean_welfare(
            n_agents=n_agents,
            benefit_to_cost_ratio=benefit_to_cost_ratio,
            distance_discount=distance_discount,
        )

    if reference == "max":
        star_welfare = star_centered_mean_welfare(
            n_agents=n_agents,
            benefit_to_cost_ratio=benefit_to_cost_ratio,
            distance_discount=distance_discount,
        )
        return max(0.0, star_welfare)

    raise ValueError(
        "reference must be one of 'star', 'empty', 'random', or 'max', "
        f"but got {reference!r}"
    )


def star_centered_mean_welfare(
    n_agents: int,
    benefit_to_cost_ratio: float,
    distance_discount: float,
) -> float:
    return ((n_agents - 1) / n_agents) * (
        2 * benefit_to_cost_ratio * distance_discount
        + (n_agents - 2) * benefit_to_cost_ratio * distance_discount**2
        - 1
    )


@lru_cache(maxsize=8)
def _er_log_binomial_coefficients(n_agents: int) -> tuple[NDArray[np.float64], ...]:
    log_factorials = np.asarray(gammaln(np.arange(n_agents) + 1), dtype=np.float64)
    rows = []
    for u in range(1, n_agents):
        discovered = np.arange(1, u + 1)
        rows.append(
            log_factorials[u] - log_factorials[discovered]
            - log_factorials[u - discovered]
        )
    return tuple(rows)


def er_discounted_distance_expectation(
    n_agents: int,
    directed_link_probability: float,
    distance_discounts: float | NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute E[delta**D] by exact BFS recursion for one or more discounts."""
    discounts = np.atleast_1d(np.asarray(distance_discounts, dtype=np.float64))
    if n_agents <= 1 or directed_link_probability == 0.0:
        return np.zeros_like(discounts)
    if directed_link_probability == 1.0:
        return discounts.copy()

    log_no_access = 2.0 * np.log1p(-directed_link_probability)
    counts = np.arange(1, n_agents)
    h = np.zeros((n_agents + 1, n_agents + 1, discounts.size), dtype=np.float64)
    for u, coefficients in enumerate(_er_log_binomial_coefficients(n_agents), start=1):
        frontier = counts[: n_agents - u]
        discovered = counts[:u]
        log_miss = frontier[:, None] * log_no_access
        log_hit = np.log(-np.expm1(log_miss))
        # Log weights avoid overflowing binomial coefficients or losing tiny p.
        weights = np.exp(
            coefficients + log_hit * discovered + log_miss * (u - discovered)
        )
        continuation = discovered[:, None] + h[u - discovered, discovered]
        h[u, frontier] = discounts * np.einsum("ft,td->fd", weights, continuation)
    return h[n_agents - 1, 1] / (n_agents - 1)


def er_centered_mean_welfare(
    n_agents: int,
    benefit_to_cost_ratio: float,
    distance_discount: float,
    directed_link_probability: float,
) -> float:
    """Return exact expected mean welfare at sponsorship probability p."""
    if n_agents <= 1:
        return 0.0
    expectation = er_discounted_distance_expectation(
        n_agents, directed_link_probability, distance_discount
    )
    return (n_agents - 1) * (
        benefit_to_cost_ratio * float(expectation[0]) - directed_link_probability
    )


def max_er_centered_mean_welfare(
    n_agents: int,
    benefit_to_cost_ratio: float,
    distance_discount: float,
    *,
    grid_points: int = 1000,
) -> float:
    """Return the best welfare on a uniform grid up to min(1, lambda * delta)."""
    if grid_points < 2:
        raise ValueError("grid_points must be at least 2")
    upper_probability = min(1.0, benefit_to_cost_ratio * distance_discount)
    if n_agents <= 1 or upper_probability <= 0.0:
        return 0.0
    probabilities = np.linspace(0.0, upper_probability, grid_points)
    return max(
        er_centered_mean_welfare(
            n_agents=n_agents,
            benefit_to_cost_ratio=benefit_to_cost_ratio,
            distance_discount=distance_discount,
            directed_link_probability=float(probability),
        )
        for probability in probabilities
    )
