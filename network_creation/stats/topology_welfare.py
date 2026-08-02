from __future__ import annotations

from functools import cache
from typing import Literal

import igraph as ig
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize_scalar  # type: ignore[import-untyped]

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


def er_centered_mean_welfare(
    n_agents: int,
    benefit_to_cost_ratio: float,
    distance_discount: float,
    directed_link_probability: float,
) -> float:
    """Return ER centered mean welfare at one directed link probability."""
    no_link_probability = (1.0 - directed_link_probability) ** 2
    with np.errstate(divide="ignore"):
        log_no_link_probability = float(
            2.0 * np.log1p(-directed_link_probability)
        )

    tail_at_d_minus_2 = 1.0
    tail_at_d_minus_1 = no_link_probability
    discount_at_distance = distance_discount
    discounted_reachability = discount_at_distance * (
        tail_at_d_minus_2 - tail_at_d_minus_1
    )

    for _distance in range(2, n_agents):
        shell_probability = tail_at_d_minus_2 - tail_at_d_minus_1
        exponent = (n_agents - 1) * shell_probability
        multiplier = (
            float(np.exp(exponent * log_no_link_probability))
            if exponent > 0.0
            else 1.0
        )
        next_tail = tail_at_d_minus_1 * multiplier
        next_tail = min(max(next_tail, 0.0), tail_at_d_minus_1)

        discount_at_distance *= distance_discount
        discounted_reachability += discount_at_distance * (
            tail_at_d_minus_1 - next_tail
        )
        tail_at_d_minus_2, tail_at_d_minus_1 = tail_at_d_minus_1, next_tail

    return (
        benefit_to_cost_ratio * (n_agents - 1) * discounted_reachability
        - (n_agents - 1) * directed_link_probability
    )


def max_er_centered_mean_welfare(
    n_agents: int,
    benefit_to_cost_ratio: float,
    distance_discount: float,
) -> float:
    """Return the maximum ER centered mean welfare over p in [0, 1]."""
    def welfare_at_probability(probability: float) -> float:
        return er_centered_mean_welfare(
            n_agents=n_agents,
            benefit_to_cost_ratio=benefit_to_cost_ratio,
            distance_discount=distance_discount,
            directed_link_probability=probability,
        )

    result = minimize_scalar(
        lambda probability: -welfare_at_probability(float(probability)),
        bounds=(0.0, 1.0),
        method="bounded",
        options={"xatol": 1e-9, "maxiter": 500},
    )
    return max(0.0, welfare_at_probability(1.0), -float(result.fun))
