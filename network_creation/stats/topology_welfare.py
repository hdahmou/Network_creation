from __future__ import annotations

import igraph as ig
from typing import Literal
import numpy as np
from numpy.typing import NDArray
from functools import cache

_ARGMAX_P_GRID = np.linspace(0.0, 1.0, 1000, dtype=float)


def total_welfare(
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


def normalized_welfare(
    graph: ig.Graph,
    benefit_to_cost_ratio: float,
    distance_discount: float,
) -> float:
    n = graph.vcount()
    return (
        total_welfare(
            graph,
            benefit_to_cost_ratio=benefit_to_cost_ratio,
            distance_discount=distance_discount,
        )
        / n
    )


@cache
def normalized_welfare_per_topology(
    topology_class: Literal["star", "empty", "random", "max"],
    n: int,
    benefit_to_cost_ratio: float,
    distance_discount: float,
) -> float:
    if topology_class == "star":
        return get_star_normalized_welfare(
            n=n,
            benefit_to_cost_ratio=benefit_to_cost_ratio,
            distance_discount=distance_discount,
        )

    if topology_class == "empty":
        return 0.0

    if topology_class == "random":
        return max(
            get_ER_normalized_welfare(
                n=n,
                benefit_to_cost_ratio=benefit_to_cost_ratio,
                distance_discount=distance_discount,
                p_directed=candidate_p,
            )
            for candidate_p in _ARGMAX_P_GRID
        )

    if topology_class == "max":
        empty_welfare = 0.0
        star_welfare = get_star_normalized_welfare(
            n=n,
            benefit_to_cost_ratio=benefit_to_cost_ratio,
            distance_discount=distance_discount,
        )
        return max(empty_welfare, star_welfare)

    raise ValueError(
        "topology_class must be one of 'star', 'empty', 'random', or 'max', "
        f"but got {topology_class!r}"
    )


def get_star_normalized_welfare(
    n: int,
    benefit_to_cost_ratio: float,
    distance_discount: float,
) -> float:
    return ((n - 1) / n) * (
        2 * benefit_to_cost_ratio * distance_discount
        + (n - 2) * benefit_to_cost_ratio * distance_discount**2
        - 1
    )


def get_ER_normalized_welfare(
    n: int,
    benefit_to_cost_ratio: float,
    distance_discount: float,
    p_directed: float,
    epsilon: float = 1e-9,
) -> float:
    lam = benefit_to_cost_ratio
    delta = distance_discount
    p = p_directed
    p_bar = 2.0 * p - p * p
    q = 1.0 - p_bar

    F = [0.0] * n
    F[0] = 1.0
    F[1] = q
    for k in range(1, n - 1):
        shell_mass = F[k - 1] - F[k]
        exponent = (n - 1) * shell_mass
        base = min(max(q, epsilon), 1.0)
        next_val = F[k] * (base**exponent)
        next_val = min(max(next_val, 0.0), F[k])
        F[k + 1] = next_val

    benefit_sum = 0.0
    delta_power = delta
    for d in range(1, n):
        prob_distance_eq_d = F[d - 1] - F[d]
        benefit_sum += delta_power * prob_distance_eq_d
        delta_power *= delta

    welfare = lam * (n - 1) * benefit_sum - (n - 1) * p

    return welfare
