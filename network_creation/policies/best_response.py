from __future__ import annotations


r"""
.. math::
    \begin{aligned}
        \text{Component Domains:} & \quad \delta \in [0, 1], \quad \frac{V}{k} \in [0, 1], \quad \eta_i(g) \in [0, n-1] \\
        \\
        \text{Summation Range:} & \quad 1 \leq \sum_{j \in N} \delta^{d(i,j,\bar{g})} \leq n \\
        \\
        \text{Raw Ratio Bounds:} & \quad -(n-1) \leq \underbrace{\frac{V}{k} \sum_{j \in N} \delta^{d(i,j,\bar{g})} - \eta_i(g)}_{x} \leq n \\
        \\
        \text{Normalization Step:} & \quad \hat{\Pi}_i = \frac{x - x_{min}}{x_{max} - x_{min}} = \frac{x + (n-1)}{2n-1} \\
        \\
        \text{Final Payoff Function:} & \quad \hat{\Pi}_i = \frac{1}{2n-1} \left[ \frac{V}{k} \sum_{j \in N} \delta^{d(i, j, \bar{g})} - \eta_i(g) + (n-1) \right] \\
        \\
        \text{Payoff Change:} & \quad \Delta \hat{\Pi}_i = \frac{1}{2n-1} \left[ \frac{V}{k} \Delta \sum_{j \in N} \delta^{d(i, j, \bar{g})} - \Delta \eta_i(g) \right] \\
        \\
        \text{where} & \quad \Delta \eta_i = 
            \begin{cases} 
                1 &: \text{link addition} \\ 
                0 &: \text{no change} \\ 
                -1 &: \text{link deletion} 
            \end{cases}
    \end{aligned}
"""


import numpy as np
from numpy.typing import NDArray

from network_creation.core.model import (
    Action,
    ActionSpace,
    ActionType,
    Observation,
    PolicyDecision,
)

from .base import Policy


def _sum_discounted_distances(
    distance_discount: float,
    distances: NDArray[np.float32],
    power_buffer: NDArray[np.float32],
) -> float:
    np.power(distance_discount, distances, out=power_buffer)
    return power_buffer.sum()


def stable_softmax_boltzmann(
    values: NDArray[np.float32],
    tau: float,
) -> NDArray[np.float64]:
    if tau <= 0.0:
        raise ValueError("tau must be > 0")
    k = len(values)
    if k == 1:
        return np.ones(1, dtype=np.float64)
    scale = np.log(k) / tau
    z = np.asarray(values, dtype=np.float64) * scale
    z = z - np.max(z)
    probs = np.exp(z)
    return probs / probs.sum()


def _sample_boltzmann(
    values: NDArray[np.float32],
    tau: float,
    rng: np.random.Generator,
) -> int:
    probabilities = stable_softmax_boltzmann(values, tau)
    return int(rng.choice(len(values), p=probabilities))


class BestResponsePolicy(Policy):
    """Select an action by Boltzmann-weighted sampling over utility deltas.

    For acting node i, each candidate action is scored as

        ΔU_i =
            λ * (1 / (2*N - 1)) * Σ_j [δ^{d*_j} - δ^{d_j}]
            - Δoutdeg(i) / (2*N - 1)

    where:
      - λ is the benefit_to_cost_ratio
      - δ is the distance discount
      - add:    Δoutdeg = +1
      - delete: Δoutdeg = -1
    """

    def __init__(
        self,
        n_nodes: int,
        seed: int,
        benefit_to_cost_ratio: float,
        distance_discount: float,
        bridge_value: float,
        inertia_term: float,
        tau: float,
    ) -> None:
        if not 0.0 < benefit_to_cost_ratio <= 1.0:
            raise ValueError("benefit_to_cost_ratio must be in (0,1]")
        if not 0.0 < distance_discount < 1.0:
            raise ValueError("distance_discount must be in (0,1)")
        if n_nodes <= 1:
            raise ValueError("n_nodes must be > 1")

        self.n_nodes = n_nodes
        self.benefit_to_cost_ratio = benefit_to_cost_ratio
        self.distance_discount = distance_discount
        self.inertia_term = inertia_term
        self.bridge_value = bridge_value
        self.tau = tau
        self._rng = np.random.default_rng(seed)

        self.lambda_over_norm = benefit_to_cost_ratio / float(2 * n_nodes - 1)
        self.one_over_norm = 1.0 / float(2 * n_nodes - 1)
        self.candidate_actions: list[Action | None] = [None] * n_nodes
        self.candidate_values = np.empty(n_nodes, dtype=np.float32)

    def reset(self, seed: int) -> None:
        self._rng = np.random.default_rng(seed)

    def get_decision(
        self, observation: Observation, action_space: ActionSpace
    ) -> PolicyDecision:
        graph = observation.graph
        node_count = graph.vcount()
        pass_action = action_space[ActionType.PASS][0]
        source_idx = pass_action.source

        baseline_dist = np.empty(node_count, dtype=np.float32)
        power_buffer = np.empty(node_count, dtype=np.float32)
        baseline_dist[:] = graph.distances(source=[source_idx], mode="all")[0]
        base_benefit = _sum_discounted_distances(
            self.distance_discount,
            baseline_dist,
            power_buffer,
        )

        add_actions = action_space[ActionType.ADD]
        delete_actions = action_space[ActionType.DELETE]
        bridge_actions = action_space[ActionType.BRIDGE]
        add_targets = [action.target for action in add_actions]

        n_candidates = 1 + len(delete_actions) + len(add_actions) + len(bridge_actions)
        assert (
            n_candidates == self.n_nodes
        ), f"Expected {self.n_nodes} candidate actions, got {n_candidates}"

        candidate_idx = 0

        add_benefit_matrix: np.ndarray | None = None
        add_target_distance_buffer: NDArray[np.float32] | None = None
        combo_distance_buffer: NDArray[np.float32] | None = None
        combo_power_buffer: NDArray[np.float32] | None = None

        if add_actions:
            add_benefit_matrix = np.empty(
                (len(add_actions), len(delete_actions) + 1),
                dtype=np.float32,
            )
            add_target_distance_buffer = np.empty(
                (len(add_actions), node_count),
                dtype=np.float32,
            )
            combo_distance_buffer = np.empty(
                (len(add_actions), node_count),
                dtype=np.float32,
            )
            combo_power_buffer = np.empty(
                (len(add_actions), node_count),
                dtype=np.float32,
            )

        # DELETE actions
        if delete_actions:
            g_delete = graph.copy()
            deleted_dist = np.empty(node_count, dtype=np.float32)
            for delete_idx, action in enumerate(delete_actions):
                eid = g_delete.get_eid(
                    source_idx, action.target, directed=True, error=True
                )
                g_delete.delete_edges([eid])

                deleted_dist[:] = g_delete.distances(source=[source_idx], mode="all")[0]
                deleted_benefit = _sum_discounted_distances(
                    self.distance_discount,
                    deleted_dist,
                    power_buffer,
                )
                delete_value = (
                    self.lambda_over_norm * (deleted_benefit - base_benefit)
                    + self.one_over_norm
                )

                self.candidate_actions[candidate_idx] = action
                self.candidate_values[candidate_idx] = delete_value
                candidate_idx += 1

                if (
                    add_benefit_matrix is not None
                    and add_target_distance_buffer is not None
                    and combo_distance_buffer is not None
                    and combo_power_buffer is not None
                ):
                    add_target_distance_buffer[:] = g_delete.distances(
                        source=add_targets, mode="all"
                    )
                    np.add(add_target_distance_buffer, 1.0, out=combo_distance_buffer)
                    np.minimum(
                        combo_distance_buffer,
                        deleted_dist[None, :],
                        out=combo_distance_buffer,
                    )
                    np.power(
                        self.distance_discount,
                        combo_distance_buffer,
                        out=combo_power_buffer,
                    )
                    add_benefit_matrix[:, delete_idx] = np.sum(
                        combo_power_buffer,
                        axis=1,
                    )

                g_delete.add_edge(source_idx, action.target)

        # ADD / SWAP actions
        if (
            add_benefit_matrix is not None
            and add_target_distance_buffer is not None
            and combo_distance_buffer is not None
            and combo_power_buffer is not None
        ):
            add_target_distance_buffer[:] = graph.distances(
                source=add_targets, mode="all"
            )
            np.add(add_target_distance_buffer, 1.0, out=combo_distance_buffer)
            np.minimum(
                combo_distance_buffer,
                baseline_dist[None, :],
                out=combo_distance_buffer,
            )
            np.power(
                self.distance_discount,
                combo_distance_buffer,
                out=combo_power_buffer,
            )

            add_only_benefit = np.sum(combo_power_buffer, axis=1)
            add_benefit_matrix[:, -1] = add_only_benefit

            add_only_values = (
                self.lambda_over_norm * (add_only_benefit - base_benefit)
                - self.one_over_norm
            )

            if delete_actions:
                swap_values = self.lambda_over_norm * (
                    add_benefit_matrix[:, :-1] - base_benefit
                )
                best_add_values = np.maximum(
                    np.max(swap_values, axis=1),
                    add_only_values,
                )
            else:
                best_add_values = add_only_values

            for action, best_add_value in zip(add_actions, best_add_values):
                self.candidate_actions[candidate_idx] = action
                self.candidate_values[candidate_idx] = best_add_value
                candidate_idx += 1

        reference_value = 0.0
        # PASS
        self.candidate_actions[candidate_idx] = pass_action
        self.candidate_values[candidate_idx] = reference_value + self.inertia_term
        candidate_idx += 1

        # BRIDGE
        if bridge_actions:
            bridge_end = candidate_idx + len(bridge_actions)
            self.candidate_actions[candidate_idx:bridge_end] = bridge_actions
            self.candidate_values[candidate_idx:bridge_end] = (
                reference_value + self.bridge_value
            )
            candidate_idx = bridge_end

        assert (
            candidate_idx == n_candidates
        ), f"Filled {candidate_idx} candidates, expected {n_candidates}"

        chosen_idx = _sample_boltzmann(
            self.candidate_values[:candidate_idx],
            self.tau,
            self._rng,
        )
        return PolicyDecision(
            actions=(self.candidate_actions[chosen_idx],),
            value={"value": float(self.candidate_values[chosen_idx])},
        )
