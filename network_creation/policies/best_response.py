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


import math

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
    return float(power_buffer.sum())


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


def _logit(p: float, eps: float = 1e-6) -> float:
    p = min(max(p, eps), 1.0 - eps)
    return float(np.log(p / (1.0 - p)))


def _tau_from_exploration(
    exploration: float,
    tau_min: float = 0.03,
    tau_max: float = 1.0,
) -> float:
    if not 0.0 <= exploration <= 1.0:
        raise ValueError("exploration must be in [0,1]")
    return float(tau_min * (tau_max / tau_min) ** exploration)


def stable_softmax_boltzmann(
    values: NDArray[np.float32],
    tau: float,
) -> NDArray[np.float64]:
    if tau <= 0.0:
        raise ValueError("tau must be > 0")
    if len(values) == 0:
        raise ValueError("values must be non-empty")
    if len(values) == 1:
        return np.ones(1, dtype=np.float64)

    z = np.asarray(values, dtype=np.float64) / tau
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
    """Select an action using a normalized one-step score + size-aware gate.

    Non-PASS actions are scored from normalized one-step utility deltas and
    mapped to a common range in [-1, 1].

    The underlying one-step payoff change is still:

        ΔΠ̂_i = (1 / (2*N - 1)) * [ λ * ΔB - Δη ]

    where:
      - λ is the benefit_to_cost_ratio
      - ΔB is the change in discounted reachability
      - Δη is the outdegree change (+1 add, -1 delete, 0 swap/bridge)

    Instead of feeding these O(1/N)-scale values directly into the policy, we
    rescale the *policy score* to make action comparison stable across N:

        raw(a) = λ * ΔB - Δη
        score(a) = raw(a) / (λ * (N - 1) + 1)

    so ADD / DELETE / SWAP scores live in [-1, 1].

    BRIDGE is also placed on the same score scale:

        BRIDGE score:
            2 * (bridge_weight * (|bridge_actions| / N)) - 1

    PASS is handled through a baseline pass-bias gate plus a correction for:
      - the best available non-PASS score
      - the current non-PASS action-space size

        P(PASS) = sigmoid(
            logit(pass_bias)
            - max(0, max_a score(a))
            - opportunity_sensitivity * availability_bonus
        )

    where:

        availability_bonus = log(1 + |A_nonpass|) / log(1 + N)

    so PASS becomes less likely when:
      - at least one promising action exists
      - the agent has many feasible non-PASS opportunities

    Conditional on not passing, an action is sampled from a Boltzmann softmax
    over the non-PASS candidate scores:

        P(a | not PASS) ∝ exp(score(a) / tau_eff)

    with a size-aware temperature:

        tau_eff = tau_base / (1 + availability_bonus)

    so larger action spaces do not become artificially diffuse.

    Exposed parameters:
      - λ  : benefit_to_cost_ratio in (0, 1]
      - δ  : distance_discount in (0, 1)
      - bridge_weight in [0, 1]
      - pass_bias in [0, 1]
      - exploration in [0, 1]
      - opportunity_sensitivity in [0, 1]
    """

    def __init__(
        self,
        n_nodes: int,
        seed: int,
        benefit_to_cost_ratio: float,
        distance_discount: float,
        bridge_weight: float,
        pass_bias: float,
        exploration: float,
        opportunity_sensitivity: float = 0.5,
    ) -> None:
        if not 0.0 < benefit_to_cost_ratio <= 1.0:
            raise ValueError("benefit_to_cost_ratio must be in (0,1]")
        if not 0.0 < distance_discount < 1.0:
            raise ValueError("distance_discount must be in (0,1)")
        if not 0.0 <= bridge_weight <= 1.0:
            raise ValueError("bridge_weight must be in [0,1]")
        if not 0.0 <= pass_bias <= 1.0:
            raise ValueError("pass_bias must be in [0,1]")
        if not 0.0 <= exploration <= 1.0:
            raise ValueError("exploration must be in [0,1]")
        if not 0.0 <= opportunity_sensitivity <= 1.0:
            raise ValueError("opportunity_sensitivity must be in [0,1]")
        if n_nodes <= 1:
            raise ValueError("n_nodes must be > 1")

        self.n_nodes = n_nodes
        self.benefit_to_cost_ratio = benefit_to_cost_ratio
        self.distance_discount = distance_discount
        self.bridge_weight = bridge_weight
        self.pass_bias = pass_bias
        self.exploration = exploration
        self.opportunity_sensitivity = opportunity_sensitivity

        self._pass_logit = _logit(pass_bias)
        self._base_tau = _tau_from_exploration(exploration)

        self._rng = np.random.default_rng(seed)
        self._score_denom = float(benefit_to_cost_ratio * (n_nodes - 1) + 1.0)
        self.candidate_actions: list[Action | None] = [None] * n_nodes
        self.candidate_values = np.empty(n_nodes, dtype=np.float32)

    def reset(self, seed: int) -> None:
        self._rng = np.random.default_rng(seed)

    def _normalize_structural_delta(self, raw_delta: float) -> float:
        score = raw_delta / self._score_denom
        return float(np.clip(score, -1.0, 1.0))

    def _availability_bonus(self, n_non_pass_candidates: int) -> float:
        if n_non_pass_candidates <= 0:
            return 0.0
        return float(
            math.log1p(n_non_pass_candidates) / math.log1p(self.n_nodes)
        )

    def _effective_tau(self, n_non_pass_candidates: int) -> float:
        if n_non_pass_candidates <= 1:
            return self._base_tau

        availability_bonus = self._availability_bonus(n_non_pass_candidates)
        tau = self._base_tau / (1.0 + availability_bonus)
        return max(0.01, float(tau))

    def _pass_probability(self, non_pass_values: NDArray[np.float32]) -> float:
        if len(non_pass_values) == 0:
            return 1.0

        best_value = max(0.0, float(np.max(non_pass_values)))
        availability_bonus = self._availability_bonus(len(non_pass_values))

        return _sigmoid(
            self._pass_logit
            - best_value
            - self.opportunity_sensitivity * availability_bonus
        )

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

        bridge_action = None
        if bridge_actions:
            bridge_action = bridge_actions[int(self._rng.integers(len(bridge_actions)))]

        add_targets = [action.target for action in add_actions]

        n_non_pass_candidates = (
            len(delete_actions) + len(add_actions) + int(bridge_action is not None)
        )
        assert (
            n_non_pass_candidates <= self.n_nodes
        ), f"Expected at most {self.n_nodes} non-pass candidates, got {n_non_pass_candidates}"

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

                raw_delete = (
                    self.benefit_to_cost_ratio * (deleted_benefit - base_benefit)
                    + 1.0
                )
                delete_value = self._normalize_structural_delta(raw_delete)

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

            add_only_raw = (
                self.benefit_to_cost_ratio * (add_only_benefit - base_benefit) - 1.0
            )
            add_only_values = np.clip(
                add_only_raw / self._score_denom,
                -1.0,
                1.0,
            ).astype(np.float32)

            if delete_actions:
                swap_raw_values = self.benefit_to_cost_ratio * (
                    add_benefit_matrix[:, :-1] - base_benefit
                )
                swap_values = np.clip(
                    swap_raw_values / self._score_denom,
                    -1.0,
                    1.0,
                ).astype(np.float32)

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

        # BRIDGE
        if bridge_action is not None:
            bridge_score = 2.0 * (
                self.bridge_weight * (len(bridge_actions) / self.n_nodes)
            ) - 1.0
            self.candidate_actions[candidate_idx] = bridge_action
            self.candidate_values[candidate_idx] = float(
                np.clip(bridge_score, -1.0, 1.0)
            )
            candidate_idx += 1

        assert (
            candidate_idx == n_non_pass_candidates
        ), f"Filled {candidate_idx} non-pass candidates, expected {n_non_pass_candidates}"

        non_pass_values = self.candidate_values[:candidate_idx]
        p_pass = self._pass_probability(non_pass_values)

        if self._rng.random() < p_pass:
            return PolicyDecision(
                actions=(pass_action,),
                value={"value": p_pass},
            )

        chosen_idx = _sample_boltzmann(
            non_pass_values,
            self._effective_tau(len(non_pass_values)),
            self._rng,
        )
        return PolicyDecision(
            actions=(self.candidate_actions[chosen_idx],),
            value={"value": float(self.candidate_values[chosen_idx])},
        )