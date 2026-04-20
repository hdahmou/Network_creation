from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from network_creation.core.model import (Action, ActionSpace, ActionType,
                                         Observation, PolicyDecision)

from .base import Policy
from .utils import _sample_argmax

ArrayF32 = NDArray[np.float32]


def _sum_discounted_distances(
    distance_discount: float,
    distances: ArrayF32,
    power_buffer: ArrayF32,
) -> float:
    np.power(distance_discount, distances, out=power_buffer)
    return float(power_buffer.sum())


def stable_softmax_boltzmann(
    values: ArrayF32,
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
    probabilities = np.exp(z)
    return probabilities / probabilities.sum()


def stable_log_mean_exp(values: ArrayF32, tau: float) -> float:
    """Return log-mean-exp(values / tau) rescaled back by tau.

    For ``tau == 0`` this reduces to ``max(values)`` so the helper matches the
    deterministic nested-best-response limit.
    """

    if len(values) == 0:
        raise ValueError("values must be non-empty")
    if tau < 0.0:
        raise ValueError("tau must be >= 0")
    if len(values) == 1:
        return float(values[0])
    if tau == 0.0:
        return float(np.max(values))

    scaled = np.asarray(values, dtype=np.float64) / tau
    scaled_max = float(np.max(scaled))
    centered = np.exp(scaled - scaled_max)
    return float(tau * (scaled_max + math.log(float(centered.mean()))))


def _sample_boltzmann(
    values: ArrayF32,
    tau: float,
    rng: np.random.Generator,
) -> int:
    probabilities = stable_softmax_boltzmann(values, tau)
    return int(rng.choice(len(values), p=probabilities))


def _sample_from_values(
    values: ArrayF32,
    tau: float,
    rng: np.random.Generator,
) -> int:
    if len(values) == 0:
        raise ValueError("values must be non-empty")
    if tau < 0.0:
        raise ValueError("tau must be >= 0")
    if tau == 0.0:
        return _sample_argmax(values, rng)
    return _sample_boltzmann(values, tau, rng)


class BestResponsePolicy(Policy):
    r"""Nested-logit local best response with one optional bridge belief term.

    Parameters
    ----------
    n_nodes:
        Total number of nodes in the full network.
    seed:
        Random seed for tie-breaking and noisy response.
    benefit_to_cost_ratio:
        The game parameter :math:`\lambda = V/k`, restricted only to be strictly positive.
    distance_discount:
        Distance discount :math:`\delta \in (0, 1)`.
    exploration:
        Temperature :math:`\tau \in [0, 1]`. ``0`` gives deterministic nested
        best response.
    outside_optimism:
        Bridge spillover optimism :math:`\omega \in [0, 1]`.
    """

    def __init__(
        self,
        n_nodes: int,
        seed: int,
        benefit_to_cost_ratio: float,
        distance_discount: float,
        exploration: float,
        outside_optimism: float = 0.0,
    ) -> None:
        if n_nodes <= 1:
            raise ValueError("n_nodes must be > 1")
        if benefit_to_cost_ratio <= 0.0:
            raise ValueError("benefit_to_cost_ratio must be > 0")
        if not 0.0 < distance_discount < 1.0:
            raise ValueError("distance_discount must be in (0,1)")
        if not 0.0 <= exploration <= 1.0:
            raise ValueError("exploration must be in [0,1]")
        if not 0.0 <= outside_optimism <= 1.0:
            raise ValueError("outside_optimism must be in [0,1]")

        self.n_nodes = n_nodes
        self.benefit_to_cost_ratio = benefit_to_cost_ratio
        self.distance_discount = distance_discount
        self.exploration = exploration
        self.outside_optimism = outside_optimism

        self._rng = np.random.default_rng(seed)
        self._score_denom = float(benefit_to_cost_ratio * (n_nodes - 1) + 1.0)
        self._direct_bridge_gain = float(distance_discount)
        self._spillover_bridge_gain = float(distance_discount**2)

    def reset(self, seed: int) -> None:
        self._rng = np.random.default_rng(seed)

    def _normalize_structural_delta(self, raw_delta: float) -> float:
        score = raw_delta / self._score_denom
        return float(np.clip(score, -1.0, 1.0))

    def _family_value(self, action_values: ArrayF32) -> float:
        return stable_log_mean_exp(action_values, tau=self.exploration)

    def _bridge_score(self, n_outside_nodes: int) -> float:
        if n_outside_nodes <= 0:
            raise ValueError("n_outside_nodes must be positive")

        spillover_targets = max(n_outside_nodes - 1, 0)
        expected_bridge_gain = self._direct_bridge_gain + (
            self.outside_optimism
            * spillover_targets
            * self._spillover_bridge_gain
        )
        raw_bridge = self.benefit_to_cost_ratio * expected_bridge_gain - 1.0
        return self._normalize_structural_delta(raw_bridge)

    def _score_add_actions(
        self,
        graph,
        baseline_benefit: float,
        baseline_powers: ArrayF32,
        add_actions: list[Action],
    ) -> ArrayF32:
        if not add_actions:
            return np.empty(0, dtype=np.float32)

        add_targets = [int(action.target) for action in add_actions]
        target_distances = np.asarray(
            graph.distances(source=add_targets, mode="all"),
            dtype=np.float32,
        )
        target_powers = np.empty_like(target_distances, dtype=np.float32)
        np.power(self.distance_discount, target_distances, out=target_powers)
        np.multiply(target_powers, self.distance_discount, out=target_powers)
        np.maximum(target_powers, baseline_powers[None, :], out=target_powers)

        add_benefits = np.sum(target_powers, axis=1, dtype=np.float32) - 1.0
        raw_add_values = (
            self.benefit_to_cost_ratio * (add_benefits - baseline_benefit) - 1.0
        )
        return np.clip(
            raw_add_values / self._score_denom,
            -1.0,
            1.0,
        ).astype(np.float32, copy=False)

    def _score_delete_actions(
        self,
        graph,
        source_idx: int,
        baseline_benefit: float,
        delete_actions: list[Action],
    ) -> ArrayF32:
        if not delete_actions:
            return np.empty(0, dtype=np.float32)

        temporary_graph = graph.copy()
        delete_distances = np.empty(graph.vcount(), dtype=np.float32)
        power_buffer = np.empty(graph.vcount(), dtype=np.float32)
        delete_values = np.empty(len(delete_actions), dtype=np.float32)

        for delete_idx, action in enumerate(delete_actions):
            if action.target is None:
                raise ValueError("DELETE actions must define a target")
            edge_id = temporary_graph.get_eid(
                source_idx,
                action.target,
                directed=True,
                error=True,
            )
            temporary_graph.delete_edges([edge_id])
            delete_distances[:] = temporary_graph.distances(
                source=[source_idx],
                mode="all",
            )[0]
            deleted_benefit = (
                _sum_discounted_distances(
                    self.distance_discount,
                    delete_distances,
                    power_buffer,
                )
                - 1.0
            )
            raw_delete = (
                self.benefit_to_cost_ratio * (deleted_benefit - baseline_benefit)
                + 1.0
            )
            delete_values[delete_idx] = self._normalize_structural_delta(raw_delete)
            temporary_graph.add_edge(source_idx, action.target)

        return delete_values

    def get_decision(
        self,
        observation: Observation,
        action_space: ActionSpace,
    ) -> PolicyDecision:
        graph = observation.graph
        source_idx = observation.node_idx
        pass_action = action_space[ActionType.PASS][0]

        baseline_distances = np.asarray(
            graph.distances(source=[source_idx], mode="all")[0],
            dtype=np.float32,
        )
        baseline_powers = np.empty(graph.vcount(), dtype=np.float32)
        np.power(self.distance_discount, baseline_distances, out=baseline_powers)
        baseline_benefit = float(baseline_powers.sum() - 1.0)

        add_actions = action_space[ActionType.ADD]
        delete_actions = action_space[ActionType.DELETE]
        bridge_actions = action_space[ActionType.BRIDGE]
        n_outside_nodes = len(observation.outside_node_idxs)

        add_values = self._score_add_actions(
            graph,
            baseline_benefit,
            baseline_powers,
            add_actions,
        )
        delete_values = self._score_delete_actions(
            graph,
            source_idx,
            baseline_benefit,
            delete_actions,
        )
        if bridge_actions:
            bridge_score = self._bridge_score(n_outside_nodes)
            bridge_values = np.full(
                len(bridge_actions),
                bridge_score,
                dtype=np.float32,
            )
        else:
            bridge_values = np.empty(0, dtype=np.float32)

        family_names: list[ActionType] = [ActionType.PASS]
        family_values: list[float] = [0.0]
        family_actions: dict[ActionType, list[Action]] = {
            ActionType.ADD: add_actions,
            ActionType.DELETE: delete_actions,
            ActionType.BRIDGE: bridge_actions,
        }
        family_action_values: dict[ActionType, ArrayF32] = {
            ActionType.ADD: add_values,
            ActionType.DELETE: delete_values,
            ActionType.BRIDGE: bridge_values,
        }

        for action_type in (ActionType.ADD, ActionType.DELETE, ActionType.BRIDGE):
            action_values = family_action_values[action_type]
            if len(action_values) == 0:
                continue
            family_names.append(action_type)
            family_values.append(self._family_value(action_values))

        family_values_array = np.asarray(family_values, dtype=np.float32)
        selected_family_index = _sample_from_values(
            family_values_array,
            tau=self.exploration,
            rng=self._rng,
        )
        selected_family = family_names[selected_family_index]
        selected_family_value = float(family_values_array[selected_family_index])

        if selected_family is ActionType.PASS:
            return PolicyDecision(
                actions=(pass_action,),
                value={
                    "value": 0.0,
                    "family": ActionType.PASS.value,
                    "family_value": selected_family_value,
                },
            )

        selected_action_values = family_action_values[selected_family]
        selected_action_index = _sample_from_values(
            selected_action_values,
            tau=self.exploration,
            rng=self._rng,
        )
        selected_action = family_actions[selected_family][selected_action_index]
        selected_value = float(selected_action_values[selected_action_index])

        return PolicyDecision(
            actions=(selected_action,),
            value={
                "value": selected_value,
                "family": selected_family.value,
                "family_value": selected_family_value,
            },
        )