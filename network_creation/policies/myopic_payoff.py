"""Myopic single-action payoff policy."""

from __future__ import annotations

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
from .utils import _distances_from_index


class MyopicPayoffPolicy(Policy):
    """Sample one local deviation and take it only when its payoff delta is positive."""

    def __init__(
        self,
        seed: int,
        benefit_to_cost_ratio: float,
        distance_discount: float,
        bridge_probability: float,
    ) -> None:
        if not 0.0 <= distance_discount < 1.0:
            raise ValueError(
                "distance_discount must satisfy 0 <= distance_discount < 1"
            )
        if not 0.0 <= benefit_to_cost_ratio <= 1.0:
            raise ValueError(
                "benefit_to_cost_ratio must satisfy 0 <= "
                "benefit_to_cost_ratio <= 1"
            )
        if not 0.0 <= bridge_probability <= 1.0:
            raise ValueError(
                "bridge_probability must satisfy 0 <= bridge_probability <= 1"
            )

        self.benefit_to_cost_ratio = benefit_to_cost_ratio
        self.distance_discount = distance_discount
        self.bridge_probability = bridge_probability
        self._rng = np.random.default_rng(seed)

    def reset(self, seed: int) -> None:
        self._rng = np.random.default_rng(seed)

    def get_decision(
        self, observation: Observation, action_space: ActionSpace
    ) -> PolicyDecision:
        pass_action = action_space[ActionType.PASS][0]

        if self._rng.random() < self.bridge_probability:
            bridge_actions = action_space[ActionType.BRIDGE]
            if not bridge_actions:
                return PolicyDecision(actions=(pass_action,), value=None)
            bridge_action = self._sample_action(bridge_actions)
            return PolicyDecision(actions=(bridge_action,), value=None)

        non_bridge_actions = (
            action_space[ActionType.ADD] + action_space[ActionType.DELETE]
        )
        if not non_bridge_actions:
            return PolicyDecision(actions=(pass_action,), value=None)

        candidate = self._sample_action(non_bridge_actions)
        delta = float(self._compute_candidate_delta(observation, candidate))
        decision_value = {
            "delta": delta,
            "considered_action": candidate,
        }

        if delta > 0.0:
            return PolicyDecision(actions=(candidate,), value=decision_value)
        return PolicyDecision(actions=(pass_action,), value=decision_value)

    def _sample_action(self, actions: list[Action]) -> Action:
        return actions[int(self._rng.integers(len(actions)))]

    def _compute_candidate_delta(
        self, observation: Observation, candidate: Action
    ) -> float:
        source_idx = candidate.source
        baseline_distances = _distances_from_index(observation.graph, source_idx)

        if candidate.type is ActionType.ADD:
            candidate_distances = self._compute_add_distances(
                observation,
                baseline_distances,
                candidate,
            )
            link_delta = -1.0
        elif candidate.type is ActionType.DELETE:
            candidate_distances = self._compute_delete_distances(
                observation,
                source_idx,
                candidate,
            )
            link_delta = 1.0
        else:
            raise ValueError(
                f"unsupported non-bridge action type: {candidate.type.value}"
            )

        baseline_benefit = np.power(
            self.distance_discount, baseline_distances
        ).sum()
        candidate_benefit = np.power(
            self.distance_discount, candidate_distances
        ).sum()
        benefit_delta = candidate_benefit - baseline_benefit
        return self.benefit_to_cost_ratio * benefit_delta + link_delta

    def _compute_add_distances(
        self,
        observation: Observation,
        baseline_distances: NDArray[np.float64],
        candidate: Action,
    ) -> NDArray[np.float64]:
        if candidate.target is None:
            raise ValueError("ADD actions must define a target")

        target_distances = _distances_from_index(observation.graph, candidate.target)
        return np.minimum(baseline_distances, 1.0 + target_distances)

    def _compute_delete_distances(
        self,
        observation: Observation,
        source_idx: int,
        candidate: Action,
    ) -> NDArray[np.float64]:
        if candidate.target is None:
            raise ValueError("DELETE actions must define a target")

        temporary_graph = observation.graph.copy()
        edge_id = temporary_graph.get_eid(
            source_idx,
            candidate.target,
            directed=True,
            error=True,
        )
        temporary_graph.delete_edges([edge_id])
        return _distances_from_index(temporary_graph, source_idx)
