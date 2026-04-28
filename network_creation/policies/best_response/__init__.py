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

from ._kernel import evaluate_best_response_terms
from ..base import Policy

FLOATING_POINT_TYPE = np.float32


def stable_softmax_boltzmann(
    values: NDArray[FLOATING_POINT_TYPE],
    tau: float,
) -> NDArray[FLOATING_POINT_TYPE]:
    if tau <= 0.0:
        raise ValueError("tau must be > 0")
    if len(values) == 0:
        raise ValueError("values must be non-empty")

    z = np.asarray(values, dtype=FLOATING_POINT_TYPE) / tau
    z = z - np.max(z)
    probs = np.exp(z)
    return probs / probs.sum()


def _sample_boltzmann(
    values: NDArray[FLOATING_POINT_TYPE],
    tau: float,
    rng: np.random.Generator,
) -> int:
    probabilities = stable_softmax_boltzmann(values, tau)
    return int(rng.choice(len(values), p=probabilities))


class BestResponsePolicy(Policy):
    def __init__(
        self,
        n_nodes: int,
        seed: int,
        benefit_to_cost_ratio: float,
        distance_discount: float,
        exploration: float,
        outside_optimism: float,
        **_: object,
    ) -> None:
        if not 0.0 < benefit_to_cost_ratio <= 1.0:
            raise ValueError("benefit_to_cost_ratio must be in (0,1]")
        if not 0.0 < distance_discount < 1.0:
            raise ValueError("distance_discount must be in (0,1)")
        if not 0.0 < exploration:
            raise ValueError("exploration must be positive")
        if not 0.0 <= outside_optimism <= 1.0:
            raise ValueError("outside_optimism must be in [0,1]")
        if n_nodes <= 1:
            raise ValueError("n_nodes must be > 1")

        self.n_nodes = n_nodes
        self.benefit_to_cost_ratio = benefit_to_cost_ratio
        self.distance_discount = distance_discount
        self.exploration = exploration
        self.outside_optimism = outside_optimism
        self._rng = np.random.default_rng(seed)

        self.candidate_actions: list[Action | None] = [None] * n_nodes
        self.candidate_values = np.empty(n_nodes, dtype=FLOATING_POINT_TYPE)

    def reset(self, seed: int) -> None:
        self._rng = np.random.default_rng(seed)

    def get_decision(
        self, observation: Observation, action_space: ActionSpace
    ) -> PolicyDecision:
        graph = observation.graph

        pass_action = action_space[ActionType.PASS][0]
        add_actions = action_space[ActionType.ADD]
        delete_actions = action_space[ActionType.DELETE]
        bridge_actions = action_space[ActionType.BRIDGE]
        bridge_action = bridge_actions[0] if bridge_actions else None

        source_idx = pass_action.source
        add_targets = [int(action.target) for action in add_actions]
        delete_targets = [int(action.target) for action in delete_actions]
        outside_node_idxs = observation.outside_node_idxs

        (
            base_score_excluding_self,
            delete_scores,
            add_only_scores,
            best_swap_scores,
            _best_swap_delete,
        ) = evaluate_best_response_terms(
            graph,
            int(source_idx),
            float(self.distance_discount),
            add_targets,
            delete_targets,
        )

        candidate_idx = 0
        # DELETE actions
        for action, delete_score in zip(delete_actions, delete_scores, strict=True):
            delete_delta = delete_score - base_score_excluding_self
            self.candidate_actions[candidate_idx] = action
            self.candidate_values[candidate_idx] = (
                self.benefit_to_cost_ratio * delete_delta + 1.0
            )
            candidate_idx += 1

        # ADD / SWAP actions
        for action, add_only_score, swap_score in zip(
            add_actions,
            add_only_scores,
            best_swap_scores,
            strict=True,
        ):
            add_only_delta = add_only_score - base_score_excluding_self
            swap_delta = swap_score - base_score_excluding_self
            add_only_value = self.benefit_to_cost_ratio * add_only_delta - 1.0
            if delete_actions:
                swap_value = self.benefit_to_cost_ratio * swap_delta
                best_add_value = max(add_only_value, swap_value)
            else:
                best_add_value = add_only_value

            self.candidate_actions[candidate_idx] = action
            self.candidate_values[candidate_idx] = best_add_value
            candidate_idx += 1

        # BRIDGE
        if bridge_action is not None:
            spillover_targets = max(len(outside_node_idxs) - 1, 0)
            expected_bridge_gain = (
                self.distance_discount
                + self.outside_optimism * spillover_targets * self.distance_discount**2
            )
            bridge_value = (
                self.benefit_to_cost_ratio
                * (expected_bridge_gain - base_score_excluding_self)
                - 1.0
            )
            self.candidate_actions[candidate_idx] = bridge_action
            self.candidate_values[candidate_idx] = bridge_value
            candidate_idx += 1

        # PASS
        self.candidate_actions[candidate_idx] = pass_action
        self.candidate_values[candidate_idx] = 0.0

        chosen_idx = _sample_boltzmann(
            values=self.candidate_values[: candidate_idx + 1],
            tau=self.exploration,
            rng=self._rng,
        )

        chosen_action = self.candidate_actions[chosen_idx]
        if chosen_action is not None and chosen_action.type is ActionType.BRIDGE:
            chosen_action = Action(
                ActionType.BRIDGE,
                source_idx,
                int(self._rng.choice(outside_node_idxs)),
            )
            self.candidate_actions[chosen_idx] = chosen_action

        return PolicyDecision(
            actions=(chosen_action,),
            value={
                "value": float(self.candidate_values[chosen_idx]),
                # "other_actions_values": list(zip(self.candidate_actions[: candidate_idx + 1],self.candidate_values[: candidate_idx + 1],)),
            },
        )
