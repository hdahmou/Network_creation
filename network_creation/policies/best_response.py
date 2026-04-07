"""Best-response payoff policy with exploration via bridge actions."""

from __future__ import annotations

import numpy as np

from network_creation.core.model import (
    Action,
    ActionSpace,
    ActionType,
    Observation,
    PolicyDecision,
)

from .base import Policy
from .utils import _distances_from_index, _sample_argmax


class BestResponsePolicy(Policy):
    """Select action with highest normalized payoff delta or bridge alternative."""

    def __init__(
        self,
        seed: int,
        benefit_to_cost_ratio: float,
        distance_discount: float,
        bridge_value_scale: float,
        inertia_term: float,
    ) -> None:
        if not 0.0 <= benefit_to_cost_ratio <= 1.0:
            raise ValueError("benefit_to_cost_ratio must be in [0,1]")
        if not 0.0 <= distance_discount < 1.0:
            raise ValueError("distance_discount must be in [0,1)")
        if bridge_value_scale < 0.0:
            raise ValueError("bridge_value_scale must be >= 0")
        if inertia_term < 0.0:
            raise ValueError("inertia_term must be >= 0")

        self.benefit_to_cost_ratio = benefit_to_cost_ratio
        self.distance_discount = distance_discount
        self.bridge_value_scale = bridge_value_scale
        self.inertia_term = inertia_term
        self._rng = np.random.default_rng(seed)

    def reset(self, seed: int) -> None:
        self._rng = np.random.default_rng(seed)

    def get_decision(
        self, observation: Observation, action_space: ActionSpace
    ) -> PolicyDecision:
        graph = observation.graph
        total_nodes = graph.vcount() + len(observation.outside_node_ids)
        pass_action = action_space[ActionType.PASS][0]

        if total_nodes <= 1:
            return PolicyDecision(actions=(pass_action,), value=None)

        name_to_index = {vertex["name"]: int(vertex.index) for vertex in graph.vs}
        source_idx = name_to_index[observation.node_id]

        baseline_dist = _distances_from_index(graph, source_idx)
        base_benefit = np.sum(self.distance_discount**baseline_dist)
        norm = float(total_nodes - 1)

        add_actions = action_space[ActionType.ADD]
        delete_actions = action_space[ActionType.DELETE]
        bridge_actions = action_space[ActionType.BRIDGE]

        add_target_indices = [name_to_index[action.target] for action in add_actions]
        delete_target_indices = [
            name_to_index[action.target] for action in delete_actions
        ]

        candidate_actions: list[Action] = []
        candidate_values: list[float] = []

        # Plain DELETE using one reusable mutable graph.
        if delete_actions:
            g_delete = graph.copy()
            for action, target_idx in zip(delete_actions, delete_target_indices):
                eid = g_delete.get_eid(
                    source_idx, target_idx, directed=True, error=True
                )
                g_delete.delete_edges([eid])

                new_dist = _distances_from_index(g_delete, source_idx)
                new_benefit = np.sum(self.distance_discount**new_dist)
                value = (
                    self.benefit_to_cost_ratio * (new_benefit - base_benefit) + 1.0
                ) / norm - self.inertia_term

                candidate_actions.append(action)
                candidate_values.append(value)

                g_delete.add_edge(source_idx, target_idx)

        # Batch distances for immediate ADD evaluation in the original graph.
        if add_actions:
            add_target_dists = np.asarray(
                graph.distances(source=add_target_indices, mode="all"),
                dtype=np.float64,
            )

            for action, target_idx, target_dist in zip(
                add_actions, add_target_indices, add_target_dists
            ):
                new_dist_add = np.minimum(baseline_dist, 1.0 + target_dist)
                new_benefit_add = np.sum(self.distance_discount**new_dist_add)
                best_add_value = (
                    self.benefit_to_cost_ratio * (new_benefit_add - base_benefit) - 1.0
                ) / norm

                if delete_actions:
                    g_add = graph.copy()
                    g_add.add_edge(source_idx, target_idx)

                    for del_target_idx in delete_target_indices:
                        eid = g_add.get_eid(
                            source_idx,
                            del_target_idx,
                            directed=True,
                            error=True,
                        )
                        g_add.delete_edges([eid])

                        new_dist = _distances_from_index(g_add, source_idx)
                        new_benefit = np.sum(self.distance_discount**new_dist)
                        value = (
                            self.benefit_to_cost_ratio * (new_benefit - base_benefit)
                        ) / norm
                        if value > best_add_value:
                            best_add_value = value

                        g_add.add_edge(source_idx, del_target_idx)

                candidate_actions.append(action)
                candidate_values.append(best_add_value - self.inertia_term)

        if bridge_actions:
            bridge_value = (
                self.bridge_value_scale * (len(bridge_actions) / norm)
                - self.inertia_term
            )
            if bridge_value > 0.0:
                bridge_action = bridge_actions[
                    int(self._rng.integers(len(bridge_actions)))
                ]
                candidate_actions.append(bridge_action)
                candidate_values.append(bridge_value)

        valid_indices = [
            idx for idx, value in enumerate(candidate_values) if value > 0.0
        ]
        if not valid_indices:
            return PolicyDecision(actions=(pass_action,), value=None)

        valid_actions = [candidate_actions[idx] for idx in valid_indices]
        valid_values = [candidate_values[idx] for idx in valid_indices]
        chosen_idx = _sample_argmax(valid_values, self._rng)

        return PolicyDecision(
            actions=(valid_actions[chosen_idx],),
            value={"value": valid_values[chosen_idx]},
        )
