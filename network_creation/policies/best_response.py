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

FLOATING_POINT_TYPE = np.float32


def _sum_discounted_distances(
    distance_discount: float,
    distances: NDArray[FLOATING_POINT_TYPE],
    power_buffer: NDArray[FLOATING_POINT_TYPE],
) -> float:
    np.power(distance_discount, distances, out=power_buffer)
    return float(power_buffer.sum())


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
        node_count = graph.vcount()

        pass_action = action_space[ActionType.PASS][0]
        add_actions = action_space[ActionType.ADD]
        delete_actions = action_space[ActionType.DELETE]
        bridge_actions = action_space[ActionType.BRIDGE]
        bridge_action = bridge_actions[0] if bridge_actions else None

        source_idx = pass_action.source
        add_targets = [action.target for action in add_actions]
        outside_node_idxs = observation.outside_node_idxs

        baseline_dist = np.empty(node_count, dtype=FLOATING_POINT_TYPE)
        power_buffer = np.empty(node_count, dtype=FLOATING_POINT_TYPE)
        baseline_dist[:] = graph.distances(source=[source_idx], mode="all")[0]
        base_benefit = _sum_discounted_distances(
            self.distance_discount,
            baseline_dist,
            power_buffer,
        )

        add_benefit_matrix: np.ndarray | None = None
        add_target_distance_buffer: NDArray[FLOATING_POINT_TYPE] | None = None
        combo_distance_buffer: NDArray[FLOATING_POINT_TYPE] | None = None
        combo_power_buffer: NDArray[FLOATING_POINT_TYPE] | None = None

        if add_actions:
            add_benefit_matrix = np.empty(
                (len(add_actions), len(delete_actions) + 1),
                dtype=FLOATING_POINT_TYPE,
            )
            add_target_distance_buffer = np.empty(
                (len(add_actions), node_count),
                dtype=FLOATING_POINT_TYPE,
            )
            combo_distance_buffer = np.empty(
                (len(add_actions), node_count),
                dtype=FLOATING_POINT_TYPE,
            )
            combo_power_buffer = np.empty(
                (len(add_actions), node_count),
                dtype=FLOATING_POINT_TYPE,
            )

        candidate_idx = 0
        if delete_actions:
            g_delete = graph.copy()
            deleted_dist = np.empty(node_count, dtype=FLOATING_POINT_TYPE)
            # DELETE actions
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
                    self.benefit_to_cost_ratio * (deleted_benefit - base_benefit) + 1.0
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
            add_only_value = (
                self.benefit_to_cost_ratio * (add_only_benefit - base_benefit) - 1.0
            )

            if delete_actions:
                swap_values = self.benefit_to_cost_ratio * (
                    add_benefit_matrix[:, :-1] - base_benefit
                )
                best_add_values = np.maximum(
                    np.max(swap_values, axis=1),
                    add_only_value,
                )
            else:
                best_add_values = add_only_value

            for action, best_add_value in zip(add_actions, best_add_values):
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
                * (
                    expected_bridge_gain - (base_benefit - 1.0)
                )  # subtract 1 for the self-distance
                + 1.0
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
                "other_actions_values": list(
                    zip(
                        self.candidate_actions[: candidate_idx + 1],
                        self.candidate_values[: candidate_idx + 1],
                    )
                ),
            },
        )
