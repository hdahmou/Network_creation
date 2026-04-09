"""Best-response payoff policy with Boltzmann action sampling."""

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


def _sum_discounted_distances(
    distance_discount: float,
    distances: NDArray[np.float32],
    power_buffer: NDArray[np.float32],
) -> float:
    np.power(distance_discount, distances, out=power_buffer)
    return power_buffer.sum()


def stable_softmax_boltzmann(
    values: NDArray[np.float32],
    temperature: float,
) -> NDArray[np.float64]:
    if temperature <= 0.0:
        raise ValueError("temperature must be > 0")

    scaled_values = np.asarray(values, dtype=np.float64) / (temperature + 1e-12)
    z = scaled_values - np.max(scaled_values)
    numerator = np.exp(z)
    denominator = np.sum(numerator)
    return numerator / denominator


def _sample_boltzmann(
    values: NDArray[np.float32],
    temperature: float,
    rng: np.random.Generator,
) -> int:
    probabilities = stable_softmax_boltzmann(values, temperature)
    return int(rng.choice(len(values), p=probabilities))


class BestResponsePolicy(Policy):
    """Select an action by Boltzmann-weighted sampling over payoff values."""

    def __init__(
        self,
        seed: int,
        benefit_to_cost_ratio: float,
        distance_discount: float,
        bridge_value_scale: float,
        inertia_term: float,
        temperature: float,
    ) -> None:
        if not 0.0 < benefit_to_cost_ratio <= 1.0:
            raise ValueError("benefit_to_cost_ratio must be in (0,1]")
        if not 0.0 <= distance_discount < 1.0:
            raise ValueError("distance_discount must be in [0,1)")
        if bridge_value_scale < 0.0:
            raise ValueError("bridge_value_scale must be >= 0")
        if inertia_term < 0.0:
            raise ValueError("inertia_term must be >= 0")
        if temperature <= 0.0:
            raise ValueError("temperature must be > 0")

        self.benefit_to_cost_ratio = benefit_to_cost_ratio
        self.distance_discount = distance_discount
        self.bridge_value_scale = bridge_value_scale
        self.inertia_term = inertia_term
        self.temperature = temperature
        self._rng = np.random.default_rng(seed)

    def reset(self, seed: int) -> None:
        self._rng = np.random.default_rng(seed)

    def get_decision(
        self, observation: Observation, action_space: ActionSpace
    ) -> PolicyDecision:
        graph = observation.graph
        node_count = graph.vcount()
        total_nodes = node_count + len(observation.outside_node_idxs)
        pass_action = action_space[ActionType.PASS][0]

        if total_nodes <= 1:
            return PolicyDecision(actions=(pass_action,), value=None)

        # Baseline state and normalization.
        source_idx = pass_action.source
        baseline_dist = np.empty(node_count, dtype=np.float32)
        distance_power_buffer = np.empty(node_count, dtype=np.float32)
        baseline_dist[:] = graph.distances(source=[source_idx], mode="all")[0]
        base_benefit = _sum_discounted_distances(
            self.distance_discount,
            baseline_dist,
            distance_power_buffer,
        )
        norm = float(total_nodes - 1)

        # Partition the action space once so each section has a single job.
        add_actions = action_space[ActionType.ADD]
        delete_actions = action_space[ActionType.DELETE]
        bridge_actions = action_space[ActionType.BRIDGE]
        add_targets = [action.target for action in add_actions]

        candidate_count = (
            1 + len(delete_actions) + len(add_actions) + len(bridge_actions)
        )
        candidate_actions: list[Action] = [pass_action] * candidate_count
        candidate_values = np.empty(candidate_count, dtype=np.float32)
        candidate_actions[0] = pass_action
        candidate_values[0] = 0.0
        candidate_idx = 1

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

        # Evaluate DELETE actions and cache the best follow-up benefit for each ADD.
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
                    distance_power_buffer,
                )
                value = (
                    self.benefit_to_cost_ratio * (deleted_benefit - base_benefit) + 1.0
                ) / norm - self.inertia_term

                candidate_actions[candidate_idx] = action
                candidate_values[candidate_idx] = value
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
                    np.add(
                        add_target_distance_buffer,
                        1.0,
                        out=combo_distance_buffer,
                    )
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

        # Evaluate ADD actions, using the best add-only or add-then-delete outcome.
        if (
            add_benefit_matrix is not None
            and add_target_distance_buffer is not None
            and combo_distance_buffer is not None
            and combo_power_buffer is not None
        ):
            add_target_distance_buffer[:] = graph.distances(
                source=add_targets, mode="all"
            )
            np.add(
                add_target_distance_buffer,
                1.0,
                out=combo_distance_buffer,
            )
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
            add_only_benefit = combo_power_buffer.sum(axis=1)
            add_benefit_matrix[:, -1] = add_only_benefit - (
                1.0 / self.benefit_to_cost_ratio
            )
            best_add_values = (
                self.benefit_to_cost_ratio
                * (np.max(add_benefit_matrix, axis=1) - base_benefit)
                / norm
            )

            for action, best_add_value in zip(add_actions, best_add_values):
                candidate_actions[candidate_idx] = action
                candidate_values[candidate_idx] = best_add_value - self.inertia_term
                candidate_idx += 1

        # Evaluate bridge exploration against PASS and keep tie-breaking random.
        if bridge_actions:
            bridge_value = (
                self.bridge_value_scale * (len(bridge_actions) / norm)
                - self.inertia_term
            )
            bridge_end = candidate_idx + len(bridge_actions)
            candidate_actions[candidate_idx:bridge_end] = bridge_actions
            candidate_values[candidate_idx:bridge_end] = bridge_value
            candidate_idx = bridge_end

        chosen_idx = _sample_boltzmann(
            candidate_values,
            self.temperature,
            self._rng,
        )
        chosen_action = candidate_actions[chosen_idx]
        return PolicyDecision(
            actions=(chosen_action,),
            value={"value": candidate_values[chosen_idx]},
        )
