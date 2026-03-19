"""Core graph construction, observation, action-space, and mutation helpers."""

from __future__ import annotations

from collections.abc import Sequence

import igraph as ig
import numpy as np
from numpy.typing import NDArray

from .model import Action, ActionSpace, ActionType, EnvironmentState, Observation


def build_graph_from_ownership_matrix(ownership_matrix: NDArray[np.int8]) -> ig.Graph:
    """Build the undirected structural graph implied by ownership data."""

    graph = ig.Graph.Adjacency(ownership_matrix.tolist(), mode="max")
    graph.vs["name"] = list(range(graph.vcount()))
    return graph


def build_state_from_ownership_matrix(
    ownership_matrix: NDArray[np.int8],
) -> EnvironmentState:
    """Create a full environment state from ownership data."""

    return EnvironmentState(
        graph=build_graph_from_ownership_matrix(ownership_matrix),
        ownership_matrix=ownership_matrix,
    )


def create_empty_state(n_agents: int) -> EnvironmentState:
    """Create the empty graph state for a fixed node set."""

    return build_state_from_ownership_matrix(
        np.zeros((n_agents, n_agents), dtype=np.int8)
    )


def get_visible_node_ids(graph: ig.Graph, node_id: int, radius: int) -> list[int]:
    """Return the node ids visible from a node within a shortest-path radius."""

    visible_node_ids = graph.neighborhood(vertices=node_id, order=radius, mode="all")
    return sorted(int(visible_node_id) for visible_node_id in visible_node_ids)


def build_observation_from_state(
    state: EnvironmentState, node_id: int, radius: int
) -> Observation:
    """Build the observation seen from one node."""

    visible_node_ids = np.asarray(
        get_visible_node_ids(state.graph, node_id, radius), dtype=np.int_
    )
    visible_mask = np.zeros(state.graph.vcount(), dtype=bool)
    visible_mask[visible_node_ids] = True

    outside_node_ids = np.flatnonzero(~visible_mask)
    paid_node_ids = np.flatnonzero(state.ownership_matrix[node_id])

    return Observation(
        node_id=node_id,
        graph=state.graph.induced_subgraph(visible_node_ids),
        outside_node_ids=outside_node_ids,
        paid_node_ids=paid_node_ids,
    )


def _get_neighbor_node_ids(
    observation: Observation, visible_node_ids: np.ndarray
) -> np.ndarray:
    source_subgraph_idx = observation.graph.vs.find(name=observation.node_id).index
    neighbor_subgraph_indices = np.asarray(
        observation.graph.neighbors(source_subgraph_idx),
        dtype=np.int_,
    )
    if neighbor_subgraph_indices.size == 0:
        return np.array([], dtype=np.int_)
    return visible_node_ids[neighbor_subgraph_indices]


def _build_actions_for_node_ids(
    action_type: ActionType, source_node_id: int, target_node_ids: np.ndarray
) -> list[Action]:
    return [
        Action(action_type, source=source_node_id, target=int(target_node_id))
        for target_node_id in target_node_ids
    ]


def get_action_space(observation: Observation) -> ActionSpace:
    """Build a structured action space from one node observation."""

    visible_node_ids = np.asarray(observation.graph.vs["name"], dtype=np.int_)
    neighbor_node_ids = _get_neighbor_node_ids(observation, visible_node_ids)
    add_node_ids = visible_node_ids[visible_node_ids != observation.node_id]

    return ActionSpace(
        actions={
            ActionType.ADD: _build_actions_for_node_ids(
                ActionType.ADD,
                observation.node_id,
                np.setdiff1d(add_node_ids, neighbor_node_ids, assume_unique=True),
            ),
            ActionType.BRIDGE: _build_actions_for_node_ids(
                ActionType.BRIDGE,
                observation.node_id,
                np.setdiff1d(
                    observation.outside_node_ids,
                    neighbor_node_ids,
                    assume_unique=True,
                ),
            ),
            ActionType.DELETE: _build_actions_for_node_ids(
                ActionType.DELETE,
                observation.node_id,
                observation.paid_node_ids,
            ),
            ActionType.PASS: [Action(ActionType.PASS, source=observation.node_id)],
        },
    )


def apply_actions(
    state: EnvironmentState, actions: Sequence[Action]
) -> EnvironmentState:
    """Apply one step of already-validated actions."""

    updated = state.ownership_matrix.copy()

    for action in actions:
        if action.type is ActionType.DELETE:
            assert action.target is not None
            updated[action.source, action.target] = 0

    for action in actions:
        if action.type in {ActionType.ADD, ActionType.BRIDGE}:
            assert action.target is not None
            updated[action.source, action.target] = 1

    return build_state_from_ownership_matrix(updated)
