"""Core graph construction, observation, action-space, and mutation helpers."""

from __future__ import annotations

from collections.abc import Sequence

import igraph as ig
import numpy as np

from .model import Action, ActionSpace, ActionType, EnvironmentState, Observation


def build_state_from_graph(graph: ig.Graph) -> EnvironmentState:
    """Create a full environment state from a directed graph."""
    graph = graph.copy()
    graph.vs["name"] = list(range(graph.vcount()))
    return EnvironmentState(graph=graph)


def create_empty_graph(n_agents: int) -> ig.Graph:
    """Create the empty directed graph for a fixed node set."""

    graph = ig.Graph(n=n_agents, directed=True)
    graph.vs["name"] = list(range(n_agents))
    return graph


def build_observation_from_state(
    state: EnvironmentState, node_id: int, radius: int
) -> Observation:
    """Build the observation seen from one node."""

    visible_node_ids = np.asarray(
        state.graph.neighborhood(vertices=node_id, order=radius, mode="all"),
        dtype=np.int_,
    )
    visible_mask = np.zeros(state.graph.vcount(), dtype=bool)
    visible_mask[visible_node_ids] = True

    outside_node_ids = np.flatnonzero(~visible_mask)

    return Observation(
        node_id=node_id,
        graph=state.graph.induced_subgraph(visible_node_ids),
        outside_node_ids=outside_node_ids,
    )


def get_action_space(observation: Observation) -> ActionSpace:
    """Build a structured action space from one node observation."""

    visible_node_ids = tuple(observation.graph.vs["name"])
    source_subgraph_idx = visible_node_ids.index(observation.node_id)
    neighbor_node_ids = {
        visible_node_ids[subgraph_idx]
        for subgraph_idx in observation.graph.neighbors(source_subgraph_idx, mode="all")
    }
    delete_node_ids = sorted(
        {
            visible_node_ids[subgraph_idx]
            for subgraph_idx in observation.graph.neighbors(
                source_subgraph_idx,
                mode="out",
            )
        }
    )
    add_node_ids = sorted(
        node_id
        for node_id in visible_node_ids
        if node_id != observation.node_id and node_id not in neighbor_node_ids
    )
    bridge_node_ids = tuple(int(node_id) for node_id in observation.outside_node_ids)

    return ActionSpace(
        actions={
            ActionType.ADD: [
                Action(ActionType.ADD, source=observation.node_id, target=node_id)
                for node_id in add_node_ids
            ],
            ActionType.BRIDGE: [
                Action(ActionType.BRIDGE, source=observation.node_id, target=node_id)
                for node_id in bridge_node_ids
            ],
            ActionType.DELETE: [
                Action(ActionType.DELETE, source=observation.node_id, target=node_id)
                for node_id in delete_node_ids
            ],
            ActionType.PASS: [Action(ActionType.PASS, source=observation.node_id)],
        },
    )


def apply_actions(
    state: EnvironmentState, actions: Sequence[Action]
) -> EnvironmentState:
    """Apply one step of already-validated actions."""

    updated_graph = state.graph.copy()

    for action in actions:
        if action.type in {ActionType.ADD, ActionType.BRIDGE}:
            updated_graph.add_edge(action.source, action.target)

    for action in actions:
        if action.type is ActionType.DELETE:
            edge_id = updated_graph.get_eid(
                action.source, action.target, directed=True, error=True
            )
            updated_graph.delete_edges([edge_id])

    return EnvironmentState(graph=updated_graph)
