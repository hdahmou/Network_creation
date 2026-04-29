"""Core graph construction, observation, action-space, and mutation helpers."""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Real
import random
from threading import Lock

import igraph as ig
import numpy as np

from .model import Action, ActionSpace, ActionType, EnvironmentState, Observation

_IGRAPH_RNG_LOCK = Lock()


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


def create_erdos_renyi_graph(n_agents: int, p: float, seed: int) -> ig.Graph:
    """Create a directed Erdős-Rényi graph for a fixed node set."""

    if isinstance(p, bool) or not isinstance(p, Real) or not 0.0 <= float(p) <= 1.0:
        raise ValueError("p must be a number in [0, 1]")

    with _IGRAPH_RNG_LOCK:
        ig.set_random_number_generator(random.Random(int(seed)))
        try:
            graph = ig.Graph.Erdos_Renyi(
                n=n_agents,
                p=float(p),
                directed=True,
                loops=False,
            )
        finally:
            ig.set_random_number_generator(random)

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

    outside_global_node_ids = np.flatnonzero(~visible_mask)

    graph = state.graph.induced_subgraph(visible_node_ids)
    local_global = Observation.create_local_global_map(graph, outside_global_node_ids)

    return Observation(
        node_idx=local_global.inverse[int(node_id)],
        graph=graph,
        outside_node_idxs=np.arange(
            graph.vcount(),
            graph.vcount() + len(outside_global_node_ids),
            dtype=np.int_,
        ),
        local_global=local_global,
    )


def get_action_space(observation: Observation) -> ActionSpace:
    """Build a structured action space from one node observation.

    Hidden nodes outside the visible region are collapsed into one generic
    BRIDGE action. The simulator later resolves the actual unseen target when
    that bridge is executed.
    """

    graph = observation.graph
    source_idx = observation.node_idx
    out_neighbor_indices = set(graph.neighbors(source_idx, mode="out"))

    add_actions = [
        Action(ActionType.ADD, source_idx, target_idx)
        for target_idx in range(graph.vcount())
        if target_idx != source_idx and target_idx not in out_neighbor_indices
    ]
    bridge_actions = []
    if len(observation.outside_node_idxs) > 0:
        bridge_actions.append(Action(ActionType.BRIDGE, source_idx))
    delete_actions = [
        Action(
            ActionType.DELETE,
            source_idx,
            target_idx,
        )
        for target_idx in graph.neighbors(
            source_idx,
            mode="out",
        )
    ]
    pass_action = [Action(ActionType.PASS, source_idx)]

    return ActionSpace(
        actions={
            ActionType.ADD: add_actions,
            ActionType.BRIDGE: bridge_actions,
            ActionType.DELETE: delete_actions,
            ActionType.PASS: pass_action,
        },
    )


def apply_actions(
    state: EnvironmentState, actions: Sequence[Action]
) -> EnvironmentState:
    """Apply one step of already-validated actions."""

    updated_graph = state.graph.copy()

    for action in actions:
        if action.type in {ActionType.ADD, ActionType.BRIDGE}:
            if action.target is None:
                raise ValueError(
                    f"{action.type.value} actions must define a target before execution"
                )
            updated_graph.add_edge(action.source, action.target)

    for action in actions:
        if action.type is ActionType.DELETE:
            edge_id = updated_graph.get_eid(
                action.source, action.target, directed=True, error=True
            )
            updated_graph.delete_edges([edge_id])

    return EnvironmentState(graph=updated_graph)
