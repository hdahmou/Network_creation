"""Environment wrapper around the graph-domain helpers."""

from __future__ import annotations

from collections.abc import Sequence

import igraph as ig

from .agent import Agent
from .graph import (
    apply_actions,
    build_observation_from_state,
    build_state_from_graph,
    get_action_space,
)
from .model import Action, ActionSpace, EnvironmentState, Observation


class Environment:
    """Thin stateful shell around the graph helpers."""

    def __init__(self, graph: ig.Graph, seed: int) -> None:
        self.seed = seed
        self._initial_graph = graph.copy()
        self.state = build_state_from_graph(self._initial_graph)
        self.node_ids = [int(node_id) for node_id in self.state.graph.vs["name"]]

    def reset(self) -> EnvironmentState:
        """Restore the initial graph state."""

        self.state = build_state_from_graph(self._initial_graph)
        return self.state

    def step(self, actions: Sequence[Action]) -> EnvironmentState:
        self.state = apply_actions(self.state, actions)
        return self.state

    def get_observation(self, agent: Agent) -> Observation:
        return build_observation_from_state(
            self.state,
            node_id=agent.id,
            radius=agent.get_visibility_distance(),
        )

    def get_action_space(self, agent: Agent) -> ActionSpace:
        return get_action_space(self.get_observation(agent))
