"""Environment wrapper around the graph-domain helpers."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from .agent import Agent
from .graph import (
    apply_actions,
    build_observation_from_state,
    build_state_from_ownership_matrix,
    get_action_space,
)
from .model import Action, ActionSpace, EnvironmentState, Observation


class Environment:
    """Thin stateful shell around the graph helpers."""

    def __init__(self, ownership_matrix: NDArray[np.int8], seed: int) -> None:
        self.seed = seed
        self._initial_ownership_matrix = np.array(
            ownership_matrix,
            dtype=np.int8,
            copy=True,
        )
        self.state = build_state_from_ownership_matrix(
            self._initial_ownership_matrix.copy()
        )
        self.node_ids = [int(node_id) for node_id in self.state.graph.vs["name"]]

    def reset(self) -> EnvironmentState:
        """Restore the initial graph state."""

        self.state = build_state_from_ownership_matrix(
            self._initial_ownership_matrix.copy()
        )
        return self.state

    def step(self, actions: Sequence[Action]) -> EnvironmentState:
        """Apply one step of already-validated actions."""

        self.state = apply_actions(self.state, actions)
        return self.state

    def get_observation(self, agent: Agent) -> Observation:
        """Build the current observation for one agent."""

        return build_observation_from_state(
            self.state, node_id=agent.id, radius=agent.radius
        )

    def get_action_space(self, agent: Agent) -> ActionSpace:
        """Build the current action space for one agent."""

        return get_action_space(self.get_observation(agent))

    def render(self) -> None:
        """Print a compact graph summary for debugging."""

        print(self.state.graph.summary())
