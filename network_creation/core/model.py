"""Core value objects for the network creation domain."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from numbers import Integral
from typing import TYPE_CHECKING, Annotated, NamedTuple, cast

import numpy as np
from numpy.typing import NDArray

from .local_global_map import LocalGlobalMap

if TYPE_CHECKING:
    import igraph as ig

    from .agent import Agent


class ActionType(str, Enum):
    """Available graph mutation actions."""

    ADD = "ADD"
    DELETE = "DELETE"
    PASS = "PASS"
    BRIDGE = "BRIDGE"


class ActorScheduler:
    """Randomly select the agents that act during one simulator step."""

    _rng: np.random.Generator
    _n_actors: int

    def __init__(
        self,
        seed: int,
        n_agents: int,
        n_actors: int,
    ) -> None:
        self._rng = np.random.default_rng(seed)
        if (
            isinstance(n_agents, bool)
            or not isinstance(n_agents, Integral)
            or n_agents < 1
        ):
            raise ValueError("n_agents must be a positive integer")
        if (
            isinstance(n_actors, bool)
            or not isinstance(n_actors, Integral)
            or not 1 <= n_actors <= n_agents
        ):
            raise ValueError(
                "ActorScheduler n_actors must satisfy 1 <= n_actors <= n_agents"
            )

        self._n_actors = int(n_actors)

    def select_agents(self, agents: tuple[Agent, ...]) -> tuple[Agent, ...]:
        n_agents = len(agents)
        if self._n_actors >= n_agents:
            return agents
        agent_indices = cast(
            NDArray[np.int_],
            self._rng.choice(
                n_agents,
                size=self._n_actors,
                replace=False,
                shuffle=True,
            ),
        )
        return tuple(agents[int(agent_index)] for agent_index in agent_indices)


class Action(NamedTuple):
    """One agent-level action proposed for a step."""

    type: ActionType
    source: int
    target: int | None = None


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    """One policy-level decision for a single agent at one step."""

    actions: tuple[Action, ...]
    value: dict[str, object] | None = None


@dataclass(slots=True)
class ActionSpace:
    """Structured local-index action-space representation for one agent."""

    actions: dict[ActionType, list[Action]]

    def __getitem__(self, action_type: ActionType) -> list[Action]:
        """Return the actions available for one action type."""

        return self.actions[action_type]


@dataclass(slots=True)
class EnvironmentState:
    """Canonical directed graph state."""

    graph: ig.Graph


@dataclass(slots=True)
class Observation:
    """Agent-local structural view."""

    node_idx: int
    graph: ig.Graph
    outside_node_idxs: Annotated[NDArray[np.int_], "shape=(n_outside_nodes,)"]
    local_global: LocalGlobalMap

    @staticmethod
    def create_local_global_map(
        graph: ig.Graph,
        outside_global_node_ids: Annotated[
            NDArray[np.int_],
            "shape=(n_outside_nodes,)",
        ],
    ) -> LocalGlobalMap:
        visible_count = graph.vcount()
        local_to_global = [int(node_id) for node_id in graph.vs["name"]]
        local_to_global.extend(int(node_id) for node_id in outside_global_node_ids)
        if len(local_to_global) != visible_count + len(outside_global_node_ids):
            raise ValueError(
                "local-to-global map size must match visible and outside nodes"
            )
        return LocalGlobalMap(local_to_global)

    def global_from_local(self, local_idx: int) -> int:
        return self.local_global[int(local_idx)]

    def local_from_global(self, node_id: int) -> int:
        return self.local_global.inverse[int(node_id)]
