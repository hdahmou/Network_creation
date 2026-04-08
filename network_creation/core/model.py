"""Core value objects for the network creation domain."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import floor
from typing import TYPE_CHECKING, Annotated, NamedTuple

from bidict import bidict
import numpy as np
from numpy.typing import NDArray

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

    def __init__(
        self,
        seed: int,
        n_agents: int,
        fraction: float,
    ) -> None:
        self._rng = np.random.default_rng(seed)
        if isinstance(n_agents, bool) or n_agents < 1:
            raise ValueError("n_agents must be positive")
        if (
            isinstance(fraction, bool)
            or not np.isfinite(fraction)
            or not 1 / n_agents <= fraction <= 1
        ):
            raise ValueError(
                "ActorScheduler fraction must satisfy 1 / n_agents <= fraction <= 1"
            )

        self._n_actors = floor(fraction * n_agents)

    def select_agents(self, agents: tuple[Agent, ...]) -> tuple[Agent, ...]:
        n_agents = len(agents)
        if self._n_actors >= n_agents:
            return agents
        agent_indices = self._rng.choice(
            n_agents,
            size=self._n_actors,
            replace=False,
            shuffle=False,
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
    local_global: bidict[int, int]

    @staticmethod
    def create_local_global_map(
        graph: ig.Graph,
        outside_global_node_ids: Annotated[
            NDArray[np.int_],
            "shape=(n_outside_nodes,)",
        ],
    ) -> bidict[int, int]:
        local_global = bidict(
            {
                int(vertex.index): int(vertex["name"])
                for vertex in graph.vs
            }
        )
        visible_count = graph.vcount()
        local_global.update(
            {
                visible_count + outside_idx: int(node_id)
                for outside_idx, node_id in enumerate(outside_global_node_ids)
            }
        )
        return local_global

    def global_from_local(self, local_idx: int) -> int:
        return self.local_global[int(local_idx)]

    def local_from_global(self, node_id: int) -> int:
        return self.local_global.inverse[int(node_id)]
