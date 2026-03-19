"""Core value objects for the network creation domain."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Annotated

import igraph as ig
import numpy as np
from numpy.typing import NDArray


class ActionType(str, Enum):
    """Available graph mutation actions."""

    ADD = "ADD"
    DELETE = "DELETE"
    PASS = "PASS"
    BRIDGE = "BRIDGE"


class SchedulingMode(str, Enum):
    """How a simulator selects acting agents at each step."""

    SEQUENTIAL = "SEQUENTIAL"
    SIMULTANEOUS = "SIMULTANEOUS"


@dataclass(frozen=True, slots=True)
class Action:
    """One agent-level action proposed for a step."""

    type: ActionType
    source: int
    target: int | None = None

    def __post_init__(self) -> None:
        if self.source < 0:
            raise ValueError("source must be non-negative")

        if self.type is ActionType.PASS:
            if self.target is not None:
                raise ValueError("PASS actions cannot define a target")
            return

        if self.target is None:
            raise ValueError(f"{self.type.value} actions require a target")
        if self.target < 0:
            raise ValueError("target must be non-negative")
        if self.target == self.source:
            raise ValueError("self-links are illegal")


@dataclass(slots=True)
class ActionSpace:
    """Structured action-space representation for one agent."""

    actions: dict[ActionType, list[Action]]

    def __getitem__(self, action_type: ActionType) -> list[Action]:
        """Return the actions available for one action type."""

        return self.actions[action_type]


@dataclass(slots=True)
class EnvironmentState:
    """Structural graph plus global link ownership state."""

    graph: ig.Graph
    ownership_matrix: Annotated[NDArray[np.int8], "shape=(n_agents, n_agents)"]


@dataclass(slots=True)
class Observation:
    """Agent-local structural view and ownership slice."""

    node_id: int
    graph: ig.Graph
    outside_node_ids: Annotated[NDArray[np.int_], "shape=(n_outside_nodes,)"]
    paid_node_ids: Annotated[NDArray[np.int_], "shape=(n_paid_nodes,)"]
