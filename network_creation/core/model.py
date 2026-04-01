"""Core value objects for the network creation domain."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Annotated

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    import igraph as ig


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


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    """One policy-level decision for a single agent at one step."""

    actions: tuple[Action, ...]
    value: dict[str, object] | None = None


@dataclass(slots=True)
class ActionSpace:
    """Structured action-space representation for one agent."""

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

    node_id: int
    graph: ig.Graph
    outside_node_ids: Annotated[NDArray[np.int_], "shape=(n_outside_nodes,)"]
