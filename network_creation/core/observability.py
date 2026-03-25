"""Observer events and protocol for simulator instrumentation."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Any, Protocol

import igraph as ig
import numpy as np
from numpy.typing import NDArray

from .graph import build_graph_from_ownership_matrix
from .model import Action, EnvironmentState, SchedulingMode


def _make_readonly_ownership_matrix(
    ownership_matrix: NDArray[np.int8],
) -> NDArray[np.int8]:
    readonly_ownership_matrix = ownership_matrix.view()
    readonly_ownership_matrix.setflags(write=False)
    return readonly_ownership_matrix


@dataclass(frozen=True)
class StateSnapshot:
    """Read-only ownership snapshot with lazy graph reconstruction."""

    ownership_matrix: NDArray[np.int8]

    @classmethod
    def from_state(cls, state: EnvironmentState) -> StateSnapshot:
        return cls(_make_readonly_ownership_matrix(state.ownership_matrix))

    @cached_property
    def graph(self) -> ig.Graph:
        return build_graph_from_ownership_matrix(self.ownership_matrix)


@dataclass(frozen=True)
class EpisodeStartEvent:
    """Episode metadata emitted before the first step."""

    episode_index: int
    seed: int
    mode: SchedulingMode
    max_steps: int
    n_agents: int
    initial_state: StateSnapshot


@dataclass(frozen=True)
class StepEvent:
    """Per-step event emitted after the environment transition."""

    step: int
    mode: SchedulingMode
    acting_agent_ids: tuple[int, ...]
    actions: tuple[Action, ...]
    modified: bool
    pre_state: StateSnapshot
    post_state: StateSnapshot


@dataclass(frozen=True)
class EpisodeEndEvent:
    """Episode summary emitted after the last executed step."""

    episode_index: int
    seed: int
    mode: SchedulingMode
    current_step: int
    last_modified_step: int
    final_state: StateSnapshot


@dataclass(frozen=True)
class StatisticResult:
    """Aggregated observer output across all episodes seen since the last reset."""

    steps: NDArray[np.int_] | None
    n_episodes: int
    outputs: dict[str, NDArray[np.float32] | tuple[NDArray[np.float32], ...]]
    sample_shape: tuple[int, ...] = ()
    dimension_names: tuple[str, ...] | None = None


class SimulationObserver(Protocol):
    """Observer interface for simulator lifecycle events across multiple episodes."""

    observation_frequency: int

    def reset(self) -> None:
        """Clear all observer state accumulated across episodes."""

    def on_episode_start(self, event: EpisodeStartEvent) -> None:
        """Handle the start of one episode."""

    def on_step_end(self, event: StepEvent) -> None:
        """Handle one completed simulator step."""

    def on_episode_end(self, event: EpisodeEndEvent) -> None:
        """Handle the end of one episode."""

    def get(self) -> Any:
        """Return the observer's current batch-level result."""
