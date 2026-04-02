"""Observer events and protocol for simulator instrumentation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import igraph as ig
import numpy as np
from numpy.typing import NDArray

from .model import EnvironmentState, PolicyDecision, SchedulingMode


@dataclass(frozen=True)
class StateView:
    """Live graph reference passed to observer events."""

    graph: ig.Graph

    @classmethod
    def from_state(cls, state: EnvironmentState) -> StateView:
        return cls(graph=state.graph)


@dataclass(frozen=True)
class EpisodeStartEvent:
    """Episode metadata emitted before the first step."""

    episode_index: int
    seed: int
    mode: SchedulingMode
    max_steps: int
    n_agents: int
    initial_state: StateView


@dataclass(frozen=True)
class StepEvent:
    """Per-step event emitted after the environment transition."""

    step: int
    mode: SchedulingMode
    acting_agent_ids: tuple[int, ...]
    decisions: tuple[PolicyDecision, ...]
    modified: bool
    pre_state: StateView
    post_state: StateView


@dataclass(frozen=True)
class EpisodeEndEvent:
    """Episode summary emitted after the last executed step."""

    episode_index: int
    seed: int
    mode: SchedulingMode
    current_step: int
    last_modified_step: int
    final_state: StateView


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
