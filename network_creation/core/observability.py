"""Observer events and protocol for simulator instrumentation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol

import igraph as ig
import numpy as np
from numpy.typing import NDArray

from .model import EnvironmentState, PolicyDecision, SchedulingMode


@dataclass(frozen=True)
class ObservationSampling:
    """Step-sampling policy shared by simulator observers."""

    kind: Literal["periodic", "terminal"]
    every: int | None = None

    def __post_init__(self) -> None:
        if self.kind == "periodic":
            if self.every is None or self.every < 1:
                raise ValueError("periodic observation sampling requires every >= 1")
            return
        if self.kind == "terminal":
            if self.every is not None:
                raise ValueError("terminal observation sampling does not accept 'every'")
            return
        raise ValueError(f"unsupported observation sampling kind: {self.kind}")

    @classmethod
    def periodic(cls, every: int) -> ObservationSampling:
        return cls(kind="periodic", every=every)

    @classmethod
    def terminal(cls) -> ObservationSampling:
        return cls(kind="terminal")

    def samples_step(self, step_index: int, max_steps: int) -> bool:
        if self.kind == "periodic":
            assert self.every is not None
            return (step_index + 1) % self.every == 0
        return max_steps > 0 and step_index == max_steps - 1

    def fixed_sample_count(self, max_steps: int) -> int:
        if self.kind == "periodic":
            assert self.every is not None
            return 1 + max_steps // self.every
        return 1


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

    observation_sampling: ObservationSampling

    def reset(self) -> None:
        """Clear all observer state accumulated across episodes."""

    def on_episode_start(self, event: EpisodeStartEvent) -> None:
        """Handle the start of one episode."""

    def on_step_end(self, event: StepEvent) -> None:
        """Handle one completed simulator step."""

    def on_episode_end(self, event: EpisodeEndEvent) -> None:
        """Handle the end of one episode."""

    def observes_step(self, step_index: int, max_steps: int) -> bool:
        """Return whether one completed simulator step should emit a step event."""
        return self.observation_sampling.samples_step(step_index, max_steps)

    def get(self) -> Any:
        """Return the observer's current batch-level result."""
