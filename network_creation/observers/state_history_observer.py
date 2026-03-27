"""Observer that stores per-step graph history for plotting."""

from __future__ import annotations

from dataclasses import dataclass

import igraph as ig
import numpy as np
from numpy.typing import NDArray

from network_creation.core.observability import (
    EpisodeEndEvent,
    EpisodeStartEvent,
    SimulationObserver,
    StepEvent,
)


@dataclass(frozen=True)
class StateHistoryResult:
    """Batch-aware graph history output."""

    steps: NDArray[np.int_]
    graphs: tuple[ig.Graph, ...]
    n_episodes: int
    runs_steps: tuple[NDArray[np.int_], ...] | None
    runs_graphs: tuple[tuple[ig.Graph, ...], ...] | None


class StateHistoryObserver(SimulationObserver):
    """Accumulate sampled graph history and optionally keep every episode."""

    def __init__(self, observation_frequency: int, keep_runs: bool) -> None:
        self.observation_frequency = observation_frequency
        self.keep_runs = keep_runs
        self.reset()

    def reset(self) -> None:
        self.n_episodes = 0
        self.n_observations = 0
        self._steps = np.empty(0, dtype=np.int_)
        self._graphs_by_observation: list[ig.Graph] = []
        self._runs_steps = [] if self.keep_runs else None
        self._runs_graphs = [] if self.keep_runs else None

    @property
    def steps(self) -> NDArray[np.int_]:
        return self._steps[: self.n_observations]

    @property
    def graphs(self) -> np.ndarray:
        return np.asarray(
            [
                graph.copy()
                for graph in self._graphs_by_observation[: self.n_observations]
            ],
            dtype=object,
        )

    def on_episode_start(self, event: EpisodeStartEvent) -> None:
        n_slots = 1 + event.max_steps // self.observation_frequency
        self.n_observations = 1
        self._steps = np.empty(n_slots, dtype=np.int_)
        self._graphs_by_observation = [ig.Graph() for _ in range(n_slots)]
        self._steps[0] = 0
        self._graphs_by_observation[0] = event.initial_state.graph.copy()

    def on_step_end(self, event: StepEvent) -> None:
        observation_idx = self.n_observations
        self._steps[observation_idx] = event.step + 1
        self._graphs_by_observation[observation_idx] = event.post_state.graph.copy()
        self.n_observations += 1

    def on_episode_end(self, event: EpisodeEndEvent) -> None:
        del event
        self.n_episodes += 1
        if self._runs_steps is not None:
            self._runs_steps.append(self.steps.copy())
        if self._runs_graphs is not None:
            self._runs_graphs.append(tuple(self.graphs))

    def get(self) -> StateHistoryResult:
        return StateHistoryResult(
            steps=self.steps.copy(),
            graphs=tuple(self.graphs),
            n_episodes=self.n_episodes,
            runs_steps=None if self._runs_steps is None else tuple(self._runs_steps),
            runs_graphs=None if self._runs_graphs is None else tuple(self._runs_graphs),
        )
