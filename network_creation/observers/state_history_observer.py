"""Observer that stores per-step graph and ownership history for plotting."""

from __future__ import annotations

from dataclasses import dataclass

import igraph as ig
import numpy as np
from numpy.typing import NDArray

from network_creation.core.graph import build_graph_from_ownership_matrix
from network_creation.core.observability import (
    EpisodeEndEvent,
    EpisodeStartEvent,
    SimulationObserver,
    StepEvent,
)


@dataclass(frozen=True)
class StateHistoryResult:
    """Batch-aware ownership history output."""

    steps: NDArray[np.int_]
    ownership_matrices: NDArray[np.int8]
    n_episodes: int
    runs_steps: tuple[NDArray[np.int_], ...] | None
    runs_ownership_matrices: tuple[NDArray[np.int8], ...] | None


class StateHistoryObserver(SimulationObserver):
    """Accumulate sampled ownership history and optionally keep every episode."""

    def __init__(self, observation_frequency: int, keep_runs: bool) -> None:
        self.observation_frequency = observation_frequency
        self.keep_runs = keep_runs
        self.reset()

    def reset(self) -> None:
        self.n_episodes = 0
        self.n_observations = 0
        self._steps = np.empty(0, dtype=np.int_)
        self._ownership_matrices = np.empty((0, 0, 0), dtype=np.int8)
        self._runs_steps = [] if self.keep_runs else None
        self._runs_ownership_matrices = [] if self.keep_runs else None

    @property
    def steps(self) -> NDArray[np.int_]:
        return self._steps[: self.n_observations]

    @property
    def ownership_matrices(self) -> NDArray[np.int8]:
        return self._ownership_matrices[: self.n_observations]

    @property
    def graphs(self) -> list[ig.Graph]:
        return [
            build_graph_from_ownership_matrix(ownership_matrix)
            for ownership_matrix in self.ownership_matrices
        ]

    def on_episode_start(self, event: EpisodeStartEvent) -> None:
        n_slots = 1 + event.max_steps // self.observation_frequency
        self.n_observations = 1
        self._steps = np.empty(n_slots, dtype=np.int_)
        self._ownership_matrices = np.empty(
            (n_slots, event.n_agents, event.n_agents),
            dtype=np.int8,
        )
        self._steps[0] = 0
        self._ownership_matrices[0] = event.initial_state.ownership_matrix

    def on_step_end(self, event: StepEvent) -> None:
        observation_idx = self.n_observations
        self._steps[observation_idx] = event.step + 1
        self._ownership_matrices[observation_idx] = event.post_state.ownership_matrix
        self.n_observations += 1

    def on_episode_end(self, event: EpisodeEndEvent) -> None:
        del event
        self.n_episodes += 1
        if self._runs_steps is not None:
            self._runs_steps.append(self.steps.copy())
        if self._runs_ownership_matrices is not None:
            self._runs_ownership_matrices.append(self.ownership_matrices.copy())

    def get(self) -> StateHistoryResult:
        return StateHistoryResult(
            steps=self.steps.copy(),
            ownership_matrices=self.ownership_matrices.copy(),
            n_episodes=self.n_episodes,
            runs_steps=None if self._runs_steps is None else tuple(self._runs_steps),
            runs_ownership_matrices=(
                None
                if self._runs_ownership_matrices is None
                else tuple(self._runs_ownership_matrices)
            ),
        )
