"""Observer that stores per-step graph and ownership history for plotting."""

from __future__ import annotations

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


class StateHistoryObserver(SimulationObserver):
    """Accumulate sampled post-step ownership history for plotting."""

    def __init__(self, observation_frequency: int) -> None:
        self.observation_frequency = observation_frequency
        self.n_observations = 0
        self._steps = np.empty(0, dtype=np.int_)
        self._ownership_matrices = np.empty((0, 0, 0), dtype=np.int8)

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
