"""Observer that tracks graph density across simulator steps."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from network_creation.core.observability import (
    EpisodeEndEvent,
    EpisodeStartEvent,
    SimulationObserver,
    StepEvent,
)


class DensityObserver(SimulationObserver):
    """Collect density values from each post-step graph."""

    label = "Density"
    x_label = "step"
    y_label = "density"
    y_min = 0.0
    y_max = 1.0

    def __init__(self, observation_frequency: int) -> None:
        self.observation_frequency = observation_frequency
        self.n_observations = 0
        self._steps = np.empty(0, dtype=np.int_)
        self._densities = np.empty(0, dtype=np.float32)

    @property
    def steps(self) -> NDArray[np.int_]:
        return self._steps[: self.n_observations]

    @property
    def densities(self) -> NDArray[np.float32]:
        return self._densities[: self.n_observations]

    def on_episode_start(self, event: EpisodeStartEvent) -> None:
        n_slots = 1 + event.max_steps // self.observation_frequency
        self.n_observations = 1
        self._steps = np.empty(n_slots, dtype=np.int_)
        self._densities = np.empty(n_slots, dtype=np.float32)
        n_nodes = event.initial_state.ownership_matrix.shape[0]
        density = (
            0.0 if n_nodes < 2 else float(event.initial_state.graph.density(loops=False))
        )
        self._steps[0] = 0
        self._densities[0] = density

    def on_step_end(self, event: StepEvent) -> None:
        n_nodes = event.post_state.ownership_matrix.shape[0]
        density = (
            0.0 if n_nodes < 2 else float(event.post_state.graph.density(loops=False))
        )

        observation_idx = self.n_observations
        self._steps[observation_idx] = event.step + 1
        self._densities[observation_idx] = density
        self.n_observations += 1

    def on_episode_end(self, event: EpisodeEndEvent) -> None:
        del event
