"""Observer that tracks graph density across simulator steps."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from network_creation.core.observability import (
    EpisodeStartEvent,
    StepEvent,
)
from network_creation.observers.statistics.episode_series import (
    EpisodeSeriesStatisticObserver,
)


class DensityObserver(EpisodeSeriesStatisticObserver):
    """Collect density values per episode and aggregate them across episodes."""

    label = "Density"
    x_label = "step"
    y_label = "density"
    y_min = 0.0
    y_max = 1.0

    @property
    def densities(self) -> NDArray[np.float32]:
        return self.values

    def compute_initial_value(self, event: EpisodeStartEvent) -> float:
        n_nodes = event.initial_state.ownership_matrix.shape[0]
        return (
            0.0
            if n_nodes < 2
            else float(event.initial_state.graph.density(loops=False))
        )

    def compute_step_value(self, event: StepEvent) -> float:
        n_nodes = event.post_state.ownership_matrix.shape[0]
        return (
            0.0 if n_nodes < 2 else float(event.post_state.graph.density(loops=False))
        )
