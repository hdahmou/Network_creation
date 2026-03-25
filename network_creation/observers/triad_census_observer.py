"""Observer that tracks selected triad census counts across simulator steps."""

from __future__ import annotations

from math import comb

import igraph as ig
import numpy as np
from numpy.typing import NDArray

from network_creation.core.observability import EpisodeStartEvent, StepEvent
from network_creation.observers.statistics.episode_tensor_series import (
    EpisodeSeriesTensorStatisticObserver,
)

TRIAD_TYPE_NAMES = (
    "003",
    "012",
    "021D",
    "021U",
    "021C",
    "030T",
    "030C",
)


def _compute_triad_proportions(
    ownership_matrix: NDArray[np.int8],
) -> NDArray[np.float32]:
    directed_graph = ig.Graph.Adjacency(ownership_matrix.tolist(), mode="directed")
    triad_census = directed_graph.triad_census()
    total_possible_triads = comb(ownership_matrix.shape[0], 3)
    return np.array(
        [
            triad_census[triad_type_name] / total_possible_triads
            for triad_type_name in TRIAD_TYPE_NAMES
        ],
        dtype=np.float32,
    )


class TriadCensusObserver(EpisodeSeriesTensorStatisticObserver):
    """Collect selected triad census proportions per sampled step."""

    label = "Triad Census"
    x_label = "step"
    y_label = "triad proportion"
    y_min = 0.0
    y_max = None
    dimension_names = TRIAD_TYPE_NAMES

    @property
    def triad_type_names(self) -> tuple[str, ...]:
        return TRIAD_TYPE_NAMES

    @property
    def triad_proportions(self) -> NDArray[np.float32]:
        return self.values

    def compute_initial_value(self, event: EpisodeStartEvent) -> NDArray[np.float32]:
        return _compute_triad_proportions(event.initial_state.ownership_matrix)

    def compute_step_value(self, event: StepEvent) -> NDArray[np.float32]:
        return _compute_triad_proportions(event.post_state.ownership_matrix)
