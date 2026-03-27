"""Observer that tracks main-hub persistence over adjacent sliding windows."""

from __future__ import annotations

import igraph as ig
import numpy as np
from numpy.typing import NDArray

from network_creation.core.observability import (
    EpisodeEndEvent,
    EpisodeStartEvent,
    StepEvent,
)
from network_creation.observers.statistics.episode_series import (
    EpisodeSeriesStatisticObserver,
)


def _get_hub_node_ids(graph: ig.Graph) -> set[int]:
    """Return all tied hub node ids for one graph state."""

    degrees = [
        len(set(graph.neighbors(node_id, mode="all")))
        for node_id in range(graph.vcount())
    ]
    max_degree = max(degrees)
    return {node_id for node_id, degree in enumerate(degrees) if degree == max_degree}


def _get_main_hub_node_ids(hub_node_ids_by_state: list[set[int]]) -> set[int]:
    """Return all tied main hubs for one sliding window."""

    counts: dict[int, int] = {}
    for hub_node_ids in hub_node_ids_by_state:
        for hub_node_id in hub_node_ids:
            counts[hub_node_id] = counts.get(hub_node_id, 0) + 1

    if not counts:
        return set()

    max_count = max(counts.values())
    return {hub_node_id for hub_node_id, count in counts.items() if count == max_count}


class HubPersistenceObserver(EpisodeSeriesStatisticObserver):
    """Collect adjacent-window hub-persistence scores per sampled step."""

    label = "Hub Persistence"
    x_label = "step"
    y_label = "hub persistence score"
    y_min = 0.0
    y_max = 1.0 + 1e-1

    def __init__(
        self,
        observation_frequency: int,
        window_size: int,
        aggregations: tuple[str, ...],
    ) -> None:
        if observation_frequency != 1:
            raise ValueError(
                "HubPersistenceObserver requires observation_frequency=1 "
                "because the ten-step window is defined on raw simulator steps"
            )
        if window_size < 1:
            raise ValueError("window_size must be positive")
        self._window_size = window_size
        super().__init__(
            observation_frequency=observation_frequency,
            aggregations=aggregations,
        )

    def _reset_episode_state(self) -> None:
        self._hub_node_ids_by_state: list[set[int]] = []
        super()._reset_episode_state()

    @property
    def hub_persistence_scores(self) -> NDArray[np.float32]:
        return self.values

    def _start_episode(self, event: EpisodeStartEvent) -> None:
        n_slots = 1 + event.max_steps
        self.n_observations = 1
        self._episode_steps = np.empty(n_slots, dtype=np.int_)
        self._episode_values = np.full(n_slots, np.nan, dtype=np.float32)
        self._episode_steps[0] = 0
        self._hub_node_ids_by_state = [_get_hub_node_ids(event.initial_state.graph)]

    def _record_step(self, event: StepEvent) -> None:
        observation_idx = self.n_observations
        self._episode_steps[observation_idx] = event.step + 1
        self._hub_node_ids_by_state.append(_get_hub_node_ids(event.post_state.graph))
        self.n_observations += 1

    def _get_episode_values(self, event: EpisodeEndEvent) -> NDArray[np.float32]:
        del event
        self._episode_values[: self.n_observations] = np.nan

        first_valid_idx = self._window_size - 1
        stop_idx = self.n_observations - self._window_size
        for idx in range(first_valid_idx, stop_idx):
            previous_window_hub_node_ids = self._hub_node_ids_by_state[
                idx - self._window_size + 1 : idx + 1
            ]
            next_window_hub_node_ids = self._hub_node_ids_by_state[
                idx + 1 : idx + 1 + self._window_size
            ]

            previous_main_hub_node_ids = _get_main_hub_node_ids(
                previous_window_hub_node_ids
            )
            next_main_hub_node_ids = _get_main_hub_node_ids(next_window_hub_node_ids)
            if not previous_main_hub_node_ids or not next_main_hub_node_ids:
                continue

            self._episode_values[idx] = float(
                len(previous_main_hub_node_ids.intersection(next_main_hub_node_ids))
                / len(previous_main_hub_node_ids)
            )

        return self.values

    def compute_initial_value(self, event: EpisodeStartEvent) -> float:
        del event
        return float("nan")

    def compute_step_value(self, event: StepEvent) -> float:
        del event
        return float("nan")
