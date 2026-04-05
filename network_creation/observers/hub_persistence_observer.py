"""Observer that tracks main-hub persistence over adjacent sliding windows."""

from __future__ import annotations

import igraph as ig
import numpy as np
from numpy.typing import NDArray

from network_creation.core.observability import (
    EpisodeEndEvent,
    EpisodeStartEvent,
    ObservationSampling,
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
        observation_sampling: ObservationSampling,
        window_size: int,
        aggregations: tuple[str, ...],
    ) -> None:
        if window_size < 1:
            raise ValueError("window_size must be positive")
        self._window_size = window_size
        super().__init__(
            observation_sampling=observation_sampling,
            aggregations=aggregations,
        )

    def _reset_episode_state(self) -> None:
        self._hub_node_ids_by_state: list[set[int]] = []
        self._raw_n_observations = 0
        self._raw_steps = np.empty(0, dtype=np.int_)
        self._raw_values = np.empty(0, dtype=np.float32)
        self.n_observations = 0
        self._sampled_steps = np.empty(0, dtype=np.int_)
        self._sampled_values = np.empty(0, dtype=np.float32)

    def observes_step(self, step_index: int, max_steps: int) -> bool:
        del step_index
        return max_steps > 0

    def _start_episode(self, event: EpisodeStartEvent) -> None:
        n_slots = 1 + event.max_steps
        self._raw_n_observations = 1
        self._raw_steps = np.empty(n_slots, dtype=np.int_)
        self._raw_values = np.full(n_slots, np.nan, dtype=np.float32)
        self._raw_steps[0] = 0
        self._sampled_steps = np.array([0], dtype=np.int_)
        self._sampled_values = np.array([np.nan], dtype=np.float32)
        self.n_observations = 1
        self._hub_node_ids_by_state = [_get_hub_node_ids(event.initial_state.graph)]

    def _record_step(self, event: StepEvent) -> None:
        observation_idx = self._raw_n_observations
        self._raw_steps[observation_idx] = event.step + 1
        self._hub_node_ids_by_state.append(_get_hub_node_ids(event.post_state.graph))
        self._raw_n_observations += 1

    def _get_episode_values(self, event: EpisodeEndEvent) -> NDArray[np.float32]:
        del event
        self._raw_values[: self._raw_n_observations] = np.nan

        first_valid_idx = self._window_size - 1
        stop_idx = self._raw_n_observations - self._window_size
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

            self._raw_values[idx] = float(
                len(previous_main_hub_node_ids.intersection(next_main_hub_node_ids))
                / len(previous_main_hub_node_ids)
            )

        raw_steps = self._raw_steps[: self._raw_n_observations]
        raw_values = self._raw_values[: self._raw_n_observations]
        if self.observation_sampling.kind == "terminal":
            valid_indices = np.flatnonzero(~np.isnan(raw_values))
            if valid_indices.size == 0:
                self._sampled_steps = np.array([0], dtype=np.int_)
                self._sampled_values = np.array([np.nan], dtype=np.float32)
            else:
                last_valid_idx = int(valid_indices[-1])
                self._sampled_steps = np.array(
                    [raw_steps[last_valid_idx]],
                    dtype=np.int_,
                )
                self._sampled_values = np.array(
                    [raw_values[last_valid_idx]],
                    dtype=np.float32,
                )
        else:
            assert self.observation_sampling.every is not None
            sample_mask = (raw_steps % self.observation_sampling.every) == 0
            self._sampled_steps = raw_steps[sample_mask].copy()
            self._sampled_values = raw_values[sample_mask].copy()
        self.n_observations = int(self._sampled_steps.shape[0])
        return self.values

    def compute_initial_value(self, event: EpisodeStartEvent) -> float:
        del event
        return float("nan")

    def compute_step_value(self, event: StepEvent) -> float:
        del event
        return float("nan")

    @property
    def steps(self) -> NDArray[np.int_]:
        return self._sampled_steps[: self.n_observations]

    @property
    def values(self) -> NDArray[np.float32]:
        return self._sampled_values[: self.n_observations]
