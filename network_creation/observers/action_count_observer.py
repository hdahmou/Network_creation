"""Observer that tracks average non-pass action counts across simulator steps."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from network_creation.core.model import ActionType
from network_creation.core.observability import (
    EpisodeStartEvent,
    StepEvent,
)
from network_creation.observers.statistics.episode_series import (
    EpisodeSeriesStatisticObserver,
)


class ActionCountObserver(EpisodeSeriesStatisticObserver):
    """Collect normalized non-pass action counts per sampled simulator step."""

    label = "Average Number of Actions"
    x_label = "step"
    y_label = "average number of actions"
    y_min = 0.0
    y_max = None

    def _reset_episode_state(self) -> None:
        self._n_agents = 0
        super()._reset_episode_state()

    def _start_episode(self, event: EpisodeStartEvent) -> None:
        self._n_agents = event.n_agents
        super()._start_episode(event)

    @property
    def average_action_counts(self) -> NDArray[np.float32]:
        return self.values

    def compute_initial_value(self, event: EpisodeStartEvent) -> float:
        del event
        return 0.0

    def compute_step_value(self, event: StepEvent) -> float:
        if self._n_agents == 0:
            return 0.0

        n_non_pass_actions = sum(
            1
            for decision in event.decisions
            for action in decision.actions
            if action.type is not ActionType.PASS
        )
        return float(n_non_pass_actions / self._n_agents)
