"""Observers for centered social welfare in the linking game."""

from __future__ import annotations


from network_creation.core.observability import (
    EpisodeStartEvent,
    ObservationSampling,
    StepEvent,
)
from network_creation.observers.statistics.episode_series import (
    EpisodeSeriesStatisticObserver,
)
from network_creation.stats.topology_welfare import (
    normalized_welfare,
    normalized_welfare_per_topology,
)


class CenteredMeanWelfareObserver(EpisodeSeriesStatisticObserver):
    """Track mean centered welfare per agent over simulator steps."""

    label = "welfare "
    x_label = "step"
    y_label = "welfare (normalized)"
    y_min = None
    y_max = None

    def __init__(
        self,
        n_agents: int,
        observation_sampling: ObservationSampling,
        benefit_to_cost_ratio: float,
        distance_discount: float,
        aggregations: tuple[str, ...],
    ) -> None:
        super().__init__(
            n_agents=n_agents,
            observation_sampling=observation_sampling,
            aggregations=aggregations,
        )
        self.benefit_to_cost_ratio = benefit_to_cost_ratio
        self.distance_discount = distance_discount
        self.max_normalized_welfare = normalized_welfare_per_topology(
            topology_class="max",
            n=self.n_agents,
            benefit_to_cost_ratio=benefit_to_cost_ratio,
            distance_discount=distance_discount,
        )

    def compute_initial_value(self, event: EpisodeStartEvent) -> float:
        return normalized_welfare(
            graph=event.initial_state.graph,
            benefit_to_cost_ratio=self.benefit_to_cost_ratio,
            distance_discount=self.distance_discount,
        ) / (self.max_normalized_welfare + 1e-8)

    def compute_step_value(self, event: StepEvent) -> float:
        return normalized_welfare(
            graph=event.post_state.graph,
            benefit_to_cost_ratio=self.benefit_to_cost_ratio,
            distance_discount=self.distance_discount,
        ) / (self.max_normalized_welfare + 1e-8)
