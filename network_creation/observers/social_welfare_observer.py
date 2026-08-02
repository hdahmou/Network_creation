"""Observers for centered social welfare in the linking game."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import numpy as np

from network_creation.core.observability import (
    EpisodeStartEvent,
    ObservationSampling,
    StepEvent,
)
from network_creation.observers.statistics.episode_series import (
    EpisodeSeriesStatisticObserver,
)
from network_creation.stats.topology_welfare import (
    WelfareReference,
    centered_mean_welfare,
    reference_centered_mean_welfare,
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
        self.max_centered_mean_welfare = reference_centered_mean_welfare(
            reference="max",
            n_agents=self.n_agents,
            benefit_to_cost_ratio=benefit_to_cost_ratio,
            distance_discount=distance_discount,
        )

    def compute_initial_value(self, event: EpisodeStartEvent) -> float:
        return centered_mean_welfare(
            graph=event.initial_state.graph,
            benefit_to_cost_ratio=self.benefit_to_cost_ratio,
            distance_discount=self.distance_discount,
        ) / (self.max_centered_mean_welfare + 1e-8)

    def compute_step_value(self, event: StepEvent) -> float:
        return centered_mean_welfare(
            graph=event.post_state.graph,
            benefit_to_cost_ratio=self.benefit_to_cost_ratio,
            distance_discount=self.distance_discount,
        ) / (self.max_centered_mean_welfare + 1e-8)

    @classmethod
    def reference_values(
        cls,
        context: Mapping[str, object],
        references: Mapping[str, float | None] | None = None,
    ) -> dict[str, float | None]:
        del cls
        result = dict(references or {})
        try:
            n_agents = int(cast(Any, context["n_agents"]))
            benefit_to_cost_ratio = float(cast(Any, context["benefit_to_cost_ratio"]))
            distance_discount = float(cast(Any, context["distance_discount"]))
        except (KeyError, TypeError, ValueError):
            return result

        def welfare(reference: WelfareReference) -> float:
            return reference_centered_mean_welfare(
                reference=reference,
                n_agents=n_agents,
                benefit_to_cost_ratio=benefit_to_cost_ratio,
                distance_discount=distance_discount,
            )

        reference_max = welfare("max")
        if not np.isfinite(reference_max) or reference_max == 0.0:
            return result

        result.update(
            {
                "empty": float(welfare("empty") / reference_max),
                "random": float(welfare("random") / reference_max),
                "star": float(welfare("star") / reference_max),
            }
        )
        return result
