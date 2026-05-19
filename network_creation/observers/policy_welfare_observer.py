"""Observer for mean welfare by policy class."""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral
from typing import Any

import numpy as np
from numpy.typing import NDArray

from network_creation.core.observability import (
    EpisodeStartEvent,
    ObservationSampling,
    StepEvent,
)
from network_creation.observers.statistics.episode_tensor_series import (
    EpisodeSeriesTensorStatisticObserver,
)
from network_creation.stats.topology_welfare import agent_payoff


class PolicyWelfareObserver(EpisodeSeriesTensorStatisticObserver):
    """Track mean normalized payoff for agents assigned to each policy class."""

    label = "Policy Welfare"
    x_label = "step"
    y_label = "normalized payoff"
    y_min = None
    y_max = None

    def __init__(
        self,
        n_agents: int,
        observation_sampling: ObservationSampling,
        benefit_to_cost_ratio: float,
        distance_discount: float,
        aggregations: tuple[str, ...],
        policy_ids: Sequence[str],
        policy_slot_by_agent: Sequence[int] | NDArray[np.integer[Any]],
        n_policy_slots: int | None = None,
    ) -> None:
        super().__init__(
            n_agents=n_agents,
            observation_sampling=observation_sampling,
            aggregations=aggregations,
        )
        self.benefit_to_cost_ratio: float = float(benefit_to_cost_ratio)
        self.distance_discount: float = float(distance_discount)
        self.policy_ids: tuple[str, ...] = tuple(
            str(policy_id) for policy_id in policy_ids
        )
        if not self.policy_ids:
            raise ValueError("policy_ids must contain at least one policy id")

        if n_policy_slots is None:
            n_policy_slots = len(self.policy_ids)
        if (
            isinstance(n_policy_slots, bool)
            or not isinstance(n_policy_slots, Integral)
            or int(n_policy_slots) < len(self.policy_ids)
        ):
            raise ValueError("n_policy_slots must be an integer >= len(policy_ids)")
        self.n_policy_slots: int = int(n_policy_slots)

        assignment = np.asarray(policy_slot_by_agent, dtype=np.int32)
        if assignment.shape != (self.n_agents,):
            raise ValueError("policy_slot_by_agent must have shape (n_agents,)")
        if np.any((assignment < 0) | (assignment >= len(self.policy_ids))):
            raise ValueError("policy_slot_by_agent contains an unknown policy slot")
        self.policy_slot_by_agent: NDArray[np.int32] = assignment.copy()
        self.dimension_names = (
            *self.policy_ids,
            *(
                f"unused_policy_{slot}"
                for slot in range(len(self.policy_ids), self.n_policy_slots)
            ),
        )

    def compute_initial_value(self, event: EpisodeStartEvent) -> NDArray[np.float32]:
        return self._policy_values(event.initial_state.graph)

    def compute_step_value(self, event: StepEvent) -> NDArray[np.float32]:
        return self._policy_values(event.post_state.graph)

    def _policy_values(self, graph: Any) -> NDArray[np.float32]:
        payoff = agent_payoff(
            graph=graph,
            benefit_to_cost_ratio=self.benefit_to_cost_ratio,
            distance_discount=self.distance_discount,
            n_agent_slots=self.n_agents,
        )
        values: NDArray[np.float32] = np.full(
            self.n_policy_slots,
            np.nan,
            dtype=np.float32,
        )
        for policy_slot in range(len(self.policy_ids)):
            selected = payoff[self.policy_slot_by_agent == policy_slot]
            finite = selected[np.isfinite(selected)]
            if finite.size:
                values[policy_slot] = np.float32(np.mean(finite, dtype=np.float32))
        return values

    @classmethod
    def fixed_output_shape_for_policy_slots(
        cls,
        *,
        max_steps: int,
        observation_sampling: ObservationSampling,
        n_policy_slots: int,
    ) -> tuple[int, ...]:
        del cls
        return (
            observation_sampling.fixed_sample_count(max_steps),
            int(n_policy_slots),
        )


__all__ = ["PolicyWelfareObserver"]
