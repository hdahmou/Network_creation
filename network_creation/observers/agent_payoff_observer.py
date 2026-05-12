"""Observer for normalized per-agent payoff in the linking game."""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral

import igraph as ig
import numpy as np
from numpy.typing import NDArray

from network_creation.core.observability import (
    EpisodeEndEvent,
    EpisodeStartEvent,
    ObservationSampling,
    StatisticResult,
    StepEvent,
)
from network_creation.observers.statistics.episode_tensor_series import (
    EpisodeSeriesTensorStatisticObserver,
)
from network_creation.stats.aggregation.base import StatisticAccumulator
from network_creation.stats.aggregation.registry import build_accumulators
from network_creation.stats.topology_welfare import agent_payoff


class AgentPayoffObserver(EpisodeSeriesTensorStatisticObserver):
    """Track normalized payoff for each agent over simulator steps."""

    label = "Agent Payoff"
    x_label = "step"
    y_label = "normalized payoff"
    y_min = None
    y_max = None
    degree_ranked_dimension_names = (
        "highest_degree",
        "second_highest_degree",
        "rest_except_top_two",
    )

    def __init__(
        self,
        n_agents: int,
        observation_sampling: ObservationSampling,
        benefit_to_cost_ratio: float,
        distance_discount: float,
        aggregations: tuple[str, ...],
        n_agent_slots: int | None = None,
    ) -> None:
        super().__init__(
            n_agents=n_agents,
            observation_sampling=observation_sampling,
            aggregations=aggregations,
        )
        if n_agent_slots is None:
            n_agent_slots = self.n_agents
        if (
            isinstance(n_agent_slots, bool)
            or not isinstance(n_agent_slots, Integral)
            or int(n_agent_slots) < self.n_agents
        ):
            raise ValueError("n_agent_slots must be an integer >= n_agents")

        self.benefit_to_cost_ratio: float = float(benefit_to_cost_ratio)
        self.distance_discount: float = float(distance_discount)
        self.n_agent_slots: int = int(n_agent_slots)
        self.dimension_names: tuple[str, ...] = tuple(
            f"agent_{agent_id}" for agent_id in range(self.n_agent_slots)
        )

    def reset(self) -> None:
        super().reset()
        self._degree_ranked_accumulators: Mapping[
            str,
            tuple[StatisticAccumulator, ...],
        ] = {
            dimension_name: build_accumulators(self.aggregations)
            for dimension_name in self.degree_ranked_dimension_names
        }
        self._last_degree_ranked_values: dict[str, NDArray[np.float32]] = {}

    def on_episode_end(self, event: EpisodeEndEvent) -> None:
        values = np.asarray(self._get_episode_values(event), dtype=np.float32)
        degree_ranked_values = self._degree_ranked_episode_values(
            values,
            final_graph=event.final_state.graph,
            episode_seed=event.seed,
        )
        self._last_degree_ranked_values = degree_ranked_values

        for dimension_name, accumulators in self._degree_ranked_accumulators.items():
            for accumulator in accumulators:
                accumulator.update(degree_ranked_values[dimension_name])

        for accumulator in self._accumulators:
            accumulator.update(values)
        self.n_episodes += 1

    def compute_initial_value(self, event: EpisodeStartEvent) -> NDArray[np.float32]:
        return agent_payoff(
            graph=event.initial_state.graph,
            benefit_to_cost_ratio=self.benefit_to_cost_ratio,
            distance_discount=self.distance_discount,
            n_agent_slots=self.n_agent_slots,
        )

    def compute_step_value(self, event: StepEvent) -> NDArray[np.float32]:
        return agent_payoff(
            graph=event.post_state.graph,
            benefit_to_cost_ratio=self.benefit_to_cost_ratio,
            distance_discount=self.distance_discount,
            n_agent_slots=self.n_agent_slots,
        )

    @classmethod
    def is_degree_ranked_dimension_name(cls, dimension_name: str | None) -> bool:
        return dimension_name in cls.degree_ranked_dimension_names

    def degree_ranked_values(
        self,
        dimension_name: str,
    ) -> NDArray[np.float32]:
        self._validate_degree_ranked_dimension_name(dimension_name)
        if dimension_name not in self._last_degree_ranked_values:
            raise ValueError("observer has not recorded a complete episode yet")
        return self._last_degree_ranked_values[dimension_name]

    def degree_ranked_result(
        self,
        dimension_name: str,
    ) -> StatisticResult:
        self._validate_degree_ranked_dimension_name(dimension_name)
        outputs = {}
        for accumulator in self._degree_ranked_accumulators[dimension_name]:
            outputs.update(accumulator.export())
        return StatisticResult(
            steps=self.steps,
            n_episodes=self.n_episodes,
            outputs=outputs,
            sample_shape=(),
            dimension_names=None,
        )

    @classmethod
    def fixed_output_shape_for_slots(
        cls,
        *,
        max_steps: int,
        observation_sampling: ObservationSampling,
        n_agent_slots: int,
    ) -> tuple[int, ...]:
        del cls
        return (
            observation_sampling.fixed_sample_count(max_steps),
            int(n_agent_slots),
        )

    def _validate_degree_ranked_dimension_name(self, dimension_name: str) -> None:
        if not self.is_degree_ranked_dimension_name(dimension_name):
            raise ValueError(f"unknown degree-ranked dimension name: {dimension_name}")

    def _degree_ranked_episode_values(
        self,
        values: NDArray[np.float32],
        *,
        final_graph: ig.Graph,
        episode_seed: int,
    ) -> dict[str, NDArray[np.float32]]:
        values = np.asarray(values, dtype=np.float32)
        if values.ndim < 2:
            raise ValueError("agent payoff values must include an agent axis")

        flat_values = values.reshape(values.shape[0], -1)
        missing = np.full(values.shape[0], np.nan, dtype=np.float32)
        ranked_agent_ids = _rank_agent_ids_by_degree(
            final_graph,
            episode_seed=episode_seed,
        )

        highest = (
            _select_agent_series(flat_values, ranked_agent_ids[0])
            if len(ranked_agent_ids) >= 1
            else missing.copy()
        )
        second_highest = (
            _select_agent_series(flat_values, ranked_agent_ids[1])
            if len(ranked_agent_ids) >= 2
            else missing.copy()
        )
        rest = (
            _mean_agent_series(flat_values, ranked_agent_ids[2:])
            if len(ranked_agent_ids) > 2
            else missing.copy()
        )
        return {
            "highest_degree": highest,
            "second_highest_degree": second_highest,
            "rest_except_top_two": rest,
        }


def _rank_agent_ids_by_degree(
    graph: ig.Graph,
    *,
    episode_seed: int,
) -> tuple[int, ...]:
    degrees = np.asarray(graph.degree(mode="all"), dtype=np.int_)
    rng = np.random.default_rng(int(episode_seed))
    ranked_agent_ids: list[int] = []
    for degree in sorted(set(int(degree) for degree in degrees), reverse=True):
        tied_agent_ids = np.flatnonzero(degrees == degree).astype(np.int_)
        rng.shuffle(tied_agent_ids)
        ranked_agent_ids.extend(int(agent_id) for agent_id in tied_agent_ids)
    return tuple(ranked_agent_ids)


def _select_agent_series(
    values: NDArray[np.float32],
    agent_id: int,
) -> NDArray[np.float32]:
    if agent_id >= values.shape[1]:
        return np.full(values.shape[0], np.nan, dtype=np.float32)
    return np.asarray(values[:, agent_id], dtype=np.float32)


def _mean_agent_series(
    values: NDArray[np.float32],
    agent_ids: tuple[int, ...],
) -> NDArray[np.float32]:
    agent_ids = tuple(agent_id for agent_id in agent_ids if agent_id < values.shape[1])
    if not agent_ids:
        return np.full(values.shape[0], np.nan, dtype=np.float32)

    selected = np.asarray(values[:, agent_ids], dtype=np.float32)
    finite_counts = np.count_nonzero(np.isfinite(selected), axis=1)
    sums = np.nansum(selected, axis=1, dtype=np.float32)
    result = np.full(values.shape[0], np.nan, dtype=np.float32)
    np.divide(
        sums,
        finite_counts,
        out=result,
        where=finite_counts > 0,
    )
    return result.astype(np.float32, copy=False)


__all__ = ["AgentPayoffObserver"]
