"""Observer for actor, link-recipient, and final-hub rankings."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from network_creation.core.model import ActionType
from network_creation.core.observability import (
    EpisodeEndEvent,
    EpisodeStartEvent,
    ObservationSampling,
    SimulationObserver,
    StepEvent,
)


@dataclass(frozen=True)
class HubRankingResult:
    """Distinct first-appearance orders and unique terminal hubs."""

    actor_order: NDArray[np.int64]
    recipient_order: NDArray[np.int64]
    final_hub_id: NDArray[np.int64]


class HubRankingObserver(SimulationObserver):
    """Record only the episode data needed for final-hub ranking analysis."""

    observation_sampling = ObservationSampling.periodic(1)

    def __init__(self, n_agents: int) -> None:
        self.n_agents = int(n_agents)
        self.reset()

    def reset(self) -> None:
        self._actor_orders: list[NDArray[np.int64]] = []
        self._recipient_orders: list[NDArray[np.int64]] = []
        self._final_hub_ids: list[int] = []
        self._actor_order: NDArray[np.int64] | None = None
        self._recipient_order: NDArray[np.int64] | None = None
        self._actor_seen: NDArray[np.bool_] | None = None
        self._recipient_seen: NDArray[np.bool_] | None = None
        self._actor_count = 0
        self._recipient_count = 0

    def on_episode_start(self, event: EpisodeStartEvent) -> None:
        del event
        self._actor_order = np.full(self.n_agents, -1, dtype=np.int64)
        self._recipient_order = np.full(self.n_agents, -1, dtype=np.int64)
        self._actor_seen = np.zeros(self.n_agents, dtype=np.bool_)
        self._recipient_seen = np.zeros(self.n_agents, dtype=np.bool_)
        self._actor_count = 0
        self._recipient_count = 0

    def on_step_end(self, event: StepEvent) -> None:
        if (
            self._actor_order is None
            or self._recipient_order is None
            or self._actor_seen is None
            or self._recipient_seen is None
        ):
            raise RuntimeError("received a step before an episode started")

        for decision in event.decisions:
            for action in decision.actions:
                actor_id = int(action.source)
                if not self._actor_seen[actor_id]:
                    self._actor_seen[actor_id] = True
                    self._actor_order[self._actor_count] = actor_id
                    self._actor_count += 1
                if (
                    action.type in {ActionType.ADD, ActionType.BRIDGE}
                    and action.target is not None
                ):
                    recipient_id = int(action.target)
                    if not self._recipient_seen[recipient_id]:
                        self._recipient_seen[recipient_id] = True
                        self._recipient_order[self._recipient_count] = recipient_id
                        self._recipient_count += 1

    def on_episode_end(self, event: EpisodeEndEvent) -> None:
        if self._actor_order is None or self._recipient_order is None:
            raise RuntimeError("received an episode end before an episode started")

        degrees = np.asarray(event.final_state.graph.degree(mode="all"), dtype=np.int64)
        hub_ids = np.flatnonzero(degrees == degrees.max())
        final_hub_id = int(hub_ids[0]) if hub_ids.size == 1 else -1
        self._actor_orders.append(self._actor_order.copy())
        self._recipient_orders.append(self._recipient_order.copy())
        self._final_hub_ids.append(final_hub_id)
        self._actor_order = None
        self._recipient_order = None
        self._actor_seen = None
        self._recipient_seen = None

    def get(self) -> HubRankingResult:
        return HubRankingResult(
            actor_order=np.asarray(self._actor_orders, dtype=np.int64),
            recipient_order=np.asarray(self._recipient_orders, dtype=np.int64),
            final_hub_id=np.asarray(self._final_hub_ids, dtype=np.int64),
        )
