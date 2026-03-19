"""Observer that prints one compact action summary per simulator step."""

from __future__ import annotations

from network_creation.core.model import ActionType
from network_creation.core.observability import (
    EpisodeEndEvent,
    EpisodeStartEvent,
    SimulationObserver,
    StepEvent,
)


class ActionPrintObserver(SimulationObserver):
    """Print one line per completed step."""

    def __init__(self, observation_frequency: int) -> None:
        self.observation_frequency = observation_frequency

    def on_episode_start(self, event: EpisodeStartEvent) -> None:
        del event

    def on_step_end(self, event: StepEvent) -> None:
        rendered_actions = ", ".join(
            self._format_action(action) for action in event.actions
        )
        print(f"step {event.step}: {rendered_actions}")

    def on_episode_end(self, event: EpisodeEndEvent) -> None:
        del event

    @staticmethod
    def _format_action(action) -> str:
        if action.type is ActionType.PASS:
            return f"{action.source} PASS"

        assert action.target is not None
        return f"{action.source} {action.type.value} {action.target}"
