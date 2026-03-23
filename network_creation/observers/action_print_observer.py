"""Observer that prints one compact action summary per simulator step."""

from __future__ import annotations

from collections import defaultdict

from network_creation.core.model import ActionType
from network_creation.core.observability import (
    EpisodeEndEvent,
    EpisodeStartEvent,
    SimulationObserver,
    StepEvent,
)


class ActionPrintObserver(SimulationObserver):
    """Print one grouped summary per completed step."""

    def __init__(self, observation_frequency: int, print_empty_steps: bool) -> None:
        self.observation_frequency = observation_frequency
        self.print_empty_steps = print_empty_steps
        self.reset()

    def reset(self) -> None:
        self.n_episodes = 0

    def on_episode_start(self, event: EpisodeStartEvent) -> None:
        del event
        if self.print_empty_steps:
            print("step 0")

    def on_step_end(self, event: StepEvent) -> None:
        actions_by_node_id: dict[int, list[str]] = defaultdict(list)
        for action in event.actions:
            actions_by_node_id[action.source].append(self._format_action(action))

        if self.print_empty_steps or event.actions:
            print(f"\nstep {event.step + 1}")
        for node_id in sorted(actions_by_node_id):
            rendered_actions = ", ".join(actions_by_node_id[node_id])
            print(f"  {node_id}: {rendered_actions}")

    def on_episode_end(self, event: EpisodeEndEvent) -> None:
        del event
        self.n_episodes += 1

    def get(self) -> None:
        return None

    @staticmethod
    def _format_action(action) -> str:
        if action.type is ActionType.PASS:
            return "PASS"

        assert action.target is not None
        return f"{action.type.value}->{action.target}"
