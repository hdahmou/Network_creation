"""Observer that prints one compact action summary per simulator step."""

from __future__ import annotations

from network_creation.core.model import Action, ActionType, PolicyDecision
from network_creation.core.observability import (
    EpisodeEndEvent,
    EpisodeStartEvent,
    ObservationSampling,
    SimulationObserver,
    StepEvent,
)


class ActionPrintObserver(SimulationObserver):
    """Print one grouped summary per completed step."""

    def __init__(
        self,
        observation_sampling: ObservationSampling,
        print_empty_steps: bool,
    ) -> None:
        self.observation_sampling = observation_sampling
        self.print_empty_steps = print_empty_steps
        self.reset()

    def reset(self) -> None:
        self.n_episodes = 0

    def on_episode_start(self, event: EpisodeStartEvent) -> None:
        del event
        if self.print_empty_steps and self.observation_sampling.kind == "periodic":
            print("step 0")

    def on_step_end(self, event: StepEvent) -> None:
        rendered_decisions_by_node_id = {
            node_id: self._format_decision(decision)
            for node_id, decision in zip(
                event.acting_agent_ids,
                event.decisions,
                strict=True,
            )
        }

        if self.print_empty_steps or any(
            rendered_decision
            for rendered_decision in rendered_decisions_by_node_id.values()
        ):
            print(f"\nstep {event.step + 1}")
        for node_id in sorted(rendered_decisions_by_node_id):
            rendered_decision = rendered_decisions_by_node_id[node_id]
            if rendered_decision:
                print(f"  {node_id}: {rendered_decision}")

    def on_episode_end(self, event: EpisodeEndEvent) -> None:
        if (
            self.observation_sampling.kind == "terminal"
            and self.print_empty_steps
            and event.current_step == 0
        ):
            print("step 0")
        self.n_episodes += 1

    def get(self) -> None:
        return None

    @staticmethod
    def _format_decision(decision: PolicyDecision) -> str:
        rendered_actions = ", ".join(
            ActionPrintObserver._format_action(action) for action in decision.actions
        )
        if decision.value is None:
            return rendered_actions
        if rendered_actions:
            return (
                f"{rendered_actions} "
                f"(value={ActionPrintObserver._format_value(decision.value)})"
            )
        return f"value={ActionPrintObserver._format_value(decision.value)}"

    @staticmethod
    def _format_value(value: object) -> str:
        if isinstance(value, dict):
            rendered_items = ", ".join(
                (
                    f"{key}={ActionPrintObserver._format_value(item_value)}"
                    for key, item_value in value.items()
                )
            )
            return "{" + rendered_items + "}"
        if isinstance(value, Action):
            return ActionPrintObserver._format_action(value)
        if isinstance(value, float):
            return str(round(value, 2))
        return str(value)

    @staticmethod
    def _format_action(action: Action) -> str:
        if action.type is ActionType.PASS:
            return "PASS"

        assert action.target is not None
        return f"{action.type.value}->{action.target}"
