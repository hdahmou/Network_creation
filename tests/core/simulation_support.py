"""Shared test doubles for simulator-driven tests."""

from __future__ import annotations

from typing import Any

from network_creation.core.model import (
    Action,
    ActionSpace,
    Observation,
    PolicyDecision,
)
from network_creation.core.observability import (
    EpisodeEndEvent,
    EpisodeStartEvent,
    ObservationSampling,
    StepEvent,
)
from network_creation.policies.base import Policy


def _local_action(observation: Observation, action: Action) -> Action:
    return Action(
        action.type,
        observation.local_from_global(action.source),
        None
        if action.target is None
        else observation.local_from_global(action.target),
    )


class ConstantPolicy(Policy):
    def __init__(self, action: Action, value: dict[str, object] | None = None) -> None:
        self.action = action
        self.value = value
        self.reset_calls = 0
        self.reset_seeds: list[int] = []
        self.get_decision_calls = 0

    def reset(self, seed: int) -> None:
        self.reset_calls += 1
        self.reset_seeds.append(seed)

    def get_decision(
        self, observation: Observation, action_space: ActionSpace
    ) -> PolicyDecision:
        del action_space
        self.get_decision_calls += 1
        return PolicyDecision(
            actions=(_local_action(observation, self.action),),
            value=self.value,
        )


class ActionListPolicy(Policy):
    def __init__(self, actions: list[Action]) -> None:
        self.actions = actions

    def reset(self, seed: int) -> None:
        del seed

    def get_decision(
        self, observation: Observation, action_space: ActionSpace
    ) -> PolicyDecision:
        del action_space
        return PolicyDecision(
            actions=tuple(
                _local_action(observation, action) for action in self.actions
            ),
            value=None,
        )


class EpisodeCallPolicy(Policy):
    def __init__(self, episode_actions: list[list[list[Action]]]) -> None:
        self.episode_actions = episode_actions
        self._episode_index = -1
        self._call_index = 0

    def reset(self, seed: int) -> None:
        del seed
        self._episode_index = min(
            self._episode_index + 1,
            len(self.episode_actions) - 1,
        )
        self._call_index = 0

    def get_decision(
        self, observation: Observation, action_space: ActionSpace
    ) -> PolicyDecision:
        del action_space
        current_episode_actions = self.episode_actions[self._episode_index]
        action_idx = min(self._call_index, len(current_episode_actions) - 1)
        self._call_index += 1
        return PolicyDecision(
            actions=tuple(
                _local_action(observation, action)
                for action in current_episode_actions[action_idx]
            ),
            value=None,
        )


class MultiActionPolicy(Policy):
    def __init__(self, actions: list[Action]) -> None:
        self.actions = actions
        self.reset_calls = 0
        self.reset_seeds: list[int] = []

    def reset(self, seed: int) -> None:
        self.reset_calls += 1
        self.reset_seeds.append(seed)

    def get_decision(
        self, observation: Observation, action_space: ActionSpace
    ) -> PolicyDecision:
        del action_space
        return PolicyDecision(
            actions=tuple(
                _local_action(observation, action) for action in self.actions
            ),
            value=None,
        )


class ScriptedPolicy(Policy):
    def __init__(self, actions_by_call: list[list[Action]]) -> None:
        self.actions_by_call = [list(actions) for actions in actions_by_call]
        self.reset_calls = 0
        self.reset_seeds: list[int] = []
        self.get_decision_calls = 0
        self._next_call_idx = 0

    def reset(self, seed: int) -> None:
        self.reset_calls += 1
        self.reset_seeds.append(seed)
        self._next_call_idx = 0

    def get_decision(
        self, observation: Observation, action_space: ActionSpace
    ) -> PolicyDecision:
        del action_space
        if self._next_call_idx >= len(self.actions_by_call):
            raise AssertionError("policy was queried more times than expected")

        self.get_decision_calls += 1
        actions = list(self.actions_by_call[self._next_call_idx])
        self._next_call_idx += 1
        return PolicyDecision(
            actions=tuple(_local_action(observation, action) for action in actions),
            value=None,
        )


class RecordingObserver:
    def __init__(self) -> None:
        self.observation_sampling = ObservationSampling.periodic(1)
        self.episode_start_event: EpisodeStartEvent | None = None
        self.step_events: list[StepEvent] = []
        self.episode_end_event: EpisodeEndEvent | None = None

    def on_episode_start(self, event: EpisodeStartEvent) -> None:
        self.episode_start_event = event

    def observes_step(self, step_index: int, max_steps: int) -> bool:
        return self.observation_sampling.samples_step(step_index, max_steps)

    def on_step_end(self, event: StepEvent) -> None:
        self.step_events.append(event)

    def on_episode_end(self, event: EpisodeEndEvent) -> None:
        self.episode_end_event = event


class BatchRecordingObserver:
    def __init__(self) -> None:
        self.observation_sampling = ObservationSampling.periodic(1)
        self.episode_start_events: list[EpisodeStartEvent] = []
        self.episode_end_events: list[EpisodeEndEvent] = []

    def reset(self) -> None:
        self.episode_start_events = []
        self.episode_end_events = []

    def on_episode_start(self, event: EpisodeStartEvent) -> None:
        self.episode_start_events.append(event)

    def observes_step(self, step_index: int, max_steps: int) -> bool:
        return self.observation_sampling.samples_step(step_index, max_steps)

    def on_step_end(self, event: StepEvent) -> None:
        del event

    def on_episode_end(self, event: EpisodeEndEvent) -> None:
        self.episode_end_events.append(event)


__all__ = [
    "ActionListPolicy",
    "BatchRecordingObserver",
    "ConstantPolicy",
    "EpisodeCallPolicy",
    "MultiActionPolicy",
    "RecordingObserver",
    "ScriptedPolicy",
    "_local_action",
]
