"""Replay policy that returns recorded actions step by step."""

from __future__ import annotations

from network_creation.core.model import (
    Action,
    ActionSpace,
    Observation,
    PolicyDecision,
)

from .base import Policy


class ReplayPolicy(Policy):
    """Return the recorded actions scheduled for each replay step."""

    def __init__(self, actions_by_step: dict[int, tuple[Action, ...]]) -> None:
        self.actions_by_step = actions_by_step
        self.next_step_id = 1

    def reset(self, seed: int) -> None:
        del seed
        self.next_step_id = 1

    def get_decision(
        self, observation: Observation, action_space: ActionSpace
    ) -> PolicyDecision:
        del observation
        del action_space

        actions = self.actions_by_step.get(self.next_step_id, ())
        self.next_step_id += 1
        return PolicyDecision(actions=actions, value=None)
