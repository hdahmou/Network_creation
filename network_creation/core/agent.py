"""Agent wrapper around a policy and node-local parameters."""

from __future__ import annotations

from dataclasses import dataclass

from ..policies.base import Policy
from .model import Action, ActionSpace, Observation


@dataclass
class Agent:
    """One acting node plus its policy configuration."""

    id: int
    radius: int
    policy: Policy
    seed: int

    def __post_init__(self) -> None:
        if self.id < 0:
            raise ValueError("id must be non-negative")
        if self.radius < 0:
            raise ValueError("radius must be non-negative")

    def reset(self) -> None:
        """Reset the internal policy state."""

        self.policy.reset()

    def act(self, observation: Observation, action_space: ActionSpace) -> list[Action]:
        """Delegate action choice to the current policy."""

        return self.policy.get_actions(observation, action_space)
