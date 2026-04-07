"""Agent wrapper around a policy and node-local parameters."""

from __future__ import annotations

from dataclasses import dataclass

from ..policies.base import Policy
from .model import ActionSpace, Observation, PolicyDecision


@dataclass
class Agent:
    """One acting node plus its policy configuration."""

    id: int
    visibility_fraction: float
    policy: Policy
    seed: int

    def __post_init__(self) -> None:
        self.visibility_fraction = float(self.visibility_fraction)

    def get_visibility_fraction(self) -> float:
        return self.visibility_fraction

    def reset(self, seed: int) -> None:
        """Reset the internal policy state from one explicit seed."""

        self.policy.reset(seed=seed)

    def act(
        self, observation: Observation, action_space: ActionSpace
    ) -> PolicyDecision:
        """Delegate decision choice to the current policy."""

        return self.policy.get_decision(observation, action_space)
