"""Agent wrapper around a policy and node-local parameters."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real

from ..policies.base import Policy
from .model import ActionSpace, Observation, PolicyDecision


@dataclass
class Agent:
    """One acting node plus its policy configuration."""

    id: int
    visibility_distance: int
    policy: Policy
    seed: int
    activation_weight: float

    def __post_init__(self) -> None:
        if (
            isinstance(self.visibility_distance, bool)
            or not isinstance(self.visibility_distance, Real)
            or not float(self.visibility_distance).is_integer()
            or int(self.visibility_distance) < 1
        ):
            raise ValueError("visibility_distance must be a positive integer")
        self.visibility_distance = int(self.visibility_distance)

        if not 0.0 < self.activation_weight < float("inf"):
            raise ValueError("activation_weight must be finite and positive")

    def get_visibility_distance(self) -> int:
        return self.visibility_distance

    def reset(self, seed: int) -> None:
        """Reset the internal policy state from one explicit seed."""

        self.policy.reset(seed=seed)

    def act(
        self, observation: Observation, action_space: ActionSpace
    ) -> PolicyDecision:
        """Delegate decision choice to the current policy."""

        return self.policy.get_decision(observation, action_space)
