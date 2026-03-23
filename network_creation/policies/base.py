"""Policy interface for node-level action selection."""

from __future__ import annotations

from abc import ABC, abstractmethod

from network_creation.core.model import Action, ActionSpace, Observation


class Policy(ABC):
    """Strategy interface for choosing one or more actions from an action space."""

    @abstractmethod
    def reset(self, seed: int) -> None:
        """Reset any internal policy state from one explicit seed."""

    @abstractmethod
    def get_actions(
        self, observation: Observation, action_space: ActionSpace
    ) -> list[Action]:
        """Choose one or more actions from the current observation and action space."""
