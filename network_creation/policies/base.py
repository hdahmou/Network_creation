"""Policy interface for node-level action selection."""

from __future__ import annotations

from abc import ABC, abstractmethod

from network_creation.core.model import Action, ActionSpace, Observation


class Policy(ABC):
    """Strategy interface for choosing one action from an action space."""

    @abstractmethod
    def reset(self) -> None:
        """Reset any internal policy state."""

    @abstractmethod
    def get_action(self, observation: Observation, action_space: ActionSpace) -> Action:
        """Choose one action from the current observation and action space."""

    @abstractmethod
    def get_params(self) -> dict[str, object]:
        """Expose policy parameters in dictionary form."""
