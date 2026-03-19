"""Random policy baseline."""

from __future__ import annotations

import numpy as np

from network_creation.core.model import Action, ActionSpace, ActionType, Observation

from .base import Policy


class RandomPolicy(Policy):
    """Uniform random choice over all currently available actions."""

    def __init__(self, seed: int) -> None:
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    def reset(self) -> None:
        """Reset the policy RNG to its initial seed."""

        self._rng = np.random.default_rng(self.seed)

    def get_action(self, observation: Observation, action_space: ActionSpace) -> Action:
        """Choose one available action uniformly at random."""

        del observation
        actions = [
            action
            for action_type in ActionType
            for action in action_space.actions.get(action_type, [])
        ]
        return actions[int(self._rng.integers(len(actions)))]

    def get_params(self) -> dict[str, object]:
        """Expose the random policy parameters."""

        return {}
