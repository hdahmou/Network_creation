"""Random policy baseline."""

from __future__ import annotations

import numpy as np

from network_creation.core.model import (
    Action,
    ActionSpace,
    ActionType,
    Observation,
    PolicyDecision,
)

from .base import Policy


class RandomPolicy(Policy):
    """Uniform random choice over all currently available actions."""

    def __init__(self, seed: int) -> None:
        self._rng = np.random.default_rng(seed)

    def reset(self, seed: int) -> None:
        self._rng = np.random.default_rng(seed)

    def get_decision(
        self, observation: Observation, action_space: ActionSpace
    ) -> PolicyDecision:
        del observation
        actions = [
            action
            for action_type in ActionType
            for action in action_space.actions.get(action_type, [])
        ]
        chosen_action = actions[int(self._rng.integers(len(actions)))]
        return PolicyDecision(actions=(chosen_action,), value=None)
