"""Random policy baseline."""

from __future__ import annotations

import numpy as np

from network_creation.core.model import (
    ActionSpace,
    ActionType,
    Observation,
    PolicyDecision,
)

from .base import Policy


class RandomPolicy(Policy):
    """Uniform random choice over all currently available actions."""

    def __init__(self, seed: int, **_: object) -> None:
        self._rng = np.random.default_rng(seed)

    def reset(self, seed: int) -> None:
        self._rng = np.random.default_rng(seed)

    def get_decision(
        self, observation: Observation, action_space: ActionSpace
    ) -> PolicyDecision:
        actions = [
            action
            for action_type in ActionType
            for action in action_space.actions.get(action_type, [])
        ]
        chosen_action = actions[int(self._rng.integers(len(actions)))]
        return PolicyDecision(actions=(chosen_action,), value=None)
