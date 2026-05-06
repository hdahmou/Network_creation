"""Random policy baseline."""

from __future__ import annotations

from numbers import Real

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
    """Random choice over currently available actions."""

    def __init__(self, seed: int, p_act: float | None = None, **_: object) -> None:
        if p_act is not None and (
            isinstance(p_act, bool)
            or not isinstance(p_act, Real)
            or not 0.0 <= p_act <= 1.0
        ):
            raise ValueError("p_act must satisfy 0 <= p_act <= 1")

        self.p_act = None if p_act is None else float(p_act)
        self._rng = np.random.default_rng(seed)

    def reset(self, seed: int) -> None:
        self._rng = np.random.default_rng(seed)

    def get_decision(
        self, observation: Observation, action_space: ActionSpace
    ) -> PolicyDecision:
        if self.p_act is not None:
            chosen_action = self._sample_with_action_probability(action_space)
            return PolicyDecision(actions=(chosen_action,), value=None)

        actions = [
            action
            for action_type in ActionType
            for action in action_space.actions.get(action_type, [])
        ]
        chosen_action = actions[int(self._rng.integers(len(actions)))]
        return PolicyDecision(actions=(chosen_action,), value=None)

    def _sample_with_action_probability(self, action_space: ActionSpace) -> Action:
        p_act = self.p_act
        if p_act is None:
            raise ValueError("p_act must be set before probability-based sampling")

        pass_action = action_space[ActionType.PASS][0]
        non_pass_actions = [
            action
            for action_type in ActionType
            if action_type is not ActionType.PASS
            for action in action_space.actions.get(action_type, [])
        ]
        if not non_pass_actions or self._rng.random() >= p_act:
            return pass_action
        return non_pass_actions[int(self._rng.integers(len(non_pass_actions)))]
