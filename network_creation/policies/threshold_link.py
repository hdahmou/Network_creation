"""Threshold-based random link creation/deletion policy."""

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


class ThresholdLinkPolicy(Policy):
    """Create outgoing links until a proportional threshold, then delete them."""

    link_threshold: float
    p_act: float
    _rng: np.random.Generator
    _is_deleting: bool

    def __init__(
        self,
        seed: int,
        link_threshold: float,
        p_act: float = 1.0,
        **_: object,
    ) -> None:
        if (
            isinstance(link_threshold, bool)
            or not isinstance(link_threshold, Real)
            or not 0.0 <= float(link_threshold) <= 1.0
        ):
            raise ValueError("link_threshold must satisfy 0 <= link_threshold <= 1")
        if (
            isinstance(p_act, bool)
            or not isinstance(p_act, Real)
            or not 0.0 <= float(p_act) <= 1.0
        ):
            raise ValueError("p_act must satisfy 0 <= p_act <= 1")

        self.link_threshold = float(link_threshold)
        self.p_act = float(p_act)
        self._rng = np.random.default_rng(seed)
        self._is_deleting = False

    def reset(self, seed: int) -> None:
        self._rng = np.random.default_rng(seed)
        self._is_deleting = False

    def get_decision(
        self, observation: Observation, action_space: ActionSpace
    ) -> PolicyDecision:
        pass_action = action_space[ActionType.PASS][0]

        possible_outgoing_edges = max(len(observation.local_global) - 1, 0)
        current_outgoing_edges = len(action_space[ActionType.DELETE])
        threshold_edges = self.link_threshold * possible_outgoing_edges
        if current_outgoing_edges >= threshold_edges:
            self._is_deleting = True

        if self._rng.random() >= self.p_act:
            return PolicyDecision(actions=(pass_action,), value=None)

        if self._is_deleting:
            candidate_actions = action_space[ActionType.DELETE]
        else:
            candidate_actions = (
                action_space[ActionType.ADD] + action_space[ActionType.BRIDGE]
            )

        if not candidate_actions:
            return PolicyDecision(actions=(pass_action,), value=None)

        chosen_action = self._sample_action(candidate_actions)
        return PolicyDecision(actions=(chosen_action,), value=None)

    def _sample_action(self, actions: list[Action]) -> Action:
        return actions[int(self._rng.integers(len(actions)))]
