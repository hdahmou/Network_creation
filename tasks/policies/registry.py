"""Policy constructor registry."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from network_creation.policies.best_response import BestResponsePolicy
from network_creation.policies.myopic_payoff import MyopicPayoffPolicy
from network_creation.policies.random import RandomPolicy


PolicyFactory = Callable[..., Any]


POLICY_REGISTRY: dict[str, PolicyFactory] = {
    "best-response": BestResponsePolicy,
    "myopic": MyopicPayoffPolicy,
    "random": RandomPolicy,
}
