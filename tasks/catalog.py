"""Application task catalog."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .policies.myopic import MyopicPolicyRun
from .policies.random import RandomPolicyRun

if TYPE_CHECKING:
    from .policies.base import PolicyRun


TASKS: dict[str, type[PolicyRun]] = {
    MyopicPolicyRun.name: MyopicPolicyRun,
    RandomPolicyRun.name: RandomPolicyRun,
}


def load_task(task_name: str) -> PolicyRun:
    try:
        task_class = TASKS[task_name]
    except KeyError as exc:
        available = ", ".join(sorted(TASKS))
        raise ValueError(
            f"Unknown task {task_name!r}. Available tasks: {available}"
        ) from exc

    return task_class()
