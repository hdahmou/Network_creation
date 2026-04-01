"""Task registry keyed by dispatch config task names."""

from __future__ import annotations

from importlib import import_module
from typing import Any, cast

from .runtime import Task


TASK_REGISTRY: dict[str, type[Task[Any]]] = {}
DEFAULT_TASK_MODULES = (
    "tasks.policies.myopic_policy",
    "tasks.policies.random_policy",
)

def register_task[TTask: Task[Any]](
    task_class: type[TTask],
) -> type[TTask]:
    name = getattr(task_class, "name", None)
    if not isinstance(name, str) or not name:
        raise ValueError("registered task class must define a non-empty name")

    existing = TASK_REGISTRY.get(name)
    if existing is not None and existing is not task_class:
        raise ValueError(f"task {name!r} already registered")

    TASK_REGISTRY[name] = cast(type[Task[Any]], task_class)
    return task_class


def _ensure_default_tasks_registered() -> None:
    for module_name in DEFAULT_TASK_MODULES:
        import_module(module_name)


def load_task(task_name: str) -> Task[Any]:
    _ensure_default_tasks_registered()
    try:
        task_class = TASK_REGISTRY[task_name]
    except KeyError as exc:
        available = ", ".join(sorted(TASK_REGISTRY))
        raise ValueError(
            f"Unknown task {task_name!r}. Available tasks: {available}"
        ) from exc

    try:
        return task_class()
    except TypeError as exc:
        raise TypeError(
            f"Task {task_name!r} must be instantiable without arguments"
        ) from exc
