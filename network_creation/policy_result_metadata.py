"""Shared interpretation of persisted policy-run metadata."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from network_creation.core.observability import ObservationSampling


def policy_composition_axis_name(
    policy_config: Any,
    parameter_names: Sequence[str],
) -> str:
    if isinstance(policy_config, Mapping):
        population = policy_config.get("population")
        if isinstance(population, Mapping):
            proportions = population.get("policy_proportions")
            if isinstance(proportions, Mapping) and "policy" in proportions:
                return f"{proportions['policy']}_proportion"
            counts = population.get("policy_counts")
            if isinstance(counts, Mapping) and "policy" in counts:
                return f"{counts['policy']}_count"

    for suffix in ("_proportion", "_count"):
        matches = [str(name) for name in parameter_names if str(name).endswith(suffix)]
        if len(matches) == 1:
            return matches[0]
    raise ValueError("could not infer policy composition axis from result metadata")


def theta_parameter_columns(
    parameter_names: Sequence[str],
    *,
    policy_config: Any,
    composition_axis_name: str,
) -> tuple[list[str], list[int]]:
    policy_ids: list[str] = []
    if isinstance(policy_config, Mapping):
        population = policy_config.get("population")
        if isinstance(population, Mapping):
            policies = population.get("policies")
            if isinstance(policies, list):
                for policy in policies:
                    if isinstance(policy, Mapping) and "id" in policy:
                        policy_ids.append(str(policy["id"]))

    bookkeeping_names = {composition_axis_name}
    bookkeeping_names.update(f"{policy_id}_count" for policy_id in policy_ids)
    keep = [
        (index, str(name))
        for index, name in enumerate(parameter_names)
        if str(name) not in bookkeeping_names
    ]
    return [name for _, name in keep], [index for index, _ in keep]


def configured_output_step_labels(
    policy_config: Any,
) -> dict[str, NDArray[np.int64]]:
    if not isinstance(policy_config, Mapping):
        return {}
    simulator = policy_config.get("simulator", {})
    raw_observers = policy_config.get("observers", {})
    if not isinstance(simulator, Mapping) or not isinstance(raw_observers, Mapping):
        return {}
    max_steps = int(simulator["max_steps"])
    defaults = raw_observers.get("defaults", {})
    if not isinstance(defaults, Mapping):
        return {}
    default_sampling = defaults.get("sampling")
    outputs = raw_observers.get("outputs", ())
    if not isinstance(outputs, Sequence):
        return {}

    labels: dict[str, NDArray[np.int64]] = {}
    for observer in outputs:
        if not isinstance(observer, Mapping):
            continue
        sampling = load_observation_sampling(
            observer.get("sampling", default_sampling)
        )
        aggregations = observer.get("aggregations", defaults.get("aggregations", ()))
        for aggregation in aggregations:
            labels[f"{observer['name']}.{aggregation}"] = observation_step_labels(
                max_steps,
                sampling,
            )
    return labels


def load_observation_sampling(raw: Any) -> ObservationSampling:
    if not isinstance(raw, Mapping):
        raise ValueError("observer sampling must be a mapping")
    kind = str(raw["kind"])
    if kind == "periodic":
        return ObservationSampling.periodic(int(raw["every"]))
    if kind == "terminal":
        return ObservationSampling.terminal()
    raise ValueError(f"unsupported observation sampling kind: {kind}")


def observation_step_labels(
    max_steps: int,
    observation_sampling: ObservationSampling,
) -> NDArray[np.int64]:
    if observation_sampling.kind == "terminal":
        return np.asarray([max_steps], dtype=np.int64)
    assert observation_sampling.every is not None
    count = observation_sampling.fixed_sample_count(max_steps)
    return np.arange(count, dtype=np.int64) * int(observation_sampling.every)


def configured_n_actors(policy_config: Any) -> int:
    if not isinstance(policy_config, Mapping):
        raise ValueError("cannot normalize time without policy configuration metadata")
    simulator = policy_config.get("simulator", {})
    if not isinstance(simulator, Mapping):
        raise ValueError("cannot normalize time without simulator metadata")
    actor_scheduler = simulator.get("actor_scheduler", {})
    if not isinstance(actor_scheduler, Mapping) or "n_actors" not in actor_scheduler:
        raise ValueError(
            "cannot normalize time without simulator.actor_scheduler.n_actors"
        )
    return int(actor_scheduler["n_actors"])


__all__ = [
    "configured_n_actors",
    "configured_output_step_labels",
    "load_observation_sampling",
    "observation_step_labels",
    "policy_composition_axis_name",
    "theta_parameter_columns",
]
