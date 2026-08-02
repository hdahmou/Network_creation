"""Shared interpretation of persisted policy-run metadata."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from network_creation.core.observability import ObservationSampling


@dataclass(frozen=True)
class PolicyOutputSchedule:
    """Population-specific raw and normalized labels for one saved output."""

    output_name: str
    sampling: ObservationSampling
    n_agents: NDArray[np.int64]
    raw_step_horizons: NDArray[np.int64]
    valid_sample_count: NDArray[np.int64]
    valid_raw_step_labels: tuple[NDArray[np.int64], ...]
    storage_raw_step_labels: NDArray[np.int64]
    last_valid_raw_step: NDArray[np.int64]
    realized_activation_opportunities_per_agent: NDArray[np.float64]


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
    raw_step_horizon: int,
) -> dict[str, NDArray[np.int64]]:
    """Return configured labels for callers with one homogeneous raw horizon."""

    return {
        output_name: observation_step_labels(raw_step_horizon, sampling)
        for output_name, sampling in _configured_output_samplings(
            policy_config
        ).items()
    }


def configured_output_sampling(
    output_name: str,
    *,
    policy_config: Any,
) -> ObservationSampling:
    """Load the observation sampling policy for one persisted output name."""

    samplings = _configured_output_samplings(policy_config)
    try:
        return samplings[str(output_name)]
    except KeyError as error:
        raise ValueError(
            f"output {output_name!r} is not configured as a saved observer output"
        ) from error


def derive_policy_output_schedule(
    output_name: str,
    *,
    policy_config: Any,
    n_agents: Any,
    raw_step_horizons: Any,
) -> PolicyOutputSchedule:
    """Derive one output's complete schedule from authoritative store metadata."""

    agents = np.asarray(n_agents, dtype=np.int64)
    horizons = np.asarray(raw_step_horizons, dtype=np.int64)
    if agents.ndim != 1 or agents.shape[0] == 0:
        raise ValueError("n_agents must be a non-empty one-dimensional array")
    if horizons.shape != agents.shape:
        raise ValueError("raw_step_horizons must align exactly with n_agents")
    if np.any(agents <= 0):
        raise ValueError("n_agents must contain only positive values")
    if np.any(horizons < 0):
        raise ValueError("raw_step_horizons must contain only non-negative values")

    sampling = configured_output_sampling(
        output_name,
        policy_config=policy_config,
    )
    valid_labels = tuple(
        observation_step_labels(int(horizon), sampling) for horizon in horizons
    )
    valid_counts = np.asarray(
        [labels.shape[0] for labels in valid_labels],
        dtype=np.int64,
    )
    last_valid_steps = np.asarray(
        [labels[-1] for labels in valid_labels],
        dtype=np.int64,
    )
    n_actors = configured_n_actors(policy_config)
    effective_actors = np.minimum(n_actors, agents).astype(np.float64)

    return PolicyOutputSchedule(
        output_name=str(output_name),
        sampling=sampling,
        n_agents=agents,
        raw_step_horizons=horizons,
        valid_sample_count=valid_counts,
        valid_raw_step_labels=valid_labels,
        storage_raw_step_labels=observation_step_labels(
            int(np.max(horizons)),
            sampling,
        ),
        last_valid_raw_step=last_valid_steps,
        realized_activation_opportunities_per_agent=np.asarray(
            horizons * effective_actors / agents,
            dtype=np.float64,
        ),
    )


def derive_compatible_output_schedule(
    output_names: Sequence[str],
    *,
    policy_config: Any,
    n_agents: Any,
    raw_step_horizons: Any,
) -> PolicyOutputSchedule:
    """Derive the shared schedule required to stack selected saved outputs."""

    names = tuple(str(name) for name in output_names)
    if not names:
        raise ValueError("at least one saved output is required")
    shared = derive_policy_output_schedule(
        names[0],
        policy_config=policy_config,
        n_agents=n_agents,
        raw_step_horizons=raw_step_horizons,
    )
    for name in names[1:]:
        if configured_output_sampling(
            name,
            policy_config=policy_config,
        ) != shared.sampling:
            raise ValueError(
                "selected outputs must share one observation sampling kind and "
                "periodic interval"
            )
    return shared


def _configured_output_samplings(
    policy_config: Any,
) -> dict[str, ObservationSampling]:
    if not isinstance(policy_config, Mapping):
        raise ValueError("policy configuration metadata must be a mapping")
    raw_observers = policy_config.get("observers")
    if not isinstance(raw_observers, Mapping):
        raise ValueError("policy configuration must contain observer metadata")
    defaults = raw_observers.get("defaults")
    if not isinstance(defaults, Mapping):
        raise ValueError("observer defaults metadata must be a mapping")
    default_sampling = defaults.get("sampling")
    outputs = raw_observers.get("outputs")
    if not isinstance(outputs, Sequence):
        raise ValueError("observer outputs metadata must be a sequence")

    samplings: dict[str, ObservationSampling] = {}
    for observer in outputs:
        if not isinstance(observer, Mapping):
            continue
        sampling = load_observation_sampling(
            observer.get("sampling", default_sampling)
        )
        aggregations = observer.get(
            "aggregations",
            defaults.get("aggregations", ()),
        )
        for aggregation in aggregations:
            samplings[f"{observer['name']}.{aggregation}"] = sampling
    return samplings


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
    "PolicyOutputSchedule",
    "configured_n_actors",
    "configured_output_sampling",
    "configured_output_step_labels",
    "derive_compatible_output_schedule",
    "derive_policy_output_schedule",
    "load_observation_sampling",
    "observation_step_labels",
    "policy_composition_axis_name",
    "theta_parameter_columns",
]
