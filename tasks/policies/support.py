"""Shared helpers for policy task preparation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from numpy.typing import NDArray

from ..dispatch.runtime import OutputField
from network_creation.core.observability import ObservationSampling
from network_creation.core.model import SchedulingMode
from .parameter_space import generate_parameter_space

if TYPE_CHECKING:
    from network_creation.core.runtime import SimulationSettings
    from network_creation.observers.statistics.episode import EpisodeStatisticObserver


type ParameterSpec = Mapping[str, object]
type ParameterConfig = Mapping[str, ParameterSpec]


@dataclass(frozen=True)
class ParameterTable:
    parameter_names: tuple[str, ...]
    parameter_values: NDArray[np.float32]
    column_index_by_name: Mapping[str, int]


@dataclass(frozen=True)
class ObserverSpec:
    name: str
    output: str
    observer_class: type[EpisodeStatisticObserver]
    kwargs: dict[str, Any]
    field: OutputField


def load_simulation_settings(raw: Mapping[str, Any]) -> SimulationSettings:
    from network_creation.core.runtime import SimulationSettings

    initial_graph = raw.get("initial_graph") or {}
    stagnation_patience = raw.get("stagnation_patience")

    return SimulationSettings(
        n_agents=int(raw["n_agents"]),
        mode=SchedulingMode(raw.get("mode", SchedulingMode.SIMULTANEOUS.value)),
        seed=int(raw["seed"]),
        max_steps=int(raw["max_steps"]),
        n_episodes=int(raw["n_episodes"]),
        stagnation_patience=None if stagnation_patience is None else int(stagnation_patience),
        initial_graph_kind=initial_graph.get("kind", "empty"),
    )


def generate_parameter_table(input_spec: Mapping[str, object]) -> ParameterTable:
    parameters = cast(ParameterConfig, input_spec["parameters"])
    sampler_seed = int(input_spec["sampler_seed"])
    log2_budget = int(input_spec["log2_budget"])
    if log2_budget < 0:
        raise ValueError("log2_budget must be non-negative")

    sample_budget = 2**log2_budget
    parameter_names, parameter_values = generate_parameter_space(
        parameters,
        sample_budget=sample_budget,
        sampler_seed=sampler_seed,
    )
    column_index_by_name = {
        name: index for index, name in enumerate(parameter_names)
    }

    return ParameterTable(
        parameter_names=parameter_names,
        parameter_values=parameter_values,
        column_index_by_name=column_index_by_name,
    )


def _load_observation_sampling(raw: Mapping[str, object]) -> ObservationSampling:
    kind = str(raw["kind"])
    if kind == "periodic":
        return ObservationSampling.periodic(int(raw["every"]))
    if kind == "terminal":
        return ObservationSampling.terminal()
    raise ValueError(f"unsupported observation sampling kind: {kind}")


def load_observers(
    raw_observers: Sequence[Mapping[str, Any]],
    simulation_settings: SimulationSettings,
) -> tuple[ObserverSpec, ...]:
    from network_creation.observers.grimoire import OBSERVER_GRIMOIRE

    observer_specs: list[ObserverSpec] = []
    for observer in raw_observers:
        name = str(observer["name"])
        output = str(observer["output"])
        observer_class = OBSERVER_GRIMOIRE[name]
        kwargs = dict(observer["kwargs"])
        observation_sampling = _load_observation_sampling(
            cast(Mapping[str, object], kwargs["observation_sampling"])
        )
        kwargs["observation_sampling"] = observation_sampling
        base_shape = observer_class.fixed_output_shape(
            max_steps=simulation_settings.max_steps,
            observation_sampling=observation_sampling,
        )
        if output == "runs":
            field_shape = (simulation_settings.n_episodes, *base_shape)
        else:
            field_shape = base_shape

        observer_specs.append(
            ObserverSpec(
                name=name,
                output=output,
                observer_class=observer_class,
                kwargs=kwargs,
                field=OutputField(
                    name=f"{name}.{output}",
                    shape=field_shape,
                    dtype=observer_class.fixed_output_dtype(),
                ),
            )
        )

    return tuple(observer_specs)
