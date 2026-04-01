"""Shared helpers for task modules."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from numpy.typing import NDArray
from dasklite.base_task import OutputField
from network_creation.core.model import SchedulingMode

if TYPE_CHECKING:
    from network_creation.core.runtime import SimulationSettings
    from network_creation.observers.statistics.episode import EpisodeStatisticObserver


type ParameterRange = Mapping[str, float]
type ParameterConfig = Mapping[str, float | ParameterRange]


@dataclass(frozen=True)
class ParameterTable:
    parameter_names: tuple[str, ...]
    parameter_values: NDArray[np.float32]
    column_index_by_name: Mapping[str, int]


@dataclass(frozen=True)
class ObserverSpec:
    # Grimoire key used in config.
    name: str
    # Aggregation selected from the observer result.
    output: str
    # Concrete observer type to instantiate at run time.
    observer_class: type[EpisodeStatisticObserver]
    # Constructor kwargs passed to the observer.
    kwargs: dict[str, Any]
    # Structured output metadata for this observer/output pair.
    field: OutputField


def load_simulation_settings(raw: Mapping[str, Any]) -> SimulationSettings:
    from network_creation.core.runtime import SimulationSettings

    initial_graph = raw.get("initial_graph") or {}
    stagnation_patience = raw.get("stagnation_patience")
    return SimulationSettings(
        n_agents=int(raw["n_agents"]),
        radius=int(raw["radius"]),
        mode=SchedulingMode(raw.get("mode", SchedulingMode.SIMULTANEOUS.value)),
        seed=int(raw["seed"]),
        max_steps=int(raw["max_steps"]),
        n_episodes=int(raw["n_episodes"]),
        stagnation_patience=None if stagnation_patience is None else int(stagnation_patience),
        initial_graph_kind=initial_graph.get("kind", "empty"),
    )


def generate_parameter_table(
    input_spec: Mapping[str, object],
) -> ParameterTable:
    parameters = cast(ParameterConfig, input_spec["parameters"])
    sampler_seed = int(input_spec["sampler_seed"])
    log2_budget = int(input_spec["log2_budget"])
    if log2_budget < 0:
        raise ValueError("log2_budget must be non-negative")
    resolved_budget = 2**log2_budget
    parameter_names = tuple(parameters)
    column_index_by_name = {
        name: index for index, name in enumerate(parameter_names)
    }

    varying_names = [
        name for name, value in parameters.items() if isinstance(value, Mapping)
    ]

    if not varying_names:
        n_rows = 1 if resolved_budget > 0 else 0
        parameter_values = np.empty(
            (n_rows, len(parameter_names)),
            dtype=np.float32,
        )
        if n_rows == 1:
            parameter_values[0] = np.asarray(
                [float(cast(float, parameters[name])) for name in parameter_names],
                dtype=np.float32,
            )
        return ParameterTable(
            parameter_names=parameter_names,
            parameter_values=parameter_values,
            column_index_by_name=column_index_by_name,
        )

    from scipy.stats import qmc

    sampler = qmc.Sobol(
        d=len(varying_names),
        scramble=True,
        optimization="random-cd",
        rng=sampler_seed,
    )
    unit_samples = np.asarray(
        sampler.random(n=resolved_budget),
        dtype=np.float32,
    )

    parameter_values = np.empty(
        (resolved_budget, len(parameter_names)),
        dtype=np.float32,
    )
    for row_index in range(resolved_budget):
        unit_index = 0
        for column_index, name in enumerate(parameter_names):
            value = parameters[name]
            if isinstance(value, Mapping):
                minimum = float(value["min"])
                maximum = float(value["max"])
                parameter_values[row_index, column_index] = minimum + (
                    unit_samples[row_index, unit_index] * (maximum - minimum)
                )
                unit_index += 1
            else:
                parameter_values[row_index, column_index] = float(value)

    return ParameterTable(
        parameter_names=parameter_names,
        parameter_values=parameter_values,
        column_index_by_name=column_index_by_name,
    )


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
        observation_frequency = int(kwargs["observation_frequency"])
        base_shape = observer_class.fixed_output_shape(
            max_steps=simulation_settings.max_steps,
            observation_frequency=observation_frequency,
        )
        observer_specs.append(
            ObserverSpec(
                name=name,
                output=output,
                observer_class=observer_class,
                kwargs=kwargs,
                field=OutputField(
                    name=f"{name}.{output}",
                    shape=(
                        (simulation_settings.n_episodes, *base_shape)
                        if output == "runs"
                        else base_shape
                    ),
                    dtype=observer_class.fixed_output_dtype(),
                ),
            )
        )
    return tuple(observer_specs)
