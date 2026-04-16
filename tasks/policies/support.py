"""Shared helpers for policy task preparation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from numpy.typing import NDArray

from ..dispatch.runtime import OutputField
from network_creation.core.observability import ObservationSampling
from network_creation.core.runtime import SimulationSettings
from .parameter_space import generate_parameter_space

if TYPE_CHECKING:
    from network_creation.observers.statistics.episode import EpisodeStatisticObserver


type ParameterSpec = Mapping[str, object]
type ParameterConfig = Mapping[str, ParameterSpec]


@dataclass(frozen=True)
class ParameterTable:
    parameter_names: tuple[str, ...]
    parameter_values: NDArray[np.float32]
    column_index_by_name: Mapping[str, int]


@dataclass(frozen=True)
class SimulationSweep:
    parameter_table: ParameterTable
    configuration_names: tuple[str, ...]
    configuration_values: NDArray[np.float32]
    column_index_by_name: Mapping[str, int]
    n_agents_values: tuple[int, ...]
    n_actors: int
    seed: int
    max_steps: int
    n_episodes: int
    stagnation_patience: int | None
    initial_graph_kind: str = "empty"

    @property
    def n_parameter_rows(self) -> int:
        return int(self.parameter_table.parameter_values.shape[0])

    @property
    def total_inputs(self) -> int:
        return len(self.n_agents_values) * self.n_parameter_rows

    def coordinates(self, input_index: int) -> tuple[int, int]:
        return input_index // self.n_parameter_rows, input_index % self.n_parameter_rows

    def configuration_at(self, input_index: int) -> NDArray[np.float32]:
        n_agents_index, parameter_row_index = self.coordinates(input_index)
        return self.configuration_values[n_agents_index, parameter_row_index]

    def resolve(self, configuration: NDArray[np.float32]) -> SimulationSettings:
        n_agents = int(configuration[self.column_index_by_name["n_agents"]])
        return SimulationSettings(
            n_agents=n_agents,
            n_actors=min(self.n_actors, n_agents),
            seed=self.seed,
            max_steps=self.max_steps,
            n_episodes=self.n_episodes,
            stagnation_patience=self.stagnation_patience,
            initial_graph_kind=self.initial_graph_kind,
        )


@dataclass(frozen=True)
class ObserverSpec:
    name: str
    output: str
    observer_class: type[EpisodeStatisticObserver]
    kwargs: dict[str, Any]
    field: OutputField


def load_simulation_sweep(
    raw: Mapping[str, Any],
    parameter_table: ParameterTable,
) -> SimulationSweep:
    initial_graph = raw.get("initial_graph") or {}
    stagnation_patience = raw.get("stagnation_patience")
    raw_n_agents = raw["n_agents"]
    values = raw_n_agents if isinstance(raw_n_agents, list) else [raw_n_agents]
    if not values:
        raise ValueError("simulation.n_agents must contain at least one value")

    n_agents_values: list[int] = []
    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 1:
            field_name = (
                f"simulation.n_agents[{index}]"
                if isinstance(raw_n_agents, list)
                else "simulation.n_agents"
            )
            raise ValueError(f"{field_name} must be a positive integer")
        n_agents_values.append(int(value))

    configuration_names = ("n_agents", *parameter_table.parameter_names)
    configuration_values = np.empty(
        (
            len(n_agents_values),
            int(parameter_table.parameter_values.shape[0]),
            len(configuration_names),
        ),
        dtype=np.float32,
    )
    configuration_values[:, :, 0] = np.asarray(
        n_agents_values,
        dtype=np.float32,
    )[:, None]
    configuration_values[:, :, 1:] = parameter_table.parameter_values[None, :, :]

    return SimulationSweep(
        parameter_table=parameter_table,
        configuration_names=configuration_names,
        configuration_values=configuration_values,
        column_index_by_name={
            name: index for index, name in enumerate(configuration_names)
        },
        n_agents_values=tuple(n_agents_values),
        n_actors=int(raw["actor_scheduler"]["n_actors"]),
        seed=int(raw["seed"]),
        max_steps=int(raw["max_steps"]),
        n_episodes=int(raw["n_episodes"]),
        stagnation_patience=(
            None if stagnation_patience is None else int(stagnation_patience)
        ),
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
    column_index_by_name = {name: index for index, name in enumerate(parameter_names)}

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
    *,
    max_steps: int,
    n_episodes: int,
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
            max_steps=max_steps,
            observation_sampling=observation_sampling,
        )
        if output == "runs":
            field_shape = (n_episodes, *base_shape)
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
